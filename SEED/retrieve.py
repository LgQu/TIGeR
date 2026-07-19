import hydra

import gc
import os, sys
import torch
from collections import OrderedDict
import numpy as np
import math
import torch.nn.functional as F
import time

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import LogitsProcessorList, set_seed, StoppingCriteriaList
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import logging
import argparse
from datasets import load_dataset
import copy
from tenacity import retry, retry_if_exception_type, stop_after_attempt
import socket

from tiger_utils import EarlyStoppingCriteria, Trie
from tiger_utils import UnconditionalDebiasLogitsProcessor
from ranking import get_img_ids
from tiger_utils import check_device, my_green, my_red

from data import get_combined_query_gallery

HOSTNAME = socket.gethostname()

uncond_context_len = None

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKENS = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

s_token = "USER:"
e_token = "ASSISTANT:"
sep = "\n"

device = "cuda"




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default='retrieval.log')
    parser.add_argument("--wait_gpu", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--bs_ranking", type=int, default=1)
    parser.add_argument("--num_beam_groups", type=int, default=1) # default 10 by haochuan
    parser.add_argument("--use_uncond", action='store_true')
    parser.add_argument("--uncond_path_prefix", default='./intermediate_results/uncond_logits_can_you_')
    parser.add_argument("-slp", "--uncond_sumlogprobs_path_prefix", default='./intermediate_results/sumlogprobs_can_you_')
    parser.add_argument("--uncond_names", default=[], nargs='+', help="input uncond precomputed beam logits")
    parser.add_argument("--uncond_version", default='v1')
    parser.add_argument("--prefix_prompt", default='Generate an image of ')
    parser.add_argument("--use_cache", action='store_true')
    parser.add_argument("--data_version", const="", nargs="?", default='tiger_bench')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument('--datasets_query', type=str, nargs='+', default=[])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', default=[])
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[])
    parser.add_argument('--nums_selected_txt_query', type=int, nargs='+', default=[])
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--model_vers", type=str, default='8b', choices=['8b', '14b'])
    parser.add_argument("--expand_prompt_v", type=str, default=None)

    parser.add_argument("--out", default=None)
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument('--diffusiondb_split', type=str, default='2m_random_1k')
    parser.add_argument('--only_parse_res', action='store_true')
    parser.add_argument("--turn", default='single_turn', help='especially for VIST')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)
    args = parser.parse_args()
    print(args)

    args.use_uncond = True

    assert args.uncond_version == 'v1'
    return args
    
def get_txt_len(input_ids, tokenizer):
    bs = input_ids.shape[0]
    text_length = None
    boi_list = torch.where(input_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])
    if len(boi_list) == 0:
        raise ValueError("No Image Token Detected!!!")
    else:
        boi_index = boi_list[1][0]
        seq_length = input_ids.shape[-1]
        text_length = boi_index
    return text_length

def compute_rank_t2i_v1(gen_img_ids, imgid2idx, gt_img_idx):
    rank = 1e20
    ranking = []
    for imgid in gen_img_ids:
        key = '_'.join([str(ii) for ii in imgid.tolist()])
        ranking.append(imgid2idx[key])
    ranking = np.array(ranking)
    tmp = np.where(ranking == gt_img_idx)[0]
    if len(tmp) > 0:
        rank = tmp[0]
    return rank, ranking

def compute_recalls(ranks):
    if len(ranks) == 0:
        return 0, 0, 0
    ranks = np.array(ranks)
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return tr1, tr5, tr10

def load_model(model_vers='8b', load_llm=True):
    tokenizer_cfg_path = 'configs/tokenizer.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True, load_image_tokenizer=False)
    transform_cfg_path = 'configs/transform.yaml'
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)
    model = None
    if load_llm:
        if model_vers == '8b':
            model_cfg = OmegaConf.load('configs/seed_llama_8b.yaml')
        else:
            model_cfg = OmegaConf.load('configs/seed_llama_14b.yaml')
        model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16, device_map='auto')
        model = model.eval() #.to(device)
    return tokenizer, transform, model

def filter_context_images(prompt, img, tokenizer):
    prompt = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
    boi_list = torch.where(prompt == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[-1]
    eoi_list = torch.where(prompt == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[-1]
    context_img = []
    filtered_img = []
    for b, e in zip(boi_list, eoi_list):
        context_img.append(prompt[0][b+1:e].tolist())
    for im in img:
        if im not in context_img:
            filtered_img.append(im)
    
    print("Filtered:", len(filtered_img), "origin:", len(img))
    return filtered_img

@torch.no_grad()
def generate(args, tokenizer, input_ids, generation_config, model, debias_logits_processor, trie, early_stopping, uncond_sumlogprobs):
    input_ids = input_ids.to(device)
    if args.use_uncond:
        generate_ids = model.generate(
            input_ids=input_ids,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            logits_processor = debias_logits_processor, 
            stopping_criteria = StoppingCriteriaList([early_stopping]),
            **generation_config
        )
    else:
        generate_ids = model.generate(
            input_ids=input_ids,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            **generation_config
        )
    
    generate_ids = re_ranking(generate_ids, model, trie, args.bs_ranking, input_ids.shape[1], uncond_sumlogprobs, tokenizer)

    generate_ids = generate_ids[:,input_ids.shape[1]-1:]
    generate_ids = torch.concat([generate_ids, tokenizer(EOI_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids.expand(generate_ids.shape[0],-1).to(device)], dim=-1)
    return generate_ids

@torch.no_grad()
def re_ranking(generate_ids, model, trie, batch_size, input_len, uncond_sumlogprobs, tokenizer):
    if generate_ids.shape[1] - 1 - input_len == NUM_IMG_TOKENS: # -1 because the last id is always 1 
        return generate_ids
    print(f'Eearly stop at step {generate_ids.shape[1] - 1 - input_len}') # -1 because the last id is always 1 
    uncond_imgid2sumlogprobs = uncond_sumlogprobs
    generate_ids_comp = [] # complete
    for gen_id in generate_ids[:, :-1]: # -1 because the last id is always 1
        prefix_seq = gen_id.tolist()
        while trie.get(prefix_seq):
            next_token = trie.get(prefix_seq)
            next_token = next_token if len(next_token) == 1 else [next_token[0]]
            prefix_seq += next_token
        generate_ids_comp.append(prefix_seq)

    generate_ids = torch.as_tensor(generate_ids_comp, device=device)
    sum_log_prob_all = []
    
    progress_bar = tqdm(range(0, len(generate_ids), batch_size), 
                    total=math.ceil(len(generate_ids) / batch_size), 
                    leave=False, 
                    desc='Re-rank')
    if 'hopper' in HOSTNAME:
        progress_bar.close()
        
    for i in progress_bar:
        gen_ids_batch = generate_ids[i:i+batch_size]
        idx = get_txt_len(gen_ids_batch, tokenizer)
        gen_ids_img = gen_ids_batch[:,input_len:] # gt
        assert gen_ids_img.shape[1] == NUM_IMG_TOKENS # 32
        logits = model(input_ids=gen_ids_batch).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        log_probs = log_probs[:,idx:idx+NUM_IMG_TOKENS,:]
        gen_ids_img_us = gen_ids_img.unsqueeze(-1) # (bs, 32, 1)
        p_all = torch.gather(log_probs, -1, gen_ids_img_us).squeeze(-1) # (bs, 32)
        sum_log_prob = p_all.sum(dim=-1) # (bs, )
        step = gen_ids_img.shape[1] - 1
        uncond_scores = []
        for local_beam_idx in range(len(gen_ids_img)):
            ids_str = '_'.join([str(i) for i in gen_ids_img[local_beam_idx][:step].tolist()])
            uncond_scores.append(uncond_imgid2sumlogprobs[ids_str])
        uncond_scores = torch.tensor(uncond_scores, dtype=log_probs.dtype)
        sum_log_prob_debias = sum_log_prob.cpu() - uncond_scores
        sum_log_prob_all.append(sum_log_prob_debias)

    sum_log_prob_all = torch.cat(sum_log_prob_all)
    sort_idx = torch.argsort(sum_log_prob_all, descending=True)
    generate_ids = generate_ids[sort_idx]
    return generate_ids


def retrieval(args, generation_config):
    sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                    args.nums_selected_txt_query, args.nums_selected_gallery, args.prefix_prompt, 
                                                                    args=args)
    n_local = len(sampled_text)
    start = 0
    if args.n_proc > 1:
        print(f"Number of processes: {args.n_proc}")
        print(f"Process ID: {args.proc_id}")
        block_size = math.ceil(len(sampled_text) / args.n_proc)
        indices = list(range(0, len(sampled_text)+block_size, block_size))
        assert len(indices) == args.n_proc + 1
        start, end = indices[args.proc_id], indices[args.proc_id+1]
        end = start + len(sampled_text[start:end]) # last block
        n_local = end - start
        print(f"Number of total data points: {len(sampled_text)}")
        print(f"Number of local data points: {end - start}")
    
    cur_saved = 0
    is_load_model = True
    if os.path.exists(args.out):
        cur_all_ranking = torch.load(args.out)
        cur_saved = len(cur_all_ranking)
        print('continue from i_local: ', cur_saved)
        if cur_saved == n_local:
            is_load_model = False
    
    if is_load_model:
        args.unconds = []
        args.uncond_sumlogprobs = []
        for i, un in enumerate(args.uncond_names):
            args.unconds.append(args.uncond_path_prefix + un + '_' + str(args.nums_selected_gallery[i]) + '.pt')
            args.uncond_sumlogprobs.append(args.uncond_sumlogprobs_path_prefix + un + '_' + str(args.nums_selected_gallery[i]) + '.pt')
        sampled_img_ids = []
        all_uncond = []
        all_uncond_sumlogprobs = []
        for i, d_g in enumerate(args.datasets_gallery):
            ids = get_img_ids(d_g, args.nums_selected_gallery[i], args)
            sampled_img_ids.append(ids)
            all_uncond.append(torch.load(args.unconds[i]))
            all_uncond_sumlogprobs.append(torch.load(args.uncond_sumlogprobs[i]))
        sampled_img_ids = torch.cat(sampled_img_ids, dim=0)
        print(f"All, sampled_img_ids.shape: {sampled_img_ids.shape}")
        uncond_results = OrderedDict()
        for i in range(len(all_uncond[0])): # i means the seq length
            uncond_results[i] = OrderedDict()
            for unc in all_uncond:
                uncond_results[i].update(unc[i])
        uncond_sumlogprobs = OrderedDict()
        for unc in all_uncond_sumlogprobs:
            uncond_sumlogprobs.update(unc)    

    if is_load_model:
        check_device(args.wait_gpu) 
        tokenizer, transform, model = load_model(args.model_vers)

        img_ids_const = [[(IMG_TOKEN.format(it)) for it in item] for item in sampled_img_ids]
        img_ids_lst = [[tokenizer(im, add_special_tokens=False).input_ids[0] for im in img] for img in img_ids_const]
        imgid2idx = OrderedDict()
        for idx, imgid_lst in enumerate(img_ids_lst):
            key = '_'.join([str(ii) for ii in imgid_lst])
            imgid2idx[key] = idx
        
        print('device: ', device)
        img_ids_tensor = torch.tensor(img_ids_lst, dtype=torch.long, device=device)
        print("img_ids_tensor.shape: ", img_ids_tensor.shape)
        print(f"#query: {len(sampled_text)}, #gallery: {img_ids_tensor.shape[0]}")
    else:
        tokenizer, _, _ = load_model(args.model_vers, load_llm=False)

    all_ranks = []
    all_valid_ranks = []
    all_ranking = []
    results_dict = []
    
    for s in range(1):
        set_seed(s)
        pbar = tqdm(range(n_local), desc="per text", total=n_local)
        for i_local in pbar:
            i_global = i_local + start
            prompt = sampled_text[i_global]
            assert len(txt2img[i_global]) == 1
            gt_img_idx = txt2img[i_global][0]
            prompt_cut = ' '.join(prompt.split(' ')[:args.max_input_length])
            len_prompt_cut = len(prompt_cut.split(' '))
            
            if i_local < cur_saved:
                ranking = cur_all_ranking[i_local]
                rank = 1e20
                tmp = np.where(ranking == gt_img_idx)[0]
                if len(tmp) > 0:
                    rank = tmp[0]
            else:
                input_tokens = tokenizer.bos_token  + s_token + " " + prompt_cut + sep + e_token + BOI_TOKEN
                print(f'len(prompt_cut) = {len_prompt_cut}, input_tokens: {input_tokens}')
                input_prompt_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
                input_prompt_ids_cut = \
                    torch.cat([input_prompt_ids[:, :args.max_input_length], input_prompt_ids[:, -args.max_input_length:]], dim=1) \
                    if input_prompt_ids.shape[1] > args.max_input_length * 2 else input_prompt_ids
                if input_prompt_ids_cut.shape[1] != input_prompt_ids.shape[1]:
                    print(f'len(input_prompt_ids) = {input_prompt_ids.shape[1]}, after cutted: {input_prompt_ids_cut.shape[1]}')

                if len(args.datasets_query) == len(args.datasets_gallery) == 1 and args.datasets_query[0] == 'vist':
                    img_ids_lst_filtered = filter_context_images(prompt, img_ids_lst, tokenizer)
                    img_ids_seq = [input_prompt_ids_cut[0].tolist() + x for x in img_ids_lst_filtered]
                else:
                    img_ids_seq = [input_prompt_ids_cut[0].tolist() + x for x in img_ids_lst]
                trie = Trie(img_ids_seq)
                context_len = len(img_ids_seq[0]) - 32 

                def prepare_debias_and_early_stopping(num_beams):
                    debias_proc = UnconditionalDebiasLogitsProcessor(uncond_results, context_len, uncond_context_len, \
                                lambda batch_id, sent: trie.get(sent.tolist()), num_beams)
                    logits_proc_list = LogitsProcessorList([debias_proc])
                    early_stopping = EarlyStoppingCriteria(lambda batch_id, sent: trie.get(sent.tolist()), num_beams, trie)
                    return logits_proc_list, early_stopping
                
                logits_proc_list, early_stopping = prepare_debias_and_early_stopping(args.num_beams)

                def halve_beam(prepare_debias_and_early_stopping):
                    def _set_parameter(retry_state):
                        print(retry_state.kwargs)
                        print(retry_state.kwargs.keys())
                        generation_config_new = copy.deepcopy(retry_state.kwargs['generation_config'])
                        half_beam  = int(generation_config_new['num_beams'] / 2)
                        print('torch.cuda.OutOfMemoryError raised. ')
                        print(f'Retry with half num_beams = {half_beam}. ')
                        generation_config_new['num_beams'] = half_beam
                        generation_config_new['num_return_sequences'] = half_beam
                        retry_state.kwargs['generation_config'] = generation_config_new
                        retry_state.kwargs['logits_proc_list'], retry_state.kwargs['early_stopping'] = \
                                prepare_debias_and_early_stopping(half_beam)
                    return _set_parameter

                @retry(
                    retry=retry_if_exception_type(torch.cuda.OutOfMemoryError),
                    stop=stop_after_attempt(7),
                    after=halve_beam(prepare_debias_and_early_stopping))
                def retry_gen(generation_config, logits_proc_list, early_stopping):
                    generate_ids = generate(args, tokenizer, input_prompt_ids_cut, generation_config, model, logits_proc_list, \
                                            trie, early_stopping, uncond_sumlogprobs)
                    return generate_ids
                
                generate_ids = retry_gen(generation_config=generation_config, logits_proc_list=logits_proc_list, early_stopping=early_stopping)
                
                gen_img_ids = generate_ids[:,1:-1]


                rank, ranking = compute_rank_t2i_v1(gen_img_ids, imgid2idx, gt_img_idx=gt_img_idx)


            all_ranking.append(ranking)
            min_gid, max_gid = query2min_max_gallery[i_global]
            ranking_print = []
            for img_idx in ranking:
                if min_gid <= img_idx <= max_gid:
                    if img_idx == gt_img_idx:
                        ranking_print.append(my_green(str(img_idx)))
                    else:
                        ranking_print.append(str(img_idx))
                else:
                    ranking_print.append(my_red(str(img_idx)))
                
            is_in = [1 if min_gid <= r <= max_gid else 0 for r in ranking]
            print('\n', f'gt: {gt_img_idx}, r: {my_green(str(rank))}, %out: {(1 - sum(is_in)/len(is_in))*100:.2f}, ' + \
                    'ranking: ' + ', '.join(ranking_print[:20]))

            all_ranks.append(rank)
            if gt_img_idx != -1:
                all_valid_ranks.append(rank)
            r1, r5, r10 = compute_recalls(all_valid_ranks)
            pbar.set_postfix_str(f'r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}, #beam={args.num_beams}')

            results_dict.append({
                'prompt': tokenizer.bos_token  + s_token + " " + prompt + sep + e_token + BOI_TOKEN, 
                'ranking_list': ranking[:20], 
                'rank': rank
            })

            if (i_local + 1) % args.save_interval == 0 and args.out is not None and i_local >= cur_saved:
                print(f'Save {args.out} with i_local {i_local}.')
                torch.save(all_ranking, args.out)

    cumsum_nq = np.cumsum([0] + args.nums_selected_txt_query)
    for i_dataset in range(len(cumsum_nq) - 1):
        vr = all_valid_ranks[cumsum_nq[i_dataset]:cumsum_nq[i_dataset + 1]]
        r1, r5, r10 = compute_recalls(vr)
        print(f'dataset: {args.datasets_query[i_dataset]}, ' + \
            f'r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}, #beam={args.num_beams}')

    if i_local >= cur_saved:
        torch.save(all_ranking, args.out)
        print(f'Save {args.out}.')

    torch.save(results_dict, args.out.replace('retr', 'results_dict'))

    print(args)

def parse_res(args):
    sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                    args.nums_selected_txt_query, args.nums_selected_gallery, args.prefix_prompt, 
                                                                    args=args)
    all_ranking = torch.load(args.out)
    assert len(sampled_text) == len(all_ranking)
    print('len(all_ranking): ', len(all_ranking))
    all_ranks = []
    all_valid_ranks = []
    pbar = tqdm(enumerate(all_ranking), total=len(all_ranking))
    for i, cur_ranking in pbar:
        rank = 1e20
        gt_img_idx = txt2img[i][0]
        tmp = np.where(cur_ranking == gt_img_idx)[0]
        if len(tmp) > 0:
            rank = tmp[0]
        all_ranks.append(rank)
        if gt_img_idx != -1:
            all_valid_ranks.append(rank)
        r1, r5, r10 = compute_recalls(all_valid_ranks)
        pbar.set_postfix_str(f'r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}, #beam={args.num_beams}')

if __name__ == '__main__':
    args = parse_args()
    log_dir = os.path.join(os.path.dirname(args.out), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,  
        filename=os.path.join(log_dir, args.log),  
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  
    )
    generation_config = {
        'temperature': 1.0,
        'num_beams': args.num_beams,
        'max_new_tokens': 32,
        'num_return_sequences': args.num_beams,
        'use_cache': args.use_cache
    }

    if args.only_parse_res:
        parse_res(args)
    else:
        retrieval(args, generation_config)





