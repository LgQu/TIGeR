import hydra
import gc
import os
import torch
from collections import OrderedDict
import numpy as np
import math
import torch.nn.functional as F
import random
import time
import multiprocessing
import socket

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from transformers import LogitsProcessorList, set_seed, StoppingCriteriaList
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import logging
import argparse
from datasets import load_dataset
import copy
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from models import build_model
from stopping_criteria import EarlyStoppingCriteria
from tiger_utils import Trie, my_red, my_green
from tiger_utils import DebiasLogitsProcessor
from tiger_utils import check_device, my_green, my_red

from data import get_combined_query_gallery

HOSTNAME = socket.gethostname()

model_path = os.environ.get('LAVIT_MODEL')
if not model_path:
    raise RuntimeError('Set LAVIT_MODEL to the LaVIT checkpoint directory.')
model_dtype = 'bf16'

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device_id = 'auto'
device = torch.device('cuda')
torch_dtype = torch.bfloat16 if model_dtype == 'bf16' else torch.float16

ratio_dict = {
        '1:1' : (1024, 1024),
        '4:3' : (896, 1152),
        '3:2' : (832, 1216),
        '16:9' : (768, 1344),
        '2:3' : (1216, 832),
        '3:4' : (1152, 896),
    }

ratio = '1:1'
height, width = ratio_dict[ratio]
BOI, EOI = 32000, 32001
PAD = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--bs_ranking", type=int, default=1)
    parser.add_argument("--num_beam_groups", type=int, default=1) # default 10 by haochuan
    parser.add_argument("--disable_uncond", action='store_true')
    parser.add_argument("--uncond_path_prefix", default='./intermediate_results/uncond_logits_can_you_')
    parser.add_argument("-slp", "--uncond_sumlogprobs_path_prefix", default='./intermediate_results/sumlogprobs_can_you_')
    parser.add_argument("--uncond_names", default=[], nargs='+', help="input uncond precomputed beam logits")
    parser.add_argument("--uncond_version", default='v1')
    parser.add_argument("--prefix_prompt", default='Generate an image of ')
    parser.add_argument("--use_cache", action='store_true')
    
    parser.add_argument('--datasets_query', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument('--nums_selected_txt_query', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument("--data_version", nargs='?', const="", default='combV3')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default='../SEED/configs/tiger/data.yaml')
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--img_id_path", required=True, default=None)
    parser.add_argument("--num_img_tokens", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--out", default=None)
    parser.add_argument('--diffusiondb_split', type=str, default='2m_random_1k')
    parser.add_argument('--only_parse_res', action='store_true')
    parser.add_argument("--turn", default='single_turn', help='especially for VIST')
    parser.add_argument("--expand_prompt_v", type=str, default=None)
    parser.add_argument('--share_mem', action='store_true')

    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)
    parser.add_argument("--wait_gpu", type=int, default=None)

    parser.add_argument('--uncond_log_prob', type=str, default=None)
    args = parser.parse_args()
    print(args)

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
        key = '_'.join([str(ii) for ii in imgid])
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

def load_model():
    model = build_model(model_path=model_path, model_dtype=model_dtype, check_safety=False, load_tokenizer=False,
                        device_id=device_id, use_xformers=True, understanding=False, local_files_only=True, 
                        load_quantizer=False, load_pixel_decoder=False)
    model.eval()
    tokenizer = model.llama_tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    llama = model.llama_model
    return tokenizer, llama

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

def complement_partial(x):
    gen_ids, trie, EOI = x
    prefix_seq = gen_ids
    while trie.get(prefix_seq):
        next_token = trie.get(prefix_seq)
        assert len(next_token) == 1
        prefix_seq += next_token
    assert prefix_seq[-1] == EOI, f'invalid seq: {prefix_seq}'
    return prefix_seq

def load_from_shared_mem(path):
    info = torch.load(path)
    ts = torch.BFloat16Tensor(torch.BFloat16Storage._new_shared_filename(*info[0]))
    ts = ts.view(info[1])
    return ts


@torch.no_grad()
def generate(args, tokenizer, input_ids, generation_config, model, debias_logits_processor, trie, early_stopping, uncond_sumlogprobs):
    input_ids = input_ids.to(device)
    if args.disable_uncond:
        raise NotImplementedError
        generate_ids = model.generate(
            input_ids=input_ids,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            **generation_config
        )
    else:
        generate_ids = model.generate(
            input_ids=input_ids,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            logits_processor = debias_logits_processor, 
            stopping_criteria = StoppingCriteriaList([early_stopping]),
            pad_token_id=tokenizer.pad_token_id,
            **generation_config
        )

    generate_ids = re_ranking(generate_ids, model, trie, args.bs_ranking, input_ids.shape[1], uncond_sumlogprobs, tokenizer)
    return generate_ids

@torch.no_grad()
def re_ranking(generate_ids, model, trie, batch_size, input_len, uncond_sumlogprobs, tokenizer):

    generate_ids_wo_pad = []
    seq_len_when_stop = []
    for seq in generate_ids:
        indices_pad = torch.where(seq == 2)[0]
        if len(indices_pad) == 0:
            assert seq[-1] == EOI, f'invalid seq: {seq}'
            seq_len_when_stop.append(len(seq) - input_len)
            generate_ids_wo_pad.append(seq)
        else:
            seq_len_when_stop.append(indices_pad[0].item() - input_len)
            generate_ids_wo_pad.append(seq[:indices_pad[0]])
    seq_len_when_stop = np.array(seq_len_when_stop)
    print("Generated sequence length when stopping, "\
            f"mean: {seq_len_when_stop.mean()}, std: {seq_len_when_stop.std()}, "\
            f"min: {seq_len_when_stop.min()}, max: {seq_len_when_stop.max()}")\

    uncond_imgid2sumlogprobs = uncond_sumlogprobs

    generate_ids_comp = [] # completed
    for gen_ids in generate_ids_wo_pad: 
        prefix_seq = gen_ids.tolist()
        while trie.get(prefix_seq):
            next_token = trie.get(prefix_seq)
            assert len(next_token) == 1
            prefix_seq += next_token
        assert prefix_seq[-1] == EOI, f'invalid seq: {prefix_seq}'
        prefix_seq = torch.as_tensor(prefix_seq, device=device)
        generate_ids_comp.append(prefix_seq)


    generate_ids = pad_sequence(generate_ids_comp, padding_value=PAD, batch_first=True) 
    img_ids = generate_ids[:, input_len - 1:]
    assert img_ids[0][0].item() == BOI
    img_ids_wo_spec = [] # without specical tokens, BOI, EOI
    sim_t2i = torch.zeros(len(generate_ids), dtype=torch.float32, device=device)
    
    progress_bar = tqdm(range(0, len(generate_ids), batch_size), 
                    total=math.ceil(len(generate_ids)/batch_size), 
                    leave=False, 
                    desc='Re-rank')
    if 'hopper' in HOSTNAME:
        progress_bar.close()

    for s in progress_bar:
        e = min(s + batch_size, len(generate_ids))
        x = generate_ids[s:e].to(device)
        bs_cur = x.shape[0]
        boi_index = torch.where(x == BOI)[1]
        eoi_index = torch.where(x == EOI)[1]
        x = x[:, :torch.max(eoi_index) + 1]

        if args.num_img_tokens == 0: # dynamic tokens
            img_token_idx = []
            for i in range(s, e):
                pad_token_indices = torch.where(img_ids[i] == PAD)[0]
                if len(pad_token_indices) != 0:
                    pad_token_idx = pad_token_indices[0].item()
                    img_token_idx.append(img_ids[i][1:pad_token_idx-1])
                else:
                    img_token_idx.append(img_ids[i][1:-1])

            img_ids_wo_spec.extend(img_token_idx)
            logits = model(input_ids=x).logits
            assert len(boi_index) == len(logits)
            sum_log_prob = []
            for i in range(len(boi_index)):
                assert len(img_token_idx[i]) == eoi_index[i] - boi_index[i] - 1
                cur_logit = logits[i, boi_index[i]:eoi_index[i]-1, :] # (cur_seq_len, n_vocab)
                assert not torch.any(torch.isnan(cur_logit))
                log_prob = F.log_softmax(cur_logit.float(), dim=-1) # (cur_seq_len, n_vocab)
                p_per_token = torch.gather(log_prob, -1, img_token_idx[i].unsqueeze(-1)).squeeze(-1) # (cur_seq_len)
                sum_log_prob.append(p_per_token.sum())

            sum_log_prob = torch.stack(sum_log_prob)
            sim_t2i[s:e] = sum_log_prob
        else:
            raise NotImplementedError

    uncond_scores = []
    for local_beam_idx in range(len(img_ids_wo_spec)):
        ids_str = '_'.join([str(i) for i in img_ids_wo_spec[local_beam_idx][:-1].tolist()])
        uncond_scores.append(uncond_imgid2sumlogprobs[ids_str])
    uncond_scores = torch.tensor(uncond_scores, dtype=sim_t2i.dtype, device=sim_t2i.device)
    sim_t2i_debias = sim_t2i - uncond_scores
    sort_idx = torch.argsort(sim_t2i_debias, descending=True)
    generate_ids_comp = [generate_ids_comp[si.item()].tolist() for si in sort_idx]

    return generate_ids_comp


def retrieval(args, generation_config):
    sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                    args.nums_selected_txt_query, args.nums_selected_gallery, args.prefix_prompt, 
                                                                    args=args)

    cur_saved = 0
    if os.path.exists(args.out):
        cur_all_ranking = torch.load(args.out)
        cur_saved = len(cur_all_ranking)
        print('continue from i_local: ', cur_saved)

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

    if cur_saved < n_local:
        print(f'Load {args.uncond_log_prob} ...')
        if not args.share_mem: 
            uncond_results = torch.load(args.uncond_log_prob)
            uncond_sumlogprobs = torch.load(args.uncond_log_prob.replace('imgid2logprob', 'imgid2sumlogprob_' + args.data_version))
        else:
            uncond_ts = load_from_shared_mem(args.uncond_log_prob)
            print('uncond_log_prob.shape: ', uncond_ts.shape)
            print('uncond_ts[0]: ', uncond_ts[0])
            uncond_imgid2tensoridx = torch.load(args.uncond_log_prob.replace('shared_mem_tensor_logprob', 'imgid2tensoridx'))
            uncond_results = {'tensor': uncond_ts, 'imgid2tensoridx': uncond_imgid2tensoridx}
            uncond_sumlogprobs = torch.load(args.uncond_log_prob.replace('shared_mem_tensor_logprob', 'imgid2sumlogprob'))
        raw_image_ids = torch.load(args.img_id_path)
        print(f"All, len(raw_image_ids): {len(raw_image_ids)}")
        if NUM_IMG_TOKENS == 0: # means use original img_ids which has different length
            print("Loading Original Img Ids...")
            image_special = torch.as_tensor([BOI, EOI], dtype=torch.long)
            image_ids = [torch.cat([image_special[:1], i, image_special[1:]]) for i in raw_image_ids]
        else:
            raise NotImplementedError
            if 'top' in args.img_id_path:
                assert str(NUM_IMG_TOKENS) in args.img_id_path
            if isinstance(raw_image_ids, list): # when topk == 256
                image_ids = torch.stack(raw_image_ids)
                tokens_special = torch.as_tensor([BOI, EOI], dtype=image_ids.dtype, device=image_ids.device)
                tokens_special = tokens_special.unsqueeze(0).expand(len(image_ids), -1)
                image_ids = torch.cat([tokens_special[:, [0]], image_ids, tokens_special[:, [1]]], dim=1)

        imgid2idx = OrderedDict()
        for idx, cur_img_ids in enumerate(raw_image_ids):
            key = '_'.join([str(ii) for ii in cur_img_ids.tolist()])
            assert key not in imgid2idx
            imgid2idx[key] = idx

        print(f"#query: {len(sampled_text)}, #gallery: {len(image_ids)}")
        img_ids_lst = image_ids

    if cur_saved < n_local:
        check_device(args.wait_gpu) 
        tokenizer, model = load_model()

    all_ranks = []
    all_valid_ranks = []
    all_ranking = []
    results_dict = []

    pbar = tqdm(range(n_local), desc="per text", total=n_local)
    for i_local in pbar:
        i_global = i_local + start
        prompt = sampled_text[i_global]
        assert len(txt2img[i_global]) == 1
        gt_img_idx = txt2img[i_global][0]
        prompt_cut = ' '.join(prompt.split(' ')[:args.max_input_length])
        len_prompt_cut = len(prompt_cut.split(' '))
        print(f'len(prompt_cut) = {len_prompt_cut}, prompt: {prompt_cut}')
        
        if i_local < cur_saved:
            ranking = cur_all_ranking[i_local]
            rank = 1e20
            tmp = np.where(ranking == gt_img_idx)[0]
            if len(tmp) > 0:
                rank = tmp[0]
        else:
            input_prompt_ids = tokenizer(tokenizer.bos_token + prompt_cut, add_special_tokens=False, return_tensors='pt', padding='longest').input_ids
            input_prompt_ids_cut = \
                torch.cat([input_prompt_ids[:, :args.max_input_length], input_prompt_ids[:, -args.max_input_length:]], dim=1) \
                if input_prompt_ids.shape[1] > args.max_input_length * 2 else input_prompt_ids
            if input_prompt_ids_cut.shape[1] != input_prompt_ids.shape[1]:
                print(f'len(input_prompt_ids) = {input_prompt_ids.shape[1]}, after cutted: {input_prompt_ids_cut.shape[1]}')

            if len(args.datasets_query) == len(args.datasets_gallery) == 1 and args.datasets_query[0] == 'vist':
                raise NotImplementedError
                img_ids_lst_filtered = filter_context_images(prompt, img_ids_lst, tokenizer)
                img_ids_seq = [input_prompt_ids_cut[0].tolist() + x for x in img_ids_lst_filtered]
            else:
                img_ids_seq = [input_prompt_ids_cut[0].tolist() + x.tolist() for x in img_ids_lst] # img_ids_lst: [BOI] ... [EOI]

            trie = Trie(img_ids_seq)
            boi_id = torch.as_tensor([BOI], dtype=input_prompt_ids_cut.dtype, device=input_prompt_ids_cut.device)
            input_prompt_ids_cut = torch.cat([input_prompt_ids_cut, 
                                    boi_id.unsqueeze(0).expand(input_prompt_ids_cut.shape[0], -1)], dim=-1)
            context_len = input_prompt_ids_cut.shape[-1] # + 1 # + 1 means BOI


            def prepare_debias_and_early_stopping(num_beams):
                debias_proc = DebiasLogitsProcessor(uncond_results, context_len, \
                            lambda batch_id, sent: trie.get(sent.tolist()), num_beams, share_mem=args.share_mem)
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
                stop=stop_after_attempt(4),
                after=halve_beam(prepare_debias_and_early_stopping))
            def retry_gen(generation_config, logits_proc_list, early_stopping):
                generate_ids = generate(args, tokenizer, input_prompt_ids_cut, generation_config, model, logits_proc_list, \
                                        trie, early_stopping, uncond_sumlogprobs)
                return generate_ids
            
            generate_ids = retry_gen(generation_config=generation_config, logits_proc_list=logits_proc_list, early_stopping=early_stopping)
            
            context_len = input_prompt_ids_cut.shape[-1]
            generate_img_ids = []
            for i in range(len(generate_ids)):       
                generate_img_ids.append(generate_ids[i][context_len:-1])

            rank, ranking = compute_rank_t2i_v1(generate_img_ids, imgid2idx, gt_img_idx=gt_img_idx)

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

        is_in = [1 if min_gid <= r <= max_gid else 0 for r in ranking[:20]]
        print('\n', f'gt: {gt_img_idx}, r: {my_green(str(rank))}, %out: {(1 - sum(is_in)/len(is_in))*100:.2f}, ' + \
                'ranking: ' + ', '.join(ranking_print[:20]))

        all_ranks.append(rank)
        if gt_img_idx != -1:
            all_valid_ranks.append(rank)
        r1, r5, r10 = compute_recalls(all_valid_ranks)
        pbar.set_postfix_str(f'r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}, #beam={args.num_beams}')

        results_dict.append({
            'prompt': prompt, 
            'ranking_list': ranking[:20], 
            'rank': rank
        })

        if (i_local + 1) % args.save_interval == 0 and args.out is not None and i_local >= cur_saved:
            print(f'Save {args.out} with i_local {i_local}.')
            torch.save(all_ranking, args.out)

    cumsum_nq = np.cumsum([0] + args.nums_selected_txt_query)
    for i_dataset in range(len(cumsum_nq) - 1):
        vr = all_ranks[cumsum_nq[i_dataset]:cumsum_nq[i_dataset + 1]]
        r1, r5, r10 = compute_recalls(vr)
        print(f'dataset: {args.datasets_query[i_dataset]}, ' + \
            f'r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}')
    
    torch.save(results_dict, args.out.replace('retr', 'results_dict'))
    if cur_saved < n_local:
        torch.save(all_ranking, args.out)
        print(f'Save {args.out}.')
    print(args)

def parse_res(args):
    raise NotImplementedError
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
    NUM_IMG_TOKENS = args.num_img_tokens


    generation_config = {
        'max_new_tokens': 256, 
        'temperature': 1.0,
        'num_beams': args.num_beams,
        'num_return_sequences': args.num_beams,
        'use_cache': args.use_cache
    }

    if args.only_parse_res:
        parse_res(args)
    else:
        retrieval(args, generation_config)





