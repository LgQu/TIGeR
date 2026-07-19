import os
import gc
import math
import argparse
import torch
import random
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import collections
import time
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from tiger_utils import check_device, my_green, my_red
from models import build_model

from data import get_combined_query_gallery

model_path = os.environ.get('LAVIT_MODEL')
if not model_path:
    raise RuntimeError('Set LAVIT_MODEL to the LaVIT checkpoint directory.')
model_dtype = 'bf16'

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device_id = 0
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

def compute_recalls(ranks):
    ranks = np.array(ranks)
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return tr1, tr5, tr10

def compute_rank_t2i(ranking, gt_img_id):
    rank = 1e20
    tmp = np.where(ranking == gt_img_id)[0]
    if len(tmp) > 0:
        rank = tmp[0]
    return rank

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--wait_gpu", type=int, default=None)
    parser.add_argument("--num_img_tokens", type=int, default=32)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out_uncond_imgid2sumlogprobs", default=None)
    parser.add_argument("--uncond", action='store_true')
    parser.add_argument("--extract_uncond_logits", action='store_true')
    parser.add_argument("--extract_uncond_logits_raw", action='store_true')
    parser.add_argument("--ip", action='store_true')
    parser.add_argument("--save_interval", type=int, default=1000000)
    parser.add_argument('--num_selected', type=int, default=None)
    parser.add_argument("--uncond_prompt", default='Give me an image.')
    parser.add_argument("--cfg_uncond_logits", default=None)
    parser.add_argument('--extract_uncond_log_prob', type=str, default=None)
    
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)

    parser.add_argument('--datasets_query', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument('--nums_selected_txt_query', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument("--data_version", default='combV3', nargs='?', const="")
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default='../SEED/configs/tiger/data.yaml')
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--img_id_path", required=True, default='')
    parser.add_argument("--prefix_prompt", nargs='?', const="", default="Generate an image of ")
    parser.add_argument("--consider_eoi", action='store_true')
    parser.add_argument("--save_postfix", nargs='?', const="", default="")

    parser.add_argument('--img_emb_path', required=True, type=str)
    parser.add_argument('--retr_ranking_list', required=True, type=str)
    parser.add_argument("--sim_i2t", type=str, default=None)
    args = parser.parse_args()
    print(args)

    NUM_IMG_TOKENS = args.num_img_tokens
    print("Loading NUM_IMG_TOKENS:", NUM_IMG_TOKENS)
    prefix_prompt = args.prefix_prompt
    print("prefix:", prefix_prompt)
    print("uncond prompt:", args.uncond_prompt)

    assert args.cfg_uncond_logits is None, 'Not implemented!'
    args.out = os.path.join(os.path.dirname(args.retr_ranking_list), 'rerank_' + args.retr_ranking_list.split('/')[-1])
    args.out = args.save_postfix.join(os.path.splitext(args.out))
    

    input_prompts = []
    sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                    args.nums_selected_txt_query, args.nums_selected_gallery, args.prefix_prompt, 
                                                                    args=args)

    TIGeR_Bench_text = sampled_text
    len_image_ids = 1
    if NUM_IMG_TOKENS == 0: # means use original img_ids which has different length
        print("Loading Original Img Ids...")
        selected_image_ids = torch.load(args.img_id_path, map_location='cpu')
        image_special = torch.as_tensor([BOI, EOI], dtype=torch.long)
        selected_image_ids = [torch.cat([image_special[:1], i, image_special[1:]]) for i in selected_image_ids]
        selected_image_ids = pad_sequence(selected_image_ids, batch_first=True, padding_value=2) # pad_token is 2
    else:
        if 'top' in args.img_id_path:
            assert str(NUM_IMG_TOKENS) in args.img_id_path
        selected_image_ids = torch.load(args.img_id_path)
        if isinstance(selected_image_ids, list): # when topk == 256
            selected_image_ids = torch.stack(selected_image_ids)
            tokens_special = torch.as_tensor([BOI, EOI], dtype=selected_image_ids.dtype, device=selected_image_ids.device)
            tokens_special = tokens_special.unsqueeze(0).expand(len(selected_image_ids), -1)
            selected_image_ids = torch.cat([tokens_special[:, [0]], selected_image_ids, tokens_special[:, [1]]], dim=1)

    check_device(args.wait_gpu) 
    text = sampled_text
    img_ids = selected_image_ids.to(device)
    print("text:", len(text), "img_ids:", img_ids.shape)

    start = 0
    n_local = len(text)
    if args.n_proc > 1:
        print(f"Number of processes: {args.n_proc}")
        print(f"Process ID: {args.proc_id}")
        block_size = math.ceil(len(text) / args.n_proc)
        indices = list(range(0, len(text)+block_size, block_size))
        assert len(indices) == args.n_proc + 1
        start, end = indices[args.proc_id], indices[args.proc_id+1]
        end = start + len(text[start:end]) # last block
        n_local = end - start
        print(f"Number of total data points: {len(text)}")
        print(f"Number of local data points: {end - start}")
    
    is_load_model = True
    cur_saved = 0
    if os.path.exists(args.out):
        cur_all_ranking = torch.load(args.out)
        cur_saved = len(cur_all_ranking)
        print('continue from i_local: ', cur_saved)
        if cur_saved == n_local:
            is_load_model = False
    
    sim_i2t = None
    if args.sim_i2t and os.path.exists(args.sim_i2t):
        sim_i2t = torch.load(args.sim_i2t)
        is_load_model = False

    if is_load_model:
        model = build_model(model_path=model_path, model_dtype=model_dtype, check_safety=False, 
                        device_id=device_id, use_xformers=True, understanding=True, local_files_only=True)
        model.eval()
        model.to(device)
        tokenizer = model.llama_tokenizer
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        llama = model.llama_model

    if not os.path.exists(args.img_emb_path):
        bs = args.bs
        tbar = tqdm(range(0, len(sampled_imgs), bs), desc='img embedding')
        with model.maybe_autocast(), torch.no_grad():
            image_pad_token = torch.tensor([32000, 32001], dtype=torch.long).to(device)
            image_pad_embeds = model.llama_model.get_input_embeddings()(image_pad_token) # [2, embed_dim]
            eos_id = model.llama_tokenizer.eos_token_id
            eos_id = torch.tensor([eos_id], dtype=torch.long).to(device)
            eos_embeds = model.llama_model.get_input_embeddings()(eos_id).unsqueeze(0).cpu()  # [1, 1, embed_dim]
            all_image_embeds_list_cpu = []
            max_token_num = -1
            for i in tbar:
                image_bs = model.process_image(sampled_imgs[i:i+bs])
                image_embeds_list = model.visual_tokenizer.encode_features(image_bs)
                batch_size = len(image_embeds_list)
                for i_b in range(batch_size):
                    all_image_embeds_list_cpu.append(
                        torch.cat([image_pad_embeds[:1], image_embeds_list[i_b], image_pad_embeds[1:]], dim=0).cpu()
                    )
                    max_token_num = max(max_token_num, len(all_image_embeds_list_cpu[-1]))

            image_attns = torch.zeros((len(sampled_imgs), max_token_num), dtype=torch.long)
            image_embeds = eos_embeds.repeat(len(sampled_imgs), max_token_num, 1)

            for i_b in range(len(sampled_imgs)):
                image_attns[i_b, -len(all_image_embeds_list_cpu[i_b]):] = 1
                image_embeds[i_b, -len(all_image_embeds_list_cpu[i_b]):] = all_image_embeds_list_cpu[i_b]
            
            print('image_embeds.shape: ', image_embeds.shape)
            print('image_attns.shape: ', image_attns.shape)
            img_emb_dict = {'emb': image_embeds, 'att': image_attns}
            torch.save(img_emb_dict, args.img_emb_path)
    else:
        img_emb_dict = torch.load(args.img_emb_path)
        image_embeds, image_attns = img_emb_dict['emb'], img_emb_dict['att']
    
    retr_ranking_list = torch.load(args.retr_ranking_list)
    assert len(retr_ranking_list) == len(text)
    
    step = 1
    is_skip = False
    all_ranks = []
    all_ori_ranks = []
    all_ranking = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    top1_freq = {}
    pbar = tqdm(range(n_local), desc="per text", total=n_local)

    batch_size = args.bs
    with torch.no_grad():
        for i in pbar:
            i_global = i + start
            ori_ranking = retr_ranking_list[i_global]
            gt_img_idx = txt2img[i_global][0]
            ori_rank = compute_rank_t2i(ori_ranking, gt_img_id=gt_img_idx)
            all_ori_ranks.append(ori_rank)

            if i < cur_saved:
                rr_ranking = cur_all_ranking[i]
                rank = 1e20
                tmp = np.where(rr_ranking == gt_img_idx)[0]
                if len(tmp) > 0:
                    rank = tmp[0]
                is_skip = True
            elif sim_i2t is not None:
                sim_i2t_i = sim_i2t[i_global][ori_ranking]
                idx_rr = np.argsort(sim_i2t_i)[::-1]
                rr_ranking = ori_ranking[idx_rr]
                rank = 1e20
                tmp = np.where(rr_ranking == gt_img_idx)[0]
                if len(tmp) > 0:
                    rank = tmp[0]
                is_skip = False
            else:
                is_skip = False
                print("Input Text:", text[i_global])
                prompt = text[i_global] # remove BOS
                if len(prompt) == 0:
                    prompt = tokenizer.bos_token + text[i_global]
                prompt_tokens = tokenizer(prompt, 
                                add_special_tokens=False, return_tensors='pt', padding='longest')
                
                prompt_embeds = llama.get_input_embeddings()(prompt_tokens.input_ids)
                tokens_tmp = tokenizer.tokenize(prompt)
                print('tokens: ', tokens_tmp)
                ori_img_embeds = [image_embeds[r] for r in ori_ranking]
                ori_img_atts = [image_attns[r] for r in ori_ranking]

                def cal_p_all(batch_size, prompt_tokens, prompt_embeds, ori_img_embeds, ori_img_atts):
                    p_all = []
                    for i_img in tqdm(range(0, len(ori_img_embeds), batch_size), desc='img batch', leave=False):
                        image_embeds_batch = torch.stack(ori_img_embeds[i_img:i_img+batch_size], dim=0).to(device) # (bs, max_len, 4096)
                        image_attns_batch = torch.stack(ori_img_atts[i_img:i_img+batch_size], dim=0).to(device)
                        prompt_embeds_batch = prompt_embeds.expand(image_embeds_batch.shape[0], -1, -1)
                        inputs_embeds = torch.cat([image_embeds_batch, prompt_embeds_batch], dim=1)
                        prompt_att_mask = prompt_tokens.attention_mask.expand(image_attns_batch.shape[0], -1).to(device)
                        attention_mask = torch.cat([image_attns_batch, prompt_att_mask], dim=1)
                        bos_index = image_embeds_batch.shape[1]
                        logits = llama(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
                        logits_prompt = logits[:, bos_index-1:-1, :] # (bs, n_seq_prompt, n_vocab) # remove BOS
                        assert not torch.any(torch.isnan(logits_prompt))
                        log_prob = F.log_softmax(logits_prompt.float(), dim=-1)
                        gt_prompt_ids = prompt_tokens.input_ids.expand(log_prob.shape[0], -1).to(device) # remove BOS
                        assert gt_prompt_ids.shape[1] == log_prob.shape[1]
                        p_per_token = torch.gather(log_prob, -1, gt_prompt_ids.unsqueeze(-1)).squeeze(-1) # (bs, n_seq_prompt)
                        p_all_batch = p_per_token.sum(dim=-1) # (bs, )
                        p_all.append(p_all_batch)
                    return p_all

                def halve_batch_size():
                    def _set_parameter(retry_state):
                        retry_state.kwargs['batch_size'] = retry_state.kwargs['batch_size'] // 2
                        print('torch.cuda.OutOfMemoryError raised. ')
                        print(f'Retry with half batch_size = {retry_state.kwargs["batch_size"]}. ')
                    return _set_parameter

                @retry(
                retry=retry_if_exception_type(torch.cuda.OutOfMemoryError),
                stop=stop_after_attempt(4),
                after=halve_batch_size())
                def retry_cal_p_all(batch_size, prompt_tokens, prompt_embeds, ori_img_embeds, ori_img_atts):
                    p_all = cal_p_all(batch_size, prompt_tokens, prompt_embeds, ori_img_embeds, ori_img_atts)
                    return p_all
                
                p_all = retry_cal_p_all(batch_size=batch_size, prompt_tokens=prompt_tokens, prompt_embeds=prompt_embeds, 
                                        ori_img_embeds=ori_img_embeds, ori_img_atts=ori_img_atts)
                p_all = torch.cat(p_all, dim=0)
                rr_ind = p_all.sort(descending=True).indices
                rr_ranking = ori_ranking[rr_ind.cpu().numpy()]
                step += 1

                top1_freq[rr_ranking[0]] = top1_freq.get(rr_ranking[0], 0) + 1
                rank = compute_rank_t2i(rr_ranking, gt_img_id=gt_img_idx)

            all_ranking.append(rr_ranking)
            min_gid, max_gid = query2min_max_gallery[i_global]
            ranking_print = []
            for img_idx in rr_ranking:
                if min_gid <= img_idx <= max_gid:
                    if img_idx == gt_img_idx:
                        ranking_print.append(my_green(str(img_idx)))
                    else:
                        ranking_print.append(str(img_idx))
                else:
                    ranking_print.append(my_red(str(img_idx)))
                
            is_in = [1 if min_gid <= r <= max_gid else 0 for r in rr_ranking[:20]]
            print('\n', f'gt: {gt_img_idx}, r: {my_green(str(rank))}, %out: {(1 - sum(is_in)/len(is_in))*100:.2f}, ' + \
                    'ranking: ' + ', '.join(ranking_print[:20]))

            all_ranks.append(rank)
            r1, r5, r10 = compute_recalls(all_ranks)
            r1_ori, r5_ori, r10_ori = compute_recalls(all_ori_ranks)
            pbar.set_postfix_str(f'r1={r1_ori:.2f}->{r1:.2f}, r5={r5_ori:.2f}->{r5:.2f}, r10={r10_ori:.2f}->{r10:.2f}')
            
            if step % 10 == 0 and not is_skip:
                print(f"Save {args.out} at step {step}.")
                torch.save(all_ranking, args.out)

        cumsum_nq = np.cumsum([0] + args.nums_selected_txt_query)
        for i_dataset in range(len(cumsum_nq) - 1):
            vr = all_ranks[cumsum_nq[i_dataset]:cumsum_nq[i_dataset + 1]]
            r1, r5, r10 = compute_recalls(vr)
            print(f'dataset: {args.datasets_query[i_dataset]}, ' + \
                f'r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}')

        top1_freq = sorted(top1_freq.items(), key=lambda item: item[1], reverse=True)
        print('top1_freq: ', top1_freq)
        if args.out is not None and not is_skip:
            print(f"Save {args.out}.")
            torch.save(all_ranking, args.out)
        
