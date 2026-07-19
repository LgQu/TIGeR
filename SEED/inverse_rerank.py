import os, sys
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
from ranking import load_model, get_img_ids, compute_rank_t2i, compute_recalls

from data import get_combined_query_gallery

device = 'cuda'
s_token = "USER:"
e_token = "ASSISTANT:"
sep = "\n"
BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'
IMG_FLAG = '<image>'
NUM_IMG_TOKENS = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000


def get_attn_mask(input_ids):
    bs = input_ids.shape[0]
    
    boi_list = torch.where(input_ids == BOI)
    eoi_list = torch.where(input_ids == EOI)
    
    if len(boi_list) == 0 and len(eoi_list) == 0:
        print("No Image Token Detected!!!")
    else:
        boi_index = boi_list[1][0]
        eoi_index = eoi_list[1][0]
        seq_length = input_ids.shape[-1]
        text_length = boi_index
        
        return text_length


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
    parser.add_argument("--cfg_uncond_logits", default=None)
    parser.add_argument('--extract_uncond_log_prob', type=str, default=None)

    parser.add_argument('--seed_model', type=str, default='seed-llama-8b-sft')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)
    parser.add_argument('--datasets_query', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument('--nums_selected_txt_query', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument("--data_version", default='tiger_bench')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--prefix_prompt", nargs='?', const="", default="Generate an image of ")
    parser.add_argument("--consider_eoi", action='store_true')
    parser.add_argument("--save_postfix", nargs='?', const="", default="")
    parser.add_argument("--sim_i2t", type=str, default=None)

    parser.add_argument('--retr_ranking_list', required=True, type=str)
    args = parser.parse_args()
    print(args)

    NUM_IMG_TOKENS = args.num_img_tokens
    print("Loading NUM_IMG_TOKENS:", NUM_IMG_TOKENS)
    prefix_prompt = args.prefix_prompt
    print("prefix:", prefix_prompt)

    assert args.cfg_uncond_logits is None, 'Not implemented!'
    args.out = os.path.join(os.path.dirname(args.retr_ranking_list), 'rerank_' + args.retr_ranking_list.split('/')[-1])
    args.out = args.save_postfix.join(os.path.splitext(args.out))
    

    input_prompts = []
    sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                    args.nums_selected_txt_query, args.nums_selected_gallery, args.prefix_prompt, 
                                                                    args=args)

    sampled_img_ids = []
    all_uncond_sumlogprobs = []
    for i, d_g in enumerate(args.datasets_gallery):
        ids = get_img_ids(d_g, args.nums_selected_gallery[i], args)
        sampled_img_ids.append(ids)
    sampled_img_ids = torch.cat(sampled_img_ids, dim=0) 
    print(f"All, sampled_img_ids.shape: {sampled_img_ids.shape}") # seq_len = 32
    sampled_img_tokens = [BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in im.view(-1).cpu().numpy()]) + EOI_TOKEN for im in sampled_img_ids]
    print('len(sampled_img_tokens) = ', len(sampled_img_tokens))

    start = 0
    n_local = len(sampled_text)
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
    
    is_load_model = True
    cur_saved = 0
    if os.path.exists(args.out):
        cur_all_ranking = torch.load(args.out)
        cur_saved = len(cur_all_ranking)
        print('continue from i_local: ', cur_saved)
        if cur_saved == n_local:
            is_load_model = False
    
    sim_i2t = None
    if os.path.exists(args.sim_i2t):
        sim_i2t = torch.load(args.sim_i2t)
        is_load_model = False

    if is_load_model:
        check_device(args.wait_gpu) 
        tokenizer, transform, model = load_model(args.seed_model)
    
    retr_ranking_list = torch.load(args.retr_ranking_list)
    assert len(retr_ranking_list) == len(sampled_text), f'{len(retr_ranking_list)}, {len(sampled_text)}'
    
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
                
                def cal_p_all(batch_size, prompt, cur_img_tokens):
                    p_all = []
                    for i_img in tqdm(range(0, len(cur_img_tokens), batch_size), desc='img batch', leave=False):
                        img_tokens_batch = cur_img_tokens[i_img:i_img + batch_size] # list, (bs, 34)
                        input_tokens = [tokenizer.bos_token + s_token + " " +\
                            it + ' Please provide an accurate and concise description of the given image. ' + \
                            sep + e_token + prompt for it in img_tokens_batch]
                        if i_img == 0:
                            print('\ninput_tokens[0]: ', input_tokens[0])
                        input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
                        input_ids = input_ids.to("cuda")
                        logits = model(input_ids=input_ids).logits
                        bot = tokenizer.convert_tokens_to_ids(':') # begin of text
                        bot_indices = torch.where(input_ids == bot)[1] # 1 means get the column index
                        bot_index = bot_indices[-1]
                        assert (bot_indices == bot_index).sum() == len(input_ids)
                        logits_prompt = logits[:, bot_index:-1, :] # (bs, n_seq_prompt, n_vocab)
                        assert not torch.any(torch.isnan(logits_prompt))
                        log_prob = F.log_softmax(logits_prompt.float(), dim=-1)
                        gt_prompt_ids = input_ids[:, bot_index+1:]
                        assert gt_prompt_ids.shape[1] == log_prob.shape[1]
                        p_per_token = torch.gather(log_prob, -1, gt_prompt_ids.unsqueeze(-1)).squeeze(-1) # (bs, n_seq_prompt)
                        p_all_batch = p_per_token.sum(dim=-1) # (bs, )
                        p_all.append(p_all_batch)
                    return p_all

                def halve_batch_size(retry_state):
                    retry_state.kwargs['batch_size'] = retry_state.kwargs['batch_size'] // 2
                    print('torch.cuda.OutOfMemoryError raised. ')
                    print(f'Retry with half batch_size = {retry_state.kwargs["batch_size"]}. ')

                @retry(
                retry=retry_if_exception_type(torch.cuda.OutOfMemoryError),
                stop=stop_after_attempt(4),
                after=halve_batch_size)
                def retry_cal_p_all(batch_size, prompt, cur_img_tokens):
                    p_all = cal_p_all(batch_size, prompt, cur_img_tokens)
                    return p_all
                
                cur_img_tokens = [sampled_img_tokens[r] for r in ori_ranking]
                p_all = retry_cal_p_all(batch_size=batch_size, prompt=sampled_text[i_global], cur_img_tokens=cur_img_tokens)
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
