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
    
def get_sim_t2i(args, t2i, img_ids, model, device='cuda'):
    sim_t2i = torch.zeros(len(t2i), dtype=torch.float32, device=device)
    log_probs_lst = []
    img_ids_wo_spec = [] # without specical tokens, BOI, EOI

    for s in tqdm(range(0, len(t2i), args.bs), desc="per text/img", unit='bs', leave=False):
        e = min(s + batch_size, len(t2i))
        x = t2i[s:e].to(device)
        bs_cur = x.shape[0]

        boi_index = torch.where(x == BOI)[1]
        eoi_index = torch.where(x == EOI)[1]

        x = x[:, :torch.max(eoi_index) + 1]

        if args.num_img_tokens == 0: # dynamic tokens
            img_token_idx = []
            for i in range(s, e):
                pad_token_indices = torch.where(img_ids[i]==2)[0]
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
                if args.extract_uncond_log_prob is not None:
                    log_probs_lst.append(log_prob.to(device='cpu', dtype=torch.bfloat16))      
                p_per_token = torch.gather(log_prob, -1, img_token_idx[i].unsqueeze(-1)).squeeze(-1) # (cur_seq_len)
                sum_log_prob.append(p_per_token.sum())

            sum_log_prob = torch.stack(sum_log_prob)

        else: # constant tokens
            if args.consider_eoi:
                img_token_idx = (img_ids[s:e, 1:]).to(device)
            else:
                img_token_idx = (img_ids[s:e, 1:-1]).to(device)
            logits = model(input_ids=x).logits
            if args.consider_eoi:
                logits = logits[:, boi_index[0]:boi_index[0]+NUM_IMG_TOKENS+1, :] # (bs, 33, n_vocab)    
            else:
                logits = logits[:, boi_index[0]:boi_index[0]+NUM_IMG_TOKENS, :] # (bs, 32, n_vocab)
            assert not torch.any(torch.isnan(logits))
            log_probs = F.log_softmax(logits.float(), dim=-1) # (bs, seq_len, n_vocab)
            img_token_idx = img_token_idx.unsqueeze(-1) # (bs, 32, 1)
            p_all = torch.gather(log_probs, -1, img_token_idx).squeeze(-1) # (bs, 32)
            sum_log_prob = p_all.sum(dim=-1) # (bs, )

        if args.cfg_uncond_logits is not None:
            cfg_uncond_batch = []
            img_ids_batch = img_input_ids[s:e][:, 1:-1]
            for i_b in range(len(img_ids_batch)):
                cur_uncond = [cfg_uncond_dict[0]['']]
                for j_s in range(1, img_ids_batch.shape[1]):  # 1, 2, ..., 31
                    ids = [str(id_) for id_ in img_ids_batch[i_b].tolist()[:j_s]]
                    ids_str = '_'.join(ids)
                    cur_uncond.append(cfg_uncond_dict[j_s][ids_str])
                cur_uncond = torch.stack(cur_uncond, dim=0)
                cfg_uncond_batch.append(cur_uncond)
            cfg_uncond_batch = torch.stack(cfg_uncond_batch, dim=0)
            cfg_uncond_batch = cfg_uncond_batch.to(device)
            alpha = 2.0
            logits = logits + alpha * (logits - cfg_uncond_batch)

        sim_t2i[s:e] = sum_log_prob
    if args.extract_uncond_log_prob is not None:
        assert len(sim_t2i) == len(log_probs_lst) == len(img_ids_wo_spec)
        imgid2logprobs = collections.OrderedDict()
        imgid2logprobs[0] = collections.OrderedDict()
        assert not torch.any(torch.isnan(log_probs_lst[0][0]))
        imgid2logprobs[0][''] = log_probs_lst[0][0]
        max_len_img_ids = max([len(i) for i in img_ids_wo_spec])
        print('max_len of img_ids_wo_spec: ', max_len_img_ids)
        for step in tqdm(range(1, max_len_img_ids), total=max_len_img_ids-1):
            imgid2logprobs[step] = collections.OrderedDict()
            for i_beam in range(len(img_ids_wo_spec)):
                if len(img_ids_wo_spec[i_beam]) > step:
                    ids = img_ids_wo_spec[i_beam][:step].tolist()
                    ids_str = '_'.join([str(id_) for id_ in ids])
                    if ids_str not in imgid2logprobs[step]:
                        assert not torch.any(torch.isnan(log_probs_lst[i_beam][step]))
                        imgid2logprobs[step][ids_str] = log_probs_lst[i_beam][step]
                    else: # more than one image are encoded into the same code
                        print('---')
                        print(step, i_beam, ids_str)
                        print(imgid2logprobs[step][ids_str] - log_probs_lst[i_beam][step])
        torch.save(imgid2logprobs, args.extract_uncond_log_prob)
        print(f'Save {args.extract_uncond_log_prob}.')
        imgid2sumlogprobs = collections.OrderedDict()
        for i_img, ids in enumerate(img_ids_wo_spec):
            ids_str = '_'.join([str(i) for i in ids.tolist()][:-1])
            imgid2sumlogprobs[ids_str] = sim_t2i[i_img].cpu()
        save_path = args.extract_uncond_log_prob.replace('logprob', 'sumlogprob')
        torch.save(imgid2sumlogprobs, save_path)
        print(f'Save {save_path}.')

    return sim_t2i
        

def shift_pad_to_front_2d(tensor, pad_value):
    shifted_tensor = torch.empty_like(tensor)
    for i, row in enumerate(tensor):
        non_pad_count = (row != pad_value).sum().item()
        
        non_pad_tensor = row[row != pad_value]
        
        pad_count = row.size(0) - non_pad_count
        
        pad_tensor = torch.full((pad_count,), pad_value, dtype=row.dtype, device=tensor.device)
        
        shifted_row = torch.cat((pad_tensor, non_pad_tensor), dim=0)
        
        shifted_tensor[i] = shifted_row
    
    return shifted_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--wait_gpu", type=int, default=None)
    parser.add_argument("--num_img_tokens", type=int, default=32)
    parser.add_argument("--out", default=None)
    parser.add_argument("--uncond_out", default=None)
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
    parser.add_argument("--data_version", const="", nargs='?', default='combV3')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default='../SEED/configs/tiger/data.yaml')
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--img_id_path", required=True, default='')
    parser.add_argument("--prefix_prompt", default='Generate an image of ')
    parser.add_argument("--consider_eoi", action='store_true')
    parser.add_argument("--uncond_factor", type=float, default=1.0)
    
    args = parser.parse_args()
    print(args)

    NUM_IMG_TOKENS = args.num_img_tokens
    print("Loading NUM_IMG_TOKENS:", NUM_IMG_TOKENS)
    prefix_prompt = args.prefix_prompt
    print("prefix:", prefix_prompt)
    print("uncond prompt:", args.uncond_prompt)

    assert args.cfg_uncond_logits is None, 'Not implemented!'

    

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
    if args.extract_uncond_logits:
        all_log_probs = []
    if args.extract_uncond_logits_raw:
        all_logits = []

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
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        sim_t2i = torch.load(args.out)
        for i in range(len(sim_t2i)):
            if torch.all(sim_t2i[i]==0):
                break
        is_load_model = (i + 1) < len(sim_t2i)
    else:
        sim_t2i = torch.zeros(n_local, len(img_ids))
    if os.path.exists(args.uncond_out):
        sim_t2i_uncond = torch.load(args.uncond_out)
    else:
        sim_t2i_uncond = None
    is_load_model = is_load_model or args.extract_uncond_log_prob is not None

    if is_load_model:
        model = build_model(model_path=model_path, model_dtype=model_dtype, check_safety=False, load_tokenizer=False,
                        device_id=device_id, use_xformers=True, understanding=False, local_files_only=True, 
                        load_quantizer=False, load_pixel_decoder=False)
        model.eval()
        tokenizer = model.llama_tokenizer
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        llama = model.llama_model
    
    step = 1
    is_skip = False
    all_ranks = []
    all_ranking = torch.zeros(n_local, len(img_ids), dtype=torch.int)
    top1_freq = {}
    pbar = tqdm(range(n_local), desc="per text", total=n_local)

    batch_size = args.bs
    with torch.no_grad():
        for i in pbar:
            i_global = i + start
            if args.extract_uncond_log_prob is not None or i_global == 0 and not os.path.exists(args.uncond_out):
                print("Running Uncond logits...")
                text_input_ids_uncond = tokenizer(tokenizer.bos_token + args.uncond_prompt, 
                            add_special_tokens=False, return_tensors='pt', padding='longest').input_ids[0].to(device)
                cur_text_input_ids_uncond = text_input_ids_uncond.unsqueeze(0).expand(img_ids.shape[0], -1).to(device)
                input_ids_uncond = torch.cat([cur_text_input_ids_uncond, img_ids], dim=-1)
                sim_t2i_uncond = get_sim_t2i(args, input_ids_uncond, img_ids, llama)
                sim_t2i_uncond = sim_t2i_uncond.cpu()
                if args.extract_uncond_log_prob is not None:
                    exit(0)
                torch.save(sim_t2i_uncond, args.uncond_out)

            if not torch.all(sim_t2i[i]==0):
                is_skip = True
                print(f"Skip {i}...")
            else:
                is_skip = False
                text_input_ids = tokenizer(tokenizer.bos_token + text[i_global], 
                                add_special_tokens=False, return_tensors='pt', padding='longest').input_ids[0]
                print("Input Text:", text[i_global])
                cur_text_input_ids = text_input_ids.unsqueeze(0).expand(img_ids.shape[0], -1).to(device)
                if args.ip:
                    input_ids = torch.cat([img_ids, cur_text_input_ids], dim=-1)
                elif args.extract_uncond_logits_raw:
                    raise NotImplementedError
                    input_ids = img_ids
                else:
                    input_ids = torch.cat([cur_text_input_ids, img_ids], dim=-1)
                    if NUM_IMG_TOKENS == 0:
                        pass
                step+=1
                sim_t2i[i] = get_sim_t2i(args, input_ids, img_ids, llama)

            sim_t2i_cur = sim_t2i[i] - sim_t2i_uncond * args.uncond_factor
            sim_t2i_cur = sim_t2i_cur / len_image_ids
            norm_sim_t2i_cond_cur = sim_t2i[i] / len_image_ids
            print('sim (cond): ', f'mean={norm_sim_t2i_cond_cur.mean().item():.2f}',  norm_sim_t2i_cond_cur[:20].long().tolist())
            print('sim (cond-uncond)', f'mean={sim_t2i_cur.mean().item():.2f}', sim_t2i_cur[:20].long().tolist())

            scores_i_cap_sorted, ranking = sim_t2i_cur.sort(descending=True)
            all_ranking[i] = ranking.cpu()
            ranking = ranking.numpy()
            top1_freq[ranking[0]] = top1_freq.get(ranking[0], 0) + 1
            gt_img_idx = txt2img[i_global][0]
            rank = compute_rank_t2i(ranking, gt_img_id=gt_img_idx)

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
            r1, r5, r10 = compute_recalls(all_ranks)
            pbar.set_postfix_str(f'r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}')
            
            
            if step % 10 == 0 and not is_skip:
                print(f"Save {args.out} at step {step}.")
                torch.save(sim_t2i, args.out)
                torch.save(all_ranking, args.out.replace('sim_t2i', 'list'))

        top1_freq = sorted(top1_freq.items(), key=lambda item: item[1], reverse=True)
        print('top1_freq: ', top1_freq)
        if args.out is not None and not is_skip:
            torch.save(sim_t2i, args.out)
            print(f"Save {args.out}.")
            torch.save(all_ranking, args.out.replace('sim_t2i', 'list'))
        
        
