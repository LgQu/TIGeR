import hydra
import argparse
from omegaconf import OmegaConf
from PIL import Image
import os, torch, gc, sys
import torch.nn.functional as F
from tqdm import tqdm
import collections
from typing import List, Tuple
import math

from models.model_tools import get_pretrained_llama_causal_model
import accelerate
import numpy as np

from data import TigerBenchDataset
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

def load_model(seed_model):
    tokenizer_cfg_path = 'configs/tokenizer.yaml'
    transform_cfg_path = 'configs/transform.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device_map="auto", load_diffusion=False)
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)
    pretrained_model_name_or_path = os.path.join(os.getenv('CKPT_ROOT'), seed_model)
    model = get_pretrained_llama_causal_model(pretrained_model_name_or_path, torch_dtype='fp16',low_cpu_mem_usage=True, device_map='auto')
    return tokenizer, transform, model


def sim_postproc(sim, idx_txt, extra_dict):
    if 'idx_context_in_gallery' in extra_dict:
        indices_remove = extra_dict['idx_context_in_gallery'][idx_txt]
        sim[indices_remove] = -torch.inf
    return sim

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

def get_attn_mask_ip(input_ids, tokenizer):
    boi_list = torch.where(input_ids == tokenizer(e_token, add_special_tokens=False).input_ids[-1]) # find the ":" in "ASSISTANT:" as begin
    
    boi_index = boi_list[1][-1]
    seq_length = input_ids.shape[-1]
    img_length = boi_index
    
    mask = torch.ones((seq_length, seq_length), dtype=torch.long, device=input_ids.device).tril_()
    mask[..., :img_length] = 1
    
    
    
    
    
    return mask, img_length

    
def inverse_prompt(input_ids, tokenizer):
    print("Input Ids:", input_ids, tokenizer(e_token, add_special_tokens=False).input_ids[-1])
    boi_list = torch.where(input_ids == tokenizer(e_token, add_special_tokens=False).input_ids[-1]) # find the ":" in "ASSISTANT:" as begin
    eoi_list = torch.where(input_ids == tokenizer(tokenizer.pad_token, add_special_tokens=False).input_ids[0])
    if len(boi_list) == 0 and len(eoi_list) == 0:
        print("No Image Token Detected!!!")
    else:
        print(boi_list, eoi_list)
        boi_index = boi_list[1][-1]
        eoi_index = input_ids.shape[-1]
        print("boi idx:", boi_index, "eoi idx:", eoi_index)
        max_length = input_ids.shape[-1]
        img_length = boi_index
        cap_length = eoi_index - boi_index - 2 # caption length
        
        img = input_ids[...,:img_length]
        cap = input_ids[...,img_length:img_length+cap_length]
        print("caption length:", cap_length, "caption shape:", cap.shape)
        assert cap_length == cap.shape[-1]
        print("max length:", max_length, "img_length:", img_length, "cap_length:", cap_length, cap.shape)
        
        inflated_batch = torch.full((input_ids.shape[0],cap_length, max_length), tokenizer.pad_token_id, dtype=torch.long)
        inflated_batch[...,:img_length] = img.unsqueeze(1).repeat(1,cap_length,1)
        inflated_batch[...,img_length:img_length+cap_length] = cap.unsqueeze(1).repeat(1,cap_length,1)
        
        mask = torch.tril(torch.ones((input_ids.shape[0],cap_length, max_length), dtype=torch.long), diagonal=img_length)
        
        print("prompt:\n",img)
        print("label:\n", cap)
        
        """
        this will construct a batch like below, label [<img>,i0,i1, ..., in,</img>] -> (1,34)
        [
            [prompt, <img>, <pad>, <pad>, ... , <pad>],
            [prompt, <img>, i0, <pad>, ...... , <pad>],
            [prompt, <img>, i0, i1, ......... , <pad>],
            ...
            [prompt, <img>, i0, i1, ... , in-1, <pad>],
        ] (len(label)-2, max_length) -> (32, max_length)
        """

        batch = inflated_batch * mask
        batch = batch.reshape(-1, batch.shape[-1])
        print("batch:\n", batch[:2], batch[-2:], batch.shape)
        return batch, img_length

def create_batch(input_ids, tokenizer):
    print("Input Ids:", input_ids)
    boi_list = torch.where(input_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])
    eoi_list = torch.where(input_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])
    
    if len(boi_list) == 0 and len(eoi_list) == 0:
        print("No Image Token Detected!!!")
    else:
        boi_index = boi_list[1][0]
        eoi_index = eoi_list[1][0]
        print("boi idx:", boi_index, "eoi idx:", eoi_index)
        max_length = input_ids.shape[-1]
        text_length = boi_index
        label_length = eoi_index - boi_index - 1 # 32
        
        text = input_ids[...,:text_length]
        label = input_ids[...,text_length:text_length+label_length]
        print("label length:", label_length, "label shape:", label.shape)
        assert label_length == label.shape[-1]
        print("max length:", max_length, "text_length:", text_length, "label_length:", label_length, label.shape)
        
        inflated_batch = torch.full((input_ids.shape[0],label_length, max_length), tokenizer.pad_token_id, dtype=torch.long)
        inflated_batch[...,:text_length] = text.unsqueeze(1).repeat(1,label_length,1)
        inflated_batch[...,text_length:text_length+label_length] = label.unsqueeze(1).repeat(1,label_length,1)
        
        mask = torch.tril(torch.ones((input_ids.shape[0],label_length, max_length), dtype=torch.long), diagonal=text_length)
        
        print("prompt:\n",text)
        print("label:\n", label)
        
        """
        this will construct a batch like below, label [<img>,i0,i1, ..., in,</img>] -> (1,34)
        [
            [prompt, <img>, <pad>, <pad>, ... , <pad>],
            [prompt, <img>, i0, <pad>, ...... , <pad>],
            [prompt, <img>, i0, i1, ......... , <pad>],
            ...
            [prompt, <img>, i0, i1, ... , in-1, <pad>],
        ] (len(label)-2, max_length) -> (32, max_length)
        """

        batch = inflated_batch * mask
        batch = batch.reshape(-1, batch.shape[-1])
        print("batch:\n", batch[:2], batch[-2:], batch.shape)
        return batch, text_length
    
def get_attn_mask(input_ids, tokenizer):
    bs = input_ids.shape[0]
    
    boi_list = torch.where(input_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])
    eoi_list = torch.where(input_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])
    
    if len(boi_list) == 0 and len(eoi_list) == 0:
        print("No Image Token Detected!!!")
    else:
        boi_index = boi_list[1][0]
        eoi_index = eoi_list[1][0]
        seq_length = input_ids.shape[-1]
        text_length = boi_index
        
        mask = torch.ones((seq_length, seq_length), device=input_ids.device).tril_()
        mask[..., :text_length] = 1 # include BOI token
        
        
        return mask, text_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out_uncond_imgid2sumlogprobs", default=None)
    parser.add_argument("--uncond", action='store_true')
    parser.add_argument("--extract_uncond_logits", action='store_true')
    parser.add_argument("--extract_uncond_logits_raw", action='store_true')
    parser.add_argument("--ip", action='store_true')
    parser.add_argument("--cog", action='store_true')
    parser.add_argument("--save_interval", type=int, default=1000000)
    parser.add_argument('--num_selected', type=int, default=None)
    parser.add_argument('--num_selected_query', type=int, default=None)
    parser.add_argument('--num_selected_gallery', type=int, default=None)
    parser.add_argument("--coco", action='store_true')
    parser.add_argument("--image_root", default="")
    parser.add_argument("--ann_root", default="")
    parser.add_argument('--cand_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='logo2k')
    parser.add_argument("--prefix_prompt", default='Generate an image of ')
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--data_version", nargs='?', const="", default='tiger_bench')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument("--turn", default='single_turn', help='especially for VIST')
    parser.add_argument("--cfg_uncond_logits", default=None)
    parser.add_argument('--seed_model', type=str, default='seed-llama-8b-sft')

    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)
    args = parser.parse_args()
    print(args)
    prefix_prompt = args.prefix_prompt
    args.cog = True

    

    ''' ################################################# Dataset ################################################# '''
    t2i_mat = torch.load(args.input)
    ids_dict = t2i_mat
    postfix = f'_{args.dataset}'
    print(ids_dict.keys())
    img_input_ids = ids_dict['prompt_img_input_ids'] if args.ip else ids_dict['img_input_ids']
    text_input_ids = ids_dict['text_input_ids']
    num_txt, num_img = text_input_ids.shape[0], img_input_ids.shape[0]
    img_input_ids = img_input_ids.to(device)
    print('img_input_ids.shape = ', img_input_ids.shape)

    
    test = TigerBenchDataset(args, prefix_prompt=prefix_prompt) 
    args.num_selected = len(test) if args.num_selected is None else args.num_selected
    args.num_selected_query = args.num_selected if args.num_selected_query is None else args.num_selected_query
    args.num_selected_gallery = args.num_selected if args.num_selected_gallery is None else args.num_selected_gallery
    extra_dict = {'idx_context_in_gallery':test.dataset.idx_context_in_gallery} if hasattr(test.dataset, 'idx_context_in_gallery') else {}
    print(f'#text: {len(test.text)}, #img: {len(test.image)}')

    assert len(img_input_ids) == len(text_input_ids)
    num_txt_selected = args.num_selected_query
        
    img_ids = torch.load(f"./intermediate_results/{args.data_version}/img_ids{postfix}.pt")
    img_ids = img_ids[:args.num_selected_gallery]
    if isinstance(img_ids, List):
        img_ids = torch.stack(img_ids)
    print("img_ids", img_ids.shape)
    
    text_ids = torch.load(f"./intermediate_results/{args.data_version}/text_ids{postfix}.pt")
    text_ids = text_ids[:args.num_selected_query]
    print("text_ids", text_ids.shape)
    
    batch_size = args.bs
    sim_t2i_old = None
    if args.out is not None and os.path.exists(args.out):
        sim_t2i_old = torch.load(args.out)

    sim_t2i_uncond = 0
    if args.out_uncond_imgid2sumlogprobs is not None and os.path.exists(args.out_uncond_imgid2sumlogprobs) and not args.uncond:
        sim_t2i_uncond = []
        imgid2sumlogprobs = torch.load(args.out_uncond_imgid2sumlogprobs)
        img_input_ids_without_special = img_input_ids[:, 1:-1]
        for i_img, ids in enumerate(img_input_ids_without_special):
            ids_str = '_'.join([str(i) for i in ids.tolist()][:-1])
            sim_t2i_uncond.append(imgid2sumlogprobs[ids_str])
        sim_t2i_uncond = torch.tensor(sim_t2i_uncond) #  (n_gallery, )
    
    if args.cfg_uncond_logits is not None:
        cfg_uncond_dict = torch.load(args.cfg_uncond_logits)

    if args.extract_uncond_logits:
        all_log_probs = []
    if args.extract_uncond_logits_raw:
        all_logits = []

    tokenizer, transform, model = load_model(args.seed_model)

    n_local = num_txt
    start = 0
    if args.n_proc > 1:
        print(f"Number of processes: {args.n_proc}")
        print(f"Process ID: {args.proc_id}")
        block_size = math.ceil(num_txt / args.n_proc)
        indices = list(range(0, num_txt+block_size, block_size))
        assert len(indices) == args.n_proc + 1
        start, end = indices[args.proc_id], indices[args.proc_id+1]
        end = start + len(text_input_ids[start:end]) # last block
        n_local = end - start
        print(f"Number of total data points: {num_txt}")
        print(f"Number of local data points: {end - start}")

    sim_t2i = torch.zeros(n_local, num_img)
    step = 1
    all_ranks = []
    pbar = tqdm(range(n_local), desc="per text", total=n_local)
    with torch.no_grad():
        for i in pbar:
            i_global = i + start
            if sim_t2i_old is not None and i < sim_t2i_old.shape[0]:
                sim_t2i[i] = sim_t2i_old[i] 
            else:
                cur_text_input_ids = text_input_ids[i_global].unsqueeze(0).expand(img_input_ids.shape[0], -1).to(device)
                if args.ip:
                    t2i = torch.cat([img_input_ids, cur_text_input_ids], dim=-1)
                elif args.extract_uncond_logits_raw:
                    t2i = img_input_ids
                else:
                    t2i = torch.cat([cur_text_input_ids, img_input_ids], dim=-1)
        
                step += 1
                per_text_ids = text_ids[i_global][text_ids[i_global] != 0][:-1].unsqueeze(0).to(device) # omit "." token
                for s in tqdm(range(0, len(t2i), batch_size), desc="per text/img", unit='bs', leave=False):
                    e = s + batch_size
                    x = t2i[s:e].to(device)
                    bs_cur = x.shape[0]
                    x = x[x!=0].reshape(bs_cur, -1) # remove padding inside batch
                    if args.ip:
                        raise NotImplementedError
                        if args.cog:
                            attn_mask, img_length = get_attn_mask_ip(x, tokenizer) # [batch_size*32, seq_len]

                            txt_token_idx = per_text_ids.expand(bs_cur, -1)
                            
                            logits = model(input_ids=x).logits
                            
                            idx = img_length 
                            txt_len = per_text_ids.shape[-1]
                            log_probs = F.log_softmax(logits.float(), dim=-1)
                            log_probs = log_probs[:,idx:idx+txt_len,:] # (bs, txt_len, 40194)

                            txt_token_idx = txt_token_idx.unsqueeze(-1) # (bs, txt_len, 1)
                            p_all = torch.gather(log_probs,-1, txt_token_idx).squeeze(-1)
                            
                            sum_log_prob = p_all.sum(dim=-1)
                        else:
                            batch, img_length = inverse_prompt(x, tokenizer) # [batch_size*32, seq_len]
                            print("batch", batch.shape)

                            txt_token_idx = per_text_ids
                            print("txt_token_idx shape:",txt_token_idx.shape)
                            
                            logits = model(input_ids=batch.to(device),attention_mask=torch.ones_like(batch).to(device)).logits
                            
                            idx = img_length 
                            txt_len = per_text_ids.shape[-1]
                            print("txt len:", txt_len, txt_token_idx[0])
                            log_probs = F.log_softmax(logits, dim=-1)
                            log_probs = log_probs[:,idx:idx+txt_len,:]
                            print('Aft logits:', logits.shape, log_probs.shape)
                            log_probs_reshape = log_probs.reshape(batch_size, -1, log_probs.shape[-2], log_probs.shape[-1])

                            
                            txt_token_idx = txt_token_idx.unsqueeze(-1).unsqueeze(-1).expand(batch_size,-1, txt_len, 1)
                            p_all = torch.gather(log_probs_reshape,-1, txt_token_idx).squeeze(-1)
                            p_all_diag = torch.diagonal(p_all, dim1=1, dim2=2)
                            sum_log_prob = p_all_diag.sum(dim=-1)
                            print("Log Softmax:", p_all_diag.shape, "Sum Log Softmax Prob:", sum_log_prob)
                        sim_t2i[i][s:e] = sum_log_prob
                        
                    else:
                        attn_mask, text_length = get_attn_mask(x, tokenizer)

                        img_token_idx = (img_ids[s:e] + image_id_shift).to(device)
                    
                        logits = model(input_ids=x).logits
                        idx = text_length
                        logits = logits[:,idx:idx+NUM_IMG_TOKENS,:] # (bs, 32, 40194)

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

                        assert not torch.any(torch.isnan(logits))
                        log_probs = F.log_softmax(logits.float(), dim=-1)
                        
                        
                        if args.extract_uncond_logits:
                            all_log_probs.append(log_probs.cpu())
                        if args.extract_uncond_logits_raw:
                            all_logits.append(logits.cpu()[:, idx:idx+NUM_IMG_TOKENS, :]) # (bs, 32, 40194)
                        
                        img_token_idx = img_token_idx.unsqueeze(-1) # (bs, 32, 1)

                        p_all = torch.gather(log_probs, -1, img_token_idx).squeeze(-1) # (bs, 32)
                        
                        sum_log_prob = p_all.sum(dim=-1) # (bs, )

                        sim_t2i[i][s:e] = sum_log_prob
                        
                    gc.collect()
                    torch.cuda.empty_cache()
            
            if args.extract_uncond_logits or args.extract_uncond_logits_raw:
                all_saved = None
                if args.extract_uncond_logits:
                    all_saved = all_log_probs
                elif args.extract_uncond_logits_raw:
                    all_saved = all_logits

                all_saved = torch.cat(all_saved, dim=0) # (bs, 32, 40194)
                all_saved = all_saved.half()
                print('all_saved.shape = ', all_saved.shape)
                img_input_ids_without_special = img_input_ids[:, 1:-1]
                print('img_input_ids_without_special.shape = ', img_input_ids_without_special.shape)
                print('###############')
                print(all_saved[:, 0, :])
                
                imgid2logprobs = collections.OrderedDict()
                imgid2logprobs[0] = collections.OrderedDict()
                imgid2logprobs[0][''] = all_saved[0, 0]
                for step in tqdm(range(1, img_input_ids_without_special.shape[-1]), total=img_input_ids_without_special.shape[-1]-1):
                    imgid2logprobs[step] = collections.OrderedDict()
                    for i_beam in range(len(img_input_ids_without_special)):
                        ids = img_input_ids_without_special[i_beam][:step].tolist()
                        ids_str = '_'.join([str(id_) for id_ in ids])
                        if ids_str not in imgid2logprobs[step]:
                            imgid2logprobs[step][ids_str] = all_saved[i_beam, step]
                        else: # more than one image are encoded into the same code
                            print('---')
                            print(step, i_beam)
                            print(ids_str)
                            print(imgid2logprobs[step][ids_str] - all_saved[i_beam, step])
                torch.save(imgid2logprobs, args.out)
                exit(0)

            sim_t2i_cur = sim_t2i[i] - sim_t2i_uncond * 1.0
            print('sim (cond): ', f'mean={sim_t2i[i].mean().item():.2f}',  sim_t2i[i][:20].long().tolist())
            print('sim (cond-uncond)', f'mean={sim_t2i_cur.mean().item():.2f}', sim_t2i_cur[:20].long().tolist())

            sim_t2i_cur = sim_postproc(sim_t2i_cur, i_global, extra_dict)
            scores_i_cap_sorted, ranking = sim_t2i_cur.sort(descending=True)
            rank = compute_rank_t2i(ranking.numpy(), gt_img_id=test.txt2img[i_global][0])
            
            print('\n', test.txt2img[i_global][0], rank, ranking.tolist()[:30])
            
            all_ranks.append(rank)
            r1, r5, r10 = compute_recalls(all_ranks)
            pbar.set_postfix_str(f'r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}')
            
            if i in [2000]: 
                print(f'#txt: {i}, r1={r1:.2f}, r5={r5:.2f}, r10={r10:.2f}')
                break

            if args.uncond:
                sim_t2i = sim_t2i[[0]].clone()
                img_input_ids_without_special = img_input_ids[:, 1:-1]
                assert sim_t2i.shape[1] == len(img_input_ids_without_special)
                print('img_input_ids_without_special.shape = ', img_input_ids_without_special.shape)
                imgid2sumlogprobs = collections.OrderedDict()
                for i_img, ids in enumerate(img_input_ids_without_special):
                    ids_str = '_'.join([str(i) for i in ids.tolist()][:-1])
                    imgid2sumlogprobs[ids_str] = sim_t2i[0, i_img]
                torch.save(imgid2sumlogprobs, args.out_uncond_imgid2sumlogprobs)
                print(f'Save {args.out_uncond_imgid2sumlogprobs}.')
                break

            if step % args.save_interval == 0: 
                print(f'Save {args.out} with step {step}.')
                torch.save(sim_t2i[:i], args.out)

    if args.out is not None:
        torch.save(sim_t2i, args.out)

    print(args)
