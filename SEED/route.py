import hydra
import argparse
from omegaconf import OmegaConf
from PIL import Image
import os, torch, gc
import torch.nn.functional as F
from tqdm import tqdm
import collections
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import math
import warnings
from collections import OrderedDict
import sys

from models.model_tools import get_pretrained_llama_causal_model
import accelerate
import numpy as np
from data import get_combined_query_gallery


tokenizer_cfg_path = 'configs/tokenizer.yaml'
transform_cfg_path = 'configs/transform.yaml'
device = 'cuda'

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg, device_map="auto", load_diffusion=False)

transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)

pretrained_model_name_or_path = os.path.join(os.getenv('CKPT_ROOT'), os.getenv('SEED_MODEL', 'seed-llama-8b-sft'))
model = get_pretrained_llama_causal_model(pretrained_model_name_or_path, torch_dtype='fp16',low_cpu_mem_usage=True, device_map='auto')
model.eval()

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


def extract_img_ids(generate_ids, tokenizer):
    boi_list = torch.where(generate_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    image_ids = torch.zeros((1, 34), dtype=torch.long)

    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        warnings.warn('No image ids found! Use default image_ids with all 0 filled. ')
    else:
        boi_index = boi_list[0]
        eoi_index = eoi_list[0]
        text_ids = generate_ids[:boi_index]
        if len(text_ids) != 0:
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        image_ids = (generate_ids[boi_index:eoi_index+1]).reshape(1,-1)

    return image_ids

class IdDataset(Dataset):
    def __init__(self, ids_dict, all_gen_ids, text_ids_uncond, ranking):
        self.img_input_ids = ids_dict['img_input_ids'] # len = 34
        self.text_input_ids = ids_dict['text_input_ids']
        self.all_gen_ids = all_gen_ids # len = 34
        self.text_ids_uncond = text_ids_uncond
        self.ranking = ranking
        assert len(self.text_input_ids) == len(self.ranking) == len(self.all_gen_ids), \
                f'{len(self.text_input_ids)}, {len(self.ranking)}, {len(self.all_gen_ids)}'
        print(f'img_input_ids={self.img_input_ids.shape}, len(text_input_ids)={len(self.text_input_ids)}')
        print(f'all_gen_ids.shape={self.all_gen_ids.shape}, len(ranking)={len(self.ranking)}')

    def __len__(self, ):
        return len(self.text_input_ids)

    def __getitem__(self, index):
        cur_txt_ids = self.text_input_ids[index]
        cur_txt_ids = cur_txt_ids[cur_txt_ids != 0] # remove paddings
        retr_img_ids = self.img_input_ids[self.ranking[index][0]]
        retr_ids = torch.cat([cur_txt_ids, retr_img_ids])
        retr_ids_uncond = torch.cat([self.text_ids_uncond, retr_img_ids])
        gen_img_ids = self.all_gen_ids[index]
        gen_ids = torch.cat([cur_txt_ids, gen_img_ids])
        gen_ids_uncond = torch.cat([self.text_ids_uncond, gen_img_ids])
        return index, retr_ids, retr_ids_uncond, gen_ids, gen_ids_uncond

def collate_fn(batch):
    batch = list(zip(*batch))
    inds, retr_ids, retr_ids_uncond, gen_ids, gen_ids_uncond = batch
    retr_ids_padded, retr_ids_uncond_padded, gen_ids_padded, gen_ids_uncond_padded = \
            (torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0) \
            for ids in [retr_ids, retr_ids_uncond, gen_ids, gen_ids_uncond])
    return inds, retr_ids_padded, retr_ids_uncond_padded, gen_ids_padded, gen_ids_uncond_padded

def get_tokenized_ids(tokenizer, ids_file_prefix, ids_q_post, ids_g_post, datasets_q, datasets_g, n_q, n_g):
        ds_q, ds_g = datasets_q, datasets_g
        q2g = [ds_g.index(q) if q in ds_g else -1 for q in ds_q]
        g2q = [ds_q.index(g) if g in ds_q else -1 for g in ds_g]
        
        assert len(datasets_q) == len(ids_q_post)
        ids_dict_q = []
        for i, ifp in enumerate(ids_q_post):
            ids_dict = torch.load(ids_file_prefix + ifp + '.pt')
            ids_dict_q.append(ids_dict)

        print('Load Data ...')
        sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                    args.nums_selected_txt_query, args.nums_selected_gallery, prefix_prompt, 
                                                                    args=args, load_img=True)
        inputs = [tokenizer.bos_token  + s_token + " " + t + sep + e_token for t in sampled_text]
        text_input_ids = tokenizer(inputs, add_special_tokens=False, return_tensors='pt', padding="longest").input_ids
        text_input_ids = text_input_ids.cpu()


        assert len(datasets_g) == len(ids_g_post)
        img_input_ids = []
        for i, igp in enumerate(ids_g_post):
            ids_dict = torch.load(ids_file_prefix + igp + '.pt') if g2q[i] == -1 else ids_dict_q[g2q[i]]
            img_input_ids.append(ids_dict['img_input_ids'][:n_g[i]])
        img_input_ids = torch.cat(img_input_ids, dim=0)
        assert len(img_input_ids) == sum(n_g)
        all_ids_dict = {'text_input_ids': text_input_ids, 'img_input_ids': img_input_ids}
        return all_ids_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_dict", default='')
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix_prompt", default='Generate an image of ')
    parser.add_argument('--uncond_txt', type=str, default='Can you generate an image?')

    parser.add_argument('--datasets_query', type=str, nargs='+', default=[])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', default=[])
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[])
    parser.add_argument('--nums_selected_txt_query', type=int, nargs='+', default=[])
    parser.add_argument('--ranking', type=str, default='')
    parser.add_argument('--ids_file_prefix', type=str, default='./intermediate_results/t2i_mat_cond_')
    parser.add_argument('--ids_files_query_postfix', type=str, nargs='+', default=[])
    parser.add_argument('--ids_files_gallery_postfix', type=str, nargs='+', default=[])
    parser.add_argument('--ids_gen_prefix', type=str, default='./generated/ids_')
    parser.add_argument('--ids_gen', type=str, nargs='+', default=[])
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--data_version", const="", nargs="?", default='tiger_bench')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--post_filtering_mask', type=str, default=None)
    args = parser.parse_args()
    print(args)
    prefix_prompt = args.prefix_prompt
    assert len(args.ids_files_query_postfix) == len(args.ids_gen)

    all_gen_ids_dict = [torch.load(args.ids_gen_prefix + ig + '.pt') for ig in args.ids_gen]
    all_gen_ids = []
    for i_gen, gen_dict in tqdm(enumerate(all_gen_ids_dict), total=len(all_gen_ids_dict), desc='extract_img_ids'):
        num_cur = len(all_gen_ids)
        num_q = args.nums_selected_txt_query[i_gen]
        if isinstance(gen_dict, dict):
            for k, v in tqdm(gen_dict.items(), total=len(gen_dict), leave=False):
                all_gen_ids.append(extract_img_ids(torch.LongTensor(v), tokenizer))
        else:
            gen_dict = gen_dict.cpu()
            for v in tqdm(gen_dict, total=len(gen_dict), leave=False):
                all_gen_ids.append(extract_img_ids(v, tokenizer))
        all_gen_ids = all_gen_ids[:num_cur + num_q]
    
    for i, x in enumerate(all_gen_ids):
        if x.shape[1] != 34:
            print(i, x.shape)


    all_gen_ids = torch.cat(all_gen_ids, dim=0)
    assert len(all_gen_ids) == sum(args.nums_selected_txt_query)

    inputs_uncond = [tokenizer.bos_token  + s_token + " " + args.uncond_txt + sep + e_token]
    text_ids_uncond = tokenizer(inputs_uncond, add_special_tokens=False, return_tensors='pt').input_ids
 
    if args.ranking.endswith('.pt'):
        all_ranking = torch.load(args.ranking)
    else:
        all_ranking = np.load(args.ranking)
    assert len(all_ranking) == sum(args.nums_selected_txt_query), f'{len(all_ranking)} != {sum(args.nums_selected_txt_query)}'

    if args.use_gt:
        for i in range(8000):
            all_ranking[sum(args.nums_selected_txt_query[:2]) + i][0] = i
    
    ids_dict = get_tokenized_ids(tokenizer, args.ids_file_prefix, args.ids_files_query_postfix, args.ids_files_gallery_postfix, \
                                        args.datasets_query, args.datasets_gallery, args.nums_selected_txt_query, args.nums_selected_gallery)

    ranking = all_ranking
    if args.post_filtering_mask is not None:
        assert len(ids_dict['text_input_ids']) == 16000
        assert not args.use_gt, 'Not implemented!'
        filtering_mask = torch.load(args.post_filtering_mask)
        all_gen_ids = [all_gen_ids[i] for i in range(len(all_gen_ids)) if filtering_mask[i]]
        all_gen_ids = torch.stack(all_gen_ids, dim=0)
        text_input_ids = [ids_dict['text_input_ids'][i] for i in range(len(ids_dict['text_input_ids'])) if filtering_mask[i]]
        selected_image_ids = [ids_dict['img_input_ids'][i] for i in range(len(ids_dict['img_input_ids'])) if filtering_mask[8000:][i]]
        text_input_ids, selected_image_ids = torch.stack(text_input_ids, dim=0), torch.stack(selected_image_ids, dim=0)
        assert len(text_input_ids) == len(selected_image_ids) * 2 == filtering_mask.sum(), f'{len(sampled_text)}, {len(sampled_imgs)}, {filtering_mask.sum()}'
        assert len(all_gen_ids) == filtering_mask.sum()
        ids_dict = {'text_input_ids': text_input_ids, 'img_input_ids': selected_image_ids}

        idx_map = []
        counter = 0
        for i in range(8000):
            if filtering_mask[8000:][i]:
                idx_map.append(counter)
                counter += 1
            else:
                idx_map.append(None)
        new_ranking = []
        for i in range(len(ranking)):
            if filtering_mask[i]:
                cur_r = [idx_map[r] for r in ranking[i] if idx_map[r] is not None]
                if len(cur_r) == 0:
                    warnings.warn(f'len(cur_r) == 0 when i = {i}')
                    g = torch.Generator()
                    g.manual_seed(i)
                    cur_r = [torch.randint(0, 3000, size=(1, ), generator=g).item()]
                new_ranking.append(cur_r)

        assert len(new_ranking) == filtering_mask.sum()
        all_ranking = new_ranking
        txt2img = OrderedDict()
        for i in range(len(new_ranking)):
            if i < len(new_ranking) // 2:
                txt2img[i] = [-1]
            else:
                txt2img[i] = [i - len(new_ranking) // 2]

    dataset = IdDataset(ids_dict, all_gen_ids, text_ids_uncond.squeeze(0), all_ranking)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=False)

    is_retr_arr = np.array([], dtype=np.float32)
    txtidx2score = collections.OrderedDict()
    pbar = tqdm(enumerate(loader), total=math.ceil(len(dataset) / args.bs), desc='Prob')
    for i_batch, batch in pbar:
        inds, retr_ids_padded, retr_ids_uncond_padded, gen_ids_padded, gen_ids_uncond_padded = batch # (bs, max_seq_len)
        log_probs = []
        for ids_padded in [retr_ids_padded, retr_ids_uncond_padded, gen_ids_padded, gen_ids_uncond_padded]:
            with torch.no_grad():
                l = model(input_ids=ids_padded).logits # (bs, max_seq_len, n_vocab)
                lp = F.log_softmax(l.float(), dim=-1)
                log_probs.append(lp)

        log_probs_retr, log_probs_retr_uncond, log_probs_gen, log_probs_gen_uncond = log_probs
        boi_inds = torch.where(retr_ids_padded == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[1]
        seq_lens = (retr_ids_padded != 0).sum(dim=1)
        boi_inds_uncond = torch.where(retr_ids_uncond_padded == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[1]

        for i in range(len(inds)):
            assert (seq_lens[i] - boi_inds[i]) == NUM_IMG_TOKENS + 2
            cur_lp_retr = log_probs_retr[i, boi_inds[i]:boi_inds[i]+NUM_IMG_TOKENS, :] # (NUM_IMG_TOKENS, n_vocab)
            cur_lp_gen = log_probs_gen[i, boi_inds[i]:boi_inds[i]+NUM_IMG_TOKENS, :]
            cur_gt_inds_retr = retr_ids_padded[i, boi_inds[i]+1:boi_inds[i]+NUM_IMG_TOKENS+1].unsqueeze(-1) # (NUM_IMG_TOKENS, 1)
            cur_gt_inds_gen = gen_ids_padded[i, boi_inds[i]+1:boi_inds[i]+NUM_IMG_TOKENS+1].unsqueeze(-1)
            p_all_retr = torch.gather(cur_lp_retr, dim=-1, index=cur_gt_inds_retr).squeeze(-1).sum()
            p_all_gen = torch.gather(cur_lp_gen, dim=-1, index=cur_gt_inds_gen).squeeze(-1).sum()
            cur_lp_retr_uncond = log_probs_retr_uncond[i, boi_inds_uncond[i]:boi_inds_uncond[i]+NUM_IMG_TOKENS, :]
            cur_lp_gen_uncond = log_probs_gen_uncond[i, boi_inds_uncond[i]:boi_inds_uncond[i]+NUM_IMG_TOKENS, :]
            cur_gt_inds_retr_uncond = retr_ids_uncond_padded[i, boi_inds_uncond[i]+1:boi_inds_uncond[i]+NUM_IMG_TOKENS+1].unsqueeze(-1)
            cur_gt_inds_gen_uncond = gen_ids_uncond_padded[i, boi_inds_uncond[i]+1:boi_inds_uncond[i]+NUM_IMG_TOKENS+1].unsqueeze(-1)
            p_all_retr_uncond = torch.gather(cur_lp_retr_uncond, dim=-1, index=cur_gt_inds_retr_uncond).squeeze(-1).sum()
            p_all_gen_uncond = torch.gather(cur_lp_gen_uncond, dim=-1, index=cur_gt_inds_gen_uncond).squeeze(-1).sum()
            txtidx2score[inds[i]] = {'retr': p_all_retr.item(), 'gen': p_all_gen.item(), \
                                    'retr_uncond': p_all_retr_uncond.item(), 'gen_uncond': p_all_gen_uncond.item()}
            is_retr = ((p_all_retr - p_all_retr_uncond) > (p_all_gen - p_all_gen_uncond)).item()
            is_retr_arr = np.append(is_retr_arr, is_retr)
            pbar.set_postfix_str(f'retr_ratio: {((is_retr_arr == 1).sum() / len(is_retr_arr) * 100):.2f}%')


    torch.save(txtidx2score, args.out)
    print(args)
