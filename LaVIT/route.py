import hydra
import argparse
from omegaconf import OmegaConf
from PIL import Image
import os
import torch, gc
import torch.nn.functional as F
from tqdm import tqdm
import collections
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import math
import warnings
import numpy as np
import random
from collections import OrderedDict

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

PAD_ID = 2
BOI, EOI = 32000, 32001


class IdDataset(Dataset):
    def __init__(self, ids_dict, all_gen_ids, text_ids_uncond, ranking):
        self.img_input_ids = ids_dict['img_input_ids']
        self.text_input_ids = ids_dict['text_input_ids']
        self.all_gen_ids = all_gen_ids 
        self.text_ids_uncond = text_ids_uncond
        self.ranking = ranking
        assert len(self.text_input_ids) == len(self.ranking) == len(self.all_gen_ids), \
                f'{len(self.text_input_ids)}, {len(self.ranking)}, {len(self.all_gen_ids)}'

    def __len__(self, ):
        return len(self.text_input_ids)

    def __getitem__(self, index):
        cur_txt_ids = self.text_input_ids[index]
        cur_txt_ids = cur_txt_ids[cur_txt_ids != PAD_ID] # remove paddings
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
            (torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=PAD_ID) \
            for ids in [retr_ids, retr_ids_uncond, gen_ids, gen_ids_uncond])
    return inds, retr_ids_padded, retr_ids_uncond_padded, gen_ids_padded, gen_ids_uncond_padded

def get_tokenized_ids(args, tokenizer):
    sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                    args.nums_selected_txt_query, args.nums_selected_gallery, args.prefix_prompt, 
                                                                    args=args)

    text_input_ids = []
    for i, text in tqdm(enumerate(sampled_text), total=len(sampled_text)):
        ids = tokenizer(tokenizer.bos_token + text, add_special_tokens=False, return_tensors='pt', padding='longest').input_ids[0]
        text_input_ids.append(ids)

    if NUM_IMG_TOKENS == 0: # means use original img_ids which has different length
        print("Loading Original Img Ids...")
        selected_image_ids = torch.load(args.img_ids_gallery, map_location='cpu')
        image_special = torch.as_tensor([BOI, EOI], dtype=torch.long)
        selected_image_ids = [torch.cat([image_special[:1], i, image_special[1:]]) for i in selected_image_ids]
    
    all_ids_dict = {'text_input_ids': text_input_ids, 'img_input_ids': selected_image_ids}
    return all_ids_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_dict", default='')
    parser.add_argument("--num_img_tokens", type=int, default=0)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--uncond_prompt", default='Give me an image.')

    parser.add_argument('--datasets_query', type=str, nargs='+', 
        default=['whoops', 'pickapic_7500', 'artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--nums_selected_txt_query', type=int, nargs='+', default=[500, 7500] + [1000] * 8)
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument("--data_version", default='combV3')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default='../SEED/configs/tiger/data.yaml')
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--img_ids_gallery", required=True, default='')
    parser.add_argument("--prefix_prompt", default='Generate an image of ')
    parser.add_argument("--consider_eoi", action='store_true')

    parser.add_argument('--ranking', type=str, default=None)
    parser.add_argument('--ids_gen_prefix', type=str, default='./generated/ids_')
    parser.add_argument('--ids_gen', type=str, nargs='+', default=[])
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--post_filtering_mask', type=str, default=None)

    args = parser.parse_args()
    print(args)
    prefix_prompt = args.prefix_prompt
    assert len(args.datasets_query) == len(args.ids_gen)

    NUM_IMG_TOKENS = args.num_img_tokens
    print("Loading NUM_IMG_TOKENS:", NUM_IMG_TOKENS)

    all_gen_ids = [torch.load(args.ids_gen_prefix + ig + '.pt', map_location='cpu') for ig in args.ids_gen]
    all_gen_ids_new = []
    for i_gen, gen_ids in tqdm(enumerate(all_gen_ids), total=len(all_gen_ids), desc='extract_img_ids'):
        all_gen_ids_new.extend(gen_ids)
    all_gen_ids = all_gen_ids_new
    assert len(all_gen_ids) == sum(args.nums_selected_txt_query)

    all_len = [len(i) for i in all_gen_ids]
    all_len = torch.FloatTensor(all_len)
    mean = all_len.mean().item()
    std = all_len.std().item()
    min_l, max_l = all_len.min().item(), all_len.max().item()
    print(f'generated img ids, mean: {mean:.2f}, std: {std:.2f}, min: {min_l}, max: {max_l}')

    model = build_model(model_path=model_path, model_dtype=model_dtype, check_safety=False, load_tokenizer=False,
                        device_id=device_id, use_xformers=True, understanding=False, local_files_only=True, 
                        load_quantizer=False)
    model.eval()
    tokenizer = model.llama_tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    llama = model.llama_model 

    text_ids_uncond = tokenizer(tokenizer.bos_token + args.uncond_prompt, 
                            add_special_tokens=False, return_tensors='pt', padding='longest').input_ids[0]
 
    if args.ranking.endswith('.pt'):
        all_ranking = torch.load(args.ranking)
    else:
        all_ranking = np.load(args.ranking)

    if args.use_gt:
        gt_id = torch.cat([torch.arange(4000), torch.arange(5000, 8000)])
        all_ranking[sum(args.nums_selected_txt_query[:2]):, 0] = gt_id

    assert len(all_ranking) == sum(args.nums_selected_txt_query), f'{len(all_ranking)} != {sum(args.nums_selected_txt_query)}'

    ids_dict = get_tokenized_ids(args, tokenizer)

    ranking = all_ranking
    if args.post_filtering_mask is not None:
        assert len(ids_dict['text_input_ids']) == 16000
        assert not args.use_gt, 'Not implemented!'
        filtering_mask = torch.load(args.post_filtering_mask)
        all_gen_ids = [all_gen_ids[i] for i in range(len(all_gen_ids)) if filtering_mask[i]]
        text_input_ids = [ids_dict['text_input_ids'][i] for i in range(len(ids_dict['text_input_ids'])) if filtering_mask[i]]
        selected_image_ids = [ids_dict['img_input_ids'][i] for i in range(len(ids_dict['img_input_ids'])) if filtering_mask[8000:][i]]
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

        

    dataset = IdDataset(ids_dict, all_gen_ids, text_ids_uncond, all_ranking)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=False)

    

    is_retr_arr = np.array([], dtype=np.float32)
    txtidx2score = collections.OrderedDict()
    pbar = tqdm(enumerate(loader), total=math.ceil(len(dataset) / args.bs), desc='Prob')
    for i_batch, batch in pbar:
        inds, retr_ids_padded, retr_ids_uncond_padded, gen_ids_padded, gen_ids_uncond_padded = batch # (bs, max_seq_len)
        log_probs = []
        for ids_padded in [retr_ids_padded, retr_ids_uncond_padded, gen_ids_padded, gen_ids_uncond_padded]:
            with torch.no_grad():
                l = llama(input_ids=ids_padded.to(device)).logits # (bs, max_seq_len, n_vocab)
                lp = F.log_softmax(l.float(), dim=-1)
                log_probs.append(lp)

        log_probs_retr, log_probs_retr_uncond, log_probs_gen, log_probs_gen_uncond = log_probs
        boi_inds_retr, eoi_inds_retr = torch.where(retr_ids_padded == BOI)[1], torch.where(retr_ids_padded == EOI)[1]
        boi_inds_gen, eoi_inds_gen = torch.where(gen_ids_padded == BOI)[1], torch.where(gen_ids_padded == EOI)[1]
        boi_inds_retr_uncond, eoi_inds_retr_uncond = torch.where(retr_ids_uncond_padded == BOI)[1], torch.where(retr_ids_uncond_padded == EOI)[1]
        boi_inds_gen_uncond, eoi_inds_gen_uncond = torch.where(gen_ids_uncond_padded == BOI)[1], torch.where(gen_ids_uncond_padded == EOI)[1]

        for i in range(len(inds)):
            cur_lp_retr = log_probs_retr[i, boi_inds_retr[i]:eoi_inds_retr[i]-1, :] # (NUM_IMG_TOKNES, n_vocab)
            cur_lp_gen = log_probs_gen[i, boi_inds_gen[i]:eoi_inds_gen[i]-1, :]
            cur_gt_inds_retr = retr_ids_padded[i, boi_inds_retr[i]+1:eoi_inds_retr[i]].unsqueeze(-1).to(device) # (NUM_IMG_TOKNES, 1)
            cur_gt_inds_gen = gen_ids_padded[i, boi_inds_gen[i]+1:eoi_inds_gen[i]].unsqueeze(-1).to(device)
            p_all_retr = torch.gather(cur_lp_retr, dim=-1, index=cur_gt_inds_retr).squeeze(-1).sum()
            p_all_gen = torch.gather(cur_lp_gen, dim=-1, index=cur_gt_inds_gen).squeeze(-1).sum()
            cur_lp_retr_uncond = log_probs_retr_uncond[i, boi_inds_retr_uncond[i]:eoi_inds_retr_uncond[i]-1, :]
            cur_lp_gen_uncond = log_probs_gen_uncond[i, boi_inds_gen_uncond[i]:eoi_inds_gen_uncond[i]-1, :]
            cur_gt_inds_retr_uncond = retr_ids_uncond_padded[i, boi_inds_retr_uncond[i]+1:eoi_inds_retr_uncond[i]].unsqueeze(-1).to(device)
            cur_gt_inds_gen_uncond = gen_ids_uncond_padded[i, boi_inds_gen_uncond[i]+1:eoi_inds_gen_uncond[i]].unsqueeze(-1).to(device)
            p_all_retr_uncond = torch.gather(cur_lp_retr_uncond, dim=-1, index=cur_gt_inds_retr_uncond).squeeze(-1).sum()
            p_all_gen_uncond = torch.gather(cur_lp_gen_uncond, dim=-1, index=cur_gt_inds_gen_uncond).squeeze(-1).sum()
            txtidx2score[inds[i]] = {'retr': p_all_retr.item(), 'gen': p_all_gen.item(), \
                                    'retr_uncond': p_all_retr_uncond.item(), 'gen_uncond': p_all_gen_uncond.item()}
            is_retr = ((p_all_retr - p_all_retr_uncond) > (p_all_gen - p_all_gen_uncond)).item()
            
            is_retr_arr = np.append(is_retr_arr, is_retr)
            pbar.set_postfix_str(f'retr_ratio: {((is_retr_arr == 1).sum() / len(is_retr_arr) * 100):.2f}%')

        print(is_retr, p_all_retr, p_all_retr_uncond, p_all_gen, p_all_gen_uncond)
        print(p_all_retr - p_all_retr_uncond, p_all_gen - p_all_gen_uncond)
        print((eoi_inds_retr[-1]-boi_inds_retr[-1]), (eoi_inds_retr_uncond[-1]-boi_inds_retr_uncond[-1]), 
                (eoi_inds_gen[-1]-boi_inds_gen[-1]), (eoi_inds_gen_uncond[-1]-boi_inds_gen_uncond[-1]))
        print('================================')

    torch.save(txtidx2score, args.out)
    print(args)
