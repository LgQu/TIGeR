import os, sys
import argparse
from PIL import Image
import open_clip
import numpy as np
import math
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import copy
from collections import OrderedDict
from itertools import accumulate
import xlwt, copy
import warnings
from collections import OrderedDict
import csv

from tiger_utils import compute_rank_t2i, create_worksheet, compute_recalls

from data import get_combined_query_gallery


class CombinedDataset(Dataset):
    def __init__(self, text_tokens, imgs_gt, imgs_gallery, ranking, txt2img, dec_dict, all_img_path, preprocess):
        self.text_tokens = text_tokens
        self.imgs_gt = imgs_gt
        self.imgs_gallery = imgs_gallery


        self.ranking = ranking
        self.preprocess = preprocess
        self.dec_dict = dec_dict
        self.txt2img = txt2img
        self.all_img_path = all_img_path

    def __len__(self, ):
        return len(self.text_tokens)
    
    def get_img(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        img = self.preprocess(img.convert('RGB'))
        return img

    def __getitem__(self, index):
        img_gt = self.get_img(self.imgs_gt[index])

        if len(self.all_img_path) > 0:
            img_gen = self.get_img(self.all_img_path[index])
        else:
            img_gen = img_gt

        if self.ranking is None:
            img_retr = img_gt
            rank = -1
        else:
            cur_ranking = self.ranking[index]
            top1_img_idx = cur_ranking[0]
            rank = compute_rank_t2i(cur_ranking, gt_img_idx=self.txt2img[index][0])
            top1_img = self.imgs_gallery[top1_img_idx]
            try:
                img_retr = self.get_img(top1_img)
            except:
                print(top1_img)
                print(top1_img_idx)
                print(len(self.imgs_gallery))
                assert False

        if self.dec_dict is not None:
            dec = self.dec_dict[index]
            s_retr = dec['retr'] - dec['retr_uncond']
            s_gen = dec['gen'] - dec['gen_uncond']
            decide = 0 if s_retr > s_gen else 1
        else:
            if self.ranking is not None:
                decide = 0
            else:
                decide = 1

        return self.text_tokens[index], img_gen, img_retr, img_gt, decide, rank


def get_txt_tokens(all_text):
    return open_clip.tokenizer.tokenize(all_text, context_length=77)

def summarize_res(res_dict, start_end_report=None):
    res_dict_new = OrderedDict()
    res_dict_report = OrderedDict()
    for k, v in res_dict.items():
        v_new = torch.cat(v)
        s, e = start_end_report if start_end_report is not None else (0, len(v_new))
        if '%' in k:
            v_new = (v_new == 0).to(dtype=torch.float32)
        res_dict_new[k] = v_new
        if k == 'rank':
            r1, r5, r10 = compute_recalls(v_new[s:e].numpy())
            res_dict_report['r@1'] = f'{r1:.2f}'
            res_dict_report['r@5'] = f'{r5:.2f}'
            res_dict_report['r@10'] = f'{r10:.2f}'
        else:
            res_dict_report[k] = f'{v_new[s:e].mean().item()*100:.2f}'

    dec_gen = res_dict_new['dec'][s:e].mean().item() - res_dict_new['gen'][s:e].mean().item()
    dec_retr = res_dict_new['dec'][s:e].mean().item() - res_dict_new['retr'][s:e].mean().item()
    res_dict_report['dec-gen'] = f'{dec_gen*100:.2f}'
    res_dict_report['dec-retr'] = f'{dec_retr*100:.2f}'
    res_dict_report['^dec'] = f'{(dec_gen + dec_retr)*100:.2f}'

    dec_gen = res_dict_new['dec(II)'][s:e].mean().item() - res_dict_new['gen(II)'][s:e].mean().item()
    dec_retr = res_dict_new['dec(II)'][s:e].mean().item() - res_dict_new['retr(II)'][s:e].mean().item()
    res_dict_report['dec-gen(II)'] = f'{dec_gen*100:.2f}'
    res_dict_report['dec-retr(II)'] = f'{dec_retr*100:.2f}'
    res_dict_report['^dec(II)'] = f'{(dec_gen + dec_retr)*100:.2f}'
    return res_dict_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="")
    parser.add_argument("--ann_root", default="")
    parser.add_argument('--cand_path', type=str, default='')
    parser.add_argument('--img_gen_dir_prefix', type=str, default='./generated/')
    parser.add_argument('--img_gen_dir_multi', type=str, default=[], nargs='+')
    parser.add_argument('--img_retr_path_multi', type=str, default=[], nargs='+')
    parser.add_argument('--dataset', type=str, default='logo2k')
    parser.add_argument('--datasets', type=str, default=[], nargs='+')
    parser.add_argument('--num_selected', type=int, default=500)
    parser.add_argument('--num_selected_txt', type=int, default=500)
    parser.add_argument("--num_selected_multi", nargs="+", default=[], type=int)
    parser.add_argument("--num_selected_txt_multi", nargs="+", default=[], type=int)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--diffusiondb_split', type=str, default='2m_random_1k')

    parser.add_argument("--prefix_prompt", const="", nargs="?", default='Generate an image of ')
    parser.add_argument('--ranking', type=str, default=None)
    parser.add_argument('--decision', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--datasets_query', type=str, nargs='+', default=[])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', default=[])
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[])
    parser.add_argument('--nums_selected_txt_query', type=int, nargs='+', default=[])
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--data_version", const="", nargs="?", default='tiger_bench')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument("--only_eval_gen", action='store_true')
    parser.add_argument("--expand_prompt_v", type=str, default=None)
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument("--out_postfix", nargs='?', const="", default="")
    parser.add_argument('--post_filtering_mask', type=str, default=None)
    parser.add_argument("--baseline_img_query_dir", type=str, default=None)
    args = parser.parse_args()
    print(args)
    device = 'cuda'
    prefix_prompt = args.prefix_prompt
    args.img_gen_dir_multi = [os.path.join(args.img_gen_dir_prefix, d) for d in args.img_gen_dir_multi]

    
    
    

    print('Load model ...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k'
    )
    model = model.to(device)
    
    print('Load Data ...')
    sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                args.nums_selected_txt_query, args.nums_selected_gallery, prefix_prompt, 
                                                                args=args, load_img=True)
    print(sampled_text[:3])

    ranking = None
    if args.ranking is not None:
        if args.ranking.endswith('.pt'):
            ranking = torch.load(args.ranking)
        else:
            ranking = np.load(args.ranking)

        if args.use_gt:
            for i in range(8000):
                ranking[sum(args.nums_selected_txt_query[:2]) + i][0] = i

        if len(ranking) == 8000 and args.post_filtering_mask is not None:
            warnings.warn('Pad ranking later ...')
        else:
            assert len(ranking) == sum(args.nums_selected_txt_query)

    query2did = [i_dset for i_dset, n_stq in enumerate(args.nums_selected_txt_query) for _ in range(n_stq) ]# query to dataset id
    query2internal_idx = [i for i_dset, n_stq in enumerate(args.nums_selected_txt_query) for i in range(n_stq) ]
    all_img_path_gen = []
    if len(args.img_gen_dir_multi) > 0:
        for index in range(len(query2did)):
            did = query2did[index]
            internal_idx = query2internal_idx[index]
            img_path = os.path.join(args.img_gen_dir_multi[did], f'{internal_idx:05d}.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(args.img_gen_dir_multi[did], f'{internal_idx}.jpg')
            all_img_path_gen.append(img_path)
    elif args.baseline_img_query_dir is not None:
        assert len(os.listdir(args.baseline_img_query_dir)) == sum(args.nums_selected_txt_query)
        for i in range(sum(args.nums_selected_txt_query)):
            all_img_path_gen.append(os.path.join(args.baseline_img_query_dir, f'{i}.jpg'))

    if args.post_filtering_mask is not None:
        if len(ranking) == 8000:
            ranking = 8000 * [np.array([1000])] + ranking
            warnings.warn('Pad ranking ...')

        assert len(sampled_text) == 16000
        assert not args.use_gt, 'Not implemented!'
        filtering_mask = torch.load(args.post_filtering_mask)
        sampled_text = [sampled_text[i] for i in range(len(sampled_text)) if filtering_mask[i]]
        sampled_imgs = [sampled_imgs[i] for i in range(len(sampled_imgs)) if filtering_mask[8000:][i]]
        gt_imgs = [gt_imgs[i] for i in range(len(gt_imgs)) if filtering_mask[i]]
        assert len(sampled_text) == len(sampled_imgs) * 2 == filtering_mask.sum(), f'{len(sampled_text)}, {len(sampled_imgs)}, {filtering_mask.sum()}'
        assert len(gt_imgs) == filtering_mask.sum()
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
                new_ranking.append(np.array(cur_r))
        assert len(new_ranking) == filtering_mask.sum()
        ranking = new_ranking
        txt2img = OrderedDict()
        for i in range(len(new_ranking)):
            if i < len(new_ranking) // 2:
                txt2img[i] = [-1]
            else:
                txt2img[i] = [i - len(new_ranking) // 2]

        all_rank = []
        for i, cur_ranking in enumerate(ranking[3000: ]):
            rank = compute_rank_t2i(cur_ranking, gt_img_idx=txt2img[i + 3000][0])
            all_rank.append(rank)

        r1, r5, r10 = compute_recalls(np.array(all_rank))

        all_img_path_gen = [all_img_path_gen[i] for i in range(len(all_img_path_gen)) if filtering_mask[i]]
        assert len(ranking) == filtering_mask.sum(), f'{len(ranking)}, {filtering_mask.sum()}'
        assert len(all_img_path_gen) == filtering_mask.sum() or len(all_img_path_gen) == 0, f'{len(all_img_path_gen)}, {filtering_mask.sum()}'
        args.datasets_query = ['whoops', 'pickapic_7500', 'logo2k', 'visual_news', 'google_landmark', 'food2k', 'inatural', 'wit']
        args.datasets_gallery = ['logo2k', 'visual_news', 'google_landmark', 'food2k', 'inatural', 'wit']
        args.nums_selected_txt_query = [500, 2500] + [500] * 6
        args.nums_selected_gallery = [500] * 6


    decision = None
    if args.decision is not None:
        if args.decision.endswith('.csv'): # VQA for decision
            id2dec = {}
            with open(args.decision, 'r') as f:
                reader = [each for each in csv.DictReader(f, delimiter='\t')]
            for r in reader:
                data_id = int(r['Img1'].split('/')[-1].split('.')[0])
                if r['Response'] == 'first':
                    id2dec[data_id] = {'retr': 1, 'gen': 0, 'retr_uncond': 0, 'gen_uncond': 0}
            decision = OrderedDict()
            for data_id in range(len(sampled_text)):
                decision[data_id] = id2dec.get(data_id, 
                                    {'retr': int(data_id%2==0), 'gen': int(data_id%2==0), 'retr_uncond': 0, 'gen_uncond': 0})
        else:
            decision = torch.load(args.decision)
            assert len(decision) == len(sampled_text)
    sampled_text_tokens = get_txt_tokens(sampled_text)

    dataset = CombinedDataset(sampled_text_tokens, gt_imgs, sampled_imgs, ranking, txt2img, decision, all_img_path_gen, preprocess)
    print('len(sampled_text) = len(dataset) = ', len(dataset))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    bs = args.batch_size
    scores_gen, scores_retr, scores_dec, scores_dec_clip, scores_gt = [], [], [], [], []
    scores_gen_ii, scores_retr_ii, scores_dec_ii, scores_dec_clip_ii = [], [], [], []
    pbar = tqdm(enumerate(loader), total=math.ceil(len(dataset) / bs), desc='Sim')
    all_decide, all_decide_clip, all_rank = [], [], []
    all_keys = ['retr', 'gen', 'dec', 'dec_clip', 'gt', \
                'retr(II)', 'gen(II)', 'dec(II)', 'dec_clip(II)', \
                'retr%', 'retr_clip%', 'rank']
    res_dict = OrderedDict({k: None for k in all_keys})
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i_batch, batch in pbar:
            text_tokens, img_gen, img_retr, img_gt, decide, rank = batch # decide: 0 -- retr, 1 -- gen
            image_features_gen = model.encode_image(img_gen.to(device))
            image_features_retr = model.encode_image(img_retr.to(device))
            image_features_gt = model.encode_image(img_gt.to(device))
            text_features = model.encode_text(text_tokens.to(device))

            image_features_gen /= image_features_gen.norm(dim=-1, keepdim=True)
            image_features_retr /= image_features_retr.norm(dim=-1, keepdim=True)
            image_features_gt /= image_features_gt.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            scores_gen.append((image_features_gen * text_features).sum(dim=-1).cpu())
            scores_retr.append((image_features_retr * text_features).sum(dim=-1).cpu())
            scores_dec.append(torch.gather(torch.stack([scores_retr[-1], scores_gen[-1]], dim=0), index=decide.unsqueeze(0), dim=0).flatten())
            dec_clip = (scores_retr[-1] < scores_gen[-1]).to(dtype=text_features.dtype)
            scores_dec_clip.append(scores_retr[-1] * (1 - dec_clip) + scores_gen[-1] * dec_clip)
            scores_gt.append((image_features_gt * text_features).sum(dim=-1).cpu())
            scores_gen_ii.append((image_features_gen * image_features_gt).sum(dim=-1).cpu())
            scores_retr_ii.append((image_features_retr * image_features_gt).sum(dim=-1).cpu())
            scores_dec_ii.append(torch.gather(torch.stack([scores_retr_ii[-1], scores_gen_ii[-1]], dim=0), index=decide.unsqueeze(0), dim=0).flatten())
            scores_dec_clip_ii.append(scores_retr_ii[-1] * (1 - dec_clip) + scores_gen_ii[-1] * dec_clip)

            all_decide.append(decide)
            all_rank.append(rank)
            all_decide_clip.append(dec_clip)
            res = [scores_retr, scores_gen, scores_dec, scores_dec_clip, scores_gt, \
                    scores_retr_ii, scores_gen_ii, scores_dec_ii, scores_dec_clip_ii, \
                    all_decide, all_decide_clip, all_rank]
            res_dict = OrderedDict({k: res[i] for i, k in enumerate(all_keys)})
            res_dict_report = summarize_res(res_dict)
            print(res_dict_report)
            print('-'*125)
    
print('#'*125)
print('#'*125)

start_lst = [0] + list(accumulate(args.nums_selected_txt_query[:-1])) + [0]
end_lst = list(accumulate(args.nums_selected_txt_query)) + [sum(args.nums_selected_txt_query)]
datasets_query = args.datasets_query + ['average']
nums_selected_txt_query = args.nums_selected_txt_query + [sum(args.nums_selected_txt_query)]
all_res_dict_report = []
for i, se in enumerate(list(zip(start_lst, end_lst))):
    res_dict_report = summarize_res(res_dict, start_end_report=se)
    all_res_dict_report.append(res_dict_report)
    print(f'dataset: {datasets_query[i]}, num: {nums_selected_txt_query[i]}, \n\t res: {res_dict_report}')
    print('-'*125)

workbook, worksheet, style0, style1, style_num = create_worksheet()
for i, k in enumerate(all_res_dict_report[0].keys()):
    worksheet.write(0, i+1, k)
for i in range(len(all_res_dict_report)):
    worksheet.write(i+1, 0, datasets_query[i])
    for j, (k, v) in enumerate(all_res_dict_report[i].items()):
        worksheet.write(i+1, j+1, v, style_num)

if args.decision is not None:
    save_name = args.decision.split('/')[-1].split('.')[:-1]
    save_name = '.'.join(save_name)
    save_name = save_name.split('_')[1:]
    save_name = '_'.join(save_name)
    path = os.path.join(os.path.dirname(args.decision), f'{save_name}{args.out_postfix}.xls')
    res_path = os.path.join(os.path.dirname(args.decision), f'res_dict_{save_name}{args.out_postfix}.pt')
else:
    save_name = 'no_decision'
    if args.ranking is not None:
        path = os.path.join(os.path.dirname(args.ranking), f'{save_name}{args.out_postfix}.xls')
        res_path = os.path.join(os.path.dirname(args.ranking), f'res_dict_{save_name}{args.out_postfix}.pt')
    else:
        assert args.out_dir is not None
        path = os.path.join(args.out_dir, f'{save_name}{args.out_postfix}.xls')
        res_path = os.path.join(args.out_dir, f'res_dict_{save_name}{args.out_postfix}.pt')

workbook.save(path)
print(f'Save {path}')
torch.save(res_dict, res_path)
