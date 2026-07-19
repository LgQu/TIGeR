import os
import argparse
import torch
import random
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import build_model

from tiger_utils import check_device
from data import get_combined_query_gallery



def generate_img(args, prompts): # highres | lowres
    query2did = [i_dset for i_dset, n_stq in enumerate(args.nums_selected_txt_query) for _ in range(n_stq)]# query to dataset id
    query2local_idx = [i for i_dset, n_stq in enumerate(args.nums_selected_txt_query) for i in range(n_stq) ]
    for dq in args.datasets_query:
        os.makedirs(os.path.join(args.out_dir, dq), exist_ok=True)
    if args.pixel_decoding == 'highres':
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
        use_xformers = True
    else:
        height, width = 512, 512
        use_xformers = False

    model = build_model(model_path=model_path, model_dtype=model_dtype, check_safety=False, load_tokenizer=True,
                device_id=device_id, use_xformers=False, understanding=False, local_files_only=True, 
                pixel_decoding=args.pixel_decoding, load_pixel_decoder=not args.only_image_ids)

    model_num_param = sum(p.numel() for p in model.parameters())
    print(f'model size: {model_num_param / 10**9:.2f}B')

    all_img_ids = []
    old_d_name = ''
    token_lens = []

    pbar = tqdm(enumerate(prompts), total=len(prompts))
    for i, prompt in pbar:
        d_name = args.datasets_query[query2did[i]]
        if d_name != old_d_name:
            img_ids_path = os.path.join(args.out_dir, f'img_ids_{d_name}.pt')
            all_img_ids = []
            if os.path.exists(img_ids_path):
                all_img_ids = torch.load(img_ids_path)

        local_i = query2local_idx[i]
        img_save_path = os.path.join(args.out_dir, d_name, f'{local_i:05d}.jpg')
        
        if local_i < len(all_img_ids):
            assert os.path.exists(img_save_path)
            token_lens.append(len(all_img_ids[local_i]))
            continue
        
        gen_args = {
                "top_p": 1.0, 
                "top_k": 50,
            } # default
        
        print(f'Prompt: {prompt}')
        if args.only_image_ids:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                img_ids = model.generate_image(prompt, width=width, height=height, 
                    guidance_scale_for_llm=args.guidance_scale_for_llm, num_return_images=1, 
                    return_image_ids=True, **gen_args)
                img_ids = img_ids[0]
        else:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                img_ids, image = model.generate_image(prompt, width=width, height=height, 
                    guidance_scale_for_llm=args.guidance_scale_for_llm, num_return_images=1, 
                    return_image_ids=True, **gen_args)
                img_ids, image = img_ids[0], image[0]
                image.save(img_save_path)
    
        all_img_ids.append(img_ids)
        old_d_name = d_name

        token_lens.append(len(img_ids))
        token_lens_ts = torch.FloatTensor(token_lens)
        mean, std = token_lens_ts.mean().item(), token_lens_ts.std().item()
        min_l, max_l = token_lens_ts.min().item(), token_lens_ts.max().item()
        pbar.set_postfix_str(f'avg_len: {mean:.2f}, std: {std:.2f}, min: {min_l}, max: {max_l}')

        if (local_i + 1) % args.save_interval == 0 or local_i + 1 == args.nums_selected_txt_query[query2did[i]]:
            img_ids_path = os.path.join(args.out_dir, f'img_ids_{d_name}.pt')
            torch.save(all_img_ids, img_ids_path)
            print(f'Save {img_ids_path} at step {local_i}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait_gpu", type=int, default=None)
    parser.add_argument("--guidance_scale_for_llm", type=int, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--pixel_decoding', type=str, default='highres')
    parser.add_argument('--model_type', type=str, choices=['bf16', 'fp16'], default='bf16')
    parser.add_argument('--datasets_query', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', 
        default=['artbench', 'logo2k', 'visual_news', 'google_landmark', 'vox_celeb', 'food2k', 'inatural', 'wit'])
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument('--nums_selected_txt_query', type=int, nargs='+', default=[1000] * 8)
    parser.add_argument("--prefix_prompt", type=str, default='Generate an image of ')
    parser.add_argument("--data_version", nargs='?', const="", default='combV3')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default='../SEED/configs/tiger/data.yaml')
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--only_image_ids", action="store_true")
    parser.add_argument("--expand_prompt_v", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    args = parser.parse_args()

    model_path = os.environ.get('LAVIT_MODEL')
    if not model_path:
        raise RuntimeError('Set LAVIT_MODEL to the LaVIT checkpoint directory.')
    model_dtype=args.model_type

    

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device_id = 'auto'
    device = torch.device('cuda')
    torch_dtype = torch.bfloat16 if model_dtype == 'bf16' else torch.float16

    sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs = get_combined_query_gallery(args.datasets_query, args.datasets_gallery, \
                                                                    args.nums_selected_txt_query, args.nums_selected_gallery, args.prefix_prompt, 
                                                                    args=args)
    check_device(args.wait_gpu) 
    generate_img(args, sampled_text)
