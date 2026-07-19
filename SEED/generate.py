import hydra
import argparse
import os, sys
import torch
from  tqdm import tqdm
from collections import OrderedDict
from datasets import load_dataset
import warnings
from retrying import retry
import traceback
import math
import re

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from pytorch_lightning import seed_everything

from preprocess import truncate_context
from data import TigerBenchDataset
from data import get_combined_query_gallery
from tiger_utils import check_device, my_green, my_red


BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKENS = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

generation_config = {
    'temperature': 1.0,
    'num_beams': 1,
    'max_new_tokens': 512,
    'top_p': 0.5,
    'do_sample': True
}

s_token = "USER:"
e_token = "ASSISTANT:"
sep = "\n"



def generate(tokenizer, input_tokens, generation_config, model):
    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to("cuda")

    generate_ids = model.generate(
        input_ids=input_ids,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]
    
    return generate_ids

def decode_image(batch_image_ids, tokenizer, batch_save_path):
    images = tokenizer.decode_image(batch_image_ids, num_inference_steps=args.timestep)
    for image, save_path in zip(images, batch_save_path):
        image.save(save_path)

def parse_image_text(generate_ids, tokenizer, save_path=None):
    image_ids = None
    len_img_ids = 0
    boi_list = torch.where(generate_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        print(texts)
    else:
        eoi_index = eoi_list[0]
        if len(boi_list) == 0 and len(eoi_list) != 0: # assume that <img> has been padded in the context
            boi_index = -1
        else:
            boi_index = boi_list[0]
            text_ids = generate_ids[:boi_index]
            if len(text_ids) != 0:
                texts = tokenizer.decode(text_ids, skip_special_tokens=True)
                print(texts)
        
        image_ids = (generate_ids[boi_index+1:eoi_index] - image_id_shift).reshape(1,-1)
        len_img_ids = image_ids.shape[1]

    
    assert len_img_ids == NUM_IMG_TOKENS, f'Invalid length of image_ids: {len_img_ids}. image_ids: {image_ids}'
    return image_ids


def retry_if_assertion_error(exception):
    print(traceback.format_exc())
    print('-'*125)
    print('Raise assertion error! Retry...')

    return isinstance(exception, AssertionError) or isinstance(exception, IndexError)

@retry(stop_max_attempt_number=20, retry_on_exception=retry_if_assertion_error)
def generate_and_parse(tokenizer, input_tokens, generation_config, model, save_path, only_gen_ids=False):
    generate_ids = generate(tokenizer, input_tokens, generation_config, model)
    
    image_ids = None
    if not only_gen_ids:
        image_ids = parse_image_text(generate_ids, tokenizer, save_path)

    return generate_ids, image_ids

def load_model(device, model_vers):
    if model_vers == '8b':
        model_cfg = OmegaConf.load('configs/seed_llama_8b.yaml')
    else:
        model_cfg = OmegaConf.load('configs/seed_llama_14b.yaml')

    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model = model.eval().to(device)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait_gpu", type=int, default=None)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--ann_root", default="")
    parser.add_argument('--model', type=str, default='seed')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cand_path', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--out_ids', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='logo2k')
    parser.add_argument('--text_set', type=str, default='top', choices=['raw', 'clip_filtered', 'top'])
    parser.add_argument('--num_selected', type=int, default=None)
    parser.add_argument('--select_range', type=int, default=[], nargs='+')
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument('--diffusiondb_split', type=str, default='2m_random_1k')
    parser.add_argument('--only_gen_ids', action='store_true')
    parser.add_argument('--ids_to_img', action='store_true')
    parser.add_argument("--data_version", const="", nargs='?', default='tiger_bench')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument("--turn", default='single_turn', help='especially for VIST')
    parser.add_argument("--truncate_mode", default=None)
    parser.add_argument("--model_vers", type=str, default='8b', choices=['8b', '14b'])
    parser.add_argument('--datasets_gallery', type=str, nargs='+', default=[])
    parser.add_argument('--nums_selected_gallery', type=int, nargs='+', default=[])
    parser.add_argument("--retr_res", default=None, type=str)
    parser.add_argument('--rand_seed', type=int, default=0)

    parser.add_argument("--prefix_prompt_load", nargs='?', const="", default="Generate an image of ")
    parser.add_argument("--prefix_prompt", nargs='?', const="", default="")
    parser.add_argument("--postfix_prompt", default='')
    parser.add_argument("--expand_prompt_v", type=str, default=None)
    parser.add_argument('--timestep', type=int, default=25)
    
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)
    args = parser.parse_args()
    print(args)
    device = "cuda"
    prefix_prompt_load = args.prefix_prompt_load
    postfix_prompt = args.postfix_prompt
    prefix_prompt = args.prefix_prompt
    args.prefix_prompt = prefix_prompt_load

    os.makedirs(args.out_dir, exist_ok=True)

    test = TigerBenchDataset(args, prefix_prompt=prefix_prompt_load)
    if args.text_set == 'top':
        all_text = test.text
    elif args.text_set == 'clip_filtered':
        raw, txt_clip_filtered, txt_top = test.dataset.get_all_text_set()
        all_text = txt_clip_filtered
    elif args.text_set == 'raw':
        raw, txt_clip_filtered, txt_top = test.dataset.get_all_text_set()
        all_text = raw

    seed_everything(args.rand_seed)


    if args.num_selected is None:
        args.num_selected = len(all_text)
    print(f'dataset: {args.dataset}, #all_text: {len(all_text)}')

    if args.num_selected is not None and len(args.select_range) == 0:
        args.select_range = [0, args.num_selected]
    elif args.num_selected is None and len(args.select_range) > 0:
        args.num_selected = args.select_range[1] - args.select_range[0]
    s_sel, e_sel = args.select_range
    all_text = all_text[s_sel:e_sel]
    print(f'local #text: {len(all_text)}')


    def seed(args):
        check_device(args.wait_gpu) 
        assert args.batch_size == 1, 'Only batch_size = 1 is valid!'
        tokenizer_cfg_path = 'configs/tokenizer.yaml'
        tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
        tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)
        bs = args.batch_size
        if args.ids_to_img:
            ids_gen = torch.load(args.out_ids)
            batch_save_path, batch_image_ids = [], []
            for i_txt, txt in tqdm(enumerate(all_text), total=len(all_text)):
                idx_save = i_txt + s_sel
                save_path = os.path.join(args.out_dir, f'{idx_save:05d}.jpg')
                if os.path.exists(save_path) and idx_save in ids_gen:
                    continue
                generate_ids = torch.tensor(ids_gen[idx_save], device=device)
                try:
                    image_ids = parse_image_text(generate_ids, tokenizer, save_path)
                except AssertionError:
                    print('Detect invalid image_ids, regenerate ...')
                    model = load_model(device, args.model_vers)
                    txt = truncate_context(txt, args.truncate_mode)
                    prompt = prefix_prompt + txt + postfix_prompt
                    pattern = re.compile(r'<img>.*?</img>', re.DOTALL)
                    print(pattern.sub('<img_ignored_for_print>', prompt))
                    input_tokens = tokenizer.bos_token  + s_token + " " + prompt + sep + e_token + BOI_TOKEN
                    while True:
                        print('try ...')
                        generate_ids, image_ids = generate_and_parse(tokenizer, input_tokens, generation_config, model, save_path, 
                                                            only_gen_ids=False)
                        print('generate_ids: ', generate_ids)
                        if image_ids is not None and image_ids.shape[1] == NUM_IMG_TOKENS:
                            ids_gen[idx_save] = generate_ids.tolist()
                            torch.save(ids_gen, args.out_ids)
                            print(f'Update {args.out_ids}')
                            break
                    del model
                    torch.cuda.empty_cache()

                batch_save_path.append(save_path)
                batch_image_ids.append(image_ids)
                if len(batch_save_path) == bs or (i_txt == len(all_text) - 1):
                    batch_image_ids = torch.cat(batch_image_ids, dim=0)
                    decode_image(batch_image_ids, tokenizer, batch_save_path)
                    batch_save_path, batch_image_ids = [], []
            exit()


        
        model = load_model(device, args.model_vers)
        model_num_param = sum(p.numel() for p in model.parameters())
        tokenizer_num_param = sum(p.numel() for p in tokenizer.image_tokenizer.parameters())

        print(f'model size: {model_num_param / 10**9:.2f}B')
        print(f'tokenizer.image_tokenizer size: {tokenizer_num_param / 10**9:.2f}B')

        os.makedirs(args.out_dir, exist_ok=True)
        if os.path.exists(args.out_ids):
            ids_gen = torch.load(args.out_ids)
        else:
            ids_gen = OrderedDict()
        
        batch_save_path, batch_image_ids = [], []
        for i_txt, txt in tqdm(enumerate(all_text), total=len(all_text)):
            idx_save = i_txt + s_sel
            save_path = os.path.join(args.out_dir, f'{idx_save:05d}.jpg')
            if os.path.exists(save_path) and idx_save in ids_gen:
                continue
            
            txt = truncate_context(txt, mode=args.truncate_mode, turn=args.turn)
            prompt = prefix_prompt + txt + postfix_prompt
            pattern = re.compile(r'<img>.*?</img>', re.DOTALL)
            if args.turn == 'single_turn':
                input_tokens = tokenizer.bos_token  + s_token + " " + prompt + sep + e_token + BOI_TOKEN
            else:
                prompt = prompt.replace('<s>', '')
                input_tokens = tokenizer.bos_token + prompt + sep + e_token

            print(input_tokens)


            generate_ids, image_ids = generate_and_parse(tokenizer, input_tokens, generation_config, model, save_path, 
                                                only_gen_ids=args.only_gen_ids)
            ids_gen[idx_save] = generate_ids.tolist()
            batch_save_path.append(save_path)
            batch_image_ids.append(image_ids)
            if not args.only_gen_ids and (len(batch_save_path) == bs or (i_txt == len(all_text) - 1)):
                batch_image_ids = torch.cat(batch_image_ids, dim=0)
                decode_image(batch_image_ids, tokenizer, batch_save_path)
                batch_save_path, batch_image_ids = [], []


            if i_txt % args.save_interval == 0:
                torch.save(ids_gen, args.out_ids)
        torch.save(ids_gen, args.out_ids)

    def sd(args):
        from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
        dtype = torch.float16
        if args.model == 'sd-21':
            model_id = "stabilityai/stable-diffusion-2-1-base"
            scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype, 
                                                            safety_checker=None, requires_safety_checker=False)
        elif args.model == 'sd-14':
            model_id = "CompVis/stable-diffusion-v1-4"
            scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype, 
                                                            safety_checker=None, requires_safety_checker=False)
        elif args.model == 'sdxl':
            model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
            scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype, 
                                                            safety_checker=None, requires_safety_checker=False)
        else:
            raise ValueError('Unknown model: ' + args.model)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()
        pipe = pipe.to(device)

        n_local = len(all_text)
        start = 0
        if args.n_proc > 1:
            print(f"Number of processes: {args.n_proc}")
            print(f"Process ID: {args.proc_id}")
            block_size = math.ceil(len(all_text) / args.n_proc)
            indices = list(range(0, len(all_text)+block_size, block_size))
            assert len(indices) == args.n_proc + 1
            start, end = indices[args.proc_id], indices[args.proc_id+1]
            end = start + len(all_text[start:end]) # last block
            n_local = end - start
            print(f"Number of total data points: {len(all_text)}")
            print(f"Number of local data points: {end - start}")

        generator = torch.Generator(device="cuda").manual_seed(args.random_seed)
        bs = args.batch_size
        idx_save = start
        for i in tqdm(range(0, n_local, bs), total=math.ceil(n_local/bs)):
            i_global = i + start
            txt_batch = []
            save_path_batch = []
            for j, txt in enumerate(all_text[i_global:i_global+bs]):
                save_path = os.path.join(args.out_dir, f'{idx_save:05d}.jpg')
                idx_save += 1
                if os.path.exists(save_path):
                    continue
                txt_batch.append(txt)
                save_path_batch.append(save_path)
            if len(txt_batch) > 0:
                pil_images = pipe(txt_batch, num_inference_steps=args.timestep, generator=generator).images
                for i_txt in range(len(txt_batch)):
                    pil_images[i_txt].save(save_path_batch[i_txt])

    if args.model == 'seed':
        seed(args)
    else:
        sd(args)
