import os, re, sys
import json
import math
from typing import List, Tuple
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from datasets import load_dataset


from data import TigerBenchDataset

from PIL import Image

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

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption
    

def truncate_context(prompt, mode='1txt', turn='single_turn'):
    if mode is None:
        return prompt

    if turn == 'single_turn':
        s, e = '<img>', '</img>'
        assert prompt.count(s) == prompt.count(e)
        lst_e = prompt.split(e)
        if mode == '1txt':
            res = lst_e[-1]
        elif mode == '2txt':
            res = lst_e[-2].split(s)[0] + ' ' + lst_e[-1]
        elif mode == '3txt':
            res = lst_e[-3].split(s)[0] + ' ' + lst_e[-2].split(s)[0] + ' ' + lst_e[-1]
        elif mode == '4txt':
            res = lst_e[-4].split(s)[0] + lst_e[-3].split(s)[0] + ' ' + lst_e[-2].split(s)[0] + ' ' + lst_e[-1]
        elif mode == '5txt':
            res = lst_e[-5].split(s)[0] + lst_e[-4].split(s)[0] + lst_e[-3].split(s)[0] + ' ' + lst_e[-2].split(s)[0] + ' ' + lst_e[-1]
        elif mode == '2txt_1img':
            res = e.join(lst_e[-2:])
        elif mode == '3txt_2img':
            res = e.join(lst_e[-3:])
        elif mode == '4txt_3img':
            res = e.join(lst_e[-4:])
        elif mode == '5txt_4img':
            res = prompt
        else:
            raise ValueError('Unknown mode: '+ mode)
    else:
        e = '</img>\n'
        assert prompt.count(e) == 4
        lst_e = prompt.split(e)
        if mode == '5txt_4img':
            res = prompt
        elif mode == '4txt_3img':
            res = e.join(lst_e[-4:])
        elif mode == '3txt_2img':
            res = e.join(lst_e[-3:])
        elif mode == '2txt_1img':
            res = e.join(lst_e[-2:])
        else:
            raise ValueError('Unknown mode: '+ mode)
        res = res.replace('<s>', '') # multi_turn_v1
    return res


    
if __name__ == "__main__":
    import hydra, torch, os
    from omegaconf import OmegaConf
    from tqdm import tqdm
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default='')
    parser.add_argument("--ann_root", default='')
    parser.add_argument('--cand_path', type=str, default='')
    parser.add_argument("--out", required=True)
    parser.add_argument("--i2t", action='store_true')
    parser.add_argument("--uncond", action='store_true')
    parser.add_argument("--bs", default=512, type=int)
    parser.add_argument("--uncond_txt", default="")
    parser.add_argument("--uncond_img", default=None, choices=[None, 'white_noise'])
    parser.add_argument('--dataset', type=str, default='logo2k')
    parser.add_argument('-d_sample', '--dataset_sample_strategy', type=str, default='top')
    parser.add_argument("--data_version", nargs='?', const="", default='tiger_bench')
    parser.add_argument("--data_split", default='test')
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument("--truncate_mode", default=None)
    parser.add_argument("--turn", default='single_turn', choices=['single_turn', 'multi_turn'], help='especially for VIST')
    parser.add_argument("--expand_prompt", action='store_true')
    parser.add_argument("--prefix_prompt", nargs='?', const="", default="Generate an image of ")
    

    args = parser.parse_args()

    device = 'cuda'
    tokenizer_cfg_path = 'configs/tokenizer.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device_map='auto', padding_side='right', load_diffusion=True)
    transform_cfg_path = 'configs/transform.yaml'
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    
    prefix_prompt = args.prefix_prompt

    ''' ################################################# Dataset ################################################# '''
    img_dir = args.image_root
    ann_file = args.ann_root

    if args.dataset == 'diffusiondb':
        test = DiffusionDB(prefix_prompt=prefix_prompt)
        assert len(test.text) == len(test.image)
    else:
        test = TigerBenchDataset(args, prefix_prompt=prefix_prompt, load_img=True)
    

    """Pre-compute text ids"""
    
    """Pre-compute img ids"""
    img_id_path = os.path.join('./intermediate_results', args.data_version, f'img_ids_{args.dataset}.pt')
    os.makedirs(os.path.dirname(img_id_path), exist_ok=True)
    if not os.path.exists(img_id_path):
        images = []
        for i, im in enumerate(tqdm(test.image, desc="Image", unit="img")):
            if isinstance(im, str):
                image = Image.open(os.path.join(img_dir, im)).convert('RGB')
            else:
                image = im.convert('RGB')
            image_tensor = transform(image)
            images.append(image_tensor)
        
        image_tensors = torch.stack(images, dim=0)
        print("Images tensors:", image_tensors.shape)
        img_ids = []
        for i_batch in tqdm(range(0, len(image_tensors), args.bs), total=math.ceil(len(image_tensors) / args.bs)):
            image_tensors_batch = image_tensors[i_batch : i_batch + args.bs]
            img_ids_batch = tokenizer.encode_image(image_torch=image_tensors_batch.to(device))
            img_ids.append(img_ids_batch)
        img_ids = torch.cat(img_ids, dim=0)
        print("Img ids:", img_ids.shape)
        torch.save(img_ids.cpu(), img_id_path)
    else:
        img_ids = torch.load(img_id_path)
        if isinstance(img_ids, List):
            img_ids = torch.stack(img_ids)
    img_tokens = [BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in im.view(-1).cpu().numpy()]) + EOI_TOKEN for im in img_ids]

    if args.uncond_img is not None:
        if args.uncond_img == 'white_noise':
            w, h = 512, 512
            seed = 42
            np.random.seed(seed)
            noise = Image.fromarray(np.random.randint(0,255,(w, h, 3),dtype=np.dtype('uint8')))
            noise.save(f'./intermediate_results/uncond_white_noise_seed{seed}.jpg')
            noise_tensor = transform(noise)
            img_ids_uncond = tokenizer.encode_image(image_torch=noise_tensor.to(device))
            print(f'img_ids_uncond.shape: {img_ids_uncond.shape} \nimg_ids_uncond: {img_ids_uncond}')

    if not args.i2t: #  text-to-image
        text_size = len(test.text)
        image_size = len(img_tokens)
        sim_t2i = torch.zeros((text_size, image_size))
        text = test.text
        img_input_ids = tokenizer(img_tokens, add_special_tokens=False, return_tensors='pt').input_ids
        if args.uncond:
            if args.uncond_img is None:
                inputs = [tokenizer.bos_token  + s_token + " " + args.uncond_txt + sep + e_token for _ in text]
            else:
                inputs = [tokenizer.bos_token  + s_token + " " +  \
                            BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in img_ids_uncond.view(-1).cpu().numpy()]) + EOI_TOKEN + \
                            " " + args.uncond_txt + sep + e_token for _ in text]

        else: 
            if args.turn == 'single_turn':
                inputs = [tokenizer.bos_token  + s_token + " " + truncate_context(t, args.truncate_mode, args.turn) + sep + e_token for t in text]
            else:
                inputs = [tokenizer.bos_token + truncate_context(t, args.truncate_mode, args.turn) + sep + e_token for t in text] # multi_turn_v1

        
        for ii in range(10):
            print(inputs[ii])
            print('-'*125)
            
        text_input_ids = tokenizer(inputs, add_special_tokens=False, return_tensors='pt', padding="longest").input_ids
        print("text_input_ids", text_input_ids.shape,"img input ids:", img_input_ids.shape)
        torch.save({'img_input_ids': img_input_ids.cpu(), 'text_input_ids': text_input_ids.cpu()}, args.out)

    else: # for inverse prompting
        text_size = len(test.text)
        image_size = len(img_tokens)
        sim_t2i = torch.zeros((text_size,image_size))
        text = test.text
    
        inputs = [tokenizer.bos_token  + s_token + " " + img + "Generate caption." + sep + e_token for img in img_tokens]
        text_input_ids = tokenizer(text, add_special_tokens=False, return_tensors='pt', padding="longest").input_ids
        txt_id_path = os.path.join('./intermediate_results', args.data_version, f'text_ids_{args.dataset}.pt')
        torch.save(text_input_ids.cpu(), txt_id_path)
        print("Inputs:", inputs[0], text[0])
        input_ids = tokenizer(inputs, add_special_tokens=False, return_tensors='pt', padding="longest").input_ids
        print("Input ids:", input_ids[0], text_input_ids[0])
        torch.save({'prompt_img_input_ids': input_ids.cpu(), 'text_input_ids': text_input_ids.cpu()}, args.out)

    
    print(f'Save {args.out}')
    print('#'*155)
