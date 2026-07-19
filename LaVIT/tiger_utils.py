import time
import os
import numpy as np
import torch

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import yaml
import xlwt
import copy
import zipfile
from transformers import LogitsProcessor


my_red = lambda str0: f"\033[31m{str0}\033[0m"
my_green = lambda str0: f"\033[32m{str0}\033[0m"
my_yellow = lambda str0: f"\033[33m{str0}\033[0m"
my_blue= lambda str0: f"\033[34m{str0}\033[0m"

def save_code_to_zip(to_zip, save_zip_name):
    exclude_dirs = ['__pycache__', 'debug', 'wandb', 'logs', 'log', 'cache', '.github', 
                    'intermediate_results', 'generated', 'images', 'results', 'model_ckpt', 
                    'ckpt', 'pretrained', 'paper_images', 'gradio_demo', 'res', 'interm_res', 
                    'output', 'demo', 'assets']
    save_zip_dir=os.path.split(os.path.abspath(save_zip_name))[0]
    if not os.path.exists(save_zip_dir):
        os.makedirs(save_zip_dir)
        print('Create New Dir %s'%save_zip_dir)
    f = zipfile.ZipFile(os.path.abspath(save_zip_name),'w',zipfile.ZIP_DEFLATED)
    if not os.path.isdir(os.path.abspath(to_zip)):
        if os.path.exists(os.path.abspath(to_zip)):
            f.write(to_zip)
            f.close()
            print('Save code: %s --> %s' % (to_zip, save_zip_name))
        else:
            print ('%s Dir not Exist.' %os.path.abspath(to_zip))
    else:
        if os.path.exists(os.path.abspath(to_zip)):
            zipList = []
            for dir,subdirs,files in os.walk(to_zip):
                if dir.split('/')[1] not in exclude_dirs:
                    for fileItem in files:
                        if fileItem.split('.')[-1] != 'npy' and fileItem.split('.')[-1] != 'pt':
                            zipList.append(os.path.join(dir,fileItem))
                    for dirItem in subdirs:
                        if dirItem not in exclude_dirs:
                            zipList.append(os.path.join(dir,dirItem))
            for i in zipList:
                f.write(i,i.replace(to_zip,''))
            f.close()
            print('Save code: %s --> %s' % (to_zip, save_zip_name))
        else:
            print('%s Dir not Exist.' % os.path.abspath(to_zip))


def compute_recalls(ranks):
    ranks = np.array(ranks)
    ranks = ranks[ranks != -1]
    if len(ranks) == 0:
        return 0, 0, 0
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return tr1, tr5, tr10

def compute_rank_t2i(ranking, gt_img_idx):
    if gt_img_idx == -1:
        return -1
    rank = 1e20
    tmp = np.where(ranking == gt_img_idx)[0]
    if len(tmp) > 0:
        rank = tmp[0]
    return rank

def check_device(device):
    if device is None:
        return 0

    t0 = time.time()
    num_check = 0
    while(1):
        free_mem = os.popen('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free').readlines()
        free_mem = [int(line.split()[2]) for line in free_mem]
        for gpuid, fm in enumerate(free_mem):
            if gpuid not in[device]:
                continue
            if fm > 22000:
                return 1
            
            time.sleep(5)
            if num_check % (12 * 5) == 0:
                print('Have been waiting for {:.1f} min.'.format((time.time()-t0)/60))
            num_check += 1

def create_worksheet():
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('eval')
    style0 = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = '等线'
    font.height = 20 * 11 # 11 is size, 20 is metric
    style0.font = font
    style1 = xlwt.XFStyle()
    font1 = copy.deepcopy(font)
    font1.height = 20 * 8
    style1.font = font1
    return workbook, worksheet, style0, style1

class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)

class DebiasLogitsProcessorV1(LogitsProcessor):
    def __init__(self, uncond, context_len, prefix_allowed_tokens_fn, num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.context_len = context_len
        self.uncond_step2imgid2logprob = uncond
        self._num_beams = num_beams
        self.debias = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = input_ids.device
        seq_len = input_ids.shape[-1]
        pre_time, if_time, find_time, find_time_query, final_time = 0, 0, 0, 0, 0
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, seq_len)):
            img_input_ids = beam_sent[..., self.context_len:] # remove input_ids context, 
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                prefix_allowed_tokens.sort() # if no sort, misalignment for logit indices  
                if img_input_ids[beam_id].shape[-1] == 0:  # First generated token.
                    uncond_scores = self.uncond_step2imgid2logprob[0][''].to(device)
                else:
                    step = len(img_input_ids[beam_id])
                    ids_str = '_'.join([str(i) for i in img_input_ids[beam_id].tolist()])
                    uncond_scores = self.uncond_step2imgid2logprob[step][ids_str].to(device)
                
                scores[beam_id, prefix_allowed_tokens] = scores[beam_id, prefix_allowed_tokens] - \
                        uncond_scores[prefix_allowed_tokens]
        return scores

class DebiasLogitsProcessor(LogitsProcessor):
    def __init__(self, uncond, context_len, prefix_allowed_tokens_fn, num_beams: int, share_mem=False, ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.context_len = context_len
        self.share_mem = share_mem
        if share_mem: 
            self.uncond_logprob = uncond['tensor']
            self.uncond_imgid2idx = uncond['imgid2tensoridx']
        else:
            self.uncond_step2imgid2logprob = uncond
        
        self._num_beams = num_beams
        self.debias = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = input_ids.device
        dtype = scores.dtype
        seq_len = input_ids.shape[-1]
        pre_time, if_time, find_time, find_time_query, final_time = 0, 0, 0, 0, 0
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, seq_len)):
            img_input_ids = beam_sent[..., self.context_len:] # remove input_ids context, 
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                prefix_allowed_tokens.sort() # if no sort, misalignment for logit indices  
                if img_input_ids[beam_id].shape[-1] == 0:  # First generated token.
                    if self.share_mem:
                        uncond_scores = self.uncond_logprob[self.uncond_imgid2idx['']].to(device=device)
                    else:
                        uncond_scores = self.uncond_step2imgid2logprob[0][''].to(device)
                    
                else:
                    step = len(img_input_ids[beam_id])
                    ids_str = '_'.join([str(i) for i in img_input_ids[beam_id].tolist()])
                    if self.share_mem:
                        uncond_scores = self.uncond_logprob[self.uncond_imgid2idx[ids_str]].to(device=device)
                    else:
                        uncond_scores = self.uncond_step2imgid2logprob[step][ids_str].to(device)
                    

                scores[beam_id, prefix_allowed_tokens] = scores[beam_id, prefix_allowed_tokens] - \
                        uncond_scores[prefix_allowed_tokens]
        return scores



if __name__ == "__main__":
    version = 'VisualDialog' 
    imgid2logprobs = torch.load(f'./interm_res/{version}/imgid2logprob_global_thr0.5.pt')


    tensor_imgid2logprobs = []
    imgid2idx = {}
    i = 0
    for step, id2logprobs in imgid2logprobs.items():
        for imgid, logprobs in id2logprobs.items():
            imgid2idx[imgid] = i
            tensor_imgid2logprobs.append(logprobs.to(dtype=torch.bfloat16))
            i += 1

    del imgid2logprobs
    tensor_imgid2logprobs = torch.stack(tensor_imgid2logprobs, dim=0)
    print(tensor_imgid2logprobs.shape)
    print(len(imgid2idx))
    print(i)
    torch.save(tensor_imgid2logprobs, f'./interm_res/{version}/tensor_logprob_global_thr0.5.pt')
    torch.save(imgid2idx, f'./interm_res/{version}/imgid2tensoridx_global_thr0.5.pt')
