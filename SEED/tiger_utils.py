import torch, math
import time
from transformers import LogitsProcessor, StoppingCriteria
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import yaml
import numpy as np
import xlwt
import copy
import os

my_red = lambda str0: f"\033[31m{str0}\033[0m"
my_green = lambda str0: f"\033[32m{str0}\033[0m"
my_yellow = lambda str0: f"\033[33m{str0}\033[0m"
my_blue= lambda str0: f"\033[34m{str0}\033[0m"


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
            if fm > 20000:
                return 1
            
            time.sleep(5)
            if num_check % (12 * 5) == 0:
                print('Have been waiting for {:.1f} min.'.format((time.time()-t0)/60))
            num_check += 1

def load_yaml(path):
    with open(path, "r") as stream:
        res = yaml.safe_load(stream)
    return res

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
    style_num = xlwt.easyxf(num_format_str='0.00')
    return workbook, worksheet, style0, style1, style_num


class EarlyStoppingCriteria(StoppingCriteria):
    def __init__(self, prefix_allowed_tokens_fn, num_beams, trie):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.num_beams = num_beams
        self.trie = trie

    def __call__(self, input_ids, scores):
        del scores
        sequence_length = input_ids.shape[-1]
        is_done = False
        for batch_id, beams in enumerate(input_ids.view(-1, self.num_beams, sequence_length)):
            is_done = True
            for sequence in beams:
                allowed = self.prefix_allowed_tokens_fn(batch_id, sequence)
                if len(allowed) > 1:
                    is_done = False
                    break
                prefix = sequence.tolist()
                while self.trie.get(prefix):
                    next_tokens = self.trie.get(prefix)
                    if len(next_tokens) > 1:
                        is_done = False
                        break
                    prefix += next_tokens
        return is_done

class UnconditionalDebiasLogitsProcessor(LogitsProcessor):
    def __init__(self, uncond, context_len, uncond_context_len, prefix_allowed_tokens_fn, num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.context_len = context_len
        self.uncond_step2imgid2logprob = uncond
        self._num_beams = num_beams
        self.debias = True
    

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = input_ids.device
        seq_len = input_ids.shape[-1]
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, seq_len)):
            img_input_ids = beam_sent[..., self.context_len:] # remove input_ids context, 
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                prefix_allowed_tokens.sort() # if no sort, misalignment for logit indices  
                if img_input_ids[beam_id].shape[-1] == 0:
                    uncond_scores = self.uncond_step2imgid2logprob[0][''].to(device)
                else:
                    step = len(img_input_ids[beam_id])
                    ids_str = '_'.join([str(i) for i in img_input_ids[beam_id].tolist()])
                    uncond_scores = self.uncond_step2imgid2logprob[step][ids_str].to(device)
                
                scores[beam_id, prefix_allowed_tokens] = scores[beam_id, prefix_allowed_tokens] - \
                        uncond_scores[prefix_allowed_tokens]

        return scores

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

if __name__ == '__main__':
    a = torch.load('trie.pt')
    
    it = iter(a)
    
    trie_seqs = []

    print(a.get([1, 3148, 1001, 29901, 3251]))
    assert False
    
    for i in it:
        trie_seqs.append(i)
        print(len(i))
        print(i)

    assert False
    print("Trie Seq:", trie_seqs[0], trie_seqs[0][:45], trie_seqs[0][:44])
    print("Trie Seq:", trie_seqs[100])
    print("Trie Seq:", trie_seqs[500])
    print("Trie Seq:", trie_seqs[999])
    common_prefix = [0] * len(a)
    longest_prefix = []
    
    for i, it in enumerate(trie_seqs):
        temp_longest_prefix = []
        for j in range(45, len(it)):
            candidate = it[:j]
            if len(a.get(candidate)) > 1:
                common_prefix[i] = j-44
                temp_longest_prefix = candidate

        longest_prefix.append(temp_longest_prefix)
        temp_longest_prefix = []
    
    print("Common_Prefix:", max(common_prefix))
    print("Longest Prefix:", len(longest_prefix))
    
    longest_prefix.sort(key=len)
    print("Longest Prefix:", len(longest_prefix[-1]), len(longest_prefix[-2]), len(longest_prefix[-3]), len(longest_prefix[-4]))
    print("Longest Prefix:", longest_prefix[-1], longest_prefix[-2], longest_prefix[-3], longest_prefix[-4])
