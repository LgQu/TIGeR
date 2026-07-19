"""Generation stopping rules used by constrained LaVIT retrieval."""

import torch
from transformers import StoppingCriteria


class EarlyStoppingCriteria(StoppingCriteria):
    def __init__(self, prefix_allowed_tokens_fn, num_beams, trie):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.num_beams = num_beams
        self.trie = trie

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        del scores, kwargs
        sequence_length = input_ids.shape[-1]
        for batch_id, beams in enumerate(input_ids.view(-1, self.num_beams, sequence_length)):
            done = True
            for sentence in beams:
                allowed = self.prefix_allowed_tokens_fn(batch_id, sentence)
                if len(allowed) > 1:
                    done = False
                    break
                prefix = sentence.tolist()
                while self.trie.get(prefix):
                    next_tokens = self.trie.get(prefix)
                    if len(next_tokens) > 1:
                        done = False
                        break
                    prefix += next_tokens
            if done:
                return True
        return False
