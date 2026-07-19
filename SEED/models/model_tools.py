import torch
import transformers

from .llama import LlamaForCausalLM

if transformers.__version__ != "4.29.2":
    raise RuntimeError("SEED requires transformers==4.29.2")


def get_pretrained_llama_causal_model(pretrained_model_name_or_path=None, torch_dtype='fp16', **kwargs):
    if torch_dtype in ('fp16', 'float16'):
        torch_dtype = torch.float16
    elif torch_dtype in ('bf16', 'bfloat16'):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    return LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )
