"""Shared SEED model and image-token helpers for retrieval and reranking."""

from __future__ import annotations

import os
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from models.model_tools import get_pretrained_llama_causal_model


def get_img_ids(name, num_selected, args):
    path = os.path.join("intermediate_results", args.data_version, f"img_ids_{name}.pt")
    image_ids = torch.load(path, map_location="cpu")
    if num_selected > len(image_ids):
        raise ValueError(f"Requested {num_selected} image IDs from {name}, found {len(image_ids)}")
    selected = image_ids[:num_selected]
    if isinstance(selected, List):
        selected = torch.stack(selected)
    return selected


def load_model(seed_model, is_load_model=True):
    tokenizer_config = OmegaConf.load("configs/tokenizer.yaml")
    tokenizer = hydra.utils.instantiate(
        tokenizer_config,
        device_map="auto",
        load_diffusion=False,
    )
    transform_config = OmegaConf.load("configs/transform.yaml")
    transform = hydra.utils.instantiate(transform_config)

    model = None
    if is_load_model:
        checkpoint_root = os.getenv("CKPT_ROOT")
        if not checkpoint_root:
            raise RuntimeError("CKPT_ROOT is required")
        model = get_pretrained_llama_causal_model(
            os.path.join(checkpoint_root, seed_model),
            torch_dtype="fp16",
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model.eval()
    return tokenizer, transform, model


def compute_recalls(ranks):
    ranks = np.asarray(ranks)
    if len(ranks) == 0:
        return 0.0, 0.0, 0.0
    return tuple(100.0 * np.count_nonzero(ranks < cutoff) / len(ranks) for cutoff in (1, 5, 10))


def compute_rank_t2i(ranking, ground_truth_image_id):
    matches = np.where(np.asarray(ranking) == ground_truth_image_id)[0]
    return matches[0] if len(matches) else float("inf")
