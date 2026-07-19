"""TIGeR-Bench dataset access used by the SEED implementation."""

from __future__ import annotations

import copy
import os
from collections import OrderedDict
from typing import Sequence

from datasets import load_dataset
from torch.utils.data import Dataset


DATASET_ID = "leigangqu/TIGeR-Bench"
SPLIT_ALIASES = {
    "whoops": "whoops",
    "pick_a_pic": "pick_a_pic",
    "pickapic_7500": "pick_a_pic",
    "logo2k": "logo2k",
    "visual_news": "visual_news",
    "google_landmark": "google_landmark",
    "food2k": "food2k",
    "inatural": "inaturalist",
    "inaturalist": "inaturalist",
    "wit": "wit",
}


class TigerBenchDataset(Dataset):
    """Compatibility wrapper around one public TIGeR-Bench split."""

    def __init__(self, args, transform=None, load_img=False, prefix_prompt="Generate an image of "):
        del load_img
        name = args.dataset
        if name not in SPLIT_ALIASES:
            choices = ", ".join(sorted(SPLIT_ALIASES))
            raise ValueError(f"Unsupported dataset {name!r}. Choose one of: {choices}")

        split = SPLIT_ALIASES[name]
        token = os.getenv("HF_TOKEN")
        cache_dir = os.getenv("HF_DATASETS_CACHE")
        dataset = load_dataset(DATASET_ID, split=split, token=token, cache_dir=cache_dir)

        images = list(dataset["image"])
        if transform is not None:
            images = [transform(image.convert("RGB")) for image in images]

        prompt = prefix_prompt + ("the logo of " if name == "logo2k" else "")
        self.image = images
        self.text = [prompt + text for text in dataset["text"]]
        self.dataset = dataset
        self.data_kw_args = {}
        self.txt2img = OrderedDict((i, [i]) for i in range(len(self.text)))
        self.img2txt = OrderedDict((i, [i]) for i in range(len(self.image)))

    def __len__(self):
        return len(self.text)


def get_combined_query_gallery(
    datasets_query: Sequence[str],
    datasets_gallery: Sequence[str],
    nums_selected_txt_query: Sequence[int],
    nums_selected_gallery: Sequence[int],
    prefix_prompt: str,
    args=None,
    load_img: bool = False,
    return_dict: bool = False,
):
    """Load and concatenate query and gallery splits with global ground-truth IDs."""
    if args is None:
        raise ValueError("args is required")
    if len(datasets_query) != len(nums_selected_txt_query):
        raise ValueError("Each query split needs a sample count")
    if len(datasets_gallery) != len(nums_selected_gallery):
        raise ValueError("Each gallery split needs a sample count")

    cache = {}

    def load(name, load_images=False):
        key = (name, load_images)
        if key not in cache:
            split_args = copy.copy(args)
            split_args.dataset = name
            cache[key] = TigerBenchDataset(
                split_args,
                prefix_prompt=prefix_prompt,
                load_img=load_images,
            )
        return cache[key]

    gallery_offsets = {}
    sampled_imgs = []
    offset = 0
    for name, count in zip(datasets_gallery, nums_selected_gallery):
        dataset = load(name, load_img)
        if count > len(dataset.image):
            raise ValueError(f"Requested {count} images from {name}, found {len(dataset.image)}")
        gallery_offsets[name] = (offset, offset + count - 1)
        sampled_imgs.extend(dataset.image[:count])
        offset += count

    sampled_text = []
    gt_imgs = []
    txt2img = OrderedDict()
    query2min_max_gallery = OrderedDict()
    query_index = 0
    for name, count in zip(datasets_query, nums_selected_txt_query):
        dataset = load(name, load_img)
        if count > len(dataset.text):
            raise ValueError(f"Requested {count} prompts from {name}, found {len(dataset.text)}")
        sampled_text.extend(dataset.text[:count])
        gt_imgs.extend(dataset.image[:count])
        bounds = gallery_offsets.get(name, (-1, -1))
        for local_index in range(count):
            target = bounds[0] + local_index if bounds[0] >= 0 and local_index <= bounds[1] - bounds[0] else -1
            txt2img[query_index] = [target]
            query2min_max_gallery[query_index] = bounds
            query_index += 1

    result = sampled_text, sampled_imgs, txt2img, query2min_max_gallery, gt_imgs
    return (*result, {}) if return_dict else result
