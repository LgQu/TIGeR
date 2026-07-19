"""Encode the TIGeR-Bench gallery with LaVIT's dynamic visual tokenizer."""

import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from data import get_combined_query_gallery
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-id-out", required=True)
    parser.add_argument("--score-out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets-gallery", nargs="+", required=True)
    parser.add_argument("--nums-selected-gallery", nargs="+", type=int, required=True)
    parser.add_argument("--datasets-query", nargs="+", required=True)
    parser.add_argument("--nums-selected-txt-query", nargs="+", type=int, required=True)
    parser.add_argument("--prefix-prompt", default="Generate an image of ")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = os.environ.get("LAVIT_MODEL")
    if not model_path:
        raise RuntimeError("Set LAVIT_MODEL to the LaVIT checkpoint directory.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    _, gallery_images, _, _, _ = get_combined_query_gallery(
        args.datasets_query,
        args.datasets_gallery,
        args.nums_selected_txt_query,
        args.nums_selected_gallery,
        args.prefix_prompt,
        args=args,
    )
    model = build_model(
        model_path=model_path,
        model_dtype="bf16",
        check_safety=False,
        load_tokenizer=True,
        device_id="auto",
        use_xformers=True,
        understanding=False,
        local_files_only=True,
        pixel_decoding="lowres",
        load_llama=False,
    )

    image_ids, prediction_scores = [], []
    for image in tqdm(gallery_images, desc="Encoding gallery", unit="image"):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            ids, scores = model.get_selected_image_ids(
                [(image, "image")], width=1024, height=1024,
                guidance_scale_for_llm=5.0, num_return_images=1,
                select_threshold=args.threshold,
            )
        image_ids.append(ids.cpu())
        prediction_scores.append(scores.cpu())

    for path in (args.img_id_out, args.score_out):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    torch.save(image_ids, args.img_id_out)
    torch.save(prediction_scores, args.score_out)


if __name__ == "__main__":
    main()
