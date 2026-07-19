# TIGeR with LaVIT

This directory contains the cleaned LaVIT implementation used for TIGeR. It downloads [TIGeR-Bench](https://huggingface.co/datasets/leigangqu/TIGeR-Bench) directly from Hugging Face and does not require a local dataset configuration file.

The pipeline covers gallery tokenization, unconditional-prior extraction, retrieval, image generation, inverse-prompt reranking, retrieval/generation routing, and evaluation.

## Setup

Create a dedicated environment because LaVIT and SEED require different `transformers` versions:

```bash
conda create -n tiger-lavit python=3.8 -y
conda activate tiger-lavit
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the official `rain1011/LaVIT-7B-v2` checkpoint and point `LAVIT_MODEL` to it:

```bash
export LAVIT_MODEL=/path/to/LaVIT-7B-v2
export GPU=0
```

The scripts use the local checkpoint only. `HF_TOKEN` is optional for the public benchmark; `HF_HOME` and `HF_DATASETS_CACHE` can select a cache directory.

## Quick start

Run from this directory. Preview the commands without loading models:

```bash
LAVIT_MODEL=/path/to/LaVIT-7B-v2 DRY_RUN=1 bash prepare_benchmark.sh
LAVIT_MODEL=/path/to/LaVIT-7B-v2 DRY_RUN=1 bash run_pipeline.sh
```

Prepare the gallery once, then run the pipeline:

```bash
bash prepare_benchmark.sh
bash run_pipeline.sh
```

These are large GPU jobs. Run or resume selected stages with `STAGES`:

```bash
STAGES=retrieve bash run_pipeline.sh
STAGES="generate rerank" bash run_pipeline.sh
STAGES="route evaluate" bash run_pipeline.sh
```

The normal order is `retrieve`, `generate`, `rerank`, `route`, `evaluate`.

## Outputs

Generated files are written below `outputs/` by default:

```text
outputs/
├── intermediate/
│   ├── gallery_image_ids.pt
│   ├── gallery_token_scores.pt
│   ├── unconditional_log_probs.pt
│   └── gallery_image_embeddings.pt
├── generated/
│   ├── img_ids_<dataset>.pt
│   └── <dataset>/*.jpg
└── results/
    ├── retrieval.pt
    ├── rerank_retrieval.pt
    ├── decision.pt
    └── ... evaluation reports
```

Set `OUTPUT_DIR` to use another location. Useful settings include `NUM_BEAMS` (default `800`), `RANK_BATCH_SIZE` (`32`), `RERANK_BATCH_SIZE` (`64`), `ROUTE_BATCH_SIZE` (`16`), `EVAL_BATCH_SIZE` (`128`), and `RANDOM_SEED` (`42`). Reduce batch sizes if CUDA runs out of memory.

## Files

- `prepare_benchmark.sh`: preprocessing entry point.
- `run_pipeline.sh`: inference and evaluation entry point.
- `tokenize_gallery.py`: dynamic visual-token extraction.
- `extract_prior.py`: unconditional image-token likelihoods.
- `retrieve.py`: constrained text-to-image beam search.
- `generate.py`: LaVIT image generation.
- `inverse_rerank.py`: image-to-text reranking.
- `route.py`: retrieval-versus-generation comparison.
- `evaluate.py`: retrieval and unified-task evaluation.
- `data.py`: Hugging Face TIGeR-Bench loader.

## License

The LaVIT-derived code is governed by the [LaVIT Community License Agreement](LICENSE) and its [use policy](USE_POLICY.md). Model weights and third-party dependencies retain their upstream terms.
