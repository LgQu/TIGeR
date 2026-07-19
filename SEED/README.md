# SEED-LLaMA backend

This directory contains the SEED-LLaMA implementation of TIGeR. It uses the public [TIGeR-Bench](https://huggingface.co/datasets/leigangqu/TIGeR-Bench) splits directly; no local dataset paths are embedded in the code.

For the complete installation guide, checkpoint layout, troubleshooting advice, and citation, see the [main README](../README.md).

## Setup

Create an environment with Python 3.8, install `requirements.txt`, and download the official SEED-LLaMA checkpoints. Configure their locations with environment variables:

```bash
python -m pip install -r requirements.txt

export CKPT_ROOT=/path/to/seed/checkpoints
export TOKENIZER_ROOT=/path/to/seed_llama_tokenizer_hf
export DIFFUSION_ROOT=/path/to/diffusion_model
export CUDA_VISIBLE_DEVICES=0
```

`CKPT_ROOT` must contain `seed-llama-8b-sft` (or `seed-llama-14b-sft`). `HF_TOKEN`, `HF_HOME`, and `HF_DATASETS_CACHE` are optional and follow Hugging Face conventions.

## Run

Precompute image tokens and unconditional gallery priors:

```bash
bash prepare_benchmark.sh all
```

Run retrieval, generation, reverse reranking, routing, and evaluation:

```bash
bash run_pipeline.sh all
```

Both entry points accept a single stage name, which is useful for resuming a run. Set `DRY_RUN=1` to print every command without loading models or writing outputs.

```bash
bash prepare_benchmark.sh tokenize
bash prepare_benchmark.sh priors
bash run_pipeline.sh retrieve
bash run_pipeline.sh generate
bash run_pipeline.sh rerank
bash run_pipeline.sh route
bash run_pipeline.sh evaluate
```

Common overrides include `MODEL_VERSION=14b`, `NUM_BEAMS`, `RANK_BATCH_SIZE`, `ROUTE_BATCH_SIZE`, `GENERATION_STEPS`, and `RANDOM_SEED`.

## Outputs

- `intermediate_results/tiger_bench`: token matrices and unconditional priors
- `generated/tiger_bench`: generated image tokens and decoded images
- `results/tiger_bench/retrieval.pt`: forward beam-search retrieval
- `results/tiger_bench/rerank_retrieval.pt`: reverse-reranked retrieval
- `results/tiger_bench/decision.pt`: generation-versus-retrieval routing scores

The implementation is derived from the official [SEED](https://github.com/AILab-CVC/SEED) codebase. Checkpoint licensing and access remain subject to the upstream project.
