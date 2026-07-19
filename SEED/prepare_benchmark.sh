#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-all}"
DATA_VERSION="${DATA_VERSION:-tiger_bench}"
MODEL_VERSION="${MODEL_VERSION:-8b}"
SEED_MODEL="${SEED_MODEL:-seed-llama-${MODEL_VERSION}-sft}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-128}"
PRIOR_BATCH_SIZE="${PRIOR_BATCH_SIZE:-64}"
UNCONDITIONAL_PROMPT="${UNCONDITIONAL_PROMPT:-Can you generate an image?}"

QUERIES=(whoops pickapic_7500 logo2k visual_news google_landmark food2k inatural wit)
GALLERIES=(logo2k visual_news google_landmark food2k inatural wit)
GALLERY_SIZE=500
OUTPUT_DIR="intermediate_results/$DATA_VERSION"

require_env() {
  if [[ -z "${!1:-}" ]]; then
    echo "Missing required environment variable: $1" >&2
    exit 2
  fi
}

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

tokenize() {
  require_env TOKENIZER_ROOT
  require_env DIFFUSION_ROOT
  run mkdir -p "$OUTPUT_DIR"

  for dataset in "${QUERIES[@]}"; do
    run python preprocess.py \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION" \
      --bs "$TOKENIZE_BATCH_SIZE" \
      --out "$OUTPUT_DIR/t2i_mat_cond_${dataset}.pt"

    run python preprocess.py \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION" \
      --bs "$TOKENIZE_BATCH_SIZE" \
      --prefix_prompt "" \
      --i2t \
      --out "$OUTPUT_DIR/i2t_mat_${dataset}.pt"
  done

  for dataset in "${GALLERIES[@]}"; do
    run python preprocess.py \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION" \
      --bs "$TOKENIZE_BATCH_SIZE" \
      --uncond \
      --uncond_txt "$UNCONDITIONAL_PROMPT" \
      --out "$OUTPUT_DIR/t2i_mat_uncond_${dataset}.pt"
  done
}

extract_priors() {
  require_env CKPT_ROOT
  require_env TOKENIZER_ROOT
  require_env DIFFUSION_ROOT

  for dataset in "${GALLERIES[@]}"; do
    run python extract_prior.py \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION" \
      --seed_model "$SEED_MODEL" \
      --bs "$PRIOR_BATCH_SIZE" \
      --num_selected "$GALLERY_SIZE" \
      --input "$OUTPUT_DIR/t2i_mat_uncond_${dataset}.pt" \
      --out_uncond_imgid2sumlogprobs "$OUTPUT_DIR/sumlogprobs_${dataset}_${GALLERY_SIZE}.pt" \
      --uncond

    run python extract_prior.py \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION" \
      --seed_model "$SEED_MODEL" \
      --bs "$PRIOR_BATCH_SIZE" \
      --num_selected "$GALLERY_SIZE" \
      --input "$OUTPUT_DIR/t2i_mat_uncond_${dataset}.pt" \
      --out "$OUTPUT_DIR/uncond_logits_${dataset}_${GALLERY_SIZE}.pt" \
      --uncond \
      --extract_uncond_logits
  done
}

case "$STAGE" in
  tokenize) tokenize ;;
  priors) extract_priors ;;
  all) tokenize; extract_priors ;;
  *) echo "Usage: $0 [tokenize|priors|all]" >&2; exit 2 ;;
esac
