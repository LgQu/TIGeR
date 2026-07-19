#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-all}"
DATA_VERSION="${DATA_VERSION:-tiger_bench}"
MODEL_VERSION="${MODEL_VERSION:-8b}"
SEED_MODEL="${SEED_MODEL:-seed-llama-${MODEL_VERSION}-sft}"
export SEED_MODEL
NUM_BEAMS="${NUM_BEAMS:-800}"
RANK_BATCH_SIZE="${RANK_BATCH_SIZE:-128}"
ROUTE_BATCH_SIZE="${ROUTE_BATCH_SIZE:-32}"
GENERATION_STEPS="${GENERATION_STEPS:-25}"
RANDOM_SEED="${RANDOM_SEED:-42}"

QUERIES=(whoops pickapic_7500 logo2k visual_news google_landmark food2k inatural wit)
QUERY_SIZES=(500 2500 500 500 500 500 500 500)
GALLERIES=(logo2k visual_news google_landmark food2k inatural wit)
GALLERY_SIZES=(500 500 500 500 500 500)

INTERMEDIATE_DIR="intermediate_results/$DATA_VERSION"
GENERATED_DIR="generated/$DATA_VERSION"
RESULTS_DIR="results/$DATA_VERSION"
RETRIEVAL_PATH="$RESULTS_DIR/retrieval.pt"
RERANKED_PATH="$RESULTS_DIR/rerank_retrieval.pt"
DECISION_PATH="$RESULTS_DIR/decision.pt"

require_env() {
  if [[ -z "${!1:-}" ]]; then
    echo "Missing required environment variable: $1" >&2
    exit 2
  fi
}

require_models() {
  require_env CKPT_ROOT
  require_env TOKENIZER_ROOT
  require_env DIFFUSION_ROOT
}

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

retrieve_images() {
  require_models
  run mkdir -p "$RESULTS_DIR"
  run python retrieve.py \
    --data_version "$DATA_VERSION" \
    --datasets_query "${QUERIES[@]}" \
    --nums_selected_txt_query "${QUERY_SIZES[@]}" \
    --datasets_gallery "${GALLERIES[@]}" \
    --nums_selected_gallery "${GALLERY_SIZES[@]}" \
    --uncond_names "${GALLERIES[@]}" \
    --uncond_path_prefix "$INTERMEDIATE_DIR/uncond_logits_" \
    --uncond_sumlogprobs_path_prefix "$INTERMEDIATE_DIR/sumlogprobs_" \
    --num_beams "$NUM_BEAMS" \
    --bs_ranking "$RANK_BATCH_SIZE" \
    --model_vers "$MODEL_VERSION" \
    --use_uncond \
    --out "$RETRIEVAL_PATH"
}

generate_images() {
  require_models
  run mkdir -p "$GENERATED_DIR/images"
  for index in "${!QUERIES[@]}"; do
    dataset="${QUERIES[$index]}"
    count="${QUERY_SIZES[$index]}"
    run python generate.py \
      --dataset "$dataset" \
      --data_version "$DATA_VERSION" \
      --model seed \
      --model_vers "$MODEL_VERSION" \
      --batch_size 1 \
      --num_selected "$count" \
      --timestep "$GENERATION_STEPS" \
      --rand_seed "$RANDOM_SEED" \
      --out_dir "$GENERATED_DIR/images/$dataset" \
      --out_ids "$GENERATED_DIR/ids_${dataset}.pt"
  done
}

rerank_images() {
  require_models
  run python inverse_rerank.py \
    --data_version "$DATA_VERSION" \
    --datasets_query "${QUERIES[@]}" \
    --nums_selected_txt_query "${QUERY_SIZES[@]}" \
    --datasets_gallery "${GALLERIES[@]}" \
    --nums_selected_gallery "${GALLERY_SIZES[@]}" \
    --seed_model "$SEED_MODEL" \
    --bs "$RANK_BATCH_SIZE" \
    --prefix_prompt "" \
    --retr_ranking_list "$RETRIEVAL_PATH" \
    --sim_i2t "$RESULTS_DIR/not_precomputed.pt"
}

route_outputs() {
  require_models
  run python route.py \
    --data_version "$DATA_VERSION" \
    --datasets_query "${QUERIES[@]}" \
    --nums_selected_txt_query "${QUERY_SIZES[@]}" \
    --datasets_gallery "${GALLERIES[@]}" \
    --nums_selected_gallery "${GALLERY_SIZES[@]}" \
    --ids_file_prefix "$INTERMEDIATE_DIR/t2i_mat_cond_" \
    --ids_files_query_postfix "${QUERIES[@]}" \
    --ids_files_gallery_postfix "${GALLERIES[@]}" \
    --ids_gen_prefix "$GENERATED_DIR/ids_" \
    --ids_gen "${QUERIES[@]}" \
    --ranking "$RERANKED_PATH" \
    --bs "$ROUTE_BATCH_SIZE" \
    --out "$DECISION_PATH"
}

evaluate_outputs() {
  run python evaluate.py \
    --data_version "$DATA_VERSION" \
    --datasets_query "${QUERIES[@]}" \
    --nums_selected_txt_query "${QUERY_SIZES[@]}" \
    --datasets_gallery "${GALLERIES[@]}" \
    --nums_selected_gallery "${GALLERY_SIZES[@]}" \
    --img_gen_dir_prefix "$GENERATED_DIR/images" \
    --img_gen_dir_multi "${QUERIES[@]}" \
    --ranking "$RERANKED_PATH" \
    --decision "$DECISION_PATH" \
    --prefix_prompt ""
}

case "$STAGE" in
  retrieve) retrieve_images ;;
  generate) generate_images ;;
  rerank) rerank_images ;;
  route) route_outputs ;;
  evaluate) evaluate_outputs ;;
  all) retrieve_images; generate_images; rerank_images; route_outputs; evaluate_outputs ;;
  *) echo "Usage: $0 [retrieve|generate|rerank|route|evaluate|all]" >&2; exit 2 ;;
esac
