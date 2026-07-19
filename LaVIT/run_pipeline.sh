#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${LAVIT_MODEL:?Set LAVIT_MODEL to the LaVIT checkpoint directory.}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAGES="${STAGES:-retrieve generate rerank route evaluate}"
NUM_BEAMS="${NUM_BEAMS:-800}"

QUERY_DATASETS=(whoops pickapic_7500 logo2k visual_news google_landmark food2k inatural wit)
QUERY_COUNTS=(500 2500 500 500 500 500 500 500)
GALLERY_DATASETS=(logo2k visual_news google_landmark food2k inatural wit)
GALLERY_COUNTS=(500 500 500 500 500 500)

INTERMEDIATE_DIR="$OUTPUT_DIR/intermediate"
RESULTS_DIR="$OUTPUT_DIR/results"
GENERATED_DIR="$OUTPUT_DIR/generated"
GALLERY_IDS="$INTERMEDIATE_DIR/gallery_image_ids.pt"
UNCONDITIONAL_PRIOR="$INTERMEDIATE_DIR/unconditional_log_probs.pt"
RETRIEVAL="$RESULTS_DIR/retrieval.pt"
RERANKED="$RESULTS_DIR/rerank_retrieval.pt"
DECISION="$RESULTS_DIR/decision.pt"

mkdir -p "$INTERMEDIATE_DIR" "$RESULTS_DIR" "$GENERATED_DIR"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != "1" ]]; then
        "$@"
    fi
}

export CUDA_VISIBLE_DEVICES="$GPU"

if has_stage retrieve; then
    run python retrieve.py \
        --num_beams "$NUM_BEAMS" --bs_ranking "${RANK_BATCH_SIZE:-32}" \
        --num_img_tokens 0 --out "$RETRIEVAL" \
        --datasets_query "${QUERY_DATASETS[@]}" \
        --datasets_gallery "${GALLERY_DATASETS[@]}" \
        --nums_selected_txt_query "${QUERY_COUNTS[@]}" \
        --nums_selected_gallery "${GALLERY_COUNTS[@]}" \
        --img_id_path "$GALLERY_IDS" \
        --uncond_log_prob "$UNCONDITIONAL_PRIOR"
fi

if has_stage generate; then
    run python generate.py \
        --out_dir "$GENERATED_DIR" --pixel_decoding highres \
        --model_type bf16 --seed "${RANDOM_SEED:-42}" \
        --datasets_query "${QUERY_DATASETS[@]}" \
        --datasets_gallery "${GALLERY_DATASETS[@]}" \
        --nums_selected_txt_query "${QUERY_COUNTS[@]}" \
        --nums_selected_gallery "${GALLERY_COUNTS[@]}"
fi

if has_stage rerank; then
    run python inverse_rerank.py \
        --bs "${RERANK_BATCH_SIZE:-64}" --num_img_tokens 0 --prefix_prompt "" \
        --datasets_query "${QUERY_DATASETS[@]}" \
        --datasets_gallery "${GALLERY_DATASETS[@]}" \
        --nums_selected_txt_query "${QUERY_COUNTS[@]}" \
        --nums_selected_gallery "${GALLERY_COUNTS[@]}" \
        --img_id_path "$GALLERY_IDS" \
        --img_emb_path "$INTERMEDIATE_DIR/gallery_image_embeddings.pt" \
        --retr_ranking_list "$RETRIEVAL"
fi

if has_stage route; then
    run python route.py \
        --bs "${ROUTE_BATCH_SIZE:-16}" --out "$DECISION" \
        --datasets_query "${QUERY_DATASETS[@]}" \
        --datasets_gallery "${GALLERY_DATASETS[@]}" \
        --nums_selected_txt_query "${QUERY_COUNTS[@]}" \
        --nums_selected_gallery "${GALLERY_COUNTS[@]}" \
        --img_ids_gallery "$GALLERY_IDS" \
        --ids_gen_prefix "$GENERATED_DIR/img_ids_" \
        --ids_gen "${QUERY_DATASETS[@]}" --ranking "$RERANKED"
fi

if has_stage evaluate; then
    run python evaluate.py \
        --datasets_query "${QUERY_DATASETS[@]}" \
        --datasets_gallery "${GALLERY_DATASETS[@]}" \
        --nums_selected_txt_query "${QUERY_COUNTS[@]}" \
        --nums_selected_gallery "${GALLERY_COUNTS[@]}" \
        --img_gen_dir_prefix "$GENERATED_DIR" \
        --img_gen_dir_multi "${QUERY_DATASETS[@]}" \
        --ranking "$RERANKED" --decision "$DECISION" \
        --batch_size "${EVAL_BATCH_SIZE:-128}" \
        --out_dir "$RESULTS_DIR"
fi

printf 'LaVIT pipeline complete. Results are in %s\n' "$RESULTS_DIR"
