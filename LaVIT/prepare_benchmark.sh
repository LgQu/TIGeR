#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${LAVIT_MODEL:?Set LAVIT_MODEL to the LaVIT checkpoint directory.}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
THRESHOLD="${THRESHOLD:-0.5}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

GALLERY_DATASETS=(logo2k visual_news google_landmark food2k inatural wit)
GALLERY_COUNTS=(500 500 500 500 500 500)
INTERMEDIATE_DIR="$OUTPUT_DIR/intermediate"
GALLERY_IDS="$INTERMEDIATE_DIR/gallery_image_ids.pt"
GALLERY_SCORES="$INTERMEDIATE_DIR/gallery_token_scores.pt"
UNCONDITIONAL_PRIOR="$INTERMEDIATE_DIR/unconditional_log_probs.pt"

mkdir -p "$INTERMEDIATE_DIR"

run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != "1" ]]; then
        "$@"
    fi
}

export CUDA_VISIBLE_DEVICES="$GPU"

run python tokenize_gallery.py \
    --datasets-query "${GALLERY_DATASETS[@]}" \
    --nums-selected-txt-query "${GALLERY_COUNTS[@]}" \
    --datasets-gallery "${GALLERY_DATASETS[@]}" \
    --nums-selected-gallery "${GALLERY_COUNTS[@]}" \
    --img-id-out "$GALLERY_IDS" \
    --score-out "$GALLERY_SCORES" \
    --threshold "$THRESHOLD"

run python extract_prior.py \
    --bs "${PRIOR_BATCH_SIZE:-16}" \
    --num_img_tokens 0 \
    --out "$INTERMEDIATE_DIR/prior_scratch_sim_t2i.pt" \
    --uncond_out "$INTERMEDIATE_DIR/prior_scratch_unconditional.pt" \
    --datasets_query "${GALLERY_DATASETS[@]}" \
    --datasets_gallery "${GALLERY_DATASETS[@]}" \
    --nums_selected_txt_query "${GALLERY_COUNTS[@]}" \
    --nums_selected_gallery "${GALLERY_COUNTS[@]}" \
    --img_id_path "$GALLERY_IDS" \
    --extract_uncond_log_prob "$UNCONDITIONAL_PRIOR"

printf 'Prepared LaVIT benchmark artifacts in %s\n' "$INTERMEDIATE_DIR"
