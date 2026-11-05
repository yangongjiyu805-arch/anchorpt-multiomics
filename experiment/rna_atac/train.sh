#!/usr/bin/env bash
# Run strict-unpaired, two-stage AnchorPT with an explicit training seed.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config.sh"

require_file "$PREPROCESSED_H5AD"
mkdir -p -- "$RUN_DIR"

export ANCHORPT_SRC_DIR="$ANCHORPT_REPO_ROOT/code/rna_atac"
export ANCHORPT_INPUT_H5AD="$PREPROCESSED_H5AD"
export ANCHORPT_OUTPUT_H5AD="$RESULT_H5AD"
export ANCHORPT_PARAMS_USED_JSON="$PARAMS_JSON"
export ANCHORPT_INPUT_LAYER=train_input
export ANCHORPT_MODALITY_KEY=modality
export ANCHORPT_TRAINING_MODE=two_stage
export ANCHORPT_RANDOM_SEED="$TRAIN_SEED"
export ANCHORPT_STAGE1_EPOCH_NUM="${ANCHORPT_STAGE1_EPOCH_NUM:-100}"
export ANCHORPT_STAGE2_EPOCH_NUM="${ANCHORPT_STAGE2_EPOCH_NUM:-80}"
export ANCHORPT_STRICT_UNPAIRED=true
export ANCHORPT_USE_TRAINING_LABELS=false
export ANCHORPT_BATCH_KEY=""
export ANCHORPT_CELLTYPE_KEY=""

"$PYTHON" "$ANCHORPT_REPO_ROOT/code/rna_atac/main_anchorpt.py"
