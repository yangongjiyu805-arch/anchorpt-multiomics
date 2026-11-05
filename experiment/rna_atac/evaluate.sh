#!/usr/bin/env bash
# Evaluate the primary shared embedding with a separate evaluation seed.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config.sh"

require_file "$RESULT_H5AD"
mkdir -p -- "$(dirname -- "$METRICS_FILE")"

"$PYTHON" "$ANCHORPT_REPO_ROOT/code/rna_atac/evaluate_metrics.py" \
  --input "$RESULT_H5AD" \
  --embedding-key latent_shared \
  --label-key cell_type \
  --modality-key modality \
  --batch-key "$EVAL_BATCH_KEY" \
  --random-state "$EVAL_SEED" \
  --n-jobs "$N_JOBS" \
  --format tsv \
  --output "$METRICS_FILE"
