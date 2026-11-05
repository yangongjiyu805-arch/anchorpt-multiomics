#!/usr/bin/env bash
# Shared experiment settings.

ANCHORPT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON="${PYTHON:-python}"
PREPROCESSED_H5AD="${PREPROCESSED_H5AD:-$ANCHORPT_REPO_ROOT/data/processed/rna_atac.h5ad}"
RUN_DIR="${RUN_DIR:-$ANCHORPT_REPO_ROOT/results/rna-atac}"
RESULT_H5AD="${RESULT_H5AD:-$RUN_DIR/result.h5ad}"
PARAMS_JSON="${PARAMS_JSON:-$RUN_DIR/params.json}"
METRICS_FILE="${METRICS_FILE:-$RUN_DIR/metrics.tsv}"

GENEACTIVITY_ANNOTATION="${GENEACTIVITY_ANNOTATION:-HAVANA}"
TOP_GENES="${TOP_GENES:-5000}"
TARGET_SUM="${TARGET_SUM:-10000}"
TRAIN_SEED="${TRAIN_SEED:-7}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_BATCH_KEY="${EVAL_BATCH_KEY:-none}"
N_JOBS="${N_JOBS:-8}"

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Required file not found: %s\n' "$1" >&2
    return 1
  fi
}
