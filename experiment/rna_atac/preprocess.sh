#!/usr/bin/env bash
# Prepare RNA–ATAC training input.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config.sh"

: "${RNA_DIR:?Set RNA_DIR to the RNA Matrix Market directory}"
: "${ATAC_DIR:?Set ATAC_DIR to the ATAC Matrix Market directory}"
: "${GTF_FILE:?Set GTF_FILE to the matching genome annotation}"

for assay_dir in "$RNA_DIR" "$ATAC_DIR"; do
  for input_file in matrix.mtx features.tsv barcodes.tsv; do
    require_file "$assay_dir/$input_file"
  done
done
require_file "$GTF_FILE"
mkdir -p -- "$(dirname -- "$PREPROCESSED_H5AD")"

"$PYTHON" "$ANCHORPT_REPO_ROOT/code/preprocessing/prepare_gxr_gtf_rna_atac.py" \
  --rna-dir "$RNA_DIR" \
  --atac-dir "$ATAC_DIR" \
  --route gtf \
  --gtf "$GTF_FILE" \
  --geneactivity-annotation "$GENEACTIVITY_ANNOTATION" \
  --rna-label-col celltype \
  --atac-label-col celltype \
  --rna-batch-col batch \
  --atac-batch-col batch \
  --top-genes "$TOP_GENES" \
  --target-sum "$TARGET_SUM" \
  --output "$PREPROCESSED_H5AD"
