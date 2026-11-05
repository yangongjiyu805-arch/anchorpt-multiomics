# AnchorPT RNA-ATAC Tutorial

This tutorial runs the complete RNA-ATAC workflow from preprocessing to evaluation.

## 1. Define paths

Replace the placeholder values before running the commands.

```bash
export REQUIREMENTS_FILE="<REQUIREMENTS_FILE>"
export PYTORCH_INDEX_URL="<PYTORCH_INDEX_URL>"
export PREPROCESS_SCRIPT="<RNA_ATAC_PREPROCESS_SCRIPT>"
export TRAIN_SCRIPT="<ANCHORPT_TRAIN_SCRIPT>"
export EVALUATION_SCRIPT="<EVALUATION_SCRIPT>"
export RNA_DIR="<RNA_INPUT_DIRECTORY>"
export ATAC_DIR="<ATAC_INPUT_DIRECTORY>"
export GTF_FILE="<GTF_ANNOTATION_FILE>"
export PREPROCESSED_H5AD="<PREPROCESSED_H5AD>"
export RESULT_H5AD="<RESULT_H5AD>"
export PARAMS_JSON="<PARAMETER_JSON>"
export METRICS_FILE="<METRICS_FILE>"
```

The RNA and ATAC directories must each contain `matrix.mtx`, `features.tsv`, and `barcodes.tsv`. An optional `metadata.csv` file may contain `celltype` and `batch` columns. The matrices must use features by cells orientation.

## 2. Install dependencies

```bash
python -m pip install -r "$REQUIREMENTS_FILE"
```

## 3. Preprocess RNA and ATAC

```bash
python "$PREPROCESS_SCRIPT" \
  --rna-dir "$RNA_DIR" \
  --atac-dir "$ATAC_DIR" \
  --route gtf \
  --gtf "$GTF_FILE" \
  --geneactivity-annotation HAVANA \
  --rna-label-col celltype \
  --atac-label-col celltype \
  --rna-batch-col batch \
  --atac-batch-col batch \
  --top-genes 5000 \
  --target-sum 10000 \
  --output "$PREPROCESSED_H5AD"
```

The generated h5ad contains `layers["counts"]`, `layers["train_input"]`, `obs["modality"]`, `obs["cell_type"]`, `obs["batch"]`, and `obs["dataset"]`.

## 4. Run AnchorPT

```bash
export ANCHORPT_INPUT_H5AD="$PREPROCESSED_H5AD"
export ANCHORPT_OUTPUT_H5AD="$RESULT_H5AD"
export ANCHORPT_PARAMS_USED_JSON="$PARAMS_JSON"
export ANCHORPT_INPUT_LAYER=train_input
export ANCHORPT_MODALITY_KEY=modality
export ANCHORPT_RANDOM_SEED=7

python "$TRAIN_SCRIPT"
```

## 5. Evaluate the embedding

```bash
python "$EVALUATION_SCRIPT" \
  --input "$RESULT_H5AD" \
  --embedding-key latent_shared \
  --label-key cell_type \
  --modality-key modality \
  --batch-key batch \
  --n-jobs 8 \
  --format tsv \
  --output "$METRICS_FILE"
```
