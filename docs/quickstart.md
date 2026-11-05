# RNA–ATAC quickstart

[Home](../README.md) · [Installation](installation.md) · [Other input routes](preprocessing.md)

This walkthrough preprocesses your own RNA and ATAC matrices, trains AnchorPT, and evaluates the shared embedding. It is a usage example, not a benchmark result or an exact manuscript experiment configuration.

Run from the repository root after [installation](installation.md). Replace the three input paths with your files.

## 1. Prepare paths

```bash
export RNA_DIR=/path/to/rna
export ATAC_DIR=/path/to/atac
export GTF_FILE=/path/to/annotation.gtf

mkdir -p data/processed results/rna-atac
```

Each assay directory must contain `matrix.mtx`, `features.tsv`, and `barcodes.tsv`. Matrices use **features × cells** orientation. Optional `metadata.csv` files can provide `celltype` and `batch` columns. See [input formats](preprocessing.md#rnaatac-from-matrix-market).

## 2. Build the training AnnData object

```bash
python pre_data/prepare_gxr_gtf_rna_atac.py \
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
  --output data/processed/rna_atac.h5ad
```

Use an annotation matching the genome assembly and chromosome naming of your ATAC peaks. `HAVANA` is an annotation-source filter; change it for your GTF as described in [preprocessing](preprocessing.md).

## 3. Train

```bash
export ANCHORPT_SRC_DIR="$PWD/src"
export ANCHORPT_INPUT_H5AD="$PWD/data/processed/rna_atac.h5ad"
export ANCHORPT_OUTPUT_H5AD="$PWD/results/rna-atac/result.h5ad"
export ANCHORPT_PARAMS_USED_JSON="$PWD/results/rna-atac/params.json"
export ANCHORPT_INPUT_LAYER=train_input
export ANCHORPT_MODALITY_KEY=modality
export ANCHORPT_TRAINING_MODE=two_stage
export ANCHORPT_RANDOM_SEED=7
export ANCHORPT_STAGE1_EPOCH_NUM=100
export ANCHORPT_STAGE2_EPOCH_NUM=80
export ANCHORPT_STRICT_UNPAIRED=true
export ANCHORPT_USE_TRAINING_LABELS=false
export ANCHORPT_BATCH_KEY=""
export ANCHORPT_CELLTYPE_KEY=""

python src/main_anchorpt.py
```

The example explicitly sets the released training defaults. For a manuscript experiment, use its full configuration rather than changing only the epoch count.

Training uses modality identities, but does not use cell-type or batch-label supervision under the strict-unpaired configuration. Those annotations can remain in the AnnData object for evaluation. The [configuration guide](configuration.md) explains the available controls.

## 4. Evaluate

```bash
python src/evaluate_metrics.py \
  --input results/rna-atac/result.h5ad \
  --embedding-key latent_shared \
  --label-key cell_type \
  --modality-key modality \
  --batch-key none \
  --random-state 0 \
  --n-jobs 8 \
  --format tsv \
  --output results/rna-atac/metrics.tsv
```

Biological labels are required for this evaluator. Use `--batch-key batch` only when `obs["batch"]` represents a meaningful nuisance variable for that benchmark. The `none` setting omits batch-correction metrics; it is not an instruction to infer batch labels.

## 5. Inspect the outputs

```python
import anndata as ad

result = ad.read_h5ad("results/rna-atac/result.h5ad")
embedding = result.obsm["latent_shared"]
print(result.shape, embedding.shape)
```

Keep `params.json`, `params_method_manifest.json`, `metrics.tsv`, and the run environment alongside the output. See [reproducibility](reproducibility.md) before comparing aggregate scores.
