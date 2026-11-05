# RNA–ATAC experiment example

[Experiments](../README.md) · [Implementation](../../code/rna_atac/README.md)

This directory contains executable preprocessing, training, and evaluation examples based on the original RNA–ATAC tutorial.

| File | Purpose |
| :--- | :--- |
| [config.sh](config.sh) | Shared paths, seeds, and preprocessing/evaluation settings |
| [preprocess.sh](preprocess.sh) | Build a gene-space training AnnData object from RNA and ATAC inputs |
| [train.sh](train.sh) | Train AnchorPT and export embeddings and effective parameters |
| [evaluate.sh](evaluate.sh) | Evaluate `obsm["latent_shared"]` |

## Run the example

Install the repository dependencies first. From the repository root, use Bash:

```bash
export RNA_DIR=/absolute/path/to/rna
export ATAC_DIR=/absolute/path/to/atac
export GTF_FILE=/absolute/path/to/annotation.gtf

bash experiment/rna_atac/preprocess.sh
bash experiment/rna_atac/train.sh
bash experiment/rna_atac/evaluate.sh
```

Each assay directory contains `matrix.mtx` (features × cells), `features.tsv`, and `barcodes.tsv`. Optional `metadata.csv` provides `celltype` and `batch` columns and uses barcodes as the first-column index. The GTF must match the genome assembly and chromosome naming of the peaks.

The annotation-source filter defaults to `HAVANA`. Set `GENEACTIVITY_ANNOTATION=none` or an appropriate source for your GTF when needed. The feature readers use the first TSV column.

All three scripts resolve the implementation relative to their own location. Their paths are quoted to support spaces in directory names. Set input paths to absolute paths when invoking the scripts from another working directory.

## Customize a run

Set these variables before invoking the scripts, or edit `config.sh`:

| Variable | Default |
| :--- | :--- |
| `PREPROCESSED_H5AD` | `data/processed/rna_atac.h5ad` under the repository root |
| `RUN_DIR` | `results/rna-atac` under the repository root |
| `RESULT_H5AD` | `$RUN_DIR/result.h5ad` |
| `PARAMS_JSON` | `$RUN_DIR/params.json` |
| `METRICS_FILE` | `$RUN_DIR/metrics.tsv` |
| `TRAIN_SEED` / `EVAL_SEED` | `7` / `0` |
| `TOP_GENES` / `TARGET_SUM` | `5000` / `10000` |
| `EVAL_BATCH_KEY` | `none` |
| `N_JOBS` | `8` |
| `PYTHON` | `python` |

Training requests 100 Stage-I and **80 Stage-II epochs**, subject to early stopping. Override `ANCHORPT_STAGE1_EPOCH_NUM` or `ANCHORPT_STAGE2_EPOCH_NUM` explicitly when needed.

For a separate output directory and seed:

```bash
export TRAIN_SEED=11
export RUN_DIR="$PWD/results/rna-atac-seed11"
bash experiment/rna_atac/train.sh
bash experiment/rna_atac/evaluate.sh
```

Supply biological labels for evaluation. Set `EVAL_BATCH_KEY=batch` only when batch correction is meaningful for the dataset. Placeholder `unknown` labels are not biological ground truth.

## Outputs

The default outputs are `result.h5ad`, `params.json`, `params_method_manifest.json`, and `metrics.tsv` inside `results/rna-atac/`. Keep them with the commit ID, environment, data/annotation versions, and split records.

These scripts call the existing implementation; they do not bundle example data, validate a fresh dependency installation, or establish reproduction of manuscript scores. See [evaluation protocol notes](../../docs/reproducibility.md).
