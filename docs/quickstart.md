# RNA–ATAC quickstart

[Install dependencies](installation.md), then run from the repository root. Supply your own data and GTF annotation.

## 1. Prepare paths

```bash
export RNA_DIR=/absolute/path/to/rna
export ATAC_DIR=/absolute/path/to/atac
export GTF_FILE=/absolute/path/to/annotation.gtf
```

Each assay directory needs `matrix.mtx` (features × cells), `features.tsv`, and `barcodes.tsv`. The GTF must match the peak genome assembly and chromosome names.

The annotation filter defaults to `HAVANA`; set `GENEACTIVITY_ANNOTATION=none` to disable it. See [input formats](preprocessing.md#rnaatac-from-matrix-market).

## 2. Build the training AnnData object

```bash
bash experiment/rna_atac/preprocess.sh
```

## 3. Train

```bash
bash experiment/rna_atac/train.sh
```

Defaults: 100/80 epochs with early stopping; training seed 7. Paths and example settings are in [config.sh](../experiment/rna_atac/config.sh); model settings are in [configuration](configuration.md).

## 4. Evaluate

```bash
bash experiment/rna_atac/evaluate.sh
```

Uses `latent_shared`, biological labels in `cell_type`, and evaluation seed 0. Set `EVAL_BATCH_KEY=batch` only for a meaningful batch covariate.

Outputs: `result.h5ad`, `params.json`, and `metrics.tsv` in `results/rna-atac/`. Change `RUN_DIR` to use another output directory.

[Evaluation protocol](reproducibility.md)
