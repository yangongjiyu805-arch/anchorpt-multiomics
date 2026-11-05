# RNA–ATAC quickstart

[Home](../README.md) · [Installation](installation.md) · [Other input routes](preprocessing.md)

The executable example in [experiment/rna_atac/](../experiment/rna_atac/) runs preprocessing, two-stage training, and evaluation. Install the [dependencies](installation.md), then run the commands below from the repository root.

## 1. Prepare paths

```bash
export RNA_DIR=/absolute/path/to/rna
export ATAC_DIR=/absolute/path/to/atac
export GTF_FILE=/absolute/path/to/annotation.gtf
```

Each assay directory must contain `matrix.mtx`, `features.tsv`, and `barcodes.tsv`. Matrices use **features × cells** orientation. Optional `metadata.csv` files use barcodes as the first-column index and can provide `celltype` and `batch` columns. See [input formats](preprocessing.md#rnaatac-from-matrix-market).

Use a GTF matching the genome assembly and chromosome naming of your ATAC peaks. The example filters annotation sources to `HAVANA`; set `GENEACTIVITY_ANNOTATION=none` to disable this filter, or select the appropriate source for your annotation.

Shared paths and settings are defined in [config.sh](../experiment/rna_atac/config.sh). Set `PREPROCESSED_H5AD`, `RUN_DIR`, `TRAIN_SEED`, or `EVAL_SEED` before running to override them.

## 2. Build the training AnnData object

```bash
bash experiment/rna_atac/preprocess.sh
```

The default output is `data/processed/rna_atac.h5ad`, with normalized values in `layers["train_input"]`. Input datasets and reference annotations are supplied separately.

## 3. Train

```bash
bash experiment/rna_atac/train.sh
```

The example requests 100 Stage-I epochs and 80 Stage-II epochs, with early stopping and training seed 7. Training uses modality identities without cell-type or batch-label supervision. Evaluation annotations can remain in the input AnnData object.

Set `ANCHORPT_STAGE1_EPOCH_NUM` and `ANCHORPT_STAGE2_EPOCH_NUM` to override the requested epoch counts. See [configuration](configuration.md) for the model and transport settings.

## 4. Evaluate

```bash
bash experiment/rna_atac/evaluate.sh
```

The evaluator selects `obsm["latent_shared"]`, uses evaluation seed 0, and writes TSV scores. Biological labels are required. Batch-correction metrics are omitted by default; set `EVAL_BATCH_KEY=batch` only when `obs["batch"]` is an interpretable nuisance variable for the benchmark.

## 5. Inspect the outputs

By default, outputs are written to `results/rna-atac/`:

| File | Contents |
| :--- | :--- |
| `result.h5ad` | Integrated AnnData and learned embeddings |
| `params.json` | Effective model, input, optimization, and seed settings |
| `metrics.tsv` | Evaluation scores |

```python
import anndata as ad

result = ad.read_h5ad("results/rna-atac/result.h5ad")
embedding = result.obsm["latent_shared"]
print(result.shape, embedding.shape)
```

Keep the outputs, configuration, input provenance, and environment together. The [evaluation protocol](reproducibility.md) defines embedding roles and aggregate scores.
