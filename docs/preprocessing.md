# Data preparation

Run from the repository root. Supply data and annotations separately.

## Training input contract

Training reads a cells × features AnnData `.h5ad` object:

| Field | Purpose |
| :--- | :--- |
| `layers["train_input"]` | Normalized training input |
| `obs["modality"]` | Modality identity |
| `layers["counts"]` | Counts retained by the preprocessors |
| `obs["cell_type"]` / `obs["batch"]` | Evaluation annotations |
| `uns["feature_list"]` | Feature indices for optional masks |

Input layer and metadata keys are configurable. For `ANCHORPT_USE_FEATURE_MASK=true`, `feature_list` keys must exactly match modality strings; values must be nonempty, unique, zero-based indices in `[0, n_vars)`. Zero padding alone does not define a mask.

## RNA–ATAC from Matrix Market

Each assay directory contains `matrix.mtx` (features × cells), `features.tsv`, and `barcodes.tsv`. Feature identifiers use the first TSV column.

Optional `metadata.csv` contains `celltype` and `batch` columns, indexed by barcodes in its first column. `meta_data.csv` takes precedence if both exist. Missing labels become `unknown`, which is unsuitable for biological evaluation.

For GTF mapping, use the [example](quickstart.md#2-build-the-training-anndata-object). Annotation coordinates and identifiers must match the data. `--geneactivity-annotation none` disables source filtering.

For a gene–region CSV with `peak` and `gene.name` columns matching the input identifiers:

```bash
python code/preprocessing/prepare_gxr_gtf_rna_atac.py \
  --rna-dir /path/to/rna --atac-dir /path/to/atac \
  --route gxr --gxr-csv /path/to/gene_region_mapping.csv \
  --top-genes 5000 --output data/processed/rna_atac.h5ad
```

## RNA–ATAC from AnnData

Supply count layers and a matching GTF:

```bash
python code/preprocessing/prepare_h5ad_gtf_rna_atac.py \
  --rna-h5ad /path/to/rna.h5ad --atac-h5ad /path/to/atac.h5ad \
  --gtf /path/to/annotation.gtf \
  --rna-counts-layer counts --atac-counts-layer counts \
  --output data/processed/rna_atac.h5ad
```

This script imports its companion preprocessor; keep both files together. Its annotation filter defaults to `HAVANA`; use `--geneactivity-annotation none` to disable it. Both RNA–ATAC routes produce gene-space training inputs for the same model.

## RNA–protein

```bash
python code/preprocessing/prepare_h5ad_hgnc_rna_protein.py \
  --rna-h5ad /path/to/rna.h5ad --protein-h5ad /path/to/protein.h5ad \
  --hgnc-json /path/to/hgnc_complete_set.json \
  --ambiguous-map-policy error --output data/processed/rna_protein.h5ad
```

The script maps protein markers through HGNC, normalizes RNA and CLR-transforms protein counts, and exports a union feature axis with `feature_list`. It also writes a protein-mapping TSV and records the HGNC hash.

The example rejects ambiguous mappings; the script defaults to selecting the first match. Inspect the mapping report and use the [multimodal runner](configuration.md#rnaprotein-and-three-modality-inputs) with feature masks.

## Three-modality unpaired splitting

```bash
python code/preprocessing/prepare_tripbmc_unpaired_split.py \
  --source-root /path/to/tripbmc/source \
  --output-dir data/processed/tripbmc_split --seed 42
```

This dataset-specific script expects fixed layouts and quotas. It assigns disjoint modality subsets using source, condition, technical stratum, and cell-type strata. Assemble the common-feature AnnData object separately.

Record input checksums, annotations, arguments, feature order, cell IDs, and splits. Report label-based split stratification separately from label-free training.
