# Data preparation

[Home](../README.md) · [Quickstart](quickstart.md) · [Configuration](configuration.md)

All commands below run from the repository root. Datasets and reference annotations must be supplied separately.

## Training input contract

The training entry points read one AnnData `.h5ad` object with cells as rows and a common feature axis as columns.

| Field | Role |
| :--- | :--- |
| `layers["train_input"]` | Default normalized input matrix |
| `obs["modality"]` | Modality identity for every cell |
| `layers["counts"]` | Preserved count representation from the bundled preprocessors |
| `obs["cell_type"]` | Biological labels for evaluation, when available |
| `obs["batch"]` | Batch labels for eligible evaluation protocols |
| `obs["dataset"]` | Dataset provenance from preprocessing |
| `uns["preprocess"]` | Preprocessing metadata from the RNA–ATAC and RNA–protein scripts |
| `uns["feature_list"]` | Modality-specific feature indices when feature masks are enabled |

The input layer and metadata keys are configurable. A zero-padded feature is not automatically recognized as unmeasured: use the extended runner and feature masks when required by the representation of your modalities.

When `ANCHORPT_USE_FEATURE_MASK=true`, `uns["feature_list"]` must be a dictionary whose keys exactly match the modality strings in `obs["modality"]`. Each value must contain nonempty, unique, zero-based column indices within `[0, n_vars)`. The extended entry point validates this contract.

## RNA–ATAC from Matrix Market

Script: [prepare_gxr_gtf_rna_atac.py](../pre_data/prepare_gxr_gtf_rna_atac.py).

Each input directory contains:

```text
rna/ or atac/
├── matrix.mtx       # Features × cells
├── features.tsv     # Feature identifiers
├── barcodes.tsv     # Cell identifiers
└── metadata.csv     # Optional cell metadata
```

Feature readers use the first TSV column. Metadata CSVs use the first column as the cell-barcode index and are aligned to `barcodes.tsv`; `meta_data.csv` is also accepted and takes precedence over `metadata.csv`. Missing biological labels fall back to `unknown`; these placeholders are not valid biological ground truth for evaluation.

RNA gene identifiers, ATAC peak coordinates, and the gene–region mapping or annotation must use compatible identifiers and genome coordinates. Inspect the script's readers when adapting multicolumn feature files.

**GTF route:** follow the [complete quickstart](quickstart.md#2-build-the-training-anndata-object). The `--geneactivity-annotation` option defaults to `none` in this script. The example uses `HAVANA`; select a source present in your annotation, or `none` to omit this source filter.

**Gene–region mapping route:**

```bash
python pre_data/prepare_gxr_gtf_rna_atac.py \
  --rna-dir /path/to/rna \
  --atac-dir /path/to/atac \
  --route gxr \
  --gxr-csv /path/to/gene_region_mapping.csv \
  --top-genes 5000 \
  --output data/processed/rna_atac.h5ad
```

The mapping CSV must contain `peak` and `gene.name` columns, matching the ATAC feature strings and RNA gene identifiers. The script constructs gene-space ATAC inputs, normalizes and log-transforms the training matrix, and selects genes using scores from both modalities.

## RNA–ATAC from AnnData

Script: [prepare_h5ad_gtf_rna_atac.py](../pre_data/prepare_h5ad_gtf_rna_atac.py).

```bash
python pre_data/prepare_h5ad_gtf_rna_atac.py \
  --rna-h5ad /path/to/rna.h5ad \
  --atac-h5ad /path/to/atac.h5ad \
  --gtf /path/to/annotation.gtf \
  --rna-counts-layer counts \
  --atac-counts-layer counts \
  --top-genes 5000 \
  --output data/processed/rna_atac.h5ad
```

This route expects count layers and uses the companion Matrix Market preprocessor for gene-activity and feature-selection functions. Keep the scripts together. Its annotation-source default is `HAVANA`; `--geneactivity-annotation none` disables that filter.

## RNA–protein

Script: [prepare_h5ad_hgnc_rna_protein.py](../pre_data/prepare_h5ad_hgnc_rna_protein.py).

```bash
python pre_data/prepare_h5ad_hgnc_rna_protein.py \
  --rna-h5ad /path/to/rna.h5ad \
  --protein-h5ad /path/to/protein.h5ad \
  --hgnc-json /path/to/hgnc_complete_set.json \
  --rna-counts-layer counts \
  --protein-counts-layer counts \
  --ambiguous-map-policy error \
  --output data/processed/rna_protein.h5ad
```

The preprocessor maps protein marker names using an HGNC JSON reference, applies RNA normalization and protein CLR transformation, and constructs a union feature axis with modality-specific feature lists. It also writes a protein-mapping TSV beside the output by default and records the HGNC file hash.

The example uses the strict `error` policy for ambiguous mappings. The script default is `first`; record whichever policy you choose. Inspect the mapping report before training. Use the extended training entry point with `ANCHORPT_USE_FEATURE_MASK=true`; see [configuration](configuration.md#rnaprotein-and-three-modality-inputs).

## Three-modality unpaired splitting

Script: [prepare_tripbmc_unpaired_split.py](../pre_data/prepare_tripbmc_unpaired_split.py).

```bash
python pre_data/prepare_tripbmc_unpaired_split.py \
  --source-root /path/to/tripbmc/source \
  --output-dir data/processed/tripbmc_split \
  --seed 42
```

This is a dataset-specific split constructor with fixed source expectations and quotas. It uses source, condition, technical stratum, and canonical cell-type fields to allocate cells. Inspect its source-layout constants and validation before applying it to a dataset.

It constructs disjoint modality allocations; it does not provide a general-purpose raw three-assay-to-AnnData pipeline. Assemble a common-feature AnnData object matching the [training input contract](#training-input-contract) before using the extended runner.

For all routes, retain input checksums, annotation versions, preprocessing arguments, feature order, cell IDs, and split assignments. Split construction using cell-type strata should be reported separately from label-free model training.
