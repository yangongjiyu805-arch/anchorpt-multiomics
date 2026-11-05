# Training configuration

[Home](../README.md) · [Quickstart](quickstart.md) · [Reproducibility](reproducibility.md)

The training scripts use `ANCHORPT_*` environment variables. Their `--help` output is brief; the entry-point source defines the complete set of options.

## Selecting the runner

| Workflow | Entry point | Explicit source directory |
| :--- | :--- | :--- |
| RNA–ATAC | [src/main_anchorpt.py](../src/main_anchorpt.py) | `$PWD/src` |
| RNA–protein or assembled three-modality data | [extended main_anchorpt.py](../src_rna_adt_three_omics/main_anchorpt.py) | `$PWD/src_rna_adt_three_omics` |

Both trees contain a package named `anchorpt`. Set `ANCHORPT_SRC_DIR` to the selected tree and run each experiment in a fresh Python process, so the intended runner is imported.

## Paths and input

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `ANCHORPT_INPUT_H5AD` | Empty; set explicitly | Training AnnData path |
| `ANCHORPT_OUTPUT_H5AD` | `output.h5ad` | Integrated output |
| `ANCHORPT_PARAMS_USED_JSON` | `params.json` | Effective parameter record |
| `ANCHORPT_METHOD_MANIFEST_JSON` | Derived from parameter path | Method-role manifest |
| `ANCHORPT_INPUT_LAYER` | `train_input` | Input layer; empty selects `.X` |
| `ANCHORPT_MODALITY_KEY` | `modality` | Modality metadata column |
| `ANCHORPT_STRICT_UNPAIRED` | `true` | Enforce absence of batch/cell-type supervision |
| `ANCHORPT_BATCH_KEY` | Empty | Training batch key |
| `ANCHORPT_CELLTYPE_KEY` | Empty | Training cell-type key |
| `ANCHORPT_USE_TRAINING_LABELS` | `false` | Allow training-label input |

Under strict-unpaired mode, nonempty batch/cell-type keys or enabled training labels raise an error. Evaluation labels can remain in `obs` without supplying them as training keys.

## Representation and optimization

These are **released code defaults**, not a complete manuscript experiment specification.

| Variable | Default |
| :--- | :--- |
| `ANCHORPT_LATENT_SHARED` | `48` |
| `ANCHORPT_LATENT_SPECIFIC` | `20` |
| `ANCHORPT_HIDDEN_LAYERS` | `[128, 128]` |
| `ANCHORPT_DROPOUT_RATE` | `0.05` |
| `ANCHORPT_BETA` | `1.25` |
| `ANCHORPT_SHARED_RECON_WEIGHT` | `1.0` |
| `ANCHORPT_TRAINING_MODE` | `two_stage`; also accepts `base` |
| `ANCHORPT_STAGE1_EPOCH_NUM` | `100`, inherited from `ANCHORPT_EPOCH_NUM` |
| `ANCHORPT_STAGE2_EPOCH_NUM` | `80` |
| `ANCHORPT_BATCH_SIZE` | `128` |
| `ANCHORPT_LR` | `7e-4` |
| `ANCHORPT_STAGE2_LR` | `2e-4` |
| `ANCHORPT_ADAPTLR` | `true` |
| `ANCHORPT_PATIENCE` | `25` |
| `ANCHORPT_BALANCED_MODALITY_BATCHES` | `true` |
| `ANCHORPT_STAGE2_TRAINABLE` | `shared_encoder` |
| `ANCHORPT_STAGE2_BASE_WEIGHT` | `1.0` |
| `ANCHORPT_STAGE2_OUTPUT_BLEND` | `1.0` |

Both training modes enable early stopping. Requested epoch counts therefore need not equal the number actually executed.

Set `ANCHORPT_RANDOM_SEED` explicitly. Without it (or `ANCHORPT_SEED`), the RNA–ATAC entry point generates a random seed, while the extended entry point defaults to `42`. The effective seed is recorded in the parameter JSON.

## State anchors and transport

| Variable | Default |
| :--- | :--- |
| `ANCHORPT_PIVOTAL_ANCHOR_NUM` | `16` |
| `ANCHORPT_PIVOTAL_ANCHOR_INIT` | `stage1_kmeans` |
| `ANCHORPT_PIVOTAL_ANCHOR_WEIGHT` | `0.15` |
| `ANCHORPT_ASSIGNMENT_TEMPERATURE` | `0.5` |
| `ANCHORPT_SINKHORN_EPSILON` | `0.05` |
| `ANCHORPT_SINKHORN_RHO` | `1.0` |
| `ANCHORPT_SINKHORN_ITERS` | `80` |
| `ANCHORPT_EMA_MOMENTUM` | `0.98` |
| `ANCHORPT_CONTRASTIVE_TEMPERATURE` | `0.08` |
| `ANCHORPT_CONTRASTIVE_WEIGHT` | `0.20` |
| `ANCHORPT_TRANSPORT_COST_LOGIT_WEIGHT` | `2.50` |
| `ANCHORPT_USE_GLOBAL_ANCHOR_TRANSPORT` | `true` |
| `ANCHORPT_STAGE1_KNN_GRAPH_WEIGHT` | `0.05` |
| `ANCHORPT_STAGE1_KNN_GRAPH_K` | `8` |
| `ANCHORPT_STAGE1_KNN_GRAPH_MAX_CELLS` | `128` |
| `ANCHORPT_STAGE2_PIVOTAL_ANCHOR_WARMUP_EPOCHS` | `5` |

Names follow the implementation. Do not identify a code coefficient with a manuscript loss coefficient without checking its use in the objective and training code.

## RNA–protein and three-modality inputs

The extended entry point adds `ANCHORPT_USE_FEATURE_MASK` (default `false`) and `ANCHORPT_MODALITY_PAIR_WEIGHTS` (unset by default). Modality pair weights are an advanced option; inspect the extended objective for pair ordering before supplying a list.

For the output of the bundled RNA–protein preprocessor:

```bash
export ANCHORPT_SRC_DIR="$PWD/src_rna_adt_three_omics"
export ANCHORPT_INPUT_H5AD="$PWD/data/processed/rna_protein.h5ad"
export ANCHORPT_OUTPUT_H5AD="$PWD/results/rna-protein/result.h5ad"
export ANCHORPT_PARAMS_USED_JSON="$PWD/results/rna-protein/params.json"
export ANCHORPT_INPUT_LAYER=train_input
export ANCHORPT_MODALITY_KEY=modality
export ANCHORPT_USE_FEATURE_MASK=true
export ANCHORPT_RANDOM_SEED=7
export ANCHORPT_STRICT_UNPAIRED=true
export ANCHORPT_USE_TRAINING_LABELS=false
export ANCHORPT_BATCH_KEY=""
export ANCHORPT_CELLTYPE_KEY=""

python src_rna_adt_three_omics/main_anchorpt.py
```

For three modalities, provide an assembled AnnData object with the same contract. Enable feature masks only when its `uns["feature_list"]` is correctly populated.

## Effective records

With `ANCHORPT_PARAMS_USED_JSON=results/run/params.json`, the derived manifest path is `results/run/params_method_manifest.json`. Optional checkpoint paths are controlled by `ANCHORPT_STAGE1_CKPT_PATH` and `ANCHORPT_STAGE2_CKPT_PATH`; they are unset by default.

Keep the parameter JSON and manifest with each run. An environment variable is only an effective setting if the selected entry point actually reads it.
