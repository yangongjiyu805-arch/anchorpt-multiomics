# Configuration

Training uses `ANCHORPT_*` environment variables. Run commands from the repository root.

## Selecting the runner

| Workflow | Entry point | `ANCHORPT_SRC_DIR` |
| :--- | :--- | :--- |
| RNA–ATAC | [main_anchorpt.py](../code/rna_atac/main_anchorpt.py) | `$PWD/code/rna_atac` |
| RNA–protein / three modalities | [main_anchorpt.py](../code/multimodal/main_anchorpt.py) | `$PWD/code/multimodal` |

Use a fresh Python process for each runner; both contain an `anchorpt` package.

## Paths and input

| Variable | Default |
| :--- | :--- |
| `ANCHORPT_INPUT_H5AD` | Required |
| `ANCHORPT_OUTPUT_H5AD` | `output.h5ad` |
| `ANCHORPT_PARAMS_USED_JSON` | `params.json` |
| `ANCHORPT_INPUT_LAYER` | `train_input`; empty selects `.X` |
| `ANCHORPT_MODALITY_KEY` | `modality` |

Strict-unpaired training is enabled by default. Keep `ANCHORPT_BATCH_KEY` and `ANCHORPT_CELLTYPE_KEY` empty and `ANCHORPT_USE_TRAINING_LABELS=false`. Evaluation labels may remain in `obs`.

## Training defaults

| Variable | Default |
| :--- | :--- |
| `ANCHORPT_LATENT_SHARED` / `ANCHORPT_LATENT_SPECIFIC` | `48` / `20` |
| `ANCHORPT_HIDDEN_LAYERS` / `ANCHORPT_DROPOUT_RATE` | `[128, 128]` / `0.05` |
| `ANCHORPT_STAGE1_EPOCH_NUM` / `ANCHORPT_STAGE2_EPOCH_NUM` | `100` / `80` |
| `ANCHORPT_LR` / `ANCHORPT_STAGE2_LR` | `7e-4` / `2e-4` |
| `ANCHORPT_BATCH_SIZE` | `128` |
| `ANCHORPT_PIVOTAL_ANCHOR_NUM` | `16` |
| `ANCHORPT_STAGE2_OUTPUT_BLEND` | `1.0` |

Early stopping can shorten a run. Set `ANCHORPT_RANDOM_SEED` explicitly; otherwise the RNA–ATAC runner generates a seed and the multimodal runner uses 42. All effective settings are saved to `params.json`.

The entry-point source lists the full set of optimization, transport, and checkpoint options.

## RNA–protein and three-modality inputs

Use a [compatible AnnData object](preprocessing.md#training-input-contract):

```bash
export ANCHORPT_SRC_DIR="$PWD/code/multimodal"
export ANCHORPT_INPUT_H5AD="$PWD/data/processed/rna_protein.h5ad"
export ANCHORPT_OUTPUT_H5AD="$PWD/results/rna-protein/result.h5ad"
export ANCHORPT_PARAMS_USED_JSON="$PWD/results/rna-protein/params.json"
export ANCHORPT_USE_FEATURE_MASK=true
export ANCHORPT_RANDOM_SEED=7
export ANCHORPT_BATCH_KEY=""
export ANCHORPT_CELLTYPE_KEY=""
export ANCHORPT_USE_TRAINING_LABELS=false
export ANCHORPT_STRICT_UNPAIRED=true
python code/multimodal/main_anchorpt.py
```

Enable feature masks only when `uns["feature_list"]` contains the required modality-specific indices.
