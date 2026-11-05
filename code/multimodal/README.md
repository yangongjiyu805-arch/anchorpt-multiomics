# Extended multimodal workflow

[Code overview](../README.md) · [Configuration](../../docs/configuration.md#rnaprotein-and-three-modality-inputs)

This workflow supports RNA–protein inputs and assembled three-modality AnnData objects. It includes optional modality-specific feature masks.

| Path | Purpose |
| :--- | :--- |
| [main_anchorpt.py](main_anchorpt.py) | Input validation, feature-mask setup, training, and export |
| [evaluate_metrics.py](evaluate_metrics.py) | Embedding evaluation |
| [anchorpt/](anchorpt/) | Extended objectives and runner with shared model components |

Run from the repository root after configuring the input and output paths:

```bash
export ANCHORPT_SRC_DIR="$PWD/code/multimodal"
python code/multimodal/main_anchorpt.py
```

When `ANCHORPT_USE_FEATURE_MASK=true`, supply modality-matched indices in `uns["feature_list"]`. See the [input contract](../../docs/preprocessing.md#training-input-contract).

Evaluate `obsm["latent_shared"]` for the primary analysis; additional outputs are described in [reproducibility](../../docs/reproducibility.md).
