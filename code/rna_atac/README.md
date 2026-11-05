# RNA–ATAC workflow

[Code overview](../README.md) · [Complete tutorial](../../docs/quickstart.md)

| Path | Purpose |
| :--- | :--- |
| [main_anchorpt.py](main_anchorpt.py) | Environment-configured training and embedding export |
| [evaluate_metrics.py](evaluate_metrics.py) | Shared-embedding evaluation |
| [anchorpt/](anchorpt/) | Model, data handling, objectives, training, and runner |

Run from the repository root after configuring the input and output paths:

```bash
export ANCHORPT_SRC_DIR="$PWD/code/rna_atac"
python code/rna_atac/main_anchorpt.py
```

See [configuration](../../docs/configuration.md) for the required environment variables and [reproducibility](../../docs/reproducibility.md) for embedding roles and evaluation settings.
