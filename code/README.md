# AnchorPT implementation

[Project home](../README.md) · [Quickstart](../docs/quickstart.md) · [Configuration](../docs/configuration.md)

The implementation is grouped by responsibility:

| Directory | Responsibility | Entry points |
| :--- | :--- | :--- |
| [preprocessing/](preprocessing/README.md) | Convert input assays and construct unpaired splits | Four dataset-preparation scripts |
| [rna_atac/](rna_atac/) | RNA–ATAC integration and evaluation | `main_anchorpt.py`, `evaluate_metrics.py` |
| [multimodal/](multimodal/README.md) | RNA–protein and assembled three-modality integration | `main_anchorpt.py`, `evaluate_metrics.py` |

Run commands from the **repository root**, where `requirements.txt` resides. Each training workflow includes an `anchorpt/` package containing the model, objectives, training routines, and runner.

Both training entry points default to 100 Stage-I epochs and 80 Stage-II epochs, subject to early stopping. See the [configuration reference](../docs/configuration.md) for all documented settings.

## Directory migration

| Previous path | Current path |
| :--- | :--- |
| `pre_data/` | `code/preprocessing/` |
| `src/` | `code/rna_atac/` |
| `src_rna_adt_three_omics/` | `code/multimodal/` |

Update existing commands and `ANCHORPT_SRC_DIR` values to the current paths. Scientific objectives and model defaults are unchanged by this directory reorganization.
