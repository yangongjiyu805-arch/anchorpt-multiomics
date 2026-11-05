# Multimodal training

Supports RNA–protein and assembled three-modality AnnData inputs.

Set input/output variables as described in [configuration](../../docs/configuration.md#rnaprotein-and-three-modality-inputs), then run from the repository root:

```bash
export ANCHORPT_SRC_DIR="$PWD/code/multimodal"
python code/multimodal/main_anchorpt.py
```

For feature masks, set `ANCHORPT_USE_FEATURE_MASK=true` and provide `uns["feature_list"]` keyed by modality. Use `obsm["latent_shared"]` for evaluation.
