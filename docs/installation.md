# Installation

[Home](../README.md) · [Quickstart](quickstart.md) · [Configuration](configuration.md)

## Obtain the code

```bash
git clone https://github.com/yangongjiyu805-arch/anchorpt-multiomics.git
cd anchorpt-multiomics
```

## Environment

Use an isolated Python environment. The commands below use Python 3.11. Check dependency and binary compatibility on the target platform before training.

The [dependency snapshot](../requirements.txt) pins PyTorch to `2.3.1+cu121` and expands `PYTORCH_INDEX_URL` during installation. Set the variable before invoking pip.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

These commands target a platform supported by the CUDA 12.1 wheel and an appropriate NVIDIA driver. The runner selects CUDA when available and CPU otherwise, but the requirements file itself is CUDA-specific. For CPU-only or macOS installations, replace the CUDA-specific PyTorch dependency with a compatible platform build.

The pins also combine PyTorch 2.3.1 with NumPy 2.0.1. Validate the NumPy–PyTorch bridge below before a long run; dependency resolution alone does not establish binary compatibility. If the bridge fails, use a compatible NumPy/PyTorch combination and record the package versions with the experiment.

## Check the installation

```bash
python - <<'PY'
import numpy as np
import torch
import scanpy as sc
import scib_metrics

print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("Scanpy:", sc.__version__)
print("CUDA available:", torch.cuda.is_available())
print("NumPy bridge:", torch.from_numpy(np.zeros((2, 2), dtype=np.float32)).shape)
PY

python code/rna_atac/main_anchorpt.py --help
python code/rna_atac/evaluate_metrics.py --help
python code/preprocessing/prepare_gxr_gtf_rna_atac.py --help
```

The GTF preprocessing route also requires `episcanpy` and a compatible annotation and peak format.

## Shell conventions

Examples use Bash, including `export` and backslash line continuation. On Windows, run them in an appropriate Bash environment or translate environment assignments to PowerShell:

```powershell
$env:PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
$env:ANCHORPT_RANDOM_SEED = "7"
```

For each experiment, retain the commit ID, Python and package versions, GPU/driver information, and the effective parameter JSON. Continue with the [quickstart](quickstart.md).
