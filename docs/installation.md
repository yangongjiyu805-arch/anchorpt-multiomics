# Installation

Run in Bash:

```bash
git clone https://github.com/yangongjiyu805-arch/anchorpt-multiomics.git
cd anchorpt-multiomics
python3.11 -m venv .venv
source .venv/bin/activate
export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

The pins target CUDA 12.1 PyTorch. CPU-only or macOS setups need a compatible PyTorch build. The GTF route requires `episcanpy`.

Check binary compatibility for the pinned PyTorch 2.3.1 / NumPy 2.0.1 combination:

```bash
python -c 'import numpy as np, torch; print(torch.__version__, np.__version__); print(torch.from_numpy(np.zeros((2, 2), dtype=np.float32)))'
```

If this fails, use compatible package versions and record them with the run.

[Run the example](quickstart.md)
