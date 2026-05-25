# Shared Manifold Domain Transfer

Domain transfer for runway approach imagery: align a synthetic simulator domain (XPlane) with a real-world domain (MSFS) by learning a shared latent manifold via M-Flow.

Data: [LARD V2](https://huggingface.co/datasets/DEEL-AI/LARD_V2) — runway approach images with 6-DOF pose labels.

## Setup

```bash
python3 -m venv .runway_domain_transfer_env
source .runway_domain_transfer_env/bin/activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
source setup_env.sh   # sets PYTHONPATH=src
```

Once sourced, your pythonpath should include the `src/` directory, allowing you to import project modules from anywhere.
## Data

```bash
# Download 1000 samples of each domain
python3 scripts/download_lard.py download --max-per-split 1000

# Diagnostic plots (pose distributions, image grid)
python3 scripts/visualize_dataset.py
```

Outputs to `data/lard/{xplane,msfs}/`.

## UMAP Baseline

Establish domain gap before training. Two embedding modes:

```bash
# Full-image mean-pool
python3 scripts/run_umap_baseline.py \
    --weights models/IN22K-vit.h.14-900e.pth.tar

# Runway-crop patch pool (focuses on runway texture)
python3 scripts/run_umap_baseline.py \
    --weights models/IN22K-vit.h.14-900e.pth.tar \
    --embedding-mode crop

# Smoke test (random weights, no checkpoint needed)
python3 scripts/run_umap_baseline.py --random-weights --max-samples 50
```

Key metric: **UMAP Silhouette Score**
- > 0.3 — clear domain gap; M-Flow training meaningful
- 0.1–0.3 — moderate gap; inspect pose coverage plot
- < 0.1 — weak gap; consider pose-conditional embeddings

The I-JEPA weights are not in the repo due to size, but can be downloaded from the project releases or Hugging Face. Place the checkpoint at `models/IN22K-vit.h.14-900e.pth.tar` before running the baseline script.
- I-JEPA ViT-H/14 checkpoint (~9.6 GB): `models/IN22K-vit.h.14-900e.pth.tar`

Figures written to `outputs/umap_baseline/{full,crop}/`.

## M-Flow Training

```bash
python3 -m shared_manifold_domain_transfer.training.train_mflow
```

Config: `configs/mflow.yaml`. Set `ijepa.weights_path` and `data.data_dir` before running.

## Results

See [RESULTS.md](RESULTS.md).
