# Shared Manifold Domain Transfer

Domain augmentation for runway approach imagery: learn a shared geometric manifold over frozen ResNet-50 embeddings from two simulator domains (X-Plane and MSFS) using a pose-conditioned M-Flow. Synthetic latent vectors at off-nominal poses are generated via pose-swap and used to finetune a bounding box regression head via a consistency loss.

Data: [LARD V2](https://huggingface.co/datasets/DEEL-AI/LARD_V2) — runway approach images with 6-DOF pose labels.

---

## Setup

```bash
python3 -m venv .runway_domain_transfer_env
source .runway_domain_transfer_env/bin/activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
source setup_env.sh   # sets PYTHONPATH=src
```

---

## Pipeline Overview

```
1. Download data
2. Train ResNet-50 bbox detector        → outputs/checkpoints/resnet_bbox/best_resnet.pt
3. Extract frozen embeddings            → outputs/embeddings/lard_20k/{d1,d2,holdout}_full.npz
4. Train M-Flow                         → outputs/checkpoints/mflow_pose_cond/best_mflow_256d.pt
5. (Optional) Probe manifold            → R^2 scores per pose variable
6. Generate synthetic embeddings        → outputs/synthetic/pose_cond_256/
7. Train bbox head (baseline + augmented) → final IoU evaluation
```

---

## Step 1 — Download Data

```bash
# Download ~20k samples per domain (used for all experiments)
python3 scripts/download_lard.py download --max-per-split 20000

# Diagnostic plots (pose distributions, image grid)
python3 scripts/visualize_dataset.py
```

Images saved to `data/lard_20k/{xplane,msfs}/images/`.

Domain splits used in training:

| Split | Source | N |
|---|---|---|
| Domain A (train) | X-Plane | 15,596 |
| Domain B nominal (train) | MSFS | 1,209 |
| Domain B off-nominal (holdout / eval) | MSFS | 13,459 |

---

## Step 2 — Train ResNet-50 Bounding Box Detector

Fine-tunes a ResNet-50 (ImageNet pretrained) end-to-end on runway bbox regression using D1 + D2 nominal data. The `avgpool` layer produces 2048-d embeddings used in all downstream steps.

```bash
PYTHONPATH=src python3 scripts/train_resnet_detector.py
```

**Saves:** `outputs/checkpoints/resnet_bbox/best_resnet.pt`
Keys: `model_state_dict`, `epoch`, `val_loss`, `cfg`

---

## Step 3 — Extract Frozen Embeddings

Loads the trained ResNet checkpoint, strips the bbox head, and runs inference over D1, D2 nominal, and the holdout split to produce `.npz` files.

```bash
PYTHONPATH=src python3 scripts/extract_resnet_embeddings.py \
    --checkpoint outputs/checkpoints/resnet_bbox/best_resnet.pt \
    --data-dir   data/lard_20k \
    --output-dir outputs/embeddings/lard_20k
```

**Saves** (all in `outputs/embeddings/lard_20k/`):

| File | Contents |
|---|---|
| `d1_full.npz` | X-Plane train embeddings |
| `d2_full.npz` | MSFS nominal embeddings |
| `holdout_full.npz` | MSFS off-nominal (eval only) |

Each `.npz` contains arrays: `embeddings (N, 2048)`, `poses (N, 6)`, `domains (N,)`, `img_paths (N,)`.

---

## Step 4 — Train M-Flow

Trains a pose-conditioned injective normalizing flow over the frozen 2048-d embeddings. The flow maps embeddings to a 256-d on-manifold coordinate `z` plus a 1792-d off-manifold noise `e`.

```bash
PYTHONPATH=src python3 -m shared_manifold_domain_transfer.training.train_mflow \
    --d1-npz  outputs/embeddings/lard_20k/d1_full.npz \
    --d2-npz  outputs/embeddings/lard_20k/d2_full.npz
```

Config: `configs/mflow.yaml`. Key hyperparameters:

| Param | Value |
|---|---|
| `manifold_dim` | 256 |
| `n_coupling_layers` | 8 |
| `cond_flow` | true (pose embedding fed to each coupling layer) |
| λ_recon | 1.0 |
| λ_nll | 0.1 |
| λ_geo | 0.01 |
| λ_pose | 1.0 |

**Saves:** `outputs/checkpoints/mflow_pose_cond/best_mflow_256d.pt`

TensorBoard logs: `outputs/runs/mflow/`

---

## Step 5 — (Optional) Probe Manifold

Fits a ridge regression from manifold coordinates `z` to each of the 6 pose dimensions to verify pose structure in the learned manifold.

```bash
PYTHONPATH=src python3 scripts/probe_manifold.py \
    --checkpoint  outputs/checkpoints/mflow_pose_cond/best_mflow_256d.pt \
    --d1-npz      outputs/embeddings/lard_20k/d1_full.npz \
    --d2-npz      outputs/embeddings/lard_20k/d2_full.npz \
    --pose-scaler outputs/checkpoints/mflow_pose_cond/pose_scaler.npy
```

Reports R^2 per pose variable. High R^2 (>0.4) confirms pose is encoded on the manifold. Key results from ablation: along-track R^2=0.632, yaw R^2=0.640.

UMAP visualizations of the manifold:
```bash
PYTHONPATH=src python3 scripts/visualize_mflow.py
```
Figures written to `outputs/umap_20k/{full,crop}/`.

---

## Step 6 — Generate Synthetic Embeddings

For each holdout pose, finds k=5 nearest neighbors in training pose space, strips the pose component from their manifold code, and injects the target pose. Decodes back to 2048-d ambient space.

```bash
PYTHONPATH=src python3 scripts/generate_synthetic.py \
    --checkpoint  outputs/checkpoints/mflow_pose_cond/best_mflow_256d.pt \
    --d1-npz      outputs/embeddings/lard_20k/d1_full.npz \
    --d2-npz      outputs/embeddings/lard_20k/d2_full.npz \
    --holdout-npz outputs/embeddings/lard_20k/holdout_full.npz \
    --output-dir  outputs/synthetic/pose_cond_256
```

Produces ~67k synthetic embeddings (5 neighbors × 13,459 holdout poses).
**Saves:** `outputs/synthetic/pose_cond_256/train_z.npz`

---

## Step 7 — Train Bounding Box Head

Trains two heads for comparison:
- **Baseline** — supervised smooth-L1 on D1 + D2 nominal real embeddings only
- **Augmented** — supervised loss + consistency loss on synthetic embeddings

```bash
PYTHONPATH=src python3 scripts/train_bbox_head.py \
    --train-npz     outputs/synthetic/pose_cond_256/train_z.npz \
    --holdout-npz   outputs/embeddings/lard_20k/holdout_full.npz
```

The consistency loss anchors bbox predictions on synthetic embeddings to their nearest real neighbor's prediction (stop-gradient):

```
L = L_sup + 0.1 * L_cons
L_cons = MSE(head(z*), sg[head(z_nn)])
```

---

## Results

| Model | Train IoU | Holdout IoU | IoU ≥ 0.25 | IoU ≥ 0.50 | Δ Mean IoU |
|---|---|---|---|---|---|
| Baseline (supervised only) | 0.325 | 0.256 | 0.456 | 0.212 | — |
| Augmented (manifold-NN consistency) | 0.326 | 0.264 | 0.474 | 0.220 | +0.009 ↑ |

See [RESULTS.md](RESULTS.md) for full ablation results.
