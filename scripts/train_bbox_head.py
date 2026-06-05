"""
Train and evaluate a runway bounding-box regression head on manifold vectors.

We treat z (manifold vector, 256-d) as a pooled RoI feature and attach a small
MLP regression head that predicts a normalised AABB [xmin, ymin, xmax, ymax]
derived from the 4-corner runway annotations in the LARD metadata.

Two-term loss (real samples + synthetic samples):

  Supervised  — smooth-L1 on real samples that have known bbox labels
  Consistency — MSE(head(z*), head(z_nn).detach())
                The synthetic z* has no bbox label; instead it should predict a
                bbox consistent with its nearest real manifold neighbor z_nn.
                Gradient flows only through the synthetic sample, not the
                neighbor (detach).

Experimental comparison:
  Baseline  — head trained with supervised loss only (D1 + D2 nominal)
  Augmented — head trained with supervised + consistency on synthetic z*

Both are evaluated on D2 holdout with mean IoU.

Prerequisites (run first):
  python3 scripts/generate_synthetic.py [options]

Usage:
  PYTHONPATH=src python3 scripts/train_bbox_head.py \\
      --train-npz     outputs/synthetic/pose_cond_256/train_z.npz \\
      --holdout-npz   outputs/synthetic/pose_cond_256/holdout_z.npz \\
      --synthetic-npz outputs/synthetic/pose_cond_256/synthetic_z.npz \\
      --data-dir      data/lard_20k \\
      --output-dir    outputs/bbox_head_results/pose_cond_256
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#  Utility: load bbox labels from LARD parquet 

def _build_bbox_index(data_dir: str) -> dict[str, np.ndarray]:
    """
    Returns {relative_img_path: np.array([xmin, ymin, xmax, ymax], float32)}
    where coordinates are normalised by image width/height to [0, 1].
    """
    import pandas as pd

    bbox_map: dict[str, np.ndarray] = {}
    for parquet in Path(data_dir).rglob("metadata.parquet"):
        df = pd.read_parquet(parquet)
        for _, row in df.iterrows():
            xs = [row["x_TR"], row["x_TL"], row["x_BL"], row["x_BR"]]
            ys = [row["y_TR"], row["y_TL"], row["y_BL"], row["y_BR"]]
            w, h = float(row["width"]), float(row["height"])
            bbox = np.array([
                min(xs) / w,
                min(ys) / h,
                max(xs) / w,
                max(ys) / h,
            ], dtype=np.float32)
            bbox_map[row["image_path"]] = bbox
    return bbox_map


def _build_airport_map(data_dir: str) -> dict[str, str]:
    """Returns {image_path: airport_code} from all parquet files under data_dir."""
    import pandas as pd
    airport_map: dict[str, str] = {}
    for parquet in Path(data_dir).rglob("metadata.parquet"):
        df = pd.read_parquet(parquet)[["image_path", "airport"]]
        airport_map.update(dict(zip(df["image_path"], df["airport"])))
    return airport_map


def _compute_same_airport_nn(
    z_synth_m: np.ndarray,      # (N_synth, 256) manifold coords of synthetic samples
    synth_airports: np.ndarray,  # (N_synth,) airport code per synthetic sample
    z_train_m: np.ndarray,      # (N_train, 256) manifold coords of real training samples
    train_airports: np.ndarray,  # (N_train,) airport code per training sample
    z_train_feat: np.ndarray,   # (N_train, D) raw features — returned for matched neighbor
    min_pool: int = 10,          # fall back to global NN if same-airport pool < min_pool
) -> tuple[np.ndarray, float, float]:
    """For each synthetic sample find the nearest real neighbor in the SAME airport.

    Falls back to the global (cross-airport) nearest neighbor when the same-airport
    training pool has fewer than min_pool samples.

    Returns:
        z_nn        — (N_synth, D) feature vector of each sample's chosen neighbor
        same_rate   — fraction of synthetic samples that got a same-airport neighbor
        fallback_rate — fraction that fell back to global NN
    """
    # Pre-compute global fallback NN for all synthetic samples
    nn_global = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn_global.fit(z_train_m)
    _, global_idx = nn_global.kneighbors(z_synth_m)
    global_idx = global_idx.squeeze()

    result = z_train_feat[global_idx].copy()   # default = global NN
    n_same, n_fallback = 0, 0

    for ap in np.unique(synth_airports):
        synth_mask = synth_airports == ap
        train_mask = train_airports == ap

        if train_mask.sum() < min_pool:
            n_fallback += int(synth_mask.sum())
            continue   # keep global NN result for these synthetic samples

        nn_ap = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
        nn_ap.fit(z_train_m[train_mask])
        _, local_idx = nn_ap.kneighbors(z_synth_m[synth_mask])
        local_idx = local_idx.squeeze()

        # Map local index back to global train index
        global_train_idx = np.where(train_mask)[0][local_idx]
        result[synth_mask] = z_train_feat[global_train_idx]
        n_same += int(synth_mask.sum())

    total = len(synth_airports)
    same_rate     = n_same     / total
    fallback_rate = n_fallback / total
    print(f"  Same-airport NN: {n_same:,} ({same_rate*100:.1f}%)  "
          f"Fallback to global: {n_fallback:,} ({fallback_rate*100:.1f}%)")
    return result.astype(np.float32), same_rate, fallback_rate


def _load_zipped(
    npz_path: str,
    bbox_map: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load feature vectors and bbox labels.

    Key priority: 'embeddings' (raw/decoded 2048-d) > 'z' (manifold 256-d).
    Also returns z_manifold (always the 256-d 'z' key, aligned to the same
    valid-bbox mask) and img_paths for downstream airport lookups.

    Returns: z_feat, z_manifold, bboxes, img_paths  (all filtered to valid bbox)
    """
    d         = np.load(npz_path, allow_pickle=True)
    z_feat    = (d["embeddings"] if "embeddings" in d else d["z"]).astype(np.float32)
    z_mani    = d["z"].astype(np.float32) if "z" in d else z_feat
    img_paths = d["img_paths"]

    valid_mask = np.array([p in bbox_map for p in img_paths])
    if not valid_mask.all():
        n_miss = (~valid_mask).sum()
        print(f"  Warning: {n_miss}/{len(valid_mask)} samples have no bbox annotation — skipped")

    paths_v  = img_paths[valid_mask]
    bboxes   = np.stack([bbox_map[p] for p in paths_v], axis=0).astype(np.float32)
    return z_feat[valid_mask], z_mani[valid_mask], bboxes, paths_v


#  Bbox regression head 

class BboxHead(nn.Module):
    """3-layer MLP: z_dim → hidden → hidden//2 → 4 (normalised AABB)."""

    def __init__(self, z_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 4),
            nn.Sigmoid(),   # outputs in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


#  IoU helpers 

def batch_iou(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Element-wise IoU for (N,4) arrays in [xmin,ymin,xmax,ymax] format."""
    inter_x1 = np.maximum(pred[:, 0], target[:, 0])
    inter_y1 = np.maximum(pred[:, 1], target[:, 1])
    inter_x2 = np.minimum(pred[:, 2], target[:, 2])
    inter_y2 = np.minimum(pred[:, 3], target[:, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter   = inter_w * inter_h

    area_p = (pred[:, 2]   - pred[:, 0])   * (pred[:, 3]   - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union  = area_p + area_t - inter

    return inter / np.maximum(union, 1e-6)


#  Training 

def _train(
    z_real:    np.ndarray,
    bbox_real: np.ndarray,
    z_synth:   np.ndarray | None,
    z_synth_nn: np.ndarray | None,   # nearest real z for each synthetic sample
    epochs:    int = 50,
    lr:        float = 1e-3,
    batch_size: int = 256,
    lam:       float = 0.1,
    device:    torch.device = torch.device("cpu"),
) -> BboxHead:
    """
    Train BboxHead.

    z_synth_nn contains the nearest real neighbour in z-space for each
    synthetic sample (pre-computed to avoid O(N²) search in the training loop).
    """
    head = BboxHead(z_dim=z_real.shape[1]).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=lr)

    z_r  = torch.tensor(z_real,    device=device)
    b_r  = torch.tensor(bbox_real, device=device)

    use_synth = z_synth is not None and lam > 0
    if use_synth:
        z_s  = torch.tensor(z_synth,    device=device)
        z_nn = torch.tensor(z_synth_nn, device=device)
        synth_ds  = TensorDataset(z_s, z_nn)
        synth_loader = DataLoader(synth_ds, batch_size=batch_size, shuffle=True)
        synth_iter   = iter(synth_loader)

    real_ds     = TensorDataset(z_r, b_r)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        head.train()
        total_sup = total_cons = 0.0
        n_batches = 0

        if use_synth:
            synth_iter = iter(synth_loader)   # reset each epoch

        for z_b, bbox_b in real_loader:
            opt.zero_grad()

            # Supervised loss on real samples
            pred_real = head(z_b)
            l_sup = F.smooth_l1_loss(pred_real, bbox_b)

            l_cons = torch.tensor(0.0, device=device)
            if use_synth:
                try:
                    z_syn_b, z_nn_b = next(synth_iter)
                except StopIteration:
                    synth_iter = iter(synth_loader)
                    z_syn_b, z_nn_b = next(synth_iter)

                pred_syn = head(z_syn_b)
                with torch.no_grad():
                    pred_nn = head(z_nn_b)   # neighbour prediction — no grad
                l_cons = F.mse_loss(pred_syn, pred_nn)

            loss = l_sup + lam * l_cons
            loss.backward()
            opt.step()

            total_sup  += l_sup.item()
            total_cons += l_cons.item()
            n_batches  += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3d}  sup={total_sup/n_batches:.4f}  "
                  f"cons={total_cons/n_batches:.4f}")

    return head


@torch.no_grad()
def _predict(head: BboxHead, z: np.ndarray, batch_size: int = 512,
             device: torch.device = torch.device("cpu")) -> np.ndarray:
    head.eval()
    preds = []
    for i in range(0, len(z), batch_size):
        z_b = torch.tensor(z[i:i+batch_size], device=device)
        preds.append(head(z_b).cpu().numpy())
    return np.concatenate(preds, axis=0)


#  CLI 

@click.command()
@click.option("--train-npz",      required=True,
              default="outputs/synthetic/pose_cond_256/train_z.npz", show_default=True)
@click.option("--holdout-npz",    required=True,
              default="outputs/synthetic/pose_cond_256/holdout_z.npz", show_default=True)
@click.option("--synthetic-npz",  default=None, show_default=True,
              help="Optional synthetic z* NPZ. If omitted, only supervised baseline is run.")
@click.option("--data-dir",       default="data/lard_20k", show_default=True)
@click.option("--output-dir",     default="outputs/bbox_head_results/pose_cond_256",
              show_default=True)
@click.option("--epochs",         default=100,  show_default=True)
@click.option("--lr",             default=1e-3, show_default=True)
@click.option("--batch-size",     default=256,  show_default=True)
@click.option("--lam",            default=0.1,  show_default=True,
              help="Weight of consistency loss")
@click.option("--n-nn",           default=1,    show_default=True,
              help="Number of nearest real neighbors per synthetic sample")
@click.option("--k-synth-per-pose", default=5,  show_default=True,
              help="k neighbors per holdout pose used in generate_synthetic.py")
@click.option("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
def main(train_npz, holdout_npz, synthetic_npz, data_dir, output_dir,
         epochs, lr, batch_size, lam, n_nn, k_synth_per_pose, device):

    device = torch.device(device)
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    #  Load bbox + airport indices 
    print("Building bbox index from parquet metadata ...")
    bbox_map    = _build_bbox_index(data_dir)
    airport_map = _build_airport_map(data_dir)
    print(f"  {len(bbox_map):,} bbox annotations loaded")

    def path_to_airport(p: str) -> str:
        return airport_map.get(p, "unknown")

    #  Load z vectors + bboxes 
    print("\nLoading training z vectors ...")
    z_train, z_train_m, bbox_train, train_paths = _load_zipped(train_npz, bbox_map)
    train_airports = np.array([path_to_airport(p) for p in train_paths])
    print(f"  {len(z_train):,} real training samples  z-dim={z_train.shape[1]}")

    print("Loading holdout z vectors ...")
    z_hold, _, bbox_hold, hold_paths = _load_zipped(holdout_npz, bbox_map)
    print(f"  {len(z_hold):,} holdout samples")

    print("Loading synthetic z vectors ...")
    z_synth = z_synth_m_arr = z_synth_nn_cross = z_synth_nn_same = None
    same_rate = fallback_rate = None
    if synthetic_npz is not None:
        synth = np.load(synthetic_npz, allow_pickle=True)
        _synth_key   = "embeddings" if "embeddings" in synth else "z"
        z_synth      = synth[_synth_key].astype(np.float32)
        z_synth_m_arr = synth["z"].astype(np.float32) if "z" in synth else z_synth
        print(f"  {len(z_synth):,} synthetic samples")

        # Airport label for each synthetic sample via holdout img_paths
        # synthetic[i*k : (i+1)*k] correspond to holdout[i]
        hold_airports_all = np.array([path_to_airport(p) for p in hold_paths])
        synth_airports    = np.repeat(hold_airports_all, k_synth_per_pose)

        #  Cross-airport NN (global, current method) 
        print(f"\nBuilding cross-airport NN index (k={n_nn}, 256-d manifold) ...")
        nn_cross = NearestNeighbors(n_neighbors=n_nn, algorithm="auto", n_jobs=-1)
        nn_cross.fit(z_train_m)
        _, nn_idx_cross = nn_cross.kneighbors(z_synth_m_arr)
        z_synth_nn_cross = z_train[nn_idx_cross].mean(axis=1).astype(np.float32)

        #  Same-airport NN (with global fallback) 
        print(f"\nBuilding same-airport NN index (256-d manifold, fallback=global) ...")
        z_synth_nn_same, same_rate, fallback_rate = _compute_same_airport_nn(
            z_synth_m_arr, synth_airports,
            z_train_m, train_airports,
            z_train,
        )
    else:
        print("  None — supervised baseline only")

    #  Training runs 
    print("\n=== Baseline (supervised only) ===")
    head_base = _train(
        z_real=z_train, bbox_real=bbox_train,
        z_synth=None,   z_synth_nn=None,
        epochs=epochs, lr=lr, batch_size=batch_size, lam=0.0, device=device,
    )
    iou_base_train = batch_iou(_predict(head_base, z_train, batch_size, device), bbox_train)
    iou_base       = batch_iou(_predict(head_base, z_hold,  batch_size, device), bbox_hold)

    iou_cross_train = iou_aug_train = iou_base_train
    iou_cross       = iou_same      = iou_base

    if z_synth is not None:
        print(f"\n=== Cross-airport consistency (lambda={lam}) ===")
        head_cross = _train(
            z_real=z_train, bbox_real=bbox_train,
            z_synth=z_synth, z_synth_nn=z_synth_nn_cross,
            epochs=epochs, lr=lr, batch_size=batch_size, lam=lam, device=device,
        )
        iou_cross_train = batch_iou(_predict(head_cross, z_train, batch_size, device), bbox_train)
        iou_cross       = batch_iou(_predict(head_cross, z_hold,  batch_size, device), bbox_hold)

        print(f"\n=== Same-airport consistency (lambda={lam}, {same_rate*100:.1f}% same-ap, "
              f"{fallback_rate*100:.1f}% fallback) ===")
        head_same = _train(
            z_real=z_train, bbox_real=bbox_train,
            z_synth=z_synth, z_synth_nn=z_synth_nn_same,
            epochs=epochs, lr=lr, batch_size=batch_size, lam=lam, device=device,
        )
        iou_same_train = batch_iou(_predict(head_same, z_train, batch_size, device), bbox_train)
        iou_same       = batch_iou(_predict(head_same, z_hold,  batch_size, device), bbox_hold)
    else:
        iou_same_train = iou_cross_train = iou_base_train
        iou_same       = iou_cross       = iou_base

    #  3-way results table 
    def _print_3way(label: str, n: int,
                    iou_b: np.ndarray, iou_c: np.ndarray, iou_s: np.ndarray):
        flag = lambda a, b: "↑" if (a-b) > 0.005 else ("↓" if (b-a) > 0.005 else "~")
        print(f"\n {label} (N={n:,}) {'':<28}")
        print(f"  {'Metric':<20}  {'Baseline':>10}  {'Cross-ap':>10}  {'Same-ap':>10}  "
              f"{'Δcross':>8}  {'Δsame':>8}")
        print("  " + "─" * 74)
        for thr in [0.25, 0.50, 0.75]:
            vb = (iou_b >= thr).mean()
            vc = (iou_c >= thr).mean()
            vs = (iou_s >= thr).mean()
            print(f"  IoU >= {thr:.2f}{'':>12} {vb:>10.4f}  {vc:>10.4f}  {vs:>10.4f}  "
                  f"{vc-vb:>+7.4f}{flag(vc,vb)}  {vs-vb:>+7.4f}{flag(vs,vb)}")
        mb, mc, ms = iou_b.mean(), iou_c.mean(), iou_s.mean()
        print(f"  Mean IoU{'':>16} {mb:>10.4f}  {mc:>10.4f}  {ms:>10.4f}  "
              f"{mc-mb:>+7.4f}{flag(mc,mb)}  {ms-mb:>+7.4f}{flag(ms,mb)}")

    _print_3way("TRAIN  (D1+D2 nominal)",
                len(z_train), iou_base_train, iou_cross_train, iou_same_train)
    _print_3way("HOLDOUT (D2 out-of-distribution)",
                len(z_hold),  iou_base,       iou_cross,       iou_same)

    n_synth_str = f"{len(z_synth):,}" if z_synth is not None else "0"
    print(f"\n  Train real: {len(z_train):,}  |  Synthetic: {n_synth_str}  "
          f"|  Holdout: {len(z_hold):,}")

    #  Plot 
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(iou_base,  bins=bins, alpha=0.5, label=f"Baseline (mean={iou_base.mean():.3f})",
            color="#4c72b0")
    ax.hist(iou_cross, bins=bins, alpha=0.5,
            label=f"Cross-airport consistency (mean={iou_cross.mean():.3f})", color="#dd8452")
    ax.hist(iou_same,  bins=bins, alpha=0.5,
            label=f"Same-airport consistency (mean={iou_same.mean():.3f})",  color="#55a868")
    for iou_arr, col in [(iou_base, "#4c72b0"), (iou_cross, "#dd8452"), (iou_same, "#55a868")]:
        ax.axvline(iou_arr.mean(), color=col, linestyle="--", linewidth=1.5)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title(f"Runway BBox Head — IoU on D2 Holdout ({len(z_hold):,} samples)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path = out / "bbox_iou_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {fig_path}")

    #  Save results 
    np.save(out / "results.npy", {
        "iou_base": iou_base, "iou_cross": iou_cross, "iou_same": iou_same,
        "iou_base_train": iou_base_train, "iou_cross_train": iou_cross_train,
        "iou_same_train": iou_same_train,
        "n_train": len(z_train), "n_synth": len(z_synth) if z_synth is not None else 0,
        "n_holdout": len(z_hold), "lam": lam, "epochs": epochs,
        "same_rate": same_rate, "fallback_rate": fallback_rate,
    })
    print(f"Results saved → {out}/results.npy")


if __name__ == "__main__":
    main()

