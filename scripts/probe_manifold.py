"""
Linear probe: z -> pose variables.

Fits a ridge regression from the manifold coordinates z (and optionally the
raw embedding x) to each of the 6 pose dimensions, reporting R^2 on a held-out
20% test split.

If R^2 from z is high, pose signal is encoded on the manifold but spread across
dimensions (invisible in 2D UMAP).  If R^2 from z is near zero but R^2 from x is
high, the flow pushed pose information off-manifold.

Usage:
  python scripts/probe_manifold.py \\
      --checkpoint  outputs/checkpoints/mflow_pose_concat/best_mflow.pt \\
      --d1-npz      outputs/embeddings/lard_20k/d1_full.npz \\
      --d2-npz      outputs/embeddings/lard_20k/d2_full.npz \\
      --pose-scaler outputs/checkpoints/mflow_pose_concat/pose_scaler.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shared_manifold_domain_transfer.models.mflow import RunwayMFlow

POSE_COLS = [
    "along_track_distance",
    "lateral_path_angle",
    "vertical_path_angle",
    "roll",
    "pitch",
    "yaw",
]


def load_checkpoint(ckpt_path: str, device: torch.device) -> RunwayMFlow:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]   # saved as plain dict
    m    = cfg["model"] if isinstance(cfg, dict) else cfg.model
    def _g(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)
    model = RunwayMFlow(
        ambient_dim=_g(m, "ambient_dim"),
        manifold_dim=_g(m, "manifold_dim"),
        n_coupling_layers=_g(m, "n_coupling_layers"),
        hidden_dim=_g(m, "hidden_dim"),
        pose_dim=_g(m, "pose_dim"),
        pose_hidden_dim=_g(m, "pose_hidden_dim"),
        cond_flow=_g(m, "cond_flow") if "cond_flow" in (m if isinstance(m, dict) else vars(m)) else False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return model


def encode_all(
    model: RunwayMFlow,
    npz_path: str,
    device: torch.device,
    batch_size: int,
    npz_path_b: str | None,
    pose_scaler: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x, z, poses) as numpy arrays."""
    raw = np.load(npz_path, allow_pickle=True)
    embeddings = raw["embeddings"].astype(np.float32)
    if npz_path_b is not None:
        raw_b = np.load(npz_path_b, allow_pickle=True)
        embeddings = np.concatenate([embeddings, raw_b["embeddings"].astype(np.float32)], axis=1)
    poses = raw["poses"].astype(np.float32)

    if pose_scaler is not None:
        pose_min = pose_scaler["pose_min"]
        pose_max = pose_scaler["pose_max"]
        denom    = pose_max - pose_min
        denom[denom == 0] = 1.0
        norm_poses = (2.0 * (poses - pose_min) / denom - 1.0).astype(np.float32)
        embeddings = np.concatenate([embeddings, norm_poses], axis=1)

    all_z = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            x_b = torch.tensor(embeddings[i:i+batch_size]).to(device)
            p_b = torch.tensor(poses[i:i+batch_size]).to(device)
            z, _, _ = model.encode(x_b, p_b)
            all_z.append(z.cpu().numpy())

    return embeddings, np.concatenate(all_z, axis=0), poses


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def probe(X_train, X_test, y_train, y_test, alpha: float = 1.0) -> float:
    """Ridge regression R^2 via closed-form solution."""
    # Add bias column
    Xtr = np.hstack([X_train, np.ones((len(X_train), 1))])
    Xte = np.hstack([X_test,  np.ones((len(X_test),  1))])
    # Ridge: (XᵀX + αI)⁻¹ Xᵀ y
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    b = Xtr.T @ y_train
    w = np.linalg.solve(A, b)
    y_pred = Xte @ w
    return r2(y_test, y_pred)


@click.command()
@click.option("--checkpoint",  required=True)
@click.option("--d1-npz",      required=True)
@click.option("--d2-npz",      default=None, help="MSFS npz (optional, probes both domains if given)")
@click.option("--d1-npz-b",    default=None)
@click.option("--d2-npz-b",    default=None)
@click.option("--pose-scaler", default=None)
@click.option("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--batch-size",  default=512)
@click.option("--alpha",       default=1.0, help="Ridge regularization strength", show_default=True)
@click.option("--test-frac",   default=0.2,  show_default=True)
def main(
    checkpoint, d1_npz, d2_npz, d1_npz_b, d2_npz_b,
    pose_scaler, device, batch_size, alpha, test_frac,
):
    dev = torch.device(device)
    model = load_checkpoint(checkpoint, dev)

    scaler = None
    if pose_scaler:
        scaler = np.load(pose_scaler, allow_pickle=True).item()
        print(f"Pose scaler loaded from {pose_scaler}")

    # Collect embeddings from both domains
    x1, z1, p1 = encode_all(model, d1_npz, dev, batch_size, d1_npz_b, scaler)
    print(f"D1 (XPlane):  {len(z1)} samples  z-dim={z1.shape[1]}  x-dim={x1.shape[1]}")

    if d2_npz:
        x2, z2, p2 = encode_all(model, d2_npz, dev, batch_size, d2_npz_b, scaler)
        print(f"D2 (MSFS):    {len(z2)} samples")
        x_all = np.concatenate([x1, x2], axis=0)
        z_all = np.concatenate([z1, z2], axis=0)
        p_all = np.concatenate([p1, p2], axis=0)
    else:
        x_all, z_all, p_all = x1, z1, p1

    # Train/test split
    n = len(z_all)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_test = max(1, int(test_frac * n))
    test_idx  = perm[:n_test]
    train_idx = perm[n_test:]

    z_tr, z_te = z_all[train_idx], z_all[test_idx]
    x_tr, x_te = x_all[train_idx], x_all[test_idx]
    p_tr, p_te = p_all[train_idx], p_all[test_idx]

    # Header
    col_w = 26
    print(f"\n{'Pose variable':<{col_w}}  {'R^2(z->pose)':>12}  {'R^2(x->pose)':>12}  {'Δ':>8}")
    print("-" * (col_w + 38))

    for i, col in enumerate(POSE_COLS):
        y_tr = p_tr[:, i]
        y_te = p_te[:, i]

        r2_z = probe(z_tr, z_te, y_tr, y_te, alpha=alpha)
        r2_x = probe(x_tr, x_te, y_tr, y_te, alpha=alpha)
        delta = r2_z - r2_x

        flag = ""
        if r2_z > 0.7:
            flag = "  strong"
        elif r2_z > 0.4:
            flag = "  moderate"
        elif r2_z < 0.1 and r2_x > 0.3:
            flag = "  off-manifold"

        print(f"{col:<{col_w}}  {r2_z:>12.4f}  {r2_x:>12.4f}  {delta:>+8.4f}{flag}")

    print()


if __name__ == "__main__":
    main()
