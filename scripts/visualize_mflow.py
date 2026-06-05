"""
M-Flow manifold visualization.

Three analyses produced as paper-quality figures:

  (A) Off-manifold norm ||e|| by domain
      Histogram of noise-head output magnitude for XPlane vs MSFS.
      XPlane (training domain) should cluster near zero; MSFS should be larger,
      quantifying how far each MSFS embedding is from the learned manifold surface.

  (B) Round-trip consistency
      For each embedding x, compute: encode -> z -> decode -> x_hat -> nearest-neighbor -> x*
      Measure ||x_hat - x*|| in ambient space, stratified by domain.
      Lower = manifold samples land close to real data points.

  (C) UMAP of z (manifold coordinates) colored by domain
      Comparison to raw I-JEPA embedding UMAP. If FD on z < FD on x,
      the flow is compressing out style variance while preserving geometry.

Usage:
  python scripts/visualize_mflow.py \\
      --checkpoint outputs/checkpoints/mflow/best_mflow.pt \\
      --d1-npz     outputs/embeddings/lard_20k/d1_crop.npz \\
      --d2-npz     outputs/embeddings/lard_20k/d2_crop.npz \\
      --output-dir outputs/mflow_viz
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score

from shared_manifold_domain_transfer.evaluation.metrics import (
    frechet_distance_between_embedding_sets,
    maximum_mean_discrepancy_rbf_kernel,
)
from shared_manifold_domain_transfer.models.mflow import RunwayMFlow
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_checkpoint(ckpt_path: str, device: torch.device) -> RunwayMFlow:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]["model"]
    model = RunwayMFlow(
        ambient_dim=cfg["ambient_dim"],
        manifold_dim=cfg["manifold_dim"],
        n_coupling_layers=cfg["n_coupling_layers"],
        hidden_dim=cfg["hidden_dim"],
        pose_dim=cfg["pose_dim"],
        pose_hidden_dim=cfg["pose_hidden_dim"],
        cond_flow=cfg.get("cond_flow", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model


def encode_npz(
    model: RunwayMFlow,
    npz_path: str,
    device: torch.device,
    batch_size: int = 512,
    npz_path_b: str | None = None,
    pose_scaler: dict | None = None,
) -> dict:
    """
    Encode all embeddings in an .npz cache through the M-Flow encoder.

    Returns dict with keys:
        x        (N, D)     original embeddings (after any concat)
        z        (N, d)     on-manifold coordinates
        e_norm   (N,)       ||e|| (off-manifold noise magnitude)
        log_prob (N,)       manifold log-probability
        x_hat    (N, D)     decoded reconstruction
        poses    (N,   6)
        domains  (N,)
        img_paths (N,)
    """
    raw = np.load(npz_path, allow_pickle=True)
    embeddings = raw["embeddings"].astype(np.float32)  # (N, D)
    if npz_path_b is not None:
        raw_b = np.load(npz_path_b, allow_pickle=True)
        embeddings = np.concatenate([embeddings, raw_b["embeddings"].astype(np.float32)], axis=1)
    poses      = raw["poses"].astype(np.float32)
    domains    = raw["domains"]
    img_paths  = raw["img_paths"]

    if pose_scaler is not None:
        pose_min = pose_scaler["pose_min"]
        pose_max = pose_scaler["pose_max"]
        denom    = pose_max - pose_min
        denom[denom == 0] = 1.0
        norm_poses = (2.0 * (poses - pose_min) / denom - 1.0).astype(np.float32)
        embeddings = np.concatenate([embeddings, norm_poses], axis=1)
    poses      = raw["poses"].astype(np.float32)
    domains    = raw["domains"]
    img_paths  = raw["img_paths"]

    all_z, all_e_norm, all_log_prob, all_x_hat = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            x_batch    = torch.tensor(embeddings[i:i+batch_size]).to(device)
            pose_batch = torch.tensor(poses[i:i+batch_size]).to(device)

            z, e, log_prob = model.encode(x_batch, pose_batch)
            x_hat = model.decode(z)

            all_z.append(z.cpu().numpy())
            all_e_norm.append(e.norm(dim=-1).cpu().numpy())
            all_log_prob.append(log_prob.cpu().numpy())
            all_x_hat.append(x_hat.cpu().numpy())

    return {
        "x":         embeddings,
        "z":         np.concatenate(all_z,        axis=0),
        "e_norm":    np.concatenate(all_e_norm,   axis=0),
        "log_prob":  np.concatenate(all_log_prob, axis=0),
        "x_hat":     np.concatenate(all_x_hat,    axis=0),
        "poses":     poses,
        "domains":   domains,
        "img_paths": img_paths,
    }


def plot_offmanifold_norm(d1: dict, d2: dict, save_path: Path) -> None:
    """Panel A: ||e|| histogram by domain."""
    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(0, max(d1["e_norm"].max(), d2["e_norm"].max()), 60)
    ax.hist(d1["e_norm"], bins=bins, alpha=0.6, label="XPlane (train domain)", color="#4878CF", density=True)
    ax.hist(d2["e_norm"], bins=bins, alpha=0.7, label="MSFS (target domain)",  color="#D65F5F", density=True)

    ax.axvline(np.median(d1["e_norm"]), color="#4878CF", linestyle="--", linewidth=1.2,
               label=f"XPlane median = {np.median(d1['e_norm']):.3f}")
    ax.axvline(np.median(d2["e_norm"]), color="#D65F5F", linestyle="--", linewidth=1.2,
               label=f"MSFS median = {np.median(d2['e_norm']):.3f}")

    ax.set_xlabel("Off-manifold norm ||e||")
    ax.set_ylabel("Density")
    ax.set_title("Off-manifold distance by domain\n(lower = closer to XPlane manifold surface)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {save_path}")


def plot_roundtrip(d1: dict, d2: dict, save_path: Path) -> None:
    """
    Panel B: round-trip consistency.
    For each x_hat = decode(encode(x)), find nearest real embedding in the combined
    pool (XPlane + MSFS), report ||x_hat - x*|| by source domain.
    """
    # Build KD-tree over all real embeddings
    all_x    = np.concatenate([d1["x"], d2["x"]], axis=0)
    all_x_hat = np.concatenate([d1["x_hat"], d2["x_hat"]], axis=0)

    log.info("Building KD-tree for nearest-neighbour lookup...")
    tree = cKDTree(all_x)

    dists, _ = tree.query(all_x_hat, k=1, workers=-1)

    n1 = len(d1["x"])
    rt_d1 = dists[:n1]
    rt_d2 = dists[n1:]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, np.percentile(dists, 99), 60)
    ax.hist(rt_d1, bins=bins, alpha=0.6, label=f"XPlane  median={np.median(rt_d1):.2f}", color="#4878CF", density=True)
    ax.hist(rt_d2, bins=bins, alpha=0.7, label=f"MSFS    median={np.median(rt_d2):.2f}", color="#D65F5F", density=True)
    ax.set_xlabel("||decode(encode(x)) − nearest real embedding||")
    ax.set_ylabel("Density")
    ax.set_title("Round-trip consistency\n(lower = manifold samples land near real data)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {save_path}")


def plot_z_umap(d1: dict, d2: dict, save_path: Path) -> dict:
    """Panel C: UMAP of z with domain coloring + gap metrics."""
    z_all     = np.concatenate([d1["z"], d2["z"]], axis=0)
    dom_all   = np.array([0] * len(d1["z"]) + [1] * len(d2["z"]))

    n_sub = min(5000, len(z_all))
    idx   = np.random.choice(len(z_all), n_sub, replace=False)
    z_sub = z_all[idx]
    d_sub = dom_all[idx]

    log.info(f"Running UMAP on {n_sub} manifold vectors...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    umap_z  = reducer.fit_transform(z_sub)

    fd  = frechet_distance_between_embedding_sets(d1["z"], d2["z"])
    mmd = maximum_mean_discrepancy_rbf_kernel(d1["z"], d2["z"])
    sil = silhouette_score(z_sub, d_sub) if len(np.unique(d_sub)) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(umap_z[d_sub == 0, 0], umap_z[d_sub == 0, 1], c="#4878CF", s=4, alpha=0.5, label="XPlane")
    ax.scatter(umap_z[d_sub == 1, 0], umap_z[d_sub == 1, 1], c="#D65F5F", s=16, alpha=0.8, label="MSFS")
    ax.set_title(f"Manifold coordinates z (64-d) — domain separation\nFD={fd:.2f}  MMD={mmd:.5f}  Silhouette={sil:.4f}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {save_path}")
    log.info(f"  z-space FD={fd:.4f}  MMD={mmd:.6f}  Silhouette={sil:.4f}")
    return {"fd": fd, "mmd": mmd, "silhouette": sil, "umap_z": umap_z, "idx": idx, "z_all": z_all, "dom_all": dom_all}


# Pose vector column names matching PoseProcessor output order
_POSE_COLS = [
    ("along_track_distance", "Along-track distance (m, negative = behind LTP)"),
    ("lateral_path_angle",   "Lateral path angle (deg, 0 = centreline)"),
    ("vertical_path_angle",  "Vertical path angle (deg, ~3deg glide)"),
    ("roll",                 "Roll (deg)"),
    ("pitch",                "Pitch (deg, aviation convention)"),
    ("yaw",                  "Yaw (deg)"),
]


def plot_z_umap_pose(d1: dict, d2: dict, umap_result: dict, save_dir: Path) -> None:
    """
    One figure per pose dimension: UMAP of z colored by pose variable value.
    Shows all sampled points (XPlane + MSFS) with a shared colormap.
    XPlane points are drawn first (smaller, semi-transparent); MSFS on top
    (larger, outlined) so the sparse MSFS coverage is still visible.
    If the flow encoded geometric structure, along_track and vertical_path_angle
    should show smooth gradients along the crescent arc.
    """
    umap_z  = umap_result["umap_z"]
    idx     = umap_result["idx"]
    dom_all = umap_result["dom_all"]

    # Recover per-point poses for the subsampled set
    poses_all = np.concatenate([d1["poses"], d2["poses"]], axis=0)  # (N+M, 6)
    poses_sub = poses_all[idx]   # (n_sub, 6)
    dom_sub   = dom_all[idx]

    mask_d1 = dom_sub == 0
    mask_d2 = dom_sub == 1

    for col_idx, (col_name, col_label) in enumerate(_POSE_COLS):
        vals = poses_sub[:, col_idx]
        p1, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
        vals_clip = np.clip(vals, p1, p99)

        fig, ax = plt.subplots(figsize=(8, 6))
        # XPlane: small filled dots
        sc = ax.scatter(
            umap_z[mask_d1, 0], umap_z[mask_d1, 1],
            c=vals_clip[mask_d1], cmap="plasma", s=4, alpha=0.5,
            vmin=p1, vmax=p99,
        )
        # MSFS: larger markers with edge so sparse points are visible
        ax.scatter(
            umap_z[mask_d2, 0], umap_z[mask_d2, 1],
            c=vals_clip[mask_d2], cmap="plasma", s=25, alpha=0.9,
            edgecolors="white", linewidths=0.4,
            vmin=p1, vmax=p99,
        )
        plt.colorbar(sc, ax=ax, label=col_label)
        n_d1, n_d2 = mask_d1.sum(), mask_d2.sum()
        ax.set_title(
            f"Manifold z \u2014 colored by {col_name}\n"
            f"XPlane n={n_d1} (small) · MSFS n={n_d2} (large outlined)"
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        out_path = save_dir / f"z_umap_pose_{col_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Saved: {out_path}")


def _build_airport_lookup(data_dir: str = "data/lard_20k") -> dict[str, str]:
    """Build {image_path -> airport} from all parquet files under data_dir."""
    lookup: dict[str, str] = {}
    for parquet_path in Path(data_dir).rglob("*.parquet"):
        df = pd.read_parquet(parquet_path, columns=["image_path", "airport"])
        lookup.update(dict(zip(df["image_path"], df["airport"])))
    return lookup


def plot_z_umap_airport(
    d1: dict,
    d2: dict,
    umap_result: dict,
    save_dir: Path,
    data_dir: str = "data/lard_20k",
) -> None:
    """
    Two figures: one for XPlane (d1) and one for MSFS (d2), each colored by
    ICAO airport code.  Airports are assigned categorical colors so you can see
    whether the flow clusters / mixes airports or preserves their geometry.
    """
    lookup = _build_airport_lookup(data_dir)

    umap_z  = umap_result["umap_z"]
    idx     = umap_result["idx"]
    dom_all = umap_result["dom_all"]

    # Build full per-point airport arrays aligned with z_all
    paths_all = np.concatenate([d1["img_paths"], d2["img_paths"]], axis=0)
    airports_all = np.array([lookup.get(p, "UNKNOWN") for p in paths_all])
    airports_sub = airports_all[idx]
    dom_sub      = dom_all[idx]

    for domain_id, domain_label in ((0, "XPlane"), (1, "MSFS")):
        mask = dom_sub == domain_id
        if mask.sum() == 0:
            continue

        xy       = umap_z[mask]
        airports = airports_sub[mask]
        unique   = sorted(set(airports))

        # Build color palette — use tab20 for up to 20, then cycle
        cmap = plt.get_cmap("tab20") if len(unique) <= 20 else plt.get_cmap("hsv")
        color_map = {ap: cmap(i / max(len(unique) - 1, 1)) for i, ap in enumerate(unique)}
        colors = np.array([color_map[ap] for ap in airports])

        fig, ax = plt.subplots(figsize=(9, 7))
        for ap in unique:
            m = airports == ap
            ax.scatter(xy[m, 0], xy[m, 1], c=[color_map[ap]], s=5, alpha=0.6, label=ap)

        ax.set_title(f"Manifold z — colored by airport ({domain_label}, n={mask.sum()})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        # Legend: compact, outside axes if many airports
        ncol = max(1, len(unique) // 10)
        legend_width = min(4.0, 0.9 * ncol + 0.5)
        fig_width = 9 + legend_width
        fig.set_size_inches(fig_width, 7)
        ax.legend(
            markerscale=3, fontsize=7, ncol=ncol,
            loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        )
        fig.tight_layout()
        out_path = save_dir / f"z_umap_airport_{domain_label.lower()}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved: {out_path}")


@click.command()
@click.option("--checkpoint",  default="outputs/checkpoints/mflow/best_mflow_crop_embeddings.pt", show_default=True)
@click.option("--d1-npz",      default="outputs/embeddings/lard_20k/d1_crop.npz", show_default=True)
@click.option("--d2-npz",      default="outputs/embeddings/lard_20k/d2_crop.npz", show_default=True)
@click.option("--d1-npz-b",    default=None, help="Second npz to concat onto d1 embeddings (concat mode)", show_default=True)
@click.option("--d2-npz-b",    default=None, help="Second npz to concat onto d2 embeddings (concat mode)", show_default=True)
@click.option("--pose-scaler",  default=None, help="Path to pose_scaler.npy (pose-concat model)", show_default=True)
@click.option("--data-dir",    default="data/lard_20k", help="Root dir for parquet metadata (airport lookup)", show_default=True)
@click.option("--output-dir",  default="outputs/mflow_viz",                        show_default=True)
@click.option("--device",      default="cuda" if torch.cuda.is_available() else "cpu", show_default=True)
@click.option("--batch-size",  default=512, show_default=True)
def main(
    checkpoint: str,
    d1_npz: str,
    d2_npz: str,
    d1_npz_b: str | None,
    d2_npz_b: str | None,
    pose_scaler: str | None,
    data_dir: str,
    output_dir: str,
    device: str,
    batch_size: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    model = load_checkpoint(checkpoint, dev)

    scaler = None
    if pose_scaler is not None:
        scaler = np.load(pose_scaler, allow_pickle=True).item()
        log.info(f"Loaded pose scaler from {pose_scaler}")

    log.info("Encoding Domain 1 (XPlane)...")
    d1 = encode_npz(model, d1_npz, dev, batch_size, npz_path_b=d1_npz_b, pose_scaler=scaler)
    log.info(f"  {len(d1['x'])} samples — median ||e||={np.median(d1['e_norm']):.4f}")

    log.info("Encoding Domain 2 (MSFS nominal)...")
    d2 = encode_npz(model, d2_npz, dev, batch_size, npz_path_b=d2_npz_b, pose_scaler=scaler)
    log.info(f"  {len(d2['x'])} samples — median ||e||={np.median(d2['e_norm']):.4f}")

    plot_offmanifold_norm(d1, d2, out / "offmanifold_norm.png")
    plot_roundtrip(d1, d2,       out / "roundtrip_consistency.png")
    umap_result = plot_z_umap(d1, d2, out / "z_umap_domain.png")

    log.info("Generating pose-colored UMAP panels...")
    plot_z_umap_pose(d1, d2, umap_result, out)

    log.info("Generating airport-colored UMAP panels...")
    plot_z_umap_airport(d1, d2, umap_result, out, data_dir=data_dir)

    metrics = {k: umap_result[k] for k in ("fd", "mmd", "silhouette")}
    print("\n--- M-Flow Gap Metrics (manifold z-space) ---")
    print(f"Fréchet Distance on z:  {metrics['fd']:.4f}   (baseline on x: 101.81)")
    print(f"MMD on z:               {metrics['mmd']:.6f}  (baseline on x:   0.008822)")
    print(f"Silhouette on z:        {metrics['silhouette']:.4f}")
    print(f"\nFigures written to {out.resolve()}/")
    for f in sorted(out.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
