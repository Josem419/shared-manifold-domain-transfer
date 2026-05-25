"""
UMAP and visualisation utilities.

Functions:
  extract_embeddings()      — run encoder over loader, return dict
  plot_domain_separation()  — UMAP 2D scatter coloured by domain
  plot_pose_coverage()      — PCA-3D of pose volumes
  plot_manifold_vectors()   — UMAP of manifold vectors during training
  plot_distance_vs_error()  — pose distance vs semantic fidelity
  plot_umap_with_holdout()  — UMAP with training + holdout points
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import umap as umap_lib
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from shared_manifold_domain_transfer.evaluation.metrics import (
    frechet_distance, maximum_mean_discrepancy
)

log = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
DOMAIN_COLORS = {0: "#2196F3", 1: "#FF5722"}
DOMAIN_LABELS = {0: "Domain 1 (XPlane)", 1: "Domain 2 (MSFS)"}


def extract_embeddings(
    encoder,
    loader,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Run all images through a frozen encoder and collect results.

    Returns:
        {
            'embeddings': (N, D),
            'domains':    (N,),
            'poses':      (N, 6),
            'img_paths':  list of str
        }
    """
    all_emb, all_dom, all_pose, all_path = [], [], [], []
    dev = torch.device(device)

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            imgs = batch["image"].to(dev)
            embs = encoder(imgs).cpu().numpy()
            all_emb.append(embs)
            all_dom.append(batch["domain"].numpy())
            all_pose.append(batch["pose_vector"].numpy())
            all_path.extend(batch["img_path"])

    return {
        "embeddings": np.concatenate(all_emb,  axis=0),
        "domains":    np.concatenate(all_dom,  axis=0),
        "poses":      np.concatenate(all_pose, axis=0),
        "img_paths":  np.array(all_path),
    }



# UMAP domain separation
def plot_domain_separation(
    embeddings_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    n_samples: int = 5000,
) -> str:
    """UMAP 2D scatter coloured by domain. Prints Fréchet distance and MMD."""
    embs    = embeddings_dict["embeddings"]
    domains = embeddings_dict["domains"]

    # Subsample for speed
    if len(embs) > n_samples:
        idx  = np.random.choice(len(embs), n_samples, replace=False)
        embs    = embs[idx]
        domains = domains[idx]

    log.info(f"Running UMAP on {len(embs)} embeddings...")
    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
    coords  = reducer.fit_transform(embs)   # (N, 2)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    for dom_id, label in DOMAIN_LABELS.items():
        mask = domains == dom_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=DOMAIN_COLORS[dom_id], label=label,
            s=8, alpha=0.6, linewidths=0,
        )
    ax.set_title("I-JEPA Embedding Space — Domain Separation (UMAP)", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)
    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "umap_domain_separation.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {save_path}")

    d1_emb = embs[domains == 0]
    d2_emb = embs[domains == 1]
    if len(d1_emb) > 10 and len(d2_emb) > 10:
        # Subsample for FD (expensive)
        n_fd = min(500, len(d1_emb), len(d2_emb))
        fd  = frechet_distance(d1_emb[:n_fd], d2_emb[:n_fd])
        mmd = maximum_mean_discrepancy(d1_emb[:n_fd], d2_emb[:n_fd])
        log.info(f"Fréchet Distance: {fd:.4f}")
        log.info(f"MMD:              {mmd:.6f}")
        print(f"\n--- Domain Gap Metrics ---")
        print(f"Fréchet Distance (FD): {fd:.4f}")
        print(f"Max Mean Discrepancy (MMD): {mmd:.6f}")
        _interpret_separation(coords, domains)

    return save_path


def _interpret_separation(coords: np.ndarray, domains: np.ndarray) -> None:
    """Print a silhouette-based summary of domain separation."""
    if len(np.unique(domains)) < 2:
        return
    score = silhouette_score(coords, domains, sample_size=min(2000, len(domains)))
    print(f"UMAP Silhouette Score: {score:.4f}")
    if score > 0.3:
        print("  -> Clear separation.")
    elif score > 0.1:
        print("  -> Moderate separation.")
    else:
        print("  -> Weak separation.")

# Pose coverage (3D PCA)
def plot_pose_coverage(
    embeddings_dict: Dict[str, np.ndarray],
    holdout_poses: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> str:
    """PCA-3D scatter of pose vectors coloured by domain, with optional holdout."""
    poses   = embeddings_dict["poses"]    # (N, 6)
    domains = embeddings_dict["domains"]  # (N,)

    all_poses = poses
    if holdout_poses is not None:
        all_poses = np.concatenate([poses, holdout_poses], axis=0)

    pca    = PCA(n_components=3)
    coords = pca.fit_transform(all_poses)[:len(poses)]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for dom_id, label in DOMAIN_LABELS.items():
        mask = domains == dom_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=DOMAIN_COLORS[dom_id], label=label, s=6, alpha=0.5,
        )

    if holdout_poses is not None:
        h_coords = pca.transform(holdout_poses)
        ax.scatter(
            h_coords[:, 0], h_coords[:, 1], h_coords[:, 2],
            c="#4CAF50", label="Holdout (MSFS out-of-volume)", s=10, alpha=0.8,
            marker="^",
        )

    ax.set_title("Pose Volume Coverage (PCA-3D)", fontsize=13)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(markerscale=3)
    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "pose_coverage.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {save_path}")
    return save_path


def plot_manifold_vectors(
    manifold_vectors: np.ndarray,
    domains: np.ndarray,
    epoch: int = 0,
    poses: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> str:
    """UMAP of manifold vectors coloured by domain, optionally by pose angle."""
    n = min(len(manifold_vectors), 3000)
    idx  = np.random.choice(len(manifold_vectors), n, replace=False)
    vecs = manifold_vectors[idx]
    doms = domains[idx]

    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
    coords  = reducer.fit_transform(vecs)

    n_plots = 2 if poses is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    for dom_id, label in DOMAIN_LABELS.items():
        mask = doms == dom_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=DOMAIN_COLORS[dom_id], label=label, s=8, alpha=0.6,
        )
    ax.set_title(f"Manifold Vectors — Domain Separation (epoch {epoch})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)

    if poses is not None:
        ax2 = axes[1]
        alt = poses[idx, 2]
        sc = ax2.scatter(coords[:, 0], coords[:, 1], c=alt, cmap="viridis", s=8, alpha=0.6)
        plt.colorbar(sc, ax=ax2, label="Vertical path angle (normalised)")
        ax2.set_title(f"Manifold Vectors — Coloured by Vertical Path Angle (epoch {epoch})")
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")

    plt.tight_layout()
    save_path = save_path or str(OUTPUTS_DIR / f"umap_manifold_vectors_epoch{epoch:03d}.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {save_path}")
    return save_path


def plot_distance_vs_error(
    results_by_model: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> str:
    """Pose distance from Domain 2 training set vs I-JEPA cosine similarity.

    Args:
        results_by_model: mapping from model name to a dict with keys
            ``pose_distances`` (M,) and ``jepa_cosine`` (M,).
    """
    model_colors = {
        "A_diffusion_only":  "#E53935",
        "B_resnet_mflow":    "#FB8C00",
        "C_ijepa_mflow":     "#43A047",
    }
    model_labels = {
        "A_diffusion_only": "A: Diffusion only (pose MLP)",
        "B_resnet_mflow":   "B: ResNet50 + M-Flow",
        "C_ijepa_mflow":    "C: I-JEPA + M-Flow (ours)",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, data in results_by_model.items():
        dists  = np.array(data["pose_distances"])
        cosine = np.array(data["jepa_cosine"])
        color  = model_colors.get(model_name, "gray")
        label  = model_labels.get(model_name, model_name)

        # Scatter (small, transparent)
        ax.scatter(dists, cosine, c=color, s=12, alpha=0.25, linewidths=0)

        # Smoothed trend line via binned mean
        n_bins = 20
        bins   = np.linspace(dists.min(), dists.max(), n_bins + 1)
        bin_centers, bin_means, bin_stds = [], [], []
        for i in range(n_bins):
            mask = (dists >= bins[i]) & (dists < bins[i + 1])
            if mask.sum() >= 3:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(cosine[mask].mean())
                bin_stds.append(cosine[mask].std())

        if bin_centers:
            bc = np.array(bin_centers)
            bm = np.array(bin_means)
            bs = np.array(bin_stds)
            ax.plot(bc, bm, color=color, linewidth=2.5, label=label)
            ax.fill_between(bc, bm - bs, bm + bs, color=color, alpha=0.15)

    ax.set_xlabel("Pose Distance from Nearest Domain 2 Training Pose", fontsize=12)
    ax.set_ylabel("I-JEPA Cosine Similarity (generated vs. ground truth)", fontsize=12)
    ax.set_title(
        "Semantic Fidelity vs. Pose Distance from Domain 2 Training Distribution",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "distance_vs_error.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    log.info(f"Saved: {save_path}")
    return save_path


def plot_umap_with_holdout(
    train_vectors: np.ndarray,
    train_domains: np.ndarray,
    holdout_vectors: np.ndarray,
    holdout_pose_dists: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> str:
    """UMAP of manifold vectors: domain1 (blue), domain2 (orange), holdout (green/gradient)."""
    all_vecs = np.concatenate([train_vectors, holdout_vectors], axis=0)
    n_train  = len(train_vectors)

    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
    all_coords = reducer.fit_transform(all_vecs)

    tr_coords = all_coords[:n_train]
    ho_coords = all_coords[n_train:]

    fig, ax = plt.subplots(figsize=(10, 8))

    for dom_id, label in DOMAIN_LABELS.items():
        mask = train_domains == dom_id
        ax.scatter(
            tr_coords[mask, 0], tr_coords[mask, 1],
            c=DOMAIN_COLORS[dom_id], label=label, s=8, alpha=0.4,
        )

    if holdout_pose_dists is not None:
        sc = ax.scatter(
            ho_coords[:, 0], ho_coords[:, 1],
            c=holdout_pose_dists, cmap="RdYlGn_r",
            s=30, alpha=0.9, marker="^", label="Holdout",
            linewidths=0.5, edgecolors="black",
        )
        plt.colorbar(sc, ax=ax, label="Pose distance from D2 train")
    else:
        ax.scatter(
            ho_coords[:, 0], ho_coords[:, 1],
            c="#4CAF50", s=30, alpha=0.9, marker="^", label="Holdout",
        )

    ax.set_title("Manifold Vectors — Training + Holdout (UMAP)", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)
    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "umap_manifold_vectors.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {save_path}")
    return save_path
