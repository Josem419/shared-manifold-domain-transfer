"""
UMAP baseline 

Extracts I-JEPA embeddings for Domain 1 (XPlane) and Domain 2 (MSFS),
runs UMAP, and prints domain-gap metrics (Fréchet Distance, MMD, silhouette).

The silhouette score interpretation:
  > 0.3  → clear separation; pretrained weights sufficient
  0.1–0.3 → marginal; review plots carefully
  < 0.1  → weak separation; I-JEPA fine-tuning needed (~1 week)

Examples:

  with pretrained I-JEPA weights
  assumes environment is sourced

  ```
  python3 scripts/run_umap_baseline.py \\
      --weights models/IN22K-vit.h.14-900e.pth.tar
  ```

  smoke test with random weights (fast, no checkpoint needed)
  the smoke test is just to make sure plumbing is working; 
  the output is meaningless for decisions on finetuning or M-Flow training.
  ```
  python3 scripts/run_umap_baseline.py --random-weights
  ```

  # GPU
  ```
  python3 scripts/run_umap_baseline.py \\
      --weights models/IN22K-vit.h.14-900e.pth.tar --device cuda
  ```
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch

from shared_manifold_domain_transfer.data_proc.dataset import make_loaders
from shared_manifold_domain_transfer.evaluation.umap_viz import (
    extract_embeddings,
    plot_domain_separation,
    plot_pose_coverage,
)
from shared_manifold_domain_transfer.models.ijepa import IJEPAEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


@click.command()
@click.option("--data-dir",      default="data/lard",    show_default=True,
              help="Root directory produced by download_lard.py.")
@click.option("--output-dir",    default="outputs/umap", show_default=True,
              help="Directory to write UMAP figures into.")
@click.option("--weights",       default=None,
              help="Path to IN22K-vit.h.14-900e.pth.tar checkpoint. "
                   "Omit to download automatically; use --random-weights to skip.")
@click.option("--random-weights", is_flag=True, default=False,
              help="Use randomly initialised encoder (fast smoke-test, no download).")
@click.option("--device",        default="cuda" if torch.cuda.is_available() else "cpu",
              show_default=True, help="Torch device.")
@click.option("--batch-size",    default=32, show_default=True)
@click.option("--num-workers",   default=4,  show_default=True)
@click.option("--max-samples",   default=None, type=int,
              help="Truncate each split to this many samples (for quick tests).")
def main(
    data_dir: str,
    output_dir: str,
    weights: str | None,
    random_weights: bool,
    device: str,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> None:
    """Run UMAP baseline and print domain-gap metrics. THese will tell us
    whther to fintuneu I-JEPA, and whteher M-FLow training is likely to succeed.

    Loads the downloaded LARD V2 data, extracts I-JEPA embeddings for both
    domains, then produces:

    \b
      outputs/umap/umap_domain_separation.png  — UMAP coloured by domain
      outputs/umap/pose_coverage.png           — PCA-3D pose volumes

    Prints Fréchet Distance, MMD, and silhouette score with an interpretation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    
    # Build DataLoaders
    log.info(f"Building DataLoaders from {data_dir}...")
    loaders = make_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )
    d1_loader  = loaders["domain1_train"]
    d2_loader  = loaders["domain2_train"]
    ho_loader  = loaders["holdout_eval"]

    n_d1 = len(loaders["_domain1_dataset"])
    n_d2 = len(loaders["_domain2_dataset"])
    n_ho = len(loaders["_holdout_dataset"])
    log.info(f"  Domain 1 (XPlane): {n_d1} images")
    log.info(f"  Domain 2 (MSFS):   {n_d2} images")
    log.info(f"  Holdout:           {n_ho} images")

    
    # 2. Load I-JEPA encoder
    if random_weights:
        log.warning("Using RANDOM encoder weights — output is meaningless for the decision gate.")
        encoder = IJEPAEncoder(weights_path=None, device=device)
    else:
        weights_path = weights  # may be None → IJEPAEncoder attempts download
        if weights_path is None:
            log.info("No --weights provided; IJEPAEncoder will attempt to download from Facebook CDN.")
        encoder = IJEPAEncoder(weights_path=weights_path, device=device)

    log.info(f"Encoder loaded on {device}.")
    # Extract embeddings for both domains + holdout
    
    log.info("Extracting Domain 1 embeddings...")
    d1_dict = extract_embeddings(encoder, d1_loader, device=device)

    log.info("Extracting Domain 2 embeddings...")
    d2_dict = extract_embeddings(encoder, d2_loader, device=device)

    log.info("Extracting holdout embeddings...")
    ho_dict = extract_embeddings(encoder, ho_loader, device=device)

    # Merge D1 + D2 for the joint UMAP plot (holdout kept separate)
    combined = {
        "embeddings": np.concatenate([d1_dict["embeddings"], d2_dict["embeddings"]], axis=0),
        "domains":    np.concatenate([d1_dict["domains"],    d2_dict["domains"]],    axis=0),
        "poses":      np.concatenate([d1_dict["poses"],      d2_dict["poses"]],      axis=0),
        "img_paths":  np.concatenate([d1_dict["img_paths"],  d2_dict["img_paths"]],  axis=0),
    }

    
    # 4. UMAP plots and metrics
    
    log.info("Running UMAP domain separation plot...")
    sep_path = plot_domain_separation(
        combined,
        save_path=str(output_path / "umap_domain_separation.png"),
    )

    log.info("Running pose coverage PCA plot...")
    cov_path = plot_pose_coverage(
        combined,
        holdout_poses=ho_dict["poses"] if len(ho_dict["poses"]) > 0 else None,
        save_path=str(output_path / "pose_coverage.png"),
    )

    print(f"\nFigures written to {output_path.resolve()}/")
    print(f"  {sep_path}")
    print(f"  {cov_path}")


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
