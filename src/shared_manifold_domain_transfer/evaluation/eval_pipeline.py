"""
Full evaluation pipeline for all three baselines.

Models:
  A — Diffusion only (pose MLP conditioning, no manifold)
  B — ResNet50 + M-Flow + Diffusion
  C — I-JEPA + M-Flow + Diffusion (ours)

For each model, for each holdout pose:
  1. Generate full image at that pose
  2. Load ground truth image from holdout set
  3. Crop both to runway region
  4. Compute pixel metrics (MSE, SSIM) + semantic fidelity (I-JEPA cosine)
  5. Compute pose distance from nearest Domain 2 training pose

Outputs:
  outputs/results_table.csv
  outputs/summary_table.csv
  outputs/distance_vs_error.png
  outputs/umap_manifold_vectors.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import click
from tqdm import tqdm

from shared_manifold_domain_transfer.evaluation.metrics import (
    pixel_metrics_mse_and_ssim, semantic_fidelity_ijepa_cosine, pose_distance_to_nearest_training_sample
)

from shared_manifold_domain_transfer.models.diffusion import ManifoldDiffusionHead
from shared_manifold_domain_transfer.evaluation.umap_viz import (
    plot_distance_vs_error, plot_umap_with_holdout
)

from shared_manifold_domain_transfer.data_proc.dataset import make_loaders
from shared_manifold_domain_transfer.models.ijepa import IJEPAEncoder
from torch.utils.data import ConcatDataset, DataLoader
from shared_manifold_domain_transfer.models.resnet_encoder import ResNetEncoder
from shared_manifold_domain_transfer.models.mflow import RunwayMFlow


log = logging.getLogger(__name__)
OUTPUTS_DIR = Path("outputs")



# Model loader helpers

def load_model_a(cfg_path: str, ckpt_path: str, device: torch.device):
    """Baseline A: Diffusion only (pose MLP conditioning)."""
    model = ManifoldDiffusionHead(conditioning="pose_mlp").to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, None, None


def load_model_b(encoder_cfg, mflow_ckpt: str, diff_ckpt: str, device: torch.device):
    """Baseline B: ResNet50 + M-Flow + Diffusion."""
    encoder = ResNetEncoder().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    mflow = RunwayMFlow().to(device)
    mflow.load_state_dict(torch.load(mflow_ckpt, map_location=device)["model_state_dict"])
    for p in mflow.parameters():
        p.requires_grad_(False)
    mflow.eval()

    diff = ManifoldDiffusionHead(conditioning="manifold").to(device)
    diff.load_state_dict(torch.load(diff_ckpt, map_location=device)["model_state_dict"])
    diff.eval()

    return diff, encoder, mflow


def load_model_c(ijepa_weights: Optional[str], mflow_ckpt: str, diff_ckpt: str, device: torch.device):
    """Model C (ours): I-JEPA + M-Flow + Diffusion."""
    encoder = IJEPAEncoder(weights_path=ijepa_weights, device=str(device))

    mflow = RunwayMFlow().to(device)
    mflow.load_state_dict(torch.load(mflow_ckpt, map_location=device)["model_state_dict"])
    for p in mflow.parameters():
        p.requires_grad_(False)
    mflow.eval()

    diff = ManifoldDiffusionHead(conditioning="manifold").to(device)
    diff.load_state_dict(torch.load(diff_ckpt, map_location=device)["model_state_dict"])
    diff.eval()

    return diff, encoder, mflow



# Per-model evaluation
@torch.no_grad()
def evaluate_model(
    model_name: str,
    diff_model,
    encoder,
    mflow,
    holdout_loader,
    train_poses: torch.Tensor,
    train_vectors: Optional[torch.Tensor],
    jepa_encoder,
    device: torch.device,
    knn_k: int = 10,
    knn_sigma: float = 1.0,
    image_size: int = 224,
) -> List[Dict]:
    """
    Evaluate one model on the full holdout set.

    Returns list of per-sample result dicts.
    """

    results = []
    diff_model.eval()

    for batch in tqdm(holdout_loader, desc=f"Evaluating {model_name}"):
        gt_images  = batch["image"].to(device)           # (B, 3, 224, 224)
        gt_crops   = batch["runway_crop"].to(device)     # (B, 3, 96, 96)
        poses      = batch["pose_vector"].to(device)     # (B, 6)
        corners    = batch["corners"].to(device)         # (B, 4, 2)

        B = gt_images.shape[0]

        # Determine conditioning vector
        if diff_model.conditioning == "pose_mlp":
            # Baseline A: condition on raw pose
            manifold_z = None
        else:
            # Models B & C: get manifold vector via kernel regression
            manifold_z = ManifoldDiffusionHead.get_manifold_vector_for_pose(
                target_pose=poses,
                train_poses=train_poses.to(device),
                train_vectors=train_vectors.to(device),
                k=knn_k,
                sigma=knn_sigma,
            )

        # Generate images
        gen_images = diff_model.generate(
            manifold_z=manifold_z,
            pose=poses if diff_model.conditioning == "pose_mlp" else None,
            image_size=image_size,
        )
        # gen_images in [-1,1] → clip and convert for metrics
        gen_images_01 = ((gen_images + 1.0) / 2.0).clamp(0, 1)

        # Crop generated images to runway region
        gen_crops = _batch_crop_runway(gen_images_01, corners, crop_size=96)
        gt_crops_01 = ((gt_images + 1.0) / 2.0).clamp(0, 1) if gt_images.min() < -0.1 else gt_images
        gt_crops_01_crop = _batch_crop_runway(gt_crops_01, corners, crop_size=96)

        # Metrics per sample
        pose_dists = pose_distance_to_nearest_training_sample(poses.cpu(), train_poses.cpu())  # (B,)

        for i in range(B):
            pix = pixel_metrics_mse_and_ssim(gen_crops[i], gt_crops_01_crop[i])
            sem = semantic_fidelity_ijepa_cosine(
                jepa_encoder,
                gen_crops[i].unsqueeze(0),
                gt_crops_01_crop[i].unsqueeze(0),
            )
            results.append({
                "model":         model_name,
                "pixel_mse":     pix["mse"],
                "ssim":          pix["ssim"],
                "jepa_cosine":   sem,
                "pose_distance": pose_dists[i].item(),
            })

    return results


def _batch_crop_runway(
    images: torch.Tensor,
    corners_norm: torch.Tensor,
    crop_size: int = 96,
) -> torch.Tensor:
    """
    Crop runway region from each image using normalised corner coordinates.
    Returns (B, 3, crop_size, crop_size).
    """
    B, C, H, W = images.shape
    crops = []
    for i in range(B):
        cx = corners_norm[i, :, 0] * W
        cy = corners_norm[i, :, 1] * H
        x1 = int(cx.min().item())
        x2 = int(cx.max().item())
        y1 = int(cy.min().item())
        y2 = int(cy.max().item())
        x1, x2 = max(0, x1), min(W, max(x2, x1 + 1))
        y1, y2 = max(0, y1), min(H, max(y2, y1 + 1))
        crop = images[i:i+1, :, y1:y2, x1:x2]
        crop = F.interpolate(crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        crops.append(crop.squeeze(0))
    return torch.stack(crops, dim=0)



# Build training pose/vector index for kernel regression
def build_train_index(
    encoder,
    mflow,
    combined_loader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build the (pose → manifold_vector) index over training data."""
    all_poses, all_z = [], []
    with torch.no_grad():
        for batch in tqdm(combined_loader, desc="Building train index"):
            imgs = batch["image"].to(device)
            pose = batch["pose_vector"].to(device)
            emb  = encoder(imgs)
            z, _, _, _ = mflow(emb, pose)
            all_poses.append(pose.cpu())
            all_z.append(z.cpu())
    return {
        "poses":   torch.cat(all_poses, dim=0),
        "vectors": torch.cat(all_z,     dim=0),
    }



# Main eval runner
def run_evaluation(
    checkpoints: Dict[str, str],
    data_dir: str,
    ijepa_weights: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    knn_k: int = 10,
    knn_sigma: float = 1.0,
    device_str: str = "cuda",
) -> None:
    """
    Full evaluation of all three models.

    Args:
        checkpoints: {
            'A_mflow': None,
            'A_diff':  'outputs/checkpoints/diffusion/best_diffusion_pose_mlp.pt',
            'B_mflow': 'outputs/checkpoints/mflow/best_mflow.pt',
            'B_diff':  'outputs/checkpoints/diffusion/best_diffusion_manifold.pt',
            'C_mflow': 'outputs/checkpoints/mflow/best_mflow.pt',
            'C_diff':  'outputs/checkpoints/diffusion/best_diffusion_manifold.pt',
        }
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    log.info(f"Evaluating on {device}")

    # Load dataloaders
    loaders = make_loaders(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)

    # I-JEPA encoder for semantic fidelity metric
    jepa_encoder = IJEPAEncoder(weights_path=ijepa_weights, device=str(device))

    # Combined train loader for index building
    combined_ds = ConcatDataset([
        loaders["_domain1_dataset"], loaders["_domain2_dataset"]
    ])
    combined_loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=False)

    all_results: Dict[str, List[Dict]] = {}

    # Model A
    if checkpoints.get("A_diff"):
        log.info("=== Evaluating Model A (Diffusion only) ===")
        model_a, _, _ = load_model_a(None, checkpoints["A_diff"], device)
        # No manifold vectors needed for A
        train_poses = torch.zeros(1, 6)  # placeholder
        train_vectors = None
        results_a = evaluate_model(
            "A_diffusion_only", model_a, None, None,
            loaders["holdout_eval"], train_poses, train_vectors,
            jepa_encoder, device, knn_k, knn_sigma,
        )
        all_results["A_diffusion_only"] = results_a

    # Model C (and B share same M-flow for now) 
    if checkpoints.get("C_diff") and checkpoints.get("C_mflow"):
        log.info("=== Evaluating Model C (I-JEPA + M-Flow) ===")
        model_c, enc_c, mflow_c = load_model_c(
            ijepa_weights, checkpoints["C_mflow"], checkpoints["C_diff"], device
        )
        train_idx = build_train_index(enc_c, mflow_c, combined_loader, device)
        results_c = evaluate_model(
            "C_ijepa_mflow", model_c, enc_c, mflow_c,
            loaders["holdout_eval"], train_idx["poses"], train_idx["vectors"],
            jepa_encoder, device, knn_k, knn_sigma,
        )
        all_results["C_ijepa_mflow"] = results_c

        # UMAP of manifold vectors
        all_domain_vecs  = train_idx["vectors"].numpy()
        all_domain_doms  = np.concatenate([
            np.zeros(len(loaders["_domain1_dataset"])),
            np.ones(len(loaders["_domain2_dataset"])),
        ])
        # Holdout manifold vectors
        holdout_z_list = []
        for batch in loaders["holdout_eval"]:
            imgs = batch["image"].to(device)
            pose = batch["pose_vector"].to(device)
            with torch.no_grad():
                emb = enc_c(imgs)
                z, _, _, _ = mflow_c(emb, pose)
            holdout_z_list.append(z.cpu())
        holdout_vecs = torch.cat(holdout_z_list, dim=0).numpy()
        holdout_dists = pose_distance_to_nearest_training_sample(
            loaders["_holdout_dataset"]._raw_poses[loaders["_holdout_dataset"]._indices],
            loaders["_domain2_sampler"].poses,
        ).numpy() if hasattr(loaders.get("_holdout_dataset", None), "_raw_poses") else None

        plot_umap_with_holdout(
            all_domain_vecs, all_domain_doms,
            holdout_vecs, holdout_dists,
        )

    if checkpoints.get("B_diff") and checkpoints.get("B_mflow"):
        log.info("=== Evaluating Model B (ResNet50 + M-Flow) ===")
        model_b, enc_b, mflow_b = load_model_b(
            None, checkpoints["B_mflow"], checkpoints["B_diff"], device
        )
        train_idx_b = build_train_index(enc_b, mflow_b, combined_loader, device)
        results_b = evaluate_model(
            "B_resnet_mflow", model_b, enc_b, mflow_b,
            loaders["holdout_eval"], train_idx_b["poses"], train_idx_b["vectors"],
            jepa_encoder, device, knn_k, knn_sigma,
        )
        all_results["B_resnet_mflow"] = results_b

    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # results_table.csv
    all_rows = []
    for model_name, rows in all_results.items():
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTPUTS_DIR / "results_table.csv", index=False)
        log.info(f"Saved results_table.csv ({len(df)} rows)")

        # summary_table.csv
        summary = df.groupby("model").agg(
            pixel_mse_mean=("pixel_mse",   "mean"),
            pixel_mse_std= ("pixel_mse",   "std"),
            ssim_mean=     ("ssim",         "mean"),
            ssim_std=      ("ssim",         "std"),
            jepa_mean=     ("jepa_cosine",  "mean"),
            jepa_std=      ("jepa_cosine",  "std"),
        ).reset_index()
        summary.to_csv(OUTPUTS_DIR / "summary_table.csv", index=False)
        log.info("Saved summary_table.csv")
        print("\n=== Summary Table ===")
        print(summary.to_string(index=False))

        # distance_vs_error.png
        plot_data = {}
        for model_name, rows in all_results.items():
            row_arr = pd.DataFrame(rows)
            plot_data[model_name] = {
                "pose_distances": row_arr["pose_distance"].values,
                "jepa_cosine":    row_arr["jepa_cosine"].values,
            }
        plot_distance_vs_error(plot_data)



# Entry point
# NOTE: this pipeline assumes trained checkpoints for models A, B, and C exist.
# It will need to be revisited once training is complete - the model interfaces,
# batch keys and kernel regression helpers may have changed.
@click.command()
@click.option("--data-dir",      default="data/lard", show_default=True)
@click.option("--ijepa-weights", default=None,        help="Path to I-JEPA checkpoint.")
@click.option("--a-diff",        default=None,        help="Model A diffusion checkpoint.")
@click.option("--b-mflow",       default=None,        help="Model B M-Flow checkpoint.")
@click.option("--b-diff",        default=None,        help="Model B diffusion checkpoint.")
@click.option("--c-mflow",       default=None,        help="Model C M-Flow checkpoint.")
@click.option("--c-diff",        default=None,        help="Model C diffusion checkpoint.")
@click.option("--batch-size",    default=16, show_default=True)
@click.option("--device",        default="cuda" if torch.cuda.is_available() else "cpu",
              show_default=True)
def main(
    data_dir: str,
    ijepa_weights: str,
    a_diff: str,
    b_mflow: str,
    b_diff: str,
    c_mflow: str,
    c_diff: str,
    batch_size: int,
    device: str,
) -> None:
    """Run full evaluation pipeline for all trained models."""
    logging.basicConfig(level=logging.INFO)
    run_evaluation(
        checkpoints={
            "A_diff":  a_diff,
            "B_mflow": b_mflow,
            "B_diff":  b_diff,
            "C_mflow": c_mflow,
            "C_diff":  c_diff,
        },
        data_dir=data_dir,
        ijepa_weights=ijepa_weights,
        batch_size=batch_size,
        device_str=device,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
