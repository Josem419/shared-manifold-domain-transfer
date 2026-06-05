"""
Evaluation pipeline: bounding box IoU on the holdout set.

Compares two bbox head checkpoints on the Domain B off-nominal holdout split:
  baseline  — head trained with supervised loss only (D1 + D2 nominal)
  augmented — head trained with supervised + manifold consistency loss

For each checkpoint:
  1. Load the bbox head MLP
  2. Run inference on holdout embeddings
  3. Compute mean IoU, IoU >= 0.25, IoU >= 0.50

Outputs:
  outputs/results_table.csv      — per-sample IoU for each model
  outputs/summary_table.csv      — aggregated metrics table
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import click
from tqdm import tqdm

log = logging.getLogger(__name__)
OUTPUTS_DIR = Path("outputs")


# Bbox head (must match train_bbox_head.py architecture)
class BboxHead(nn.Module):
    def __init__(self, in_dim: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute IoU for two arrays of [xmin, ymin, xmax, ymax] boxes. (N,)"""
    ix1 = np.maximum(pred[:, 0], gt[:, 0])
    iy1 = np.maximum(pred[:, 1], gt[:, 1])
    ix2 = np.minimum(pred[:, 2], gt[:, 2])
    iy2 = np.minimum(pred[:, 3], gt[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_p + area_g - inter
    return np.where(union > 0, inter / union, 0.0)


@torch.no_grad()
def evaluate_bbox_head(
    model_name: str,
    ckpt_path: str,
    holdout_npz: str,
    device: torch.device,
    in_dim: int = 2048,
) -> List[Dict]:
    """
    Load a bbox head checkpoint and evaluate on holdout embeddings.

    holdout_npz must contain:
      embeddings  (N, in_dim)
      bboxes      (N, 4)   normalised [xmin, ymin, xmax, ymax]
    """
    data = np.load(holdout_npz)
    embeddings = torch.from_numpy(data["embeddings"]).float().to(device)
    gt_bboxes  = data["bboxes"].astype(np.float32)           # (N, 4)

    head = BboxHead(in_dim=in_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()

    preds = []
    batch_size = 256
    for i in tqdm(range(0, len(embeddings), batch_size), desc=f"Evaluating {model_name}"):
        batch = embeddings[i : i + batch_size]
        preds.append(head(batch).cpu().numpy())
    pred_bboxes = np.concatenate(preds, axis=0)              # (N, 4)

    iou_scores = _iou(pred_bboxes, gt_bboxes)

    results = []
    for i in range(len(iou_scores)):
        results.append({
            "model": model_name,
            "iou":   float(iou_scores[i]),
        })
    return results


def run_evaluation(
    baseline_ckpt: str,
    augmented_ckpt: str,
    holdout_npz: str,
    in_dim: int = 2048,
    device_str: str = "cuda",
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    log.info(f"Evaluating on {device}")

    all_rows: List[Dict] = []

    if baseline_ckpt:
        all_rows += evaluate_bbox_head("baseline", baseline_ckpt, holdout_npz, device, in_dim)

    if augmented_ckpt:
        all_rows += evaluate_bbox_head("augmented", augmented_ckpt, holdout_npz, device, in_dim)

    if not all_rows:
        log.warning("No checkpoints provided — nothing to evaluate.")
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUTS_DIR / "results_table.csv", index=False)
    log.info(f"Saved results_table.csv ({len(df)} rows)")

    summary = df.groupby("model")["iou"].agg(
        mean_iou="mean",
        iou_ge_025=lambda s: (s >= 0.25).mean(),
        iou_ge_050=lambda s: (s >= 0.50).mean(),
    ).reset_index()
    summary.to_csv(OUTPUTS_DIR / "summary_table.csv", index=False)
    log.info("Saved summary_table.csv")
    print("\n=== Summary Table ===")
    print(summary.to_string(index=False))


@click.command()
@click.option("--baseline-ckpt",  default=None, help="Baseline bbox head checkpoint.")
@click.option("--augmented-ckpt", default=None, help="Augmented bbox head checkpoint.")
@click.option("--holdout-npz",    required=True, help="Holdout embeddings NPZ with bboxes.")
@click.option("--in-dim",         default=2048, show_default=True)
@click.option("--device",         default="cuda" if torch.cuda.is_available() else "cpu",
              show_default=True)
def main(
    baseline_ckpt: str,
    augmented_ckpt: str,
    holdout_npz: str,
    in_dim: int,
    device: str,
) -> None:
    """Evaluate baseline and augmented bbox heads on the holdout set."""
    logging.basicConfig(level=logging.INFO)
    run_evaluation(
        baseline_ckpt=baseline_ckpt,
        augmented_ckpt=augmented_ckpt,
        holdout_npz=holdout_npz,
        in_dim=in_dim,
        device_str=device,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
