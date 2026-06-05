"""
Extract I-JEPA full-image embeddings for the MSFS holdout split and save as NPZ.

The holdout split = MSFS images whose poses fall OUTSIDE the Domain 2 nominal
corridor (13,459 samples).  These are the hard cases used for evaluation.

Usage:
  PYTHONPATH=src python3 scripts/extract_holdout_npz.py \\
      --data-dir  data/lard_20k \\
      --weights   models/IN22K-vit.h.14-900e.pth.tar \\
      --output    outputs/embeddings/lard_20k/d2_holdout_full.npz

If the output file already exists the script exits immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shared_manifold_domain_transfer.data_proc.dataset import make_loaders
from shared_manifold_domain_transfer.models.ijepa import IJEPAEncoder


@click.command()
@click.option("--data-dir",  default="data/lard_20k", show_default=True)
@click.option("--weights",   default="models/IN22K-vit.h.14-900e.pth.tar", show_default=True)
@click.option("--output",    default="outputs/embeddings/lard_20k/d2_holdout_full.npz", show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--device",    default="cuda" if torch.cuda.is_available() else "cpu", show_default=True)
def main(data_dir, weights, output, batch_size, device):
    output = Path(output)
    if output.exists():
        print(f"Already exists: {output}  — skipping.")
        return

    print(f"Loading I-JEPA encoder from {weights} ...")
    encoder = IJEPAEncoder(weights_path=weights, device=device)
    encoder.eval()

    print("Building dataloaders ...")
    loaders = make_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    loader = loaders["holdout_eval"]
    print(f"Holdout split: {len(loader.dataset):,} samples")

    all_emb, all_poses, all_domains, all_paths = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding holdout"):
            imgs    = batch["image"].to(device)
            embs    = encoder(imgs).cpu().numpy()
            all_emb.append(embs)
            all_poses.append(batch["pose_vector"].numpy())
            all_domains.append(batch["domain"].numpy())
            all_paths.extend(batch["img_path"])

    data = {
        "embeddings": np.concatenate(all_emb,    axis=0).astype(np.float32),
        "poses":      np.concatenate(all_poses,  axis=0).astype(np.float32),
        "domains":    np.concatenate(all_domains, axis=0),
        "img_paths":  np.array(all_paths),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output.with_suffix(""), **data)
    print(f"Saved {len(data['embeddings']):,} embeddings → {output}")


if __name__ == "__main__":
    main()
