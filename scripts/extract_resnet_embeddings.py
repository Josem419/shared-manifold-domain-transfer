"""
Extract frozen ResNet-50 backbone embeddings (2048-d) for D1, D2 nominal, holdout.

Loads the best_resnet.pt checkpoint trained by train_resnet_detector.py,
strips the bbox head, and runs inference over each split.
  embeddings  (N, 2048)  float32
  poses       (N, 6)     float32
  domains     (N,)       int
  img_paths   (N,)       str

Usage:
  PYTHONPATH=src python3 scripts/extract_resnet_embeddings.py \\
      --checkpoint outputs/checkpoints/resnet_bbox/best_resnet.pt \\
      --data-dir   data/lard_20k \\
      --output-dir outputs/embeddings/lard_20k
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shared_manifold_domain_transfer.data_proc.dataset import (
    DOMAIN2_LIMITS,
    LARDDataset,
)
from shared_manifold_domain_transfer.data_proc.pose import PoseVolumeSampler
from scripts.train_resnet_detector import ResNetBboxDetector


def _load_backbone(ckpt_path: str, device: torch.device) -> ResNetBboxDetector:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNetBboxDetector(freeze_bn=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path}  epoch={ckpt['epoch']}  val_iou={ckpt.get('val_iou', '?'):.3f}")
    return model


def _encode_dataset(model: ResNetBboxDetector, dataset, device: torch.device,
                    batch_size: int, num_workers: int) -> dict:
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    all_emb, all_poses, all_domains, all_paths = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding", leave=False):
            imgs = batch["image"].to(device)
            emb  = model.features(imgs).cpu().numpy()
            all_emb.append(emb)
            all_poses.append(batch["pose_vector"].numpy())
            all_domains.append(batch["domain"].numpy())
            all_paths.extend(batch["img_path"])
    return {
        "embeddings": np.concatenate(all_emb,    axis=0).astype(np.float32),
        "poses":      np.concatenate(all_poses,  axis=0).astype(np.float32),
        "domains":    np.concatenate(all_domains, axis=0),
        "img_paths":  np.array(all_paths),
    }


@click.command()
@click.option("--checkpoint",   default="outputs/checkpoints/resnet_bbox/best_resnet.pt",
              show_default=True)
@click.option("--data-dir",     default="data/lard_20k",                   show_default=True)
@click.option("--output-dir",   default="outputs/embeddings/lard_20k",     show_default=True)
@click.option("--batch-size",   default=128,  show_default=True)
@click.option("--num-workers",  default=4,    show_default=True)
@click.option("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
def main(checkpoint, data_dir, output_dir, batch_size, num_workers, device):
    device = torch.device(device)
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = _load_backbone(checkpoint, device)

    # Build datasets (same split logic as make_loaders) 
    print("\nBuilding D1 (XPlane) ...")
    d1 = LARDDataset(data_dir=data_dir, split="domain1", augment=False)
    pose_proc = d1.pose_processor

    print("Building D2 all → fit hull ...")
    d2_all = LARDDataset(data_dir=data_dir, split="domain2",
                         pose_processor=pose_proc, augment=False)
    sampler = PoseVolumeSampler(d2_all._norm_poses[d2_all._indices],
                                limits=DOMAIN2_LIMITS)

    print("Building D2 nominal ...")
    d2 = LARDDataset(data_dir=data_dir, split="domain2",
                     pose_processor=pose_proc, domain2_sampler=sampler, augment=False)

    print("Building holdout ...")
    holdout = LARDDataset(data_dir=data_dir, split="holdout",
                          pose_processor=pose_proc, domain2_sampler=sampler, augment=False)

    splits = [
        ("d1_resnet",      d1,      f"{len(d1):,} D1 samples"),
        ("d2_resnet",      d2,      f"{len(d2):,} D2 nominal samples"),
        ("holdout_resnet", holdout, f"{len(holdout):,} holdout samples"),
    ]

    for name, ds, desc in splits:
        npz_path = out / f"{name}.npz"
        if npz_path.exists():
            print(f"\n{npz_path} already exists — skipping")
            continue
        print(f"\nEncoding {desc} ...")
        data = _encode_dataset(model, ds, device, batch_size, num_workers)
        np.savez(npz_path.with_suffix(""), **data)
        print(f"  Saved {npz_path}  shape={data['embeddings'].shape}")

    print("\nAll embeddings extracted.")


if __name__ == "__main__":
    main()
