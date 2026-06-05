"""
Fine-tune a ResNet-50 backbone on runway bounding-box regression.

Architecture
  ResNet-50 (ImageNet pretrained) -> avgpool -> 2048-d -> BboxHead(4) -> sigmoid

Labels
  Derived from 4-corner LARD annotations -> normalised AABB [xmin, ymin, xmax, ymax]

Training set
  D1 (XPlane, 15 596) + D2 nominal (MSFS, 1 209)
  Validation: random 10% of combined set

Saves
  outputs/checkpoints/resnet_bbox/best_resnet.pt
    keys: model_state_dict, epoch, val_loss, cfg

Usage:
  PYTHONPATH=src python3 scripts/train_resnet_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shared_manifold_domain_transfer.data_proc.dataset import LARDDataset


#  Model 
class ResNetBboxDetector(nn.Module):
    """ResNet-50 backbone + linear bbox head."""

    def __init__(self, freeze_bn: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Drop the final FC
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 2048, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)   # (B, 2048)
        return self.head(feat)               # (B, 4) in [0,1]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2048-d features (no head)."""
        return self.backbone(x).flatten(1)


#  Bbox helpers 

def corners_to_bbox(corners: torch.Tensor) -> torch.Tensor:
    """corners (B,4,2) -> bbox (B,4) = [xmin,ymin,xmax,ymax] in [0,1]."""
    xmin = corners[:, :, 0].min(dim=1).values
    xmax = corners[:, :, 0].max(dim=1).values
    ymin = corners[:, :, 1].min(dim=1).values
    ymax = corners[:, :, 1].max(dim=1).values
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def batch_iou_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ix1 = torch.max(pred[:, 0], target[:, 0])
    iy1 = torch.max(pred[:, 1], target[:, 1])
    ix2 = torch.min(pred[:, 2], target[:, 2])
    iy2 = torch.min(pred[:, 3], target[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    ap = (pred[:, 2]   - pred[:, 0])   * (pred[:, 3]   - pred[:, 1])
    at = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + at - inter
    return inter / union.clamp(min=1e-6)


#  Training loop 
@click.command()
@click.option("--data-dir",    default="data/lard_20k",                    show_default=True)
@click.option("--output-dir",  default="outputs/checkpoints/resnet_bbox",  show_default=True)
@click.option("--epochs",      default=30,   show_default=True)
@click.option("--lr",          default=1e-4, show_default=True)
@click.option("--batch-size",  default=32,   show_default=True)
@click.option("--val-split",   default=0.1,  show_default=True)
@click.option("--num-workers", default=4,    show_default=True)
@click.option("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
def main(data_dir, output_dir, epochs, lr, batch_size, val_split, num_workers, device):
    device = torch.device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    #  Build combined D1 + D2 nominal dataset 
    print("Loading D1 (XPlane) ...")
    d1 = LARDDataset(data_dir=data_dir, split="domain1", augment=True)
    pose_proc = d1.pose_processor

    print("Loading D2 nominal ...")
    d2_all = LARDDataset(data_dir=data_dir, split="domain2",
                         pose_processor=pose_proc, augment=True)
    from shared_manifold_domain_transfer.data_proc.pose import PoseVolumeSampler
    from shared_manifold_domain_transfer.data_proc.dataset import DOMAIN2_LIMITS
    sampler = PoseVolumeSampler(d2_all._norm_poses[d2_all._indices],
                                limits=DOMAIN2_LIMITS)
    d2 = LARDDataset(data_dir=data_dir, split="domain2",
                     pose_processor=pose_proc, domain2_sampler=sampler, augment=True)

    full_ds = torch.utils.data.ConcatDataset([d1, d2])
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    print(f"Train: {n_train:,}  Val: {n_val:,}")

    #  Model 
    model = ResNetBboxDetector(freeze_bn=True).to(device)
    opt   = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_loss = float("inf")
    cfg = {"ambient_dim": 2048, "image_size": 224}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()  # keep BN frozen
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train", leave=False):
            imgs  = batch["image"].to(device)
            bbox  = corners_to_bbox(batch["corners"].to(device))
            pred  = model(imgs)
            loss  = F.smooth_l1_loss(pred, bbox)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                bbox = corners_to_bbox(batch["corners"].to(device))
                pred = model(imgs)
                val_loss += F.smooth_l1_loss(pred, bbox).item()
                val_iou  += batch_iou_torch(pred, bbox).mean().item()
        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:>3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"val_iou={val_iou:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
            }, out / "best_resnet.pt")
            print(f"  ✓ Saved best  (val={val_loss:.4f}  iou={val_iou:.3f})")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoint -> {out}/best_resnet.pt")


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
