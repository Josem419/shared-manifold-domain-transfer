"""
Image transforms for LARD data.

Two transform factories:
  get_train_transforms()  — with augmentation
  get_val_transforms()    — deterministic resize + normalise only

ImageNet mean/std is used for I-JEPA compatibility (pretrained on ImageNet22K).
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ViT-H/14 was pretrained at 224×224. The patch embedding is a Conv2d with
# kernel_size=14 and stride=14, producing a 16×16=256 patch grid. Positional
# embeddings are fixed-length (257 including CLS), so any other input size
# requires interpolating them — and the Facebook checkpoint was not trained
# with that interpolation. We keep all images at exactly 224×224 to avoid it.
IJEPA_IMAGE_SIZE = 224

# Fixed size for pixel-level evaluation metrics (MSE, SSIM) on the runway crop.
# The crop AABB varies per image (derived from 4-corner annotations), so we
# resize it to a common size to make cross-image comparisons meaningful.
# NOTE: this is NOT the size fed to I-JEPA for semantic fidelity — that metric
# re-encodes the crop at IJEPA_IMAGE_SIZE=224 (see evaluation/metrics.py).
RUNWAY_CROP_EVAL_SIZE = 96


def get_train_transforms(image_size: int = IJEPA_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.0),   # runway geometry is not horizontally symmetric
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = IJEPA_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_crop_transforms(crop_size: int = RUNWAY_CROP_EVAL_SIZE) -> transforms.Compose:
    """
    Resize + normalise a runway crop for pixel-level metrics (MSE, SSIM).
    Uses RUNWAY_CROP_EVAL_SIZE=96 by default — compact but consistent.
    For I-JEPA semantic fidelity, the crop must be re-resized to
    IJEPA_IMAGE_SIZE=224 before encoding (see evaluation/metrics.py).
    """
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor):
    """Undo ImageNet normalisation for visualisation. Returns tensor in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
