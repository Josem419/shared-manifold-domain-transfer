"""
LARD V2 Dataset loader.

LARDDataset — PyTorch Dataset wrapping HuggingFace DEEL-AI/LARD_V2.
make_loaders()  — factory that returns train/val/holdout DataLoaders.

Domain definitions
------------------
Domain 1 (label=0): XPlane — full approach pose volume
Domain 2 (label=1): MSFS nominal approach volume only
Holdout  (label=1): MSFS poses outside nominal volume  

The nominal vs. full split is determined by PoseVolumeSampler: we compute the
convex hull of all MSFS training poses and label anything outside it as holdout.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import sys
import argparse
import ast

import torch
from torch.utils.data import Dataset, DataLoader

from shared_manifold_domain_transfer.data_proc.pose import (
    PoseProcessor,
    PoseVolumeSampler,
    DOMAIN2_LIMITS,
)
from shared_manifold_domain_transfer.data_proc.transforms import (
    get_train_transforms,
    get_val_transforms,
    IJEPA_IMAGE_SIZE,
)

log = logging.getLogger(__name__)

# LARD V2 simulator column values (lower-cased for matching)
_XPLANE_TAGS  = {"xplane", "x-plane", "xp11", "xp12"}
_MSFS_TAGS    = {"msfs", "msfs2020", "microsoft flight simulator", "flsim"}

# Corner coordinate columns — LARD stores 4 corners as separate x/y columns
# or as a single array. We try both conventions.
_CORNER_X_COLS = ["x1", "x2", "x3", "x4"]
_CORNER_Y_COLS = ["y1", "y2", "y3", "y4"]
_CORNER_ALT_COLS = [
    "corner1_x", "corner2_x", "corner3_x", "corner4_x",
    "corner1_y", "corner2_y", "corner3_y", "corner4_y",
]
# hf_config is the reliable persistent column ("xplane" or "flsim");
# sim_tag is only added in-memory by visualize_dataset.py, not saved to parquet.
_SIM_CANDIDATES = ["hf_config", "simulator", "sim", "source", "dataset", "domain"]
_IMG_CANDIDATES = ["image", "img", "file", "filename", "filepath", "path", "image_path"]


class LARDDataset(Dataset):
    """
    Args:
        data_dir: root directory containing downloaded LARD V2 data.
                  Expects parquet/CSV metadata and image files underneath.
        split:    one of 'domain1', 'domain2', 'holdout', 'all'
        pose_processor: fitted PoseProcessor for normalisation. Pass None to
                        build and fit one from this split's data.
        augment: whether to apply colour jitter / augmentation.
        image_size: target image size for the full image (default 224 — I-JEPA
                    requires exactly this size).
        max_samples: if set, truncate dataset (useful for smoke tests).

    The runway crop is NOT pre-computed here. Corner annotations are returned
    as normalised [0,1] coordinates so the caller can crop at whatever size
    makes sense (e.g. eval_pipeline crops to the natural AABB size).
    """

    DOMAIN1 = 0   # XPlane
    DOMAIN2 = 1   # MSFS

    def __init__(
        self,
        data_dir: str,
        split: str = "domain1",
        pose_processor: Optional[PoseProcessor] = None,
        domain2_sampler: Optional[PoseVolumeSampler] = None,
        augment: bool = False,
        image_size: int = IJEPA_IMAGE_SIZE,
        max_samples: Optional[int] = None,
    ) -> None:
        assert split in ("domain1", "domain2", "holdout", "all"), \
            f"split must be domain1|domain2|holdout|all, got '{split}'"

        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.image_size = image_size

        self.img_transform = (get_train_transforms(image_size) if augment
                              else get_val_transforms(image_size))

        # Load metadata
        self._meta = self._load_metadata()

        # Detect simulator column
        sim_col = self._resolve(self._meta.columns.tolist(), _SIM_CANDIDATES)
        if sim_col is None:
            raise ValueError(
                f"Cannot find simulator column in metadata. Columns: {self._meta.columns.tolist()[:15]}"
            )
        self._sim_col = sim_col

        # Identify image path column
        self._img_col = self._resolve(self._meta.columns.tolist(), _IMG_CANDIDATES)

        # Build / use PoseProcessor
        if pose_processor is None:
            log.info("Fitting PoseProcessor on this split's data...")
            self.pose_processor = PoseProcessor()
            self.pose_processor.fit(self._meta)
        else:
            self.pose_processor = pose_processor

        # Store both raw (physical units) and normalised poses.
        # _raw_poses  — used with ApproachLimits.is_valid() (physical corridor check)
        # _norm_poses — used with PoseVolumeSampler.is_inside() (convex hull)
        self._raw_poses  = self.pose_processor.transform_raw(self._meta)
        self._norm_poses = self.pose_processor.transform(self._meta)

        # Build domain2 hull sampler if not provided
        if domain2_sampler is None and split in ("holdout",):
            raise ValueError(
                "PoseVolumeSampler for Domain 2 must be provided when split='holdout'. "
                "Build it from the domain2 split first."
            )
        self._domain2_sampler = domain2_sampler

        # Filter rows to the requested split
        self._indices = self._build_indices()

        if max_samples is not None:
            self._indices = self._indices[:max_samples]

        log.info(f"LARDDataset: split='{split}', n_samples={len(self._indices)}")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict:
        row_idx = self._indices[idx]
        row = self._meta.iloc[row_idx]

        # Load image
        image_pil = self._load_image(row)
        width, height = image_pil.size

        # Parse corners shape(4, 2) in pixel space, then normalise to [0,1]
        corners_px = self._parse_corners(row)          # (4, 2) float32
        corners_norm = corners_px / np.array([width, height], dtype=np.float32)  # (4, 2) in [0,1]

        # Apply transforms to full image only
        image_tensor = self.img_transform(image_pil)   # (3, image_size, image_size)

        # Pose vector
        pose_row = row.to_frame().T  # single-row DataFrame
        pose_vec = self.pose_processor.transform(pose_row)[0]  # (6,)

        # Domain label
        sim_tag = str(row[self._sim_col]).lower().strip()
        domain  = self.DOMAIN1 if sim_tag in _XPLANE_TAGS else self.DOMAIN2

        img_path = str(row[self._img_col]) if self._img_col else f"row_{row_idx}"

        return {
            "image":       image_tensor,                                    # (3, 224, 224)
            "pose_vector": torch.tensor(pose_vec, dtype=torch.float32),     # (6,)
            "corners":     torch.tensor(corners_norm, dtype=torch.float32), # (4, 2) in [0,1]
            "domain":      torch.tensor(domain, dtype=torch.long),
            "img_path":    img_path,
        }

    # Internal helpers
    def _load_metadata(self) -> pd.DataFrame:
        """Try parquet first, then CSV. Searches data_dir recursively."""
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        csv_files     = list(self.data_dir.rglob("*.csv"))

        if parquet_files:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            meta = pd.concat(dfs, ignore_index=True)
            log.info(f"Loaded {len(parquet_files)} parquet files, {len(meta)} rows")
            return meta
        if csv_files:
            dfs = [pd.read_csv(f) for f in csv_files]
            meta = pd.concat(dfs, ignore_index=True)
            log.info(f"Loaded {len(csv_files)} CSV files, {len(meta)} rows")
            return meta

        raise FileNotFoundError(
            f"No parquet or CSV metadata found under {self.data_dir}. "
            "Run scripts/download_lard.py first."
        )

    def _build_indices(self) -> List[int]:
        sim_tags = self._meta[self._sim_col].str.lower().str.strip()

        if self.split == "all":
            return list(range(len(self._meta)))

        if self.split == "domain1":
            mask = sim_tags.isin(_XPLANE_TAGS)
            return list(np.where(mask.values)[0])

        # For domain2 and holdout: restrict to MSFS rows
        msfs_mask = sim_tags.isin(_MSFS_TAGS)
        msfs_idx  = np.where(msfs_mask.values)[0]

        if self.split == "domain2":
            # MSFS nominal: inside Domain 2 hull (hull check + corridor limits)
            if self._domain2_sampler is None:
                # First time building — return all MSFS rows as domain2
                return list(msfs_idx)
            inside = self._domain2_sampler.is_inside(
                self._norm_poses[msfs_idx],
                raw_poses=self._raw_poses[msfs_idx],
            )
            return list(msfs_idx[inside])

        if self.split == "holdout":
            # MSFS poses outside domain2 nominal volume
            outside = self._domain2_sampler.is_outside(
                self._norm_poses[msfs_idx],
                raw_poses=self._raw_poses[msfs_idx],
            )
            return list(msfs_idx[outside])

        return []

    def _load_image(self, row) -> Image.Image:
        """Load PIL image from path stored in metadata."""
        if self._img_col is None:
            raise ValueError("Image path column not found in metadata.")

        img_path = str(row[self._img_col])

        # Check if it's a relative path
        if not os.path.isabs(img_path):
            img_path = str(self.data_dir / img_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        return Image.open(img_path).convert("RGB")

    def _parse_corners(self, row) -> np.ndarray:
        """Parse 4 corner pixel coordinates → (4, 2) float32 array (x, y)."""
        cols = self._meta.columns.tolist()

        # Try convention 1: x1,y1,x2,y2,x3,y3,x4,y4
        if all(c in cols for c in _CORNER_X_COLS + _CORNER_Y_COLS):
            xs = [float(row[c]) for c in _CORNER_X_COLS]
            ys = [float(row[c]) for c in _CORNER_Y_COLS]
            return np.array(list(zip(xs, ys)), dtype=np.float32)

        # Try convention 2: corner1_x, corner1_y, ...
        alt_x_cols = [f"corner{i}_x" for i in range(1, 5)]
        alt_y_cols = [f"corner{i}_y" for i in range(1, 5)]
        if all(c in cols for c in alt_x_cols + alt_y_cols):
            xs = [float(row[c]) for c in alt_x_cols]
            ys = [float(row[c]) for c in alt_y_cols]
            return np.array(list(zip(xs, ys)), dtype=np.float32)

        # Try convention 3: corners as a serialised list column
        corner_candidates = [c for c in cols if "corner" in c.lower()]
        if corner_candidates:
            raw = row[corner_candidates[0]]
            if isinstance(raw, str):
                data = ast.literal_eval(raw)
                corners = np.array(data, dtype=np.float32).reshape(4, 2)
                return corners

        log.warning("Could not parse corner coordinates — using full image as runway region.")
        # Fallback: corners at the four image corners (norm [0,1])
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    @staticmethod
    def _resolve(columns: List[str], candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        return None

# Dataloaders Factory
def make_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = IJEPA_IMAGE_SIZE,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Returns:
        {
            'domain1_train': DataLoader,
            'domain2_train': DataLoader,
            'holdout_eval':  DataLoader,
        }
    """
    # Step 1: Build domain1 dataset to fit PoseProcessor on combined data
    log.info("Building Domain 1 (XPlane) dataset...")
    d1 = LARDDataset(
        data_dir=data_dir,
        split="domain1",
        augment=True,
        image_size=image_size,
        max_samples=max_samples,
    )
    pose_proc = d1.pose_processor

    # Step 2: Build all-MSFS dataset to fit Domain 2 hull
    log.info("Building Domain 2 (MSFS) dataset to fit pose hull...")
    d2_all = LARDDataset(
        data_dir=data_dir,
        split="domain2",
        pose_processor=pose_proc,
        augment=True,
        image_size=image_size,
    )
    domain2_sampler = PoseVolumeSampler(
        d2_all._norm_poses[d2_all._indices],
        limits=DOMAIN2_LIMITS,
    )

    # Step 3: Re-build domain2 using the hull (nominal only)
    log.info("Re-filtering Domain 2 to nominal volume...")
    d2 = LARDDataset(
        data_dir=data_dir,
        split="domain2",
        pose_processor=pose_proc,
        domain2_sampler=domain2_sampler,
        augment=True,
        image_size=image_size,
        max_samples=max_samples,
    )

    # Step 4: Holdout — MSFS outside Domain 2 nominal
    log.info("Building holdout (MSFS outside nominal volume)...")
    holdout = LARDDataset(
        data_dir=data_dir,
        split="holdout",
        pose_processor=pose_proc,
        domain2_sampler=domain2_sampler,
        augment=False,
        image_size=image_size,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "domain1_train": DataLoader(d1, shuffle=True, **loader_kwargs),
        "domain2_train": DataLoader(d2, shuffle=True, **loader_kwargs),
        "holdout_eval":  DataLoader(holdout, shuffle=False, **loader_kwargs),
        # Expose datasets for downstream use (e.g. building pose index)
        "_domain1_dataset":  d1,
        "_domain2_dataset":  d2,
        "_holdout_dataset":  holdout,
        "_pose_processor":   pose_proc,
        "_domain2_sampler":  domain2_sampler,
    }

# quick test of dataset and loaders
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", help="Path to downloaded LARD V2 data")
    parser.add_argument("--max_samples", type=int, default=8)
    args = parser.parse_args()

    loaders = make_loaders(
        data_dir=args.data_dir,
        batch_size=4,
        num_workers=0,
        max_samples=args.max_samples,
    )

    for name, loader in loaders.items():
        if name.startswith("_"):
            continue
        batch = next(iter(loader))
        print(f"\n{name}:")
        print(f"  image:       {batch['image'].shape}")
        print(f"  pose_vector: {batch['pose_vector'].shape}")
        print(f"  corners:     {batch['corners'].shape}  (normalised [0,1], crop at eval time)")
        print(f"  domain:      {batch['domain'].tolist()}")

    sys.exit(0)
