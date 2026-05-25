"""
Download LARD V2 from HuggingFace.

LARD V2 is organised as separate HuggingFace configs per simulator:
    xplane   — XPlane 12           (our Domain 1)
    flsim    — Microsoft Flight Sim (our Domain 2 + Holdout)
    arcgis   — ArcGIS imagery
    bingmaps — Bing Maps imagery
    ges      — Google Earth Studio

Each config has "train" and "test" splits.

Examples:

  # inspect schema without downloading
  python scripts/download_lard.py inspect

  # download 50 samples from xplane and flsim
  python scripts/download_lard.py download --max-per-split 50

  # full download of specific configs
  python scripts/download_lard.py download --config xplane --config flsim
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import click
import pandas as pd
from datasets import load_dataset as hf_load
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATASET_ID = "DEEL-AI/LARD_V2"

# Maps HuggingFace config name → local subdirectory name
CONFIG_DIR_MAP = {
    "xplane":   "xplane",
    "flsim":    "msfs",
    "arcgis":   "arcgis",
    "bingmaps": "bing_maps",
    "ges":      "google_earth",
}
DEFAULT_CONFIGS = ["xplane", "flsim"]


def inspect_dataset(configs: list[str]) -> None:
    """Print dataset schema and a sample row for each config."""
    for config in configs:
        log.info(f"Inspecting config '{config}'...")
        ds = hf_load(DATASET_ID, config, split="train", streaming=True)
        sample = next(iter(ds))
        print(f"\n=== Config: {config} ===")
        for k, v in sample.items():
            if k == "image":
                if isinstance(v, PILImage.Image):
                    print(f"  image: PIL Image {v.size} mode={v.mode}")
                else:
                    print(f"  image: {type(v).__name__}")
            else:
                print(f"  {k}: {type(v).__name__} = {str(v)[:80]}")


def download_lard(
    output_dir: str,
    configs: list[str] = DEFAULT_CONFIGS,
    hf_splits: list[str] = ("train",),
    max_per_split: int | None = None,
) -> None:
    """Download LARD V2 configs and save images + parquet metadata to disk.

    When max_per_split is set, uses HuggingFace streaming mode so only the
    requested rows are downloaded (no full shard files needed).

    Output structure::

        output_dir/
          xplane/
            images/
            metadata.parquet   # one row per image; image_path is relative
          msfs/
            images/
            metadata.parquet
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        sim_dir_name = CONFIG_DIR_MAP.get(config, config)
        sim_dir      = output_dir / sim_dir_name
        img_dir      = sim_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict] = []

        for hf_split in hf_splits:
            log.info(f"Loading {DATASET_ID} config='{config}' split='{hf_split}'...")

            if max_per_split is not None:
                # Streaming avoids downloading full shard files (~400 MB each)
                # when we only need a small sample.
                ds_iter = hf_load(DATASET_ID, config, split=hf_split, streaming=True)
                ds_iter = ds_iter.take(max_per_split)
                items   = list(ds_iter)
            else:
                ds    = hf_load(DATASET_ID, config, split=hf_split)
                items = list(ds)

            log.info(f"  {len(items)} samples to save")

            for i, item in enumerate(items):
                # --- image ---
                img = item.get("image")
                if img is None:
                    log.warning(f"No image at index {i}; skipping")
                    continue
                if not isinstance(img, PILImage.Image):
                    try:
                        img = PILImage.open(BytesIO(img)).convert("RGB")
                    except Exception as exc:
                        log.warning(f"Could not decode image {i}: {exc}; skipping")
                        continue

                img_filename = f"{config}_{hf_split}_{i:06d}.jpg"
                img_path     = img_dir / img_filename
                img.save(img_path, quality=95)

                # --- metadata row ---
                row = {k: v for k, v in item.items() if k != "image"}
                row["image_path"] = str(img_path.relative_to(output_dir))
                row["hf_split"]   = hf_split
                row["hf_config"]  = config
                all_rows.append(row)

                if (i + 1) % 50 == 0:
                    log.info(f"  [{config}/{hf_split}] {i+1}/{len(items)}")

        if not all_rows:
            log.warning(f"No rows collected for config '{config}'")
            continue

        df = pd.DataFrame(all_rows)
        parquet_path = sim_dir / "metadata.parquet"

        # Append to existing file if present
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(parquet_path, index=False)
        log.info(f"Saved {len(df)} rows → {parquet_path}")

    log.info("\nDownload complete.")
    _print_summary(output_dir)


def _print_summary(output_dir: Path) -> None:
    print(f"\n=== Download Summary ===")
    print(f"Output: {output_dir.resolve()}")
    for sim_dir in sorted(output_dir.iterdir()):
        if not sim_dir.is_dir():
            continue
        meta_file = sim_dir / "metadata.parquet"
        img_dir   = sim_dir / "images"
        n_imgs    = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_meta    = len(pd.read_parquet(meta_file)) if meta_file.exists() else 0
        print(f"  {sim_dir.name:15s}: {n_imgs:6d} images,  {n_meta:6d} metadata rows")


@click.group()
def cli():
    """Download and inspect the LARD V2 runway approach dataset from HuggingFace."""


@cli.command("download")
@click.option("--output-dir",    default="data/lard", show_default=True,
              help="Root directory for downloaded images and parquet metadata.")
@click.option("--config",        "configs", multiple=True, default=DEFAULT_CONFIGS,
              show_default=True,
              help="HF config to download (repeat for multiple: --config xplane --config flsim).")
@click.option("--hf-split",      "hf_splits", multiple=True, default=("train",),
              show_default=True,
              help="HF split to download (repeat for multiple).")
@click.option("--max-per-split", type=int, default=None,
              help="Max samples per (config, split) pair. Uses streaming — no full shard download.")
def download_cmd(output_dir, configs, hf_splits, max_per_split):
    """Download LARD V2 images and metadata to OUTPUT_DIR.

    Saves one subdirectory per simulator with images/ and metadata.parquet.
    When --max-per-split is set, uses HuggingFace streaming so only the
    requested rows are fetched (avoids downloading ~400 MB shard files).
    """
    download_lard(
        output_dir=output_dir,
        configs=list(configs),
        hf_splits=list(hf_splits),
        max_per_split=max_per_split,
    )


@cli.command("inspect")
@click.option("--config", "configs", multiple=True, default=DEFAULT_CONFIGS,
              show_default=True,
              help="HF config to inspect (repeat for multiple).")
def inspect_cmd(configs):
    """Print schema and one sample row per config without downloading images."""
    inspect_dataset(list(configs))


if __name__ == "__main__":
    cli()

