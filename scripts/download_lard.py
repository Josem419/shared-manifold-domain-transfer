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


def _flush_rows(rows: list[dict], parquet_path: Path) -> None:
    """Append a batch of metadata rows to the parquet file on disk."""
    df = pd.DataFrame(rows)
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(parquet_path, index=False)


def download_lard(
    output_dir: str,
    configs: list[str] = DEFAULT_CONFIGS,
    hf_splits: list[str] = ("train",),
    max_per_split: int | None = None,
    save_every: int = 200,
) -> None:
    """Download LARD V2 configs and save images + parquet metadata to disk.

    Always uses HuggingFace streaming so images are fetched and saved one at a
    time without loading the full dataset into memory.  Metadata is flushed to
    disk every *save_every* images so progress is preserved if the process is
    interrupted.

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
        parquet_path = sim_dir / "metadata.parquet"

        # Track already-downloaded indices so a resumed run skips them.
        done_indices: set[str] = set()
        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path, columns=["image_path"])
            done_indices = set(existing_df["image_path"].tolist())
            log.info(f"  Resuming — {len(done_indices)} images already saved.")

        for hf_split in hf_splits:
            log.info(f"Streaming {DATASET_ID} config='{config}' split='{hf_split}'...")

            ds_iter = hf_load(DATASET_ID, config, split=hf_split, streaming=True)
            if max_per_split is not None:
                ds_iter = ds_iter.take(max_per_split)

            pending_rows: list[dict] = []
            saved = 0
            skipped = 0

            for i, item in enumerate(ds_iter):
                img_filename = f"{config}_{hf_split}_{i:06d}.jpg"
                rel_path     = f"{sim_dir_name}/images/{img_filename}"

                # Skip already-downloaded images (resume support).
                if rel_path in done_indices:
                    skipped += 1
                    continue

                # --- decode image ---
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

                img_path = img_dir / img_filename
                img.save(img_path, quality=95)
                # Explicitly close to release memory immediately.
                img.close()

                # --- accumulate metadata ---
                row = {k: v for k, v in item.items() if k != "image"}
                row["image_path"] = rel_path
                row["hf_split"]   = hf_split
                row["hf_config"]  = config
                pending_rows.append(row)
                saved += 1

                if saved % save_every == 0:
                    _flush_rows(pending_rows, parquet_path)
                    pending_rows = []
                    log.info(f"  [{config}/{hf_split}] {saved} saved (flushed to parquet)")

            # Flush any remaining rows.
            if pending_rows:
                _flush_rows(pending_rows, parquet_path)

            log.info(
                f"  [{config}/{hf_split}] done — {saved} saved, {skipped} skipped (already existed)"
            )

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
              help="Max samples per (config, split) pair.")
@click.option("--save-every",    type=int, default=200, show_default=True,
              help="Flush metadata to parquet after this many images (lower = more resume-friendly).")
def download_cmd(output_dir, configs, hf_splits, max_per_split, save_every):
    """Download LARD V2 images and metadata to OUTPUT_DIR.

    Always uses HuggingFace streaming — images are fetched and saved one at a
    time so the full dataset is never loaded into memory.  Progress is flushed
    to parquet every --save-every images so the download can be safely resumed.
    """
    download_lard(
        output_dir=output_dir,
        configs=list(configs),
        hf_splits=list(hf_splits),
        max_per_split=max_per_split,
        save_every=save_every,
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

