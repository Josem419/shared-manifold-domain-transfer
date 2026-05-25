"""
Visualize the downloaded LARD V2 sample.

Examples:

  python scripts/visualize_dataset.py --data-dir data/lard
  python scripts/visualize_dataset.py --data-dir data/lard --output-dir outputs/viz

Produces four figures:

  1. sample_images.png          \u2014 image grid with runway corners + pose overlay
  2. approach_distributions.png \u2014 histograms of the 3 key approach parameters
  3. approach_cone_2d.png       \u2014 scatter: lateral/glidepath vs along-track
  4. odd_coverage.png           \u2014 runway_in_cone counts + roll/pitch distributions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")

from shared_manifold_domain_transfer.data_proc.domain_odd import (
    DOMAIN1_LIMITS,
    DOMAIN2_LIMITS,
)
from shared_manifold_domain_transfer.data_proc.pose import PoseProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Colour palette: XPlane = blue, MSFS = orange
COLOURS = {"xplane": "#2196F3", "msfs": "#FF9800"}


# Data loading
def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load and merge parquet files from xplane/ and msfs/ sub-dirs."""
    dfs = []
    for sim_tag, colour in COLOURS.items():
        meta_path = data_dir / sim_tag / "metadata.parquet"
        if not meta_path.exists():
            log.warning(f"No parquet at {meta_path} — skipping")
            continue
        df = pd.read_parquet(meta_path)
        df["sim_tag"] = sim_tag
        dfs.append(df)
        log.info(f"Loaded {len(df)} rows from {sim_tag}")

    if not dfs:
        raise FileNotFoundError(f"No parquet files found under {data_dir}. Run download_lard.py first.")

    return pd.concat(dfs, ignore_index=True)


def load_image(row: pd.Series, data_dir: Path) -> Optional[Image.Image]:
    """Load a PIL image from the relative image_path stored in metadata."""
    img_path = data_dir / str(row["image_path"])
    if not img_path.exists():
        return None
    try:
        return Image.open(img_path).convert("RGB")
    except Exception:
        return None


def plot_sample_images(
    df: pd.DataFrame,
    data_dir: Path,
    output_path: Path,
    n_per_sim: int = 6,
) -> None:
    """Grid of images with runway corners drawn and key pose values as title."""
    fig, axes = plt.subplots(
        2, n_per_sim, figsize=(n_per_sim * 3, 7),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    fig.suptitle("LARD V2 Sample Images", fontsize=14, y=1.01)

    for row_idx, (sim_tag, colour) in enumerate(COLOURS.items()):
        subset = df[df["sim_tag"] == sim_tag]
        if subset.empty:
            continue
        sample = subset.sample(min(n_per_sim, len(subset)), random_state=42)

        for col_idx, (_, row) in enumerate(sample.iterrows()):
            ax = axes[row_idx][col_idx]
            img = load_image(row, data_dir)
            if img is None:
                ax.axis("off")
                continue

            ax.imshow(img)

            # Draw runway corners if available
            w, h = img.size
            corner_cols = [("x_TL", "y_TL"), ("x_TR", "y_TR"),
                           ("x_BR", "y_BR"), ("x_BL", "y_BL")]
            corners = []
            for xc, yc in corner_cols:
                if xc in row and yc in row and pd.notna(row[xc]) and pd.notna(row[yc]):
                    corners.append((float(row[xc]), float(row[yc])))
            if len(corners) == 4:
                xs = [c[0] for c in corners] + [corners[0][0]]
                ys = [c[1] for c in corners] + [corners[0][1]]
                ax.plot(xs, ys, "-", color="lime", linewidth=1.5, alpha=0.9)

            # Pose annotation — values are already in approach frame
            atd = row.get("along_track_distance", float("nan"))
            lat = row.get("lateral_path_angle",  float("nan"))
            vrt = row.get("vertical_path_angle", float("nan"))
            ax.set_title(
                f"d={atd:.0f}m\nlat={lat:.1f}° vrt={vrt:.1f}°",
                fontsize=6.5, color=colour, pad=2,
            )
            ax.axis("off")

            # Sim label on first image
            if col_idx == 0:
                ax.set_ylabel(sim_tag.upper(), fontsize=9, color=colour, rotation=0,
                              labelpad=40, va="center")
                ax.yaxis.set_label_position("left")

        # Fill unused slots
        for col_idx in range(len(sample), n_per_sim):
            axes[row_idx][col_idx].axis("off")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {output_path}")


def plot_approach_distributions(df: pd.DataFrame, output_path: Path) -> None:
    """Histograms of along_track_distance, lateral_path_angle, vertical_path_angle."""
    params = [
        ("along_track_distance", "Along-track distance  (m)",
         DOMAIN1_LIMITS.along_track_range,  DOMAIN2_LIMITS.along_track_range),
        ("lateral_path_angle",  "Lateral path angle  (\u00b0)",
         DOMAIN1_LIMITS.lateral_range,      DOMAIN2_LIMITS.lateral_range),
        ("vertical_path_angle", "Vertical path angle  (\u00b0)",
         DOMAIN1_LIMITS.vertical_deg_range, DOMAIN2_LIMITS.vertical_deg_range),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("LARD V2 Approach Parameter Distributions", fontsize=13)

    for ax, (col, xlabel, d1_range, d2_range) in zip(axes, params):
        if col not in df.columns:
            ax.set_title(f"{col}\n(not in data)")
            continue

        for sim_tag, colour in COLOURS.items():
            subset = df[(df["sim_tag"] == sim_tag) & df[col].notna()][col]
            if subset.empty:
                continue
            ax.hist(subset, bins=30, alpha=0.6, color=colour,
                    label=sim_tag.upper(), density=True, edgecolor="none")

        # ODD corridor overlays
        ax.axvspan(*d1_range, alpha=0.08, color="blue",  label="D1 ODD")
        ax.axvspan(*d2_range, alpha=0.15, color="orange", label="D2 nominal")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {output_path}")


def plot_approach_cone(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plots showing the 2D approach cone."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LARD V2 Approach Cone Coverage", fontsize=13)

    # Left: lateral vs along-track (top-down view of approach corridor)
    ax = axes[0]
    for sim_tag, colour in COLOURS.items():
        sub = df[df["sim_tag"] == sim_tag]
        if sub.empty or "along_track_distance" not in df.columns:
            continue
        ax.scatter(sub["along_track_distance"], sub["lateral_path_angle"],
                   s=8, alpha=0.6, color=colour, label=sim_tag.upper())

    _draw_corridor_box(ax, DOMAIN1_LIMITS.along_track_range,
                       DOMAIN1_LIMITS.lateral_range,
                       "blue", "D1 ODD")
    _draw_corridor_box(ax, DOMAIN2_LIMITS.along_track_range,
                       DOMAIN2_LIMITS.lateral_range,
                       "orange", "D2 nominal")
    ax.set_xlabel("Along-track distance  (m)", fontsize=10)
    ax.set_ylabel("Lateral path angle  (°)", fontsize=10)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)
    ax.set_title("Top-down (horizontal) corridor", fontsize=10)

    # Right: vertical vs along-track (side-view, glidepath)
    ax = axes[1]
    for sim_tag, colour in COLOURS.items():
        sub = df[df["sim_tag"] == sim_tag]
        if sub.empty or "vertical_path_angle" not in df.columns:
            continue
        ax.scatter(sub["along_track_distance"], sub["vertical_path_angle"],
                   s=8, alpha=0.6, color=colour, label=sim_tag.upper())

    _draw_corridor_box(ax, DOMAIN1_LIMITS.along_track_range,
                       DOMAIN1_LIMITS.vertical_deg_range,
                       "blue", "D1 ODD")
    _draw_corridor_box(ax, DOMAIN2_LIMITS.along_track_range,
                       DOMAIN2_LIMITS.vertical_deg_range,
                       "orange", "D2 nominal")
    ax.set_xlabel("Along-track distance  (m)", fontsize=10)
    ax.set_ylabel("Vertical path angle  (°)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title("Side-view (glidepath) corridor", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {output_path}")


def _draw_corridor_box(ax, x_range, y_range, colour, label):
    rect = mpatches.Rectangle(
        (x_range[0], y_range[0]),
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        linewidth=1.5, edgecolor=colour, facecolor=colour,
        alpha=0.12, label=label,
    )
    ax.add_patch(rect)

def plot_odd_coverage(df: pd.DataFrame, output_path: Path) -> None:
    """runway_in_cone counts and attitude angle distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("ODD Coverage and Aircraft Attitude", fontsize=13)

    # Left: runway_in_cone bar chart
    ax = axes[0]
    if "runway_in_cone" in df.columns:
        for i, (sim_tag, colour) in enumerate(COLOURS.items()):
            sub = df[df["sim_tag"] == sim_tag]["runway_in_cone"].value_counts()
            x = np.arange(len(sub)) + i * 0.4
            ax.bar(x, sub.values, width=0.35, color=colour,
                   label=sim_tag.upper(), alpha=0.8)
            for xi, label in zip(x, sub.index):
                ax.text(xi, -max(sub.values) * 0.04, str(label),
                        ha="center", fontsize=7, rotation=15)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("runway_in_cone", fontsize=10)
        ax.set_xticks([])
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "runway_in_cone\nnot found", ha="center",
                transform=ax.transAxes)

    # Middle + Right: roll and pitch histograms
    for ax, col, xlabel in [
        (axes[1], "roll",  "Roll  (°)"),
        (axes[2], "pitch", "Pitch  (°)"),
    ]:
        if col not in df.columns:
            ax.set_title(f"{col} not found")
            continue
        for sim_tag, colour in COLOURS.items():
            sub = df[(df["sim_tag"] == sim_tag) & df[col].notna()][col]
            ax.hist(sub, bins=25, alpha=0.6, color=colour,
                    label=sim_tag.upper(), density=True, edgecolor="none")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {output_path}")


@click.command()
@click.option("--data-dir",   default="data/lard",    show_default=True,
              help="Root directory produced by download_lard.py.")
@click.option("--output-dir", default="outputs/viz",  show_default=True,
              help="Directory to write figures into.")
@click.option("--n-images",   default=6, show_default=True,
              help="Images per simulator in the sample grid.")
def main(data_dir: str, output_dir: str, n_images: int) -> None:
    """Visualize the downloaded LARD V2 sample.

    Loads parquet metadata from DATA_DIR/{xplane,msfs}/metadata.parquet and
    the corresponding JPEG images, then writes four diagnostic figures to
    OUTPUT_DIR.
    """
    data_path   = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading metadata from {data_path}")
    df = load_metadata(data_path)
    log.info(f"Total rows: {len(df)} | columns: {df.columns.tolist()}")

    # Convert raw LARD columns to approach frame:
    #   along_track_distance  positive km  →  negative metres
    #   pitch                 graphics Z-up (~86°) →  aviation horizon (~-4°)
    proc = PoseProcessor().fit(df)
    approach = proc.transform_raw(df)           # (N, 6) approach frame
    df = df.copy()
    df["along_track_distance"] = approach[:, 0]  # negative metres
    df["pitch"]                = approach[:, 4]  # aviation degrees

    print("\n=== Dataset summary (approach frame) ===")
    for sim_tag in df["sim_tag"].unique():
        sub = df[df["sim_tag"] == sim_tag]
        print(f"  {sim_tag:10s}: {len(sub):5d} images")
        for col in ("along_track_distance", "lateral_path_angle", "vertical_path_angle"):
            if col in sub.columns:
                v = sub[col].dropna()
                print(f"    {col:28s}: [{v.min():.1f}, {v.max():.1f}]  mean={v.mean():.2f}")

    log.info("Generating figures...")
    plot_sample_images(df, data_path, output_path / "sample_images.png", n_per_sim=n_images)
    plot_approach_distributions(df, output_path / "approach_distributions.png")
    plot_approach_cone(df, output_path / "approach_cone_2d.png")
    plot_odd_coverage(df, output_path / "odd_coverage.png")

    print(f"\nAll figures saved to {output_path.resolve()}")


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
