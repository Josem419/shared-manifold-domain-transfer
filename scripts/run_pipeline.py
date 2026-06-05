"""
End-to-end pipeline runner.

Stages:
  1. download        — download LARD V2 from HuggingFace
  2. umap            — UMAP domain separation baseline
  3. train_resnet    — fine-tune ResNet-50 bbox detector
  4. extract_embeds  — extract frozen ResNet-50 embeddings
  5. train_mflow     — train pose-conditioned M-Flow
  6. generate_synth  — generate synthetic embeddings via pose-swap
  7. train_bbox_head — train bbox head (baseline + augmented)
  8. eval            — evaluate bbox IoU on holdout set

Usage:
  python scripts/run_pipeline.py --stage all
  python scripts/run_pipeline.py --stage train_mflow
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STAGES = ["download", "umap", "train_resnet", "extract_embeds", "train_mflow", "generate_synth", "train_bbox_head", "eval"]
ROOT = Path(__file__).parent.parent


def run_stage(stage: str, extra_args: list[str]) -> None:
    cmd_map = {
        "download": [
            sys.executable, str(ROOT / "scripts" / "download_lard.py"),
        ],
        "umap": [
            sys.executable, str(ROOT / "scripts" / "run_umap_baseline.py"),
        ],
        "train_resnet": [
            sys.executable, str(ROOT / "scripts" / "train_resnet_detector.py"),
        ],
        "extract_embeds": [
            sys.executable, str(ROOT / "scripts" / "extract_resnet_embeddings.py"),
        ],
        "train_mflow": [
            sys.executable, "-m",
            "shared_manifold_domain_transfer.training.train_mflow",
        ],
        "generate_synth": [
            sys.executable, str(ROOT / "scripts" / "generate_synthetic.py"),
        ],
        "train_bbox_head": [
            sys.executable, str(ROOT / "scripts" / "train_bbox_head.py"),
        ],
        "eval": [
            sys.executable, "-m",
            "shared_manifold_domain_transfer.evaluation.eval_pipeline",
        ],
    }

    if stage not in cmd_map:
        log.error(f"Unknown stage '{stage}'. Valid: {list(cmd_map.keys())}")
        sys.exit(1)

    cmd = cmd_map[stage] + extra_args
    log.info(f"Running stage '{stage}': {' '.join(cmd)}")

    result = subprocess.run(cmd, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    if result.returncode != 0:
        log.error(f"Stage '{stage}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    log.info(f"Stage '{stage}' complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full domain transfer pipeline")
    parser.add_argument(
        "--stage", default="all",
        choices=["all"] + STAGES,
        help="Pipeline stage to run",
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to the stage")
    args = parser.parse_args()

    stages_to_run = STAGES if args.stage == "all" else [args.stage]

    for stage in stages_to_run:
        run_stage(stage, args.extra)

    log.info("\n=== Pipeline complete. ===")
    log.info("Check outputs/ for plots and results.")


if __name__ == "__main__":
    main()
