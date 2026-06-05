"""
M-Flow training script.

Usage:
  python -m shared_manifold_domain_transfer.training.train_mflow
  python -m shared_manifold_domain_transfer.training.train_mflow +profile=local
  python -m shared_manifold_domain_transfer.training.train_mflow +profile=aws
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra

from shared_manifold_domain_transfer.data_proc.dataset import make_loaders
from shared_manifold_domain_transfer.models.ijepa import IJEPAEncoder
from shared_manifold_domain_transfer.models.mflow import RunwayMFlow
from shared_manifold_domain_transfer.evaluation.umap_viz import (
    plot_manifold_vectors,
)

log = logging.getLogger(__name__)


# Embedding cache helpers
def extract_and_cache_embeddings(
    encoder,
    loader: DataLoader,
    cache_path: str,
    device: torch.device,
    npz_path: str | None = None,
) -> Dict[str, np.ndarray]:
    """
    Run all images through the frozen I-JEPA encoder and cache results to disk.
    If cache exists, load and return without re-running encoder.

    Checks in order:
      1. cache_path (.npy dict) — written by this function on first run
      2. npz_path (.npz)       — pre-built by run_umap_baseline.py; loaded and
                                  re-saved as .npy so subsequent runs are fast
      3. Run encoder from scratch and save to cache_path

    Returns:
        {
            'embeddings': (N, 1280),
            'poses':      (N, 6),
            'domains':    (N,),
            'img_paths':  list of str
        }
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        log.info(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True).item()
        return data

    if npz_path is not None:
        npz_path = Path(npz_path)
        if npz_path.exists():
            log.info(f"Loading pre-built embeddings from {npz_path}")
            raw  = np.load(npz_path, allow_pickle=True)
            data = {k: raw[k] for k in raw.files}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, data)
            log.info(f"Re-saved as {cache_path} for future runs")
            return data

    log.info("Extracting I-JEPA embeddings (this runs once and is cached)...")
    all_emb, all_pose, all_domain, all_paths = [], [], [], []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            imgs   = batch["image"].to(device)
            embs   = encoder(imgs).cpu().numpy()
            poses  = batch["pose_vector"].numpy()
            domains = batch["domain"].numpy()
            paths  = batch["img_path"]

            all_emb.append(embs)
            all_pose.append(poses)
            all_domain.append(domains)
            all_paths.extend(paths)

    data = {
        "embeddings": np.concatenate(all_emb,    axis=0),
        "poses":      np.concatenate(all_pose,   axis=0),
        "domains":    np.concatenate(all_domain, axis=0),
        "img_paths":  np.array(all_paths),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, data)
    log.info(f"Cached {len(data['embeddings'])} embeddings to {cache_path}")
    return data


class EmbeddingDataset(torch.utils.data.Dataset):
    """Thin Dataset wrapper over cached embeddings + poses."""

    def __init__(self, embeddings: np.ndarray, poses: np.ndarray, domains: np.ndarray) -> None:
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.poses      = torch.tensor(poses,      dtype=torch.float32)
        self.domains    = torch.tensor(domains,    dtype=torch.long)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embeddings[idx],
            "pose":      self.poses[idx],
            "domain":    self.domains[idx],
        }


# Training loop
class MFlowTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Training on {self.device}")

    def run(self) -> None:

        cfg = self.cfg

        # Only load the heavy I-JEPA encoder if embedding caches don't already exist
        # (avoids wasting ~2.4 GB of RAM/VRAM when prebuilt NPZs are provided)
        cache_dir = Path(cfg.ijepa.embeddings_cache_dir)
        run_name  = cfg.run.get("name", "default")
        d1_cache  = cache_dir / f"domain1_train_{run_name}.npy"
        d2_cache  = cache_dir / f"domain2_train_{run_name}.npy"
        d1_npz    = cfg.ijepa.get("d1_prebuilt_npz")
        d2_npz    = cfg.ijepa.get("d2_prebuilt_npz")

        need_encoder = not (
            (d1_cache.exists() or (d1_npz and Path(d1_npz).exists())) and
            (d2_cache.exists() or (d2_npz and Path(d2_npz).exists()))
        )

        if need_encoder:
            log.info("Loading I-JEPA encoder (no prebuilt cache found)...")
            encoder = IJEPAEncoder(
                weights_path=cfg.ijepa.get("weights_path"),
                device=str(self.device),
            )
            encoder.eval()
        else:
            log.info("Prebuilt NPZ/cache found — skipping I-JEPA encoder load.")
            encoder = None

        # Build dataloaders (only needed if encoder will be used)
        loaders = {}
        if need_encoder:
            log.info("Building dataloaders...")
            loaders = make_loaders(
                data_dir=cfg.data.data_dir,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.data.num_workers,
                image_size=cfg.data.image_size,
            )

        d1_data = extract_and_cache_embeddings(
            encoder, loaders.get("domain1_train"),
            d1_cache, self.device,
            npz_path=d1_npz,
        )
        d2_data = extract_and_cache_embeddings(
            encoder, loaders.get("domain2_train"),
            d2_cache, self.device,
            npz_path=d2_npz,
        )

        # Optional second npz: concatenate embeddings along feature axis (e.g. crop+full)
        npz_b_d1 = cfg.ijepa.get("d1_prebuilt_npz_b")
        npz_b_d2 = cfg.ijepa.get("d2_prebuilt_npz_b")
        if npz_b_d1 and npz_b_d2:
            log.info("Concat mode: loading second embedding set and concatenating...")
            raw_b1 = np.load(npz_b_d1, allow_pickle=True)
            raw_b2 = np.load(npz_b_d2, allow_pickle=True)
            d1_data["embeddings"] = np.concatenate(
                [d1_data["embeddings"], raw_b1["embeddings"]], axis=1
            )
            d2_data["embeddings"] = np.concatenate(
                [d2_data["embeddings"], raw_b2["embeddings"]], axis=1
            )
            log.info(f"  concat embedding dim: {d1_data['embeddings'].shape[1]}")

        # Optional pose concatenation: normalize pose to [-1, 1] using XPlane (d1) range,
        # then append to the embedding vector.  Scaler is saved so inference can replicate.
        if cfg.ijepa.get("concat_pose", False):
            log.info("Pose-concat mode: fitting MinMax scaler on XPlane (d1) poses...")
            pose_min = d1_data["poses"].min(axis=0)   # (6,)
            pose_max = d1_data["poses"].max(axis=0)   # (6,)
            denom    = pose_max - pose_min
            denom[denom == 0] = 1.0  # avoid div-by-zero for constant dimensions

            def _norm_pose(poses: np.ndarray) -> np.ndarray:
                return (2.0 * (poses - pose_min) / denom - 1.0).astype(np.float32)

            d1_data["embeddings"] = np.concatenate(
                [d1_data["embeddings"], _norm_pose(d1_data["poses"])], axis=1
            )
            d2_data["embeddings"] = np.concatenate(
                [d2_data["embeddings"], _norm_pose(d2_data["poses"])], axis=1
            )
            log.info(f"  pose-concat embedding dim: {d1_data['embeddings'].shape[1]}")

            # Save scaler alongside checkpoint so inference/viz can reproduce it
            ckpt_dir = Path(cfg.checkpoint.save_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            scaler = {"pose_min": pose_min, "pose_max": pose_max}
            np.save(ckpt_dir / "pose_scaler.npy", scaler)
            log.info(f"  Pose scaler saved → {ckpt_dir}/pose_scaler.npy")

        # Combine both domains
        embeddings = np.concatenate([d1_data["embeddings"], d2_data["embeddings"]], axis=0)
        poses      = np.concatenate([d1_data["poses"],      d2_data["poses"]],      axis=0)
        domains    = np.concatenate([d1_data["domains"],    d2_data["domains"]],    axis=0)

        # 80/20 train/val split
        n = len(embeddings)
        perm = np.random.permutation(n)
        n_train = int(0.8 * n)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        train_ds = EmbeddingDataset(embeddings[train_idx], poses[train_idx], domains[train_idx])
        val_ds   = EmbeddingDataset(embeddings[val_idx],   poses[val_idx],   domains[val_idx])

        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.training.batch_size,
            shuffle=False, num_workers=0,
        )

        # 3. Build M-Flow model
        model = RunwayMFlow(
            ambient_dim=cfg.model.ambient_dim,
            manifold_dim=cfg.model.manifold_dim,
            n_coupling_layers=cfg.model.n_coupling_layers,
            hidden_dim=cfg.model.hidden_dim,
            pose_dim=cfg.model.pose_dim,
            pose_hidden_dim=cfg.model.pose_hidden_dim,
            cond_flow=cfg.model.get("cond_flow", False),
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"M-Flow parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.max_epochs
        )

        # train loop
        save_dir = Path(cfg.checkpoint.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_ctr  = 0
        accum_steps   = cfg.training.gradient_accumulation_steps

        for epoch in range(1, cfg.training.max_epochs + 1):
            model.train()
            train_losses: list[dict] = []
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)):
                x    = batch["embedding"].to(self.device)
                pose = batch["pose"].to(self.device)

                loss, loss_dict = model.compute_loss(
                    x, pose,
                    lambda_likelihood=cfg.training.lambda_likelihood,
                    lambda_geometry=cfg.training.lambda_geometry,
                    lambda_pose=cfg.training.get("lambda_pose", 0.0),
                )
                (loss / accum_steps).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                train_losses.append(loss_dict)

            scheduler.step()

            mean_train = {k: np.mean([d[k] for d in train_losses]) for k in train_losses[0]}
            pose_str = f"  pose={mean_train['pose']:.4f}" if mean_train.get("pose", 0) > 0 else ""
            log.info(
                f"Epoch {epoch:3d} | "
                f"recon={mean_train['recon']:.4f}  "
                f"nll={mean_train['nll']:.4f}  "
                f"geo={mean_train['geometry']:.4f}"
                f"{pose_str}  "
                f"total={mean_train['total']:.4f}"
            )

            # Validation
            if epoch % cfg.training.val_every_n_epochs == 0:
                model.eval()
                val_losses: list[dict] = []

                all_z, all_domains = [], []
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
                        x    = batch["embedding"].to(self.device)
                        pose = batch["pose"].to(self.device)
                        _, loss_dict = model.compute_loss(x, pose,
                            lambda_likelihood=cfg.training.lambda_likelihood,
                            lambda_geometry=cfg.training.lambda_geometry,
                            lambda_pose=cfg.training.get("lambda_pose", 0.0),
                        )
                        val_losses.append(loss_dict)

                        z, _, _, _ = model(x, pose)
                        all_z.append(z.cpu().numpy())
                        all_domains.append(batch["domain"].numpy())

                mean_val = {k: np.mean([d[k] for d in val_losses]) for k in val_losses[0]}
                log.info(
                    f"  VAL | recon={mean_val['recon']:.4f}  "
                    f"nll={mean_val['nll']:.4f}  total={mean_val['total']:.4f}"
                )

                # UMAP of manifold vectors
                all_z_cat = np.concatenate(all_z, axis=0)
                all_d_cat = np.concatenate(all_domains, axis=0)
                try:
                    plot_manifold_vectors(all_z_cat, all_d_cat, epoch=epoch)
                except Exception as e:
                    log.warning(f"UMAP plot failed: {e}")

                # Checkpoint
                val_loss = mean_val["total"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_ctr  = 0
                    ckpt_path = save_dir / "best_mflow.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "cfg": OmegaConf.to_container(cfg),
                    }, ckpt_path)
                    log.info(f"  Checkpoint saved: {ckpt_path}  (val_loss={val_loss:.4f})")
                else:
                    patience_ctr += 1
                    log.info(f"  No improvement. Patience: {patience_ctr}/{cfg.training.early_stopping_patience}")
                    if patience_ctr >= cfg.training.early_stopping_patience:
                        log.info("Early stopping triggered.")
                        break

        log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

        best_recon = min(d["recon"] for d in train_losses)
        log.info(f"  Best val total loss:   {best_val_loss:.4f}")
        log.info(f"  Final train recon MSE: {best_recon:.4f}")
        if best_recon < 0.5:
            log.info("  Reconstruction MSE < 0.5. M-Flow looking good.")
        else:
            log.warning(
                f"  Reconstruction MSE = {best_recon:.4f} > 0.5. "
                "Consider: (1) lower manifold_dim, (2) train longer, "
                "(3) increase hidden_dim."
            )


@hydra.main(config_path="../../../configs", config_name="mflow", version_base=None)
def main(cfg: DictConfig) -> None:
    # Apply profile overrides
    if "profile" in cfg:
        profile = cfg.profile
        if isinstance(profile, str) and profile in cfg.get("profiles", {}):
            profile_cfg = cfg.profiles[profile]
            cfg = OmegaConf.merge(cfg, profile_cfg)
            log.info(f"Applied profile: {profile}")

    log.info(OmegaConf.to_yaml(cfg))
    trainer = MFlowTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
