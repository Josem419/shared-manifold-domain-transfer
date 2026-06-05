"""
Generate synthetic manifold vectors z* at target poses using the M-Flow.

Architecture insight
--------------------
The M-Flow encodes: z = encoder_output + pose_emb  (additive pose injection).
Therefore z decomposes cleanly as:
    z_content = z - pose_emb   (appearance / scene content)
    z = z_content + pose_emb

To generate z* at a new target pose p*:
  1. Compute pose_emb* = pose_encoder(p*)
  2. Find k nearest training samples in normalised pose space
  3. z* = z_content[knn] + pose_emb*

This swaps the pose component while keeping content from the training manifold.

Usage:
  PYTHONPATH=src python3 scripts/generate_synthetic.py \
      --checkpoint outputs/checkpoints/mflow_pose_cond/best_mflow_256d.pt \
      --d1-npz     outputs/embeddings/lard_20k/d1_full.npz \
      --d2-npz     outputs/embeddings/lard_20k/d2_full.npz \
      --holdout-npz outputs/embeddings/lard_20k/holdout_full.npz \
      --output-dir outputs/synthetic/pose_cond_256
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from shared_manifold_domain_transfer.models.mflow import RunwayMFlow


def _load_model(ckpt_path: str, device: torch.device) -> RunwayMFlow:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ckpt["cfg"]["model"]
    model = RunwayMFlow(
        ambient_dim=m["ambient_dim"],
        manifold_dim=m["manifold_dim"],
        n_coupling_layers=m["n_coupling_layers"],
        hidden_dim=m["hidden_dim"],
        pose_dim=m["pose_dim"],
        pose_hidden_dim=m["pose_hidden_dim"],
        cond_flow=m.get("cond_flow", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path}  epoch={ckpt['epoch']}  val={ckpt['val_loss']:.2f}  "
          f"manifold_dim={m['manifold_dim']}  cond_flow={m.get('cond_flow', False)}")
    return model


def _encode_npz(model: RunwayMFlow, npz_path: str, device: torch.device,
                batch_size: int = 512) -> dict:
    """Encode raw embeddings -> z, z_content, pose_emb, x_decoded, poses, img_paths."""
    raw = np.load(npz_path, allow_pickle=True)
    X         = raw["embeddings"].astype(np.float32)
    poses     = raw["poses"].astype(np.float32)
    img_paths = raw["img_paths"] if "img_paths" in raw else np.array([""] * len(X))

    all_z, all_z_content, all_pose_emb, all_x_dec = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_b = torch.tensor(X[i:i+batch_size]).to(device)
            p_b = torch.tensor(poses[i:i+batch_size]).to(device)
            z_b, _, _ = model.encode(x_b, p_b)
            pe_b = model.pose_encoder(p_b)
            x_dec_b = model.decode(z_b)
            all_z.append(z_b.cpu().numpy())
            all_z_content.append((z_b - pe_b).cpu().numpy())
            all_pose_emb.append(pe_b.cpu().numpy())
            all_x_dec.append(x_dec_b.cpu().numpy())

    return {
        "z":         np.concatenate(all_z,         axis=0),
        "z_content": np.concatenate(all_z_content, axis=0),
        "pose_emb":  np.concatenate(all_pose_emb,  axis=0),
        "x_decoded": np.concatenate(all_x_dec,     axis=0),  # ambient reconstruction
        "x_raw":     X,                                       # original input embeddings
        "poses":     poses,
        "img_paths": img_paths,
    }


@click.command()
@click.option("--checkpoint",   required=True,
              default="outputs/checkpoints/mflow_pose_cond/best_mflow_256d.pt",
              show_default=True)
@click.option("--d1-npz",       required=True,
              default="outputs/embeddings/lard_20k/d1_full.npz", show_default=True)
@click.option("--d2-npz",       required=True,
              default="outputs/embeddings/lard_20k/d2_full.npz", show_default=True)
@click.option("--holdout-npz",  required=True,
              default="outputs/embeddings/lard_20k/holdout_full.npz", show_default=True)
@click.option("--output-dir",   default="outputs/synthetic/pose_cond_256", show_default=True)
@click.option("--k-neighbors",  default=5,    show_default=True,
              help="Number of nearest-neighbor content vectors per target pose")
@click.option("--batch-size",   default=512,  show_default=True)
@click.option("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
def main(checkpoint, d1_npz, d2_npz, holdout_npz, output_dir,
         k_neighbors, batch_size, device):

    device = torch.device(device)
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = _load_model(checkpoint, device)

    # 1. Encode training data (D1 + D2 nominal)
    print("\nEncoding D1 (XPlane) ...")
    d1 = _encode_npz(model, d1_npz, device, batch_size)
    print(f"  {len(d1['z']):,} samples  z-dim={d1['z'].shape[1]}")

    print("Encoding D2 nominal ...")
    d2 = _encode_npz(model, d2_npz, device, batch_size)
    print(f"  {len(d2['z']):,} samples")

    z_train      = np.concatenate([d1["z"],         d2["z"]],         axis=0)
    z_content_tr = np.concatenate([d1["z_content"], d2["z_content"]], axis=0)
    pose_train   = np.concatenate([d1["poses"],     d2["poses"]],     axis=0)

    img_paths_train = np.concatenate([d1["img_paths"], d2["img_paths"]], axis=0)
    # Use raw (un-decoded) ResNet embeddings for the bbox head supervised loss
    x_raw_train     = np.concatenate([d1["x_raw"], d2["x_raw"]], axis=0)
    np.savez(out / "train_z.npz",
             z=z_train, poses=pose_train, img_paths=img_paths_train,
             embeddings=x_raw_train,
             z_d1=d1["z"], poses_d1=d1["poses"], img_paths_d1=d1["img_paths"],
             z_d2=d2["z"], poses_d2=d2["poses"], img_paths_d2=d2["img_paths"])
    print(f"Saved train_z.npz  ({len(z_train):,} training samples)")

    # 2. Encode holdout
    print("\nEncoding D2 holdout ...")
    holdout = _encode_npz(model, holdout_npz, device, batch_size)
    print(f"  {len(holdout['z']):,} samples")
    # Use raw (un-decoded) ResNet embeddings for the bbox head holdout evaluation
    np.savez(out / "holdout_z.npz",
             z=holdout["z"], poses=holdout["poses"],
             embeddings=holdout["x_raw"], img_paths=holdout["img_paths"])
    print("Saved holdout_z.npz")

    # 3. Normalise pose spaces for k-NN
    print("\nBuilding pose k-NN index ...")
    scaler = StandardScaler().fit(pose_train)
    pose_train_s   = scaler.transform(pose_train)
    pose_holdout_s = scaler.transform(holdout["poses"])

    # 4. Compute pose_emb for all holdout poses
    target_poses = holdout["poses"]   # (M, 6)
    pose_embs_holdout = []
    with torch.no_grad():
        for i in range(0, len(target_poses), batch_size):
            p_t = torch.tensor(target_poses[i:i+batch_size].astype(np.float32)).to(device)
            pose_embs_holdout.append(model.pose_encoder(p_t).cpu().numpy())
    pose_embs_holdout = np.concatenate(pose_embs_holdout, axis=0)  # (M, dim)

    # 5. z* = z_content[knn] + pose_emb*
    print(f"Generating synthetic z* (content-swap, k={k_neighbors}) ...")
    all_z_star, all_poses_star = [], []

    pose_batch = 1024
    for i in tqdm(range(0, len(target_poses), pose_batch), desc="Generating"):
        p_s  = pose_holdout_s[i:i+pose_batch]   # (B, 6) normalised
        pe   = pose_embs_holdout[i:i+pose_batch] # (B, dim)
        b    = len(p_s)

        # Pairwise squared distances in normalised pose space: (B, N_train)
        dists = ((p_s[:, None, :] - pose_train_s[None, :, :]) ** 2).sum(-1)

        knn_idx = np.argpartition(dists, k_neighbors, axis=1)[:, :k_neighbors]  # (B, k)

        content_knn = z_content_tr[knn_idx]    # (B, k, dim)
        pe_exp      = pe[:, None, :]            # (B, 1, dim)
        z_star_b    = content_knn + pe_exp      # (B, k, dim)

        all_z_star.append(z_star_b.reshape(-1, z_star_b.shape[-1]))
        all_poses_star.append(np.repeat(target_poses[i:i+b], k_neighbors, axis=0))

    z_star     = np.concatenate(all_z_star,     axis=0).astype(np.float32)
    poses_star = np.concatenate(all_poses_star, axis=0).astype(np.float32)

    # 6. Decode z* → ambient space (2048-d ResNet features or 1280-d I-JEPA)
    print("Decoding z* → ambient features ...")
    all_x_star = []
    with torch.no_grad():
        for i in range(0, len(z_star), batch_size):
            z_b = torch.tensor(z_star[i:i+batch_size]).to(device)
            x_b = model.decode(z_b)
            all_x_star.append(x_b.cpu().numpy())
    x_star = np.concatenate(all_x_star, axis=0).astype(np.float32)

    np.savez(out / "synthetic_z.npz", z=z_star, embeddings=x_star, poses=poses_star)
    print(f"Saved synthetic_z.npz  ({len(z_star):,} synthetic samples "
          f"at {len(target_poses):,} target poses x {k_neighbors} neighbors)")
    print(f"  z* shape: {z_star.shape}  (manifold)  "
          f"x* shape: {x_star.shape}  (ambient/decoded)")

    print(f"\nAll outputs in {out}/")
    print("  train_z.npz     -- D1+D2 training z + decoded ambient + poses")
    print("  holdout_z.npz   -- D2 holdout z + poses (evaluation)")
    print("  synthetic_z.npz -- generated z* + decoded x* at holdout poses")


if __name__ == "__main__":
    main()
