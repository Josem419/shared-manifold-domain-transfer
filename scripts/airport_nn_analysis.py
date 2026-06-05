"""
Airport analysis of consistency-loss nearest neighbors.

For each synthetic z* (256-d manifold), finds its k=1 nearest neighbor in the
training set using the same manifold-space NN index as train_bbox_head.py, then
checks whether the retrieved neighbor is from the same airport as the synthetic
sample's source holdout pose.

Usage:
    PYTHONPATH=src python3 scripts/airport_nn_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

TRAIN_NPZ   = "outputs/synthetic/resnet_cond_256/train_z.npz"
SYNTH_NPZ   = "outputs/synthetic/resnet_cond_256/synthetic_z.npz"
HOLDOUT_NPZ = "outputs/synthetic/resnet_cond_256/holdout_z.npz"
XPLANE_PARQUET = "data/lard_20k/xplane/metadata.parquet"
MSFS_PARQUET   = "data/lard_20k/msfs/metadata.parquet"
K_SYNTH_PER_POSE = 5   # must match generate_synthetic.py --k-neighbors


def main():
    #  Build airport lookup from parquet metadata 
    df = pd.concat([
        pd.read_parquet(XPLANE_PARQUET)[["image_path", "airport"]],
        pd.read_parquet(MSFS_PARQUET)[["image_path", "airport"]],
    ])
    airport_map: dict[str, str] = dict(zip(df["image_path"], df["airport"]))

    def path_to_airport(p: str) -> str:
        return airport_map.get(p, airport_map.get(p.split("/")[-1], "unknown"))

    #  Load NPZs 
    tr = np.load(TRAIN_NPZ,   allow_pickle=True)
    sy = np.load(SYNTH_NPZ,   allow_pickle=True)
    ho = np.load(HOLDOUT_NPZ, allow_pickle=True)

    z_train_m = tr["z"].astype(np.float32)   # (N_train, 256)
    z_synth_m = sy["z"].astype(np.float32)   # (N_synth, 256)

    train_paths = tr["img_paths"]
    hold_paths  = ho["img_paths"]

    # Airport labels 
    train_airports = np.array([path_to_airport(p) for p in train_paths])

    # Synthetic samples: K_SYNTH_PER_POSE per holdout pose in order
    synth_airports = np.repeat(
        [path_to_airport(p) for p in hold_paths], K_SYNTH_PER_POSE
    )

    print("Train airport distribution:")
    for ap, cnt in zip(*np.unique(train_airports, return_counts=True)):
        print(f"  {ap}: {cnt}")

    print(f"\nSynth airport distribution (k={K_SYNTH_PER_POSE} per holdout pose):")
    for ap, cnt in zip(*np.unique(synth_airports, return_counts=True)):
        print(f"  {ap}: {cnt}")

    #  Build manifold-space NN index 
    print("\nBuilding NN index (256-d manifold z-space) ...")
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(z_train_m)
    _, nn_idx = nn.kneighbors(z_synth_m)
    nn_idx = nn_idx.squeeze()

    nn_airports = train_airports[nn_idx]

    # Same-airport rate
    same = (synth_airports == nn_airports).mean()
    print(f"\nSame-airport NN rate:   {same:.3f}  ({same*100:.1f}%)")
    print(f"Cross-airport NN rate:  {1-same:.3f}  ({(1-same)*100:.1f}%)")

    print("\nPer holdout-airport breakdown:")
    for ap in np.unique(synth_airports):
        mask = synth_airports == ap
        rate = (nn_airports[mask] == ap).mean()
        print(f"  synth={ap}  same-airport rate: {rate:.3f}")


if __name__ == "__main__":
    main()
