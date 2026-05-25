"""
Pose preprocessing for LARD V2: runway-relative 6-DOF approach vectors.

LARD already provides aircraft pose in runway-relative coordinates, so no
geodetic conversion is required.  The 6-DOF vector used throughout this
codebase is:

    [0] along_track_distance   (metres, negative — aircraft behind LTP)
    [1] lateral_path_angle     (degrees, 0 = on centreline)
    [2] vertical_path_angle    (degrees, negative — aircraft above runway)
    [3] roll                   (degrees)
    [4] pitch                  (degrees)
    [5] yaw                    (degrees)

The LARD V2 Operational Design Domain (ODD), from the dataset README:
    along_track:   -6 000 m to  -280 m  (from Landing Threshold Point)
    lateral:       ±3°                  (from centreline)
    vertical:      -1.8° to -5.2°       (w.r.t. Vertical Reference Point)
    pitch:         -15° to +5°
    roll/yaw:      segment-dependent (see LARD paper)

Classes:
    PoseProcessor      — fit normalisation stats; transform df → (N,6)
    ApproachLimits     — physical approach-corridor filter (metres / degrees)
    PoseVolumeSampler  — convex hull in normalised 6-DOF space + corridor limits
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Column name resolution (LARD V2 uses consistent names, but keep fallbacks)
# ---------------------------------------------------------------------------

_ALONG_TRACK_CANDIDATES  = ["along_track_distance"]
_LATERAL_ANGLE_CANDIDATES  = ["lateral_path_angle"]
_VERTICAL_ANGLE_CANDIDATES = ["vertical_path_angle"]
_ROLL_CANDIDATES   = ["roll", "roll_deg", "Roll"]
_PITCH_CANDIDATES  = ["pitch", "pitch_deg", "Pitch"]
_YAW_CANDIDATES    = ["yaw", "yaw_deg", "heading", "heading_deg", "Yaw", "Heading"]


def _resolve(columns: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def _ensure_degrees(values: np.ndarray, col: str) -> np.ndarray:
    """Convert to degrees if the column name implies radians."""
    if "rad" in col.lower():
        return np.rad2deg(values)
    return values  # LARD stores angles in degrees


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PoseStats:
    """Per-dimension mean and std computed from training split."""
    mean: np.ndarray  # (6,)
    std:  np.ndarray  # (6,)


@dataclass
class ApproachLimits:
    """Hard approach-corridor limits in physical LARD units.

    All thresholds are in the native units of the LARD columns:
        along_track_distance  — metres (negative values, aircraft behind LTP)
        lateral_path_angle    — degrees (signed, 0 = on centreline)
        vertical_path_angle   — degrees (negative, aircraft descending)

    These limits are applied to the **raw** (un-normalised) pose vectors.
    They are independent of the convex-hull test, which operates in
    normalised space.

    Usage example::

        raw   = proc.transform_raw(df)    # physical units
        norm  = proc.transform(df)        # normalised
        mask  = DOMAIN2_LIMITS.is_valid(raw) & sampler.is_inside(norm)
    """
    along_track_range:    Tuple[float, float] = (-6000.0, -280.0)
    max_lateral_deg:      float = 3.0
    vertical_deg_range:   Tuple[float, float] = (-5.2, -1.8)

    def is_valid(self, raw_poses: np.ndarray) -> np.ndarray:
        """Return (N,) boolean mask: True where all corridor limits are met.

        Args:
            raw_poses: (N, 6) array in physical LARD units
                       columns: [along_track, lateral, vertical, roll, pitch, yaw]
        """
        along    = raw_poses[:, 0]
        lateral  = raw_poses[:, 1]
        vertical = raw_poses[:, 2]

        return (
            (along   >= self.along_track_range[0])
            & (along   <= self.along_track_range[1])
            & (np.abs(lateral) <= self.max_lateral_deg)
            & (vertical >= self.vertical_deg_range[0])
            & (vertical <= self.vertical_deg_range[1])
        )


# Domain corridors grounded in the LARD V2 ODD.
#
# DOMAIN1_LIMITS — full LARD ODD (XPlane covers this entire volume).
# DOMAIN2_LIMITS — tighter nominal corridor for MSFS.  Images from MSFS that
#                  fall outside DOMAIN2_LIMITS but inside DOMAIN1_LIMITS form
#                  the holdout split.  Tune these values once real data has
#                  been downloaded and the actual distribution is known.
DOMAIN1_LIMITS = ApproachLimits(
    along_track_range  = (-3000.0, -280.0),
    max_lateral_deg    = 3.0,
    vertical_deg_range = (-5.2, -1.8),
)

DOMAIN2_LIMITS = ApproachLimits(
    along_track_range  = (-2500.0, -280.0),  # tighter along-track window
    max_lateral_deg    = 1.5,                # ±1.5° vs ±3° for D1
    vertical_deg_range = (-3.5, -2.5),       # narrower glidepath band
)


# ---------------------------------------------------------------------------
# PoseProcessor
# ---------------------------------------------------------------------------

class PoseProcessor:
    """Converts LARD V2 metadata rows into normalised 6-DOF pose vectors.

    The processor reads LARD's native runway-relative columns directly —
    no geodetic projection is required.

    Typical use::

        proc = PoseProcessor()
        proc.fit(domain1_df)                      # compute mean/std
        d1_norm = proc.transform(domain1_df)      # (N, 6) normalised
        d1_raw  = proc.transform_raw(domain1_df)  # (N, 6) physical units
    """

    def __init__(self, stats: Optional[PoseStats] = None) -> None:
        self.stats = stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PoseProcessor":
        """Compute normalisation stats from a training-split dataframe."""
        raw = self._batch_to_approach_frame(df)
        self.stats = PoseStats(mean=raw.mean(0), std=raw.std(0) + 1e-8)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return normalised (N, 6) pose array.  Requires fit() first."""
        raw = self._batch_to_approach_frame(df)
        if self.stats is not None:
            return (raw - self.stats.mean) / self.stats.std
        return raw

    def transform_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Return un-normalised (N, 6) pose array in physical LARD units."""
        return self._batch_to_approach_frame(df)

    def transform_single(
        self,
        along_track_m: float,
        lateral_angle_deg: float,
        vertical_angle_deg: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> np.ndarray:
        """Convert a single pose to a normalised 6-DOF vector."""
        raw = np.array(
            [along_track_m, lateral_angle_deg, vertical_angle_deg,
             roll_deg, pitch_deg, yaw_deg],
            dtype=np.float32,
        )
        if self.stats is not None:
            raw = (raw - self.stats.mean) / self.stats.std
        return raw

    def inverse_transform(self, poses: np.ndarray) -> np.ndarray:
        """Undo normalisation; returns physical-unit vectors."""
        if self.stats is None:
            return poses
        return poses * self.stats.std + self.stats.mean

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _batch_to_approach_frame(self, df: pd.DataFrame) -> np.ndarray:
        cols = df.columns.tolist()

        along_col    = _resolve(cols, _ALONG_TRACK_CANDIDATES)   or _raise("along_track_distance not found")
        lateral_col  = _resolve(cols, _LATERAL_ANGLE_CANDIDATES)  or _raise("lateral_path_angle not found")
        vertical_col = _resolve(cols, _VERTICAL_ANGLE_CANDIDATES) or _raise("vertical_path_angle not found")
        roll_col     = _resolve(cols, _ROLL_CANDIDATES)           or _raise("roll not found")
        pitch_col    = _resolve(cols, _PITCH_CANDIDATES)          or _raise("pitch not found")
        yaw_col      = _resolve(cols, _YAW_CANDIDATES)            or _raise("yaw not found")

        along    = df[along_col].values.astype(np.float32)
        lateral  = _ensure_degrees(df[lateral_col].values.astype(np.float32),  lateral_col)
        vertical = _ensure_degrees(df[vertical_col].values.astype(np.float32), vertical_col)
        rolls    = _ensure_degrees(df[roll_col].values.astype(np.float32),  roll_col)
        pitches  = _ensure_degrees(df[pitch_col].values.astype(np.float32), pitch_col)
        yaws     = _ensure_degrees(df[yaw_col].values.astype(np.float32),   yaw_col)

        return np.column_stack([along, lateral, vertical, rolls, pitches, yaws])


def _raise(msg: str):
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# PoseVolumeSampler
# ---------------------------------------------------------------------------

class PoseVolumeSampler:
    """Convex-hull membership in normalised 6-DOF pose space.

    The hull defines the pose volume of a training domain.  An optional
    ApproachLimits filter can be ANDed with the hull test, but it requires
    the caller to also pass the un-normalised raw poses because the limits
    are in physical units.

    Usage::

        sampler = PoseVolumeSampler(d2_norm, limits=DOMAIN2_LIMITS)

        # full test: hull AND corridor
        mask = sampler.is_inside(d1_norm, raw_poses=d1_raw)

        # hull only (no raw poses available)
        mask = sampler.is_inside(d1_norm)
    """

    def __init__(
        self,
        poses: np.ndarray,
        limits: Optional[ApproachLimits] = None,
    ) -> None:
        """
        Args:
            poses:  (N, 6) normalised 6-DOF pose vectors (from PoseProcessor.transform).
            limits: optional ApproachLimits for physical-unit corridor check.
        """
        self.poses  = poses
        self.limits = limits
        # ConvexHull requires at least d+1 = 7 points in 6-D
        self.hull = ConvexHull(poses) if len(poses) >= 7 else None

    def is_inside(
        self,
        query_norm: np.ndarray,
        raw_poses:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Check hull membership AND (optionally) approach corridor.

        Args:
            query_norm: (M, 6) normalised pose vectors.
            raw_poses:  (M, 6) un-normalised poses in physical LARD units.
                        Required when self.limits is not None; ignored otherwise.
        Returns:
            mask: (M,) boolean array — True = inside volume.
        """
        if self.hull is None:
            return np.zeros(len(query_norm), dtype=bool)

        # Half-space representation: A @ x + b <= 0  (scipy convention)
        A = self.hull.equations[:, :-1]   # (n_facets, 6)
        b = -self.hull.equations[:, -1]   # (n_facets,)
        slack = query_norm @ A.T          # (M, n_facets)
        mask  = np.all(slack <= b[None, :] + 1e-10, axis=1)

        # Physical corridor filter (requires raw un-normalised poses)
        if self.limits is not None and raw_poses is not None:
            mask &= self.limits.is_valid(raw_poses)

        return mask

    def is_outside(
        self,
        query_norm: np.ndarray,
        raw_poses:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return ~self.is_inside(query_norm, raw_poses=raw_poses)

    def pose_distance_to_hull(self, query_poses: np.ndarray) -> np.ndarray:
        """Min Euclidean distance from each query to the nearest training pose.

        Args:
            query_poses: (M, 6) normalised
        Returns:
            distances: (M,) float array
        """
        diff  = query_poses[:, None, :] - self.poses[None, :, :]  # (M, N, 6)
        dists = np.linalg.norm(diff, axis=-1)                      # (M, N)
        return dists.min(axis=1)                                    # (M,)


# ---------------------------------------------------------------------------
# Unit tests (python -m shared_manifold_domain_transfer.data.pose)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== PoseProcessor / ApproachLimits / PoseVolumeSampler unit tests ===\n")

    # ---- Test 1: _batch_to_approach_frame reads correct columns ----------
    df = pd.DataFrame({
        "along_track_distance": [-3000.0, -1500.0, -500.0],
        "lateral_path_angle":   [   -1.0,     0.5,    2.0],
        "vertical_path_angle":  [   -3.5,    -3.0,   -2.5],
        "roll":   [ 0.0,  1.0, -1.0],
        "pitch":  [-3.0, -3.5, -4.0],
        "yaw":    [  0.0,  1.0,  2.0],
    })
    proc = PoseProcessor()
    raw = proc.transform_raw(df)
    assert raw.shape == (3, 6), f"Expected (3,6), got {raw.shape}"
    assert raw[0, 0] == -3000.0, "along_track_distance passthrough failed"
    print(f"  [PASS] _batch_to_approach_frame: shape={raw.shape}")
    print(f"         row[0] = {raw[0]}")

    # ---- Test 2: fit/transform normalises correctly ----------------------
    proc.fit(df)
    normed = proc.transform(df)
    assert normed.shape == (3, 6)
    # After normalisation the mean should be ~0
    assert np.abs(normed.mean(0)).max() < 0.1, "Normalised mean should be near 0"
    print(f"  [PASS] fit/transform: normalised mean ≈ 0")

    # ---- Test 3: inverse_transform recovers raw -------------------------
    recovered = proc.inverse_transform(normed)
    assert np.allclose(recovered, raw, atol=1e-4), "inverse_transform failed"
    print(f"  [PASS] inverse_transform: recovers raw values")

    # ---- Test 4: ApproachLimits.is_valid --------------------------------
    lim = DOMAIN2_LIMITS
    valid = lim.is_valid(raw)
    # row 0: along_track=-3000 (D2 min is -5000, OK), lateral=-1 (OK), vertical=-3.5 (in [-4,-2.5], OK)
    # row 1: along_track=-1500 (OK), lateral=0.5 (OK), vertical=-3.0 (OK)
    # row 2: along_track=-500 (in [-5000,-400] OK), lateral=2.0 (>1.5 — FAIL)
    assert valid[0] and valid[1] and not valid[2], \
        f"ApproachLimits.is_valid gave unexpected result: {valid}"
    print(f"  [PASS] ApproachLimits.is_valid: {valid} (row 2 correctly rejected)")

    # ---- Test 5: DOMAIN1_LIMITS is strictly wider than DOMAIN2_LIMITS ---
    valid_d1 = DOMAIN1_LIMITS.is_valid(raw)
    assert valid_d1.sum() >= valid.sum(), "D1 should accept at least as many as D2"
    print(f"  [PASS] DOMAIN1 ≥ DOMAIN2 acceptance: D1={valid_d1.sum()}, D2={valid.sum()}")

    # ---- Test 6: PoseVolumeSampler hull membership ----------------------
    rng = np.random.default_rng(42)
    training_norm = rng.standard_normal((50, 6)).astype(np.float32)
    sampler = PoseVolumeSampler(training_norm)
    inside_train = sampler.is_inside(training_norm[:5])
    assert inside_train.all(), "Training poses should all be inside their own hull"
    print(f"  [PASS] Hull: {inside_train.sum()}/5 training poses inside (expect 5)")

    far = np.full((1, 6), 100.0, dtype=np.float32)
    assert not sampler.is_inside(far)[0], "Far outlier should be outside hull"
    print(f"  [PASS] Hull: far outlier correctly outside")

    # ---- Test 7: hull + corridor filter ---------------------------------
    sampler2 = PoseVolumeSampler(training_norm, limits=DOMAIN2_LIMITS)
    # Pass identical norm and raw; since raw is synthetic N(0,1), most will
    # fail the physical limits — the main check is that it runs without error
    inside_with_limits = sampler2.is_inside(training_norm, raw_poses=training_norm)
    assert inside_with_limits.sum() <= inside_train.sum()
    print(f"  [PASS] Hull + limits: {inside_with_limits.sum()} ≤ {inside_train.sum()} (limits only restrict)")

    print("\nAll tests passed.")
    sys.exit(0)
