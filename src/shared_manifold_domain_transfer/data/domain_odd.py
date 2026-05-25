"""Operational Design Domain (ODD) definitions for LARD V2 runway approach.

All values are in the *approach frame* produced by PoseProcessor — i.e.
after the two raw-LARD corrections have been applied:

    along_track_distance : metres, negative  (aircraft behind LTP)
    lateral_path_angle   : degrees, signed   (0 = on centreline)
    vertical_path_angle  : degrees, positive (ILS glideslope magnitude, ~3°)
    roll / pitch / yaw   : degrees           (pitch in aviation convention)

Domain definitions
------------------
DOMAIN1_LIMITS  — full LARD V2 ODD; XPlane 12 data covers this volume.
DOMAIN2_LIMITS  — tighter nominal corridor for MSFS data.
                  MSFS images outside D2 but inside D1 form the holdout split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ApproachLimits:
    """Physical approach-corridor limits in approach-frame units.

    Used both as a hard filter (``is_valid``) and to draw ODD corridor boxes
    in visualisations.  Values are in the approach frame, not raw LARD units.
    """
    along_track_range:  Tuple[float, float] = (-6000.0, -280.0)
    max_lateral_deg:    float               = 3.0
    vertical_deg_range: Tuple[float, float] = (1.8, 5.2)

    # Convenience accessors for plot helpers --------------------------------

    @property
    def lateral_range(self) -> Tuple[float, float]:
        """Symmetric lateral corridor as a (min, max) tuple."""
        return (-self.max_lateral_deg, self.max_lateral_deg)

    # Mask ------------------------------------------------------------------

    def is_valid(self, raw_poses: np.ndarray) -> np.ndarray:
        """Return (N,) boolean mask: True where all corridor limits are met.

        Args:
            raw_poses: (N, 6) approach-frame array
                       columns: [along_track, lateral, vertical, roll, pitch, yaw]
        """
        along    = raw_poses[:, 0]
        lateral  = raw_poses[:, 1]
        vertical = raw_poses[:, 2]
        return (
            (along   >= self.along_track_range[0])
            & (along <= self.along_track_range[1])
            & (np.abs(lateral) <= self.max_lateral_deg)
            & (vertical >= self.vertical_deg_range[0])
            & (vertical <= self.vertical_deg_range[1])
        )


# ---------------------------------------------------------------------------
# Named domain instances
# ---------------------------------------------------------------------------

DOMAIN1_LIMITS = ApproachLimits(
    along_track_range  = (-3000.0, -280.0),
    max_lateral_deg    = 3.0,
    vertical_deg_range = (1.8, 5.2),
)

DOMAIN2_LIMITS = ApproachLimits(
    along_track_range  = (-2500.0, -280.0),
    max_lateral_deg    = 1.5,
    vertical_deg_range = (2.5, 3.5),
)
