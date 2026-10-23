"""Dust2 radar-visibility proxy for parsed demo ticks.

The demo `spotted` flag is not a usable radar signal (see README), so we
approximate "a CT can see a T" with three pieces of geometry:

- horizontal FOV: the target must be within ``HALF_FOV`` degrees of the CT yaw;
- line of sight: the segment between both players must stay on walkable cells
  of the released Dust2 mask, sampled every ``LOS_STEP`` units;
- smoke occlusion: an active smoke blocks any sample point within
  ``SMOKE_RADIUS`` units.

Coordinates and yaw come from Awpy: X/Y are map units, yaw is degrees in
[-180, 180] with 0 pointing along +X (validated against 1,303 cross-side kills).

This is a lower bound on visibility: it ignores elevation (2D mask), wall
penetration, and utility other than smoke.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAGNET_ROOT = PROJECT_ROOT.parent / "TAGNet-CS2-main"
DEFAULT_MASK = TAGNET_ROOT / "data" / "map" / "dust2_walkable_mask.npz"

MAX_DISTANCE = 4000.0
HALF_FOV = 50.0
LOS_STEP = 8.0
SMOKE_RADIUS = 144.0


def wrap_degrees(degrees: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-180, 180)."""
    return (np.asarray(degrees) + 180.0) % 360.0 - 180.0


class Dust2Visibility:
    """Map-geometry visibility model for de_dust2."""

    def __init__(self, mask_path: Path | str = DEFAULT_MASK) -> None:
        data = np.load(mask_path)
        self.mask = data["mask"].astype(bool)
        self.x_edges = data["x_edges"].astype(float)
        self.y_edges = data["y_edges"].astype(float)
        self.cell_x = float(self.x_edges[1] - self.x_edges[0])
        self.cell_y = float(self.y_edges[1] - self.y_edges[0])

    def _cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((points[..., 0] - self.x_edges[0]) / self.cell_x).astype(int)
        iy = np.floor((points[..., 1] - self.y_edges[0]) / self.cell_y).astype(int)
        return (
            np.clip(ix, 0, self.mask.shape[0] - 1),
            np.clip(iy, 0, self.mask.shape[1] - 1),
        )

    def line_of_sight(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        smokes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Whether each start->end segment is unobstructed.

        Parameters
        ----------
        starts, ends: (R, 2) arrays of X/Y positions.
        smokes: optional (S, 2) centres of smokes active at this tick.
        """
        starts = np.atleast_2d(np.asarray(starts, dtype=float))
        ends = np.atleast_2d(np.asarray(ends, dtype=float))
        if len(starts) == 0:
            return np.zeros(0, dtype=bool)

        distance = np.linalg.norm(ends - starts, axis=1)
        samples = np.maximum(2, np.ceil(distance / LOS_STEP).astype(int) + 1)
        max_samples = int(samples.max())
        fractions = np.arange(max_samples, dtype=float)[None, :] / np.maximum(
            samples - 1, 1
        )[:, None]
        valid = fractions <= 1.0
        points = starts[:, None, :] + fractions[:, :, None] * (ends - starts)[:, None, :]

        ix, iy = self._cells(points)
        walkable = self.mask[ix, iy]
        walkable[~valid] = True  # padding is not an obstacle
        blocked = ~walkable.all(axis=1)

        if smokes is not None and len(smokes) > 0:
            for centre in np.asarray(smokes, dtype=float):
                inside = ((points - centre) ** 2).sum(axis=2) < SMOKE_RADIUS**2
                blocked |= (inside & valid).any(axis=1)
        return ~blocked

    def visibility_matrix(
        self,
        ct_xy: np.ndarray,
        ct_yaw: np.ndarray,
        t_xy: np.ndarray,
        smokes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an (n_ct, n_t) boolean matrix: CT i can see T j at this tick.

        Dead players must be removed by the caller.
        """
        ct_xy = np.asarray(ct_xy, dtype=float).reshape(-1, 2)
        t_xy = np.asarray(t_xy, dtype=float).reshape(-1, 2)
        ct_yaw = np.asarray(ct_yaw, dtype=float).reshape(-1)
        if len(ct_xy) == 0 or len(t_xy) == 0:
            return np.zeros((len(ct_xy), len(t_xy)), dtype=bool)

        delta = t_xy[None, :, :] - ct_xy[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        angle = np.degrees(np.arctan2(delta[..., 1], delta[..., 0]))
        angle_diff = np.abs(wrap_degrees(ct_yaw[:, None] - angle))
        candidate = (distance <= MAX_DISTANCE) & (angle_diff <= HALF_FOV)

        visible = np.zeros_like(candidate)
        ct_idx, t_idx = np.nonzero(candidate)
        if len(ct_idx):
            visible[ct_idx, t_idx] = self.line_of_sight(
                ct_xy[ct_idx], t_xy[t_idx], smokes
            )
        return visible
