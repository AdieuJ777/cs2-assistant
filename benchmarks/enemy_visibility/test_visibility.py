"""Unit tests for the Dust2 visibility proxy.

Run:
    uv run --with numpy python benchmarks/enemy_visibility/test_visibility.py
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dust2_visibility", HERE / "visibility.py")
visibility = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(visibility)


def synthetic_mask() -> str:
    mask = np.ones((100, 100), dtype=bool)
    mask[50, :] = False  # a wall at x ~= 800-816 for every y
    x_edges = np.arange(0, 1601, 16, dtype=float)
    y_edges = np.arange(0, 1601, 16, dtype=float)
    handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(handle.name, mask=mask, x_edges=x_edges, y_edges=y_edges)
    handle.close()
    return handle.name


class VisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = synthetic_mask()
        cls.model = visibility.Dust2Visibility(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        Path(cls.path).unlink(missing_ok=True)

    def test_wrap_degrees(self) -> None:
        self.assertAlmostEqual(float(visibility.wrap_degrees(190.0)), -170.0)
        self.assertAlmostEqual(float(visibility.wrap_degrees(-190.0)), 170.0)
        self.assertAlmostEqual(float(visibility.wrap_degrees(0.0)), 0.0)

    def test_line_of_sight_open_and_blocked(self) -> None:
        start = np.array([[100.0, 100.0]])
        self.assertTrue(self.model.line_of_sight(start, np.array([[700.0, 100.0]]))[0])
        self.assertFalse(self.model.line_of_sight(start, np.array([[900.0, 100.0]]))[0])

    def test_smoke_blocks_line_of_sight(self) -> None:
        start = np.array([[100.0, 100.0]])
        end = np.array([[700.0, 100.0]])
        smoke = np.array([[400.0, 100.0]])
        self.assertFalse(self.model.line_of_sight(start, end, smoke)[0])

    def test_fov_and_wall_in_visibility_matrix(self) -> None:
        ct_xy = np.array([[100.0, 100.0]])
        t_xy = np.array([[700.0, 100.0], [900.0, 100.0], [100.0, 700.0]])
        visible = self.model.visibility_matrix(ct_xy, np.array([0.0]), t_xy)
        self.assertTrue(visible[0, 0])   # ahead, clear
        self.assertFalse(visible[0, 1])  # ahead, behind the wall
        self.assertFalse(visible[0, 2])  # outside the 50 degree half-FOV

    def test_dead_or_out_of_range(self) -> None:
        ct_xy = np.array([[0.0, 0.0]])
        t_xy = np.array([[visibility.MAX_DISTANCE + 100.0, 0.0]])
        visible = self.model.visibility_matrix(ct_xy, np.array([0.0]), t_xy)
        self.assertFalse(visible[0, 0])


if __name__ == "__main__":
    unittest.main()
