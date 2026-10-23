"""Tests for the radar screenshot parser.

These tests need Pillow, numpy and scipy; they are skipped in the stdlib-only
test run. Run them with:

    uv run --with pillow --with numpy --with scipy \
        python -m unittest live.tests.test_radar_parser -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT = PROJECT_ROOT / "data" / "radar_capture" / "ingame_mid.jpg"
CONFIG_PATH = PROJECT_ROOT / "live" / "models" / "radar_config_1280x960.json"
SCREENSHOT_2560 = PROJECT_ROOT / "data" / "radar_capture" / "ingame_2560_clean.jpg"
CONFIG_2560 = PROJECT_ROOT / "live" / "models" / "radar_config_2560x1440.json"
GRID_PATH = PROJECT_ROOT / "live" / "models" / "area_grid.npz"

try:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(PROJECT_ROOT / "live"))
    from radar_parser import (
        classify_area,
        load_area_grid,
        load_config,
        parse_frame,
        pixel_to_world,
    )

    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

HAVE_FIXTURES = SCREENSHOT.exists() and CONFIG_PATH.exists()


@unittest.skipUnless(HAVE_DEPS and HAVE_FIXTURES, "radar fixtures or vision deps missing")
class RadarParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.image = np.asarray(Image.open(SCREENSHOT).convert("RGB"))
        cls.self_world = tuple(cls.config["self_world_at_calibration"])
        cls.parsed = parse_frame(cls.image, cls.config, cls.self_world)

    def test_center_maps_to_self(self) -> None:
        center = self.config["radar_center"]
        x, y = pixel_to_world(center[0], center[1], self.config, self.self_world)
        self.assertAlmostEqual(x, self.self_world[0], places=3)
        self.assertAlmostEqual(y, self.self_world[1], places=3)

    def test_enemy_detected_on_t_side(self) -> None:
        self.assertGreaterEqual(len(self.parsed["enemies"]), 1)
        enemy = self.parsed["enemies"][0]
        self.assertLess(enemy["y"], 1000)
        self.assertIn(enemy["area"], {"A", "B", "Mid", "CT", "other"})

    def test_teammates_detected(self) -> None:
        self.assertGreaterEqual(len(self.parsed["teammates"]), 1)
        for mate in self.parsed["teammates"]:
            self.assertIn(mate["area"], {"A", "B", "Mid", "CT", "other"})
            self.assertEqual(mate["health"], 100)

    def test_area_grid(self) -> None:
        grid = load_area_grid()
        if grid is None:
            self.skipTest("area grid not built")
        # TSpawn is outside the four macro areas in the training labels.
        self.assertIn(classify_area(-760, -795, {}, grid), {"other"})
        self.assertEqual(classify_area(1095, 2617, {}, grid), "A")
        self.assertEqual(classify_area(-1724, 2245, {}, grid), "B")

    def test_label_only_localization_1280(self) -> None:
        config = load_config(CONFIG_PATH)
        image = np.asarray(Image.open(SCREENSHOT).convert("RGB"))
        parsed = parse_frame(image, config, self_world=None)
        self.assertIsNotNone(parsed["self"])
        self.assertEqual(parsed["self"]["source"], "radar_labels")
        expected = config["self_world_at_calibration"]
        distance = (
            (parsed["self"]["x"] - expected[0]) ** 2
            + (parsed["self"]["y"] - expected[1]) ** 2
        ) ** 0.5
        self.assertLess(distance, 100.0)

    def test_static_mode_ignores_self_position(self) -> None:
        static = json.loads(json.dumps(self.config))
        static["mode"] = "static"
        other_self = (-1000.0, 0.0)
        parsed = parse_frame(self.image, static, other_self)
        self.assertEqual(len(parsed["enemies"]), len(self.parsed["enemies"]))
        self.assertAlmostEqual(
            parsed["enemies"][0]["x"], self.parsed["enemies"][0]["x"], places=1
        )


@unittest.skipUnless(
    HAVE_DEPS and SCREENSHOT_2560.exists() and CONFIG_2560.exists(),
    "2560x1440 radar fixtures missing",
)
class RadarParser2560Tests(unittest.TestCase):
    def test_2560_layout(self) -> None:
        config = load_config(CONFIG_2560)
        image = np.asarray(Image.open(SCREENSHOT_2560).convert("RGB"))
        parsed = parse_frame(image, config, tuple(config["self_world_at_calibration"]))
        self.assertGreaterEqual(len(parsed["enemies"]), 1)
        self.assertGreaterEqual(len(parsed["teammates"]), 1)
        for enemy in parsed["enemies"]:
            self.assertIn(enemy["area"], {"A", "B", "Mid", "CT", "other"})
        self.assertEqual(config["radar_roi"]["size"], 486)

    def test_label_only_localization_2560(self) -> None:
        config = load_config(CONFIG_2560)
        image = np.asarray(Image.open(SCREENSHOT_2560).convert("RGB"))
        parsed = parse_frame(image, config, self_world=None)
        self.assertIsNotNone(parsed["self"])
        self.assertEqual(parsed["self"]["source"], "radar_labels")
        expected = config["self_world_at_calibration"]
        distance = (
            (parsed["self"]["x"] - expected[0]) ** 2
            + (parsed["self"]["y"] - expected[1]) ** 2
        ) ** 0.5
        self.assertLess(distance, 100.0)


if __name__ == "__main__":
    unittest.main()
