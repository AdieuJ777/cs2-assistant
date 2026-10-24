"""End-to-end offline check: radar screenshot -> state card -> candidates.

The vision dependencies are optional. Run with:

    uv run --with pillow --with numpy --with scipy \
        python -m unittest live.tests.test_midround_integration -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT = PROJECT_ROOT / "data" / "radar_capture" / "ingame_mid.jpg"
CONFIG_PATH = PROJECT_ROOT / "live" / "models" / "radar_config_1280x960.json"

sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(PROJECT_ROOT / "live"))
    from radar_parser import load_config, parse_frame

    from live.midround.candidate_builder import build_candidates, validate_candidate
    from live.midround.route_graph import RouteGraph
    from live.midround.state_card import build_state_card

    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

HAVE_FIXTURES = SCREENSHOT.exists() and CONFIG_PATH.exists()


def simulated_gsi(x: float, y: float) -> dict:
    return {
        "map": {
            "name": "de_dust2",
            "round": 5,
            "team_ct": {"score": 5},
            "team_t": {"score": 3},
        },
        "round": {"phase": "live", "bomb": "carried"},
        "player": {
            "team": "CT",
            "state": {"health": 100, "armor": 100, "equipment_value": 4200},
            "position": {"x": x, "y": y},
            "forward": {"x": 1.0, "y": 0.0},
        },
        "phase_countdowns": {"phase_ends_in": "38.0"},
    }


@unittest.skipUnless(HAVE_DEPS and HAVE_FIXTURES, "radar fixtures or vision deps missing")
class MidroundIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = RouteGraph.load()
        cls.config = load_config(CONFIG_PATH)
        image = np.asarray(Image.open(SCREENSHOT).convert("RGB"))
        self_world = tuple(cls.config["self_world_at_calibration"])
        cls.parsed = parse_frame(image, cls.config, self_world)
        cls.card = build_state_card(
            simulated_gsi(*self_world),
            {
                "teammates": cls.parsed["teammates"],
                "enemies": cls.parsed["enemies"],
                "enemy_alive_count": cls.parsed["enemy_alive_count"],
                "self_area": cls.parsed["self"]["area"],
                "self_x": self_world[0],
                "self_y": self_world[1],
            },
            {
                "action": "HOLD",
                "stage1_proba": 0.4,
                "destination": None,
                "destination_proba": 0.6,
                "threshold": 0.6,
                "warnings": [],
            },
            now=1760000000.0,
            radar_age_s=0.2,
        )

    def test_card_uses_parsed_radar(self) -> None:
        self.assertIsNotNone(self.card["self"]["area"])
        self.assertGreaterEqual(self.card["enemies"]["visible_count"], 1)
        self.assertGreaterEqual(self.card["team"]["alive_count"], 1)
        self.assertEqual(self.card["perception"]["warnings"], [])

    def test_candidates_are_legal(self) -> None:
        candidates = build_candidates(self.card, self.graph)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertLessEqual(len(candidates), 4)
        for candidate in candidates:
            ok, reason = validate_candidate(candidate, self.card, self.graph)
            self.assertTrue(ok, f"{candidate['id']}: {reason}")
            self.assertLessEqual(len(candidate["call_hint"]), 20)


if __name__ == "__main__":
    unittest.main()
