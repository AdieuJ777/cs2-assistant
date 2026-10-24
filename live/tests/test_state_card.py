"""Tests for the mid-round state card."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.state_card import build_state_card  # noqa: E402


def gsi_payload(
    *,
    phase: str = "live",
    position: dict | None = None,
    health: int = 100,
    seconds_left: float | None = 38.0,
) -> dict:
    payload = {
        "map": {
            "name": "de_dust2",
            "round": 5,
            "team_ct": {"score": 5},
            "team_t": {"score": 3},
        },
        "round": {"phase": phase, "bomb": "carried"},
        "player": {
            "team": "CT",
            "state": {
                "health": health,
                "armor": 100,
                "equipment_value": 4200,
                "defusekit": True,
                "smoke": 1,
                "flashbang": 1,
                "hegrenade": 0,
                "incgrenade": 0,
            },
            "position": position if position is not None else {"x": -346.0, "y": 1345.0},
            "forward": {"x": 1.0, "y": 0.0},
            "weapons": {"weapon_0": {"name": "weapon_m4a1_silencer", "state": "active"}},
        },
    }
    payload["phase_countdowns"] = (
        {"phase_ends_in": str(seconds_left)} if seconds_left is not None else {}
    )
    return payload


def radar_payload(
    *,
    with_enemies: bool = True,
    with_teammates: bool = True,
    teammate_health: bool = True,
    self_x: float | None = None,
    self_y: float | None = None,
) -> dict:
    payload: dict = {
        "enemies": [{"x": 1094.0, "y": 2617.0, "area": "A", "visible": True}]
        if with_enemies
        else [],
        "enemy_alive_count": 5,
        "self_spots_enemy": True,
    }
    if with_teammates:
        mates = [
            {"x": -1724.0, "y": 2245.0, "area": "B"},
            {"x": 1094.0, "y": 2617.0, "area": "A"},
        ]
        if teammate_health:
            mates[0]["health"] = 80
            mates[1]["health"] = 45
        payload["teammates"] = mates
    if self_x is not None and self_y is not None:
        payload["self_x"] = self_x
        payload["self_y"] = self_y
        payload["self_area"] = "Mid"
        payload["self_source"] = "radar_labels"
    return payload


def rotation(action: str = "ROTATE_B") -> dict:
    return {
        "action": action,
        "destination": action,
        "stage1_proba": 0.64,
        "destination_proba": 0.72,
        "threshold": 0.6,
        "warnings": [],
    }


class StateCardTests(unittest.TestCase):
    def test_live_round_card(self) -> None:
        card = build_state_card(
            gsi_payload(), radar_payload(), rotation(), now=1760000000.0, radar_age_s=0.1
        )
        self.assertEqual(card["schema_version"], 1)
        self.assertEqual(card["map"], "de_dust2")
        self.assertEqual(card["side"], "ct")
        self.assertEqual(card["round"]["number"], 5)
        self.assertEqual(card["round"]["phase"], "live")
        self.assertAlmostEqual(card["round"]["time_left"], 38.0)
        self.assertAlmostEqual(card["round"]["seconds_since_freeze"], 77.0)
        self.assertEqual(card["round"]["score_ct"], 5)
        self.assertEqual(card["round"]["score_t"], 3)

        self.assertEqual(card["self"]["area"], "Mid")
        self.assertEqual(card["self"]["health"], 100)
        self.assertEqual(card["self"]["weapon"], "weapon_m4a1_silencer")
        self.assertTrue(card["self"]["defuse_kit"])
        self.assertEqual(card["self"]["utility"]["smoke"], 1)
        self.assertEqual(card["self"]["utility"]["flash"], 1)

        self.assertEqual(card["team"]["alive_count"], 3)
        self.assertEqual(card["team"]["health_source"], "radar")
        self.assertEqual(card["enemies"]["visible_count"], 1)
        self.assertEqual(card["enemies"]["alive_count"], 5)
        self.assertEqual(card["enemies"]["alive_count_source"], "radar")
        self.assertEqual(card["enemies"]["unknown_count"], 4)
        self.assertEqual(card["enemies"]["nearest_area"], "A")
        self.assertEqual(card["model"]["action"], "ROTATE_B")
        self.assertTrue(card["model"]["available"])
        self.assertEqual(card["perception"]["warnings"], [])
        json.dumps(card)  # must be plain JSON-serialisable

    def test_without_radar_marks_defaults(self) -> None:
        card = build_state_card(gsi_payload(), None, rotation("NO_RADAR"))
        self.assertEqual(card["enemies"]["alive_count"], 5)
        self.assertEqual(card["enemies"]["alive_count_source"], "default_5")
        self.assertEqual(card["team"]["health_source"], "default_100")
        self.assertIsNone(card["perception"]["teammates_source"])
        self.assertIn("no_radar", card["perception"]["warnings"])
        self.assertIn("enemy_alive_count_default", card["perception"]["warnings"])
        self.assertFalse(card["model"]["available"])
        self.assertIsNone(card["model"]["action"])

    def test_radar_position_fallback(self) -> None:
        gsi = gsi_payload()
        gsi["player"].pop("position")
        card = build_state_card(
            gsi, radar_payload(self_x=-346.0, self_y=1345.0), rotation()
        )
        self.assertEqual(card["perception"]["self_source"], "radar_labels")
        self.assertEqual(card["self"]["x"], -346.0)
        self.assertEqual(card["self"]["area"], "Mid")

    def test_missing_teammate_health_uses_default(self) -> None:
        card = build_state_card(
            gsi_payload(), radar_payload(teammate_health=False), rotation()
        )
        self.assertEqual(card["team"]["health_source"], "default_100")
        self.assertEqual(
            [member["health"] for member in card["team"]["members"]], [100, 100]
        )
        self.assertIn("teammate_health_default", card["perception"]["warnings"])

    def test_missing_teammates_is_warned(self) -> None:
        card = build_state_card(
            gsi_payload(), radar_payload(with_teammates=False), rotation()
        )
        self.assertEqual(card["team"]["alive_count"], 1)
        self.assertIn("teammate_positions_missing", card["perception"]["warnings"])

    def test_bomb_planted(self) -> None:
        gsi = gsi_payload(seconds_left=18.0)
        gsi["round"] = {"phase": "live", "bomb": "planted", "bomb_site": "A"}
        card = build_state_card(gsi, radar_payload(), rotation())
        self.assertEqual(card["bomb"]["state"], "planted")
        self.assertEqual(card["bomb"]["site"], "A")
        self.assertAlmostEqual(card["bomb"]["timer"], 18.0)

    def test_freeze_phase(self) -> None:
        card = build_state_card(
            gsi_payload(phase="freezetime", seconds_left=15.0), None, rotation("BUY_PHASE")
        )
        self.assertEqual(card["round"]["phase"], "freeze")
        self.assertAlmostEqual(card["round"]["time_left"], 15.0)
        self.assertFalse(card["model"]["available"])

    def test_dead_self(self) -> None:
        card = build_state_card(gsi_payload(health=0), radar_payload(), rotation())
        self.assertFalse(card["self"]["alive"])

    def test_no_position_warns(self) -> None:
        gsi = gsi_payload()
        gsi["player"].pop("position")
        card = build_state_card(gsi, radar_payload(), rotation())
        self.assertIsNone(card["self"]["x"])
        self.assertIn("self_position_missing", card["perception"]["warnings"])
        self.assertIn("self_area_unknown", card["perception"]["warnings"])


if __name__ == "__main__":
    unittest.main()
