"""Tests for the stdlib rotation policy and live advisor."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "live" / "models" / "rotation_policy.json"
sys.path.insert(0, str(PROJECT_ROOT / "live"))

from rotation_policy import (  # noqa: E402
    RotationAdvisor,
    RotationPolicy,
    state_to_features,
)

try:
    from live.gsi_server import AppState  # noqa: E402
except ImportError:  # pragma: no cover
    AppState = None  # type: ignore


def base_state() -> dict:
    return {
        "map": "de_dust2",
        "side": "ct",
        "phase": "live",
        "round_num": 5,
        "round_phase": "first_half",
        "score_diff": 1,
        "seconds_since_freeze": 20.0,
        "seconds_since_previous_event": 3.0,
        "self": {"x": 0, "y": 1500, "yaw": 0, "health": 100, "armor": 100, "equip": 4000},
        "teammates": [
            {"x": -500, "y": 200, "health": 90},
            {"x": 900, "y": 1800, "health": 100},
        ],
        "enemies": [{"x": 1200, "y": 1600, "area": "A"}],
        "self_spots_enemy": True,
        "enemy_alive_count": 3,
        "newly_visible_count": 1,
    }


@unittest.skipUnless(MODEL_PATH.exists(), "rotation policy artifact not exported")
class RotationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = RotationPolicy.load(MODEL_PATH)
        cls.advisor = RotationAdvisor.load(MODEL_PATH)

    def test_tree_traversal_matches_sklearn(self) -> None:
        for example in self.policy.artifact["parity_examples"]:
            result = self.policy.predict_encoded(example["encoded"])
            self.assertAlmostEqual(
                result["stage1_proba"], example["stage1_proba"], places=9
            )
            self.assertEqual(
                self.policy.artifact["actions"].index(result["destination_argmax"]),
                example["stage2_action"],
            )

    def test_state_features(self) -> None:
        features = state_to_features(self.policy.artifact, base_state())
        self.assertEqual(features["ct_alive_count"], 3)
        self.assertEqual(features["count_A"] + features["count_B"] + features["count_Mid"] + features["count_CT"], 3)
        self.assertEqual(features["enemy_visible_count"], 1)
        self.assertEqual(features["enemy_unknown_count"], 2)
        self.assertEqual(features["enemy_visible_A"], 1)
        self.assertAlmostEqual(features["enemy_nearest_distance"], math.hypot(1200, 100))
        self.assertIn(features["nearest_area"], {"A", "B", "Mid", "CT"})

    def test_advise_with_radar(self) -> None:
        result = self.advisor.advise(base_state())
        self.assertTrue(result["radar_available"])
        self.assertIn(result["action"], {"HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT"})
        self.assertIn(result["confidence"], {"low", "medium", "high"})
        self.assertIsNotNone(result["stage1_proba"])

    def test_no_radar_is_not_guessed(self) -> None:
        state = base_state()
        state["enemies"] = []
        result = self.advisor.advise(state)
        self.assertEqual(result["action"], "NO_RADAR")
        self.assertFalse(result["radar_available"])

    def test_non_dust2_and_freeze(self) -> None:
        state = base_state()
        state["map"] = "de_mirage"
        self.assertEqual(self.advisor.advise(state)["action"], "NOT_DUST2")
        state = base_state()
        state["phase"] = "freeze"
        self.assertEqual(self.advisor.advise(state)["action"], "BUY_PHASE")

    def test_unknown_area_falls_back(self) -> None:
        state = base_state()
        state["enemies"] = [{"x": 1200, "y": 1600, "area": "Spawn"}]
        result = self.advisor.advise(state)
        self.assertTrue(result["radar_available"])
        self.assertIn(result["action"], {"HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT"})


def gsi_payload(phase: str = "live", x: float = 0.0, y: float = 1500.0) -> dict:
    return {
        "map": {
            "name": "de_dust2",
            "round": 5,
            "team_ct": {"score": 3},
            "team_t": {"score": 2},
        },
        "round": {"phase": phase},
        "player": {
            "team": "CT",
            "state": {"health": 100, "armor": 100, "money": 4000, "equipment_value": 4000},
            "position": {"x": x, "y": y},
            "forward": {"x": 1, "y": 0},
        },
        "phase_countdowns": {"phase_ends_in": "95.0"},
    }


def radar_payload(
    with_teammates: bool = True,
    with_enemies: bool = True,
    self_x: float | None = None,
    self_y: float | None = None,
) -> dict:
    payload = {
        "enemies": [{"x": 1200, "y": 1600, "area": "A", "visible": True}]
        if with_enemies
        else [],
        "enemy_alive_count": 3,
        "self_spots_enemy": True,
    }
    if self_x is not None and self_y is not None:
        payload["self_x"] = self_x
        payload["self_y"] = self_y
        payload["self_area"] = "Mid"
        payload["self_source"] = "radar_labels"
    if with_teammates:
        payload["teammates"] = [
            {"x": -500, "y": 200, "health": 90},
            {"x": 900, "y": 1800, "health": 100},
        ]
    return payload


@unittest.skipUnless(MODEL_PATH.exists() and AppState is not None, "artifact missing")
class ServerIntegrationTests(unittest.TestCase):
    def test_live_round_with_radar(self) -> None:
        state = AppState(RotationAdvisor.load(MODEL_PATH), shadow_log=None)
        state.update(gsi_payload())
        state.update_radar(radar_payload())
        rotation = state.snapshot()["rotation"]
        self.assertIsNotNone(rotation)
        self.assertTrue(rotation["radar_available"])
        self.assertIn(
            rotation["action"],
            {"HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT"},
        )

    def test_live_round_without_radar(self) -> None:
        state = AppState(RotationAdvisor.load(MODEL_PATH), shadow_log=None)
        state.update(gsi_payload())
        rotation = state.snapshot()["rotation"]
        self.assertEqual(rotation["action"], "NO_RADAR")

    def test_seconds_since_freeze_from_countdown(self) -> None:
        state = AppState(RotationAdvisor.load(MODEL_PATH), shadow_log=None)
        state.update(gsi_payload())
        state.update_radar(radar_payload())
        rotation = state.snapshot()["rotation"]
        self.assertAlmostEqual(
            rotation["features"]["seconds_since_freeze"], 20.0, places=1
        )

    def test_seconds_since_freeze_falls_back_to_local_timer(self) -> None:
        state = AppState(RotationAdvisor.load(MODEL_PATH), shadow_log=None)
        payload = gsi_payload()
        payload["phase_countdowns"] = {}
        state.update(payload)
        state.update_radar(radar_payload())
        rotation = state.snapshot()["rotation"]
        # The local timer starts at the first live phase, not at 115.
        self.assertLess(rotation["features"]["seconds_since_freeze"], 5.0)

    def test_radar_self_fallback_when_gsi_has_no_position(self) -> None:
        state = AppState(RotationAdvisor.load(MODEL_PATH), shadow_log=None)
        state.update(gsi_payload(x=0.0, y=0.0))
        state.update_radar(radar_payload(self_x=-73.0, self_y=943.0))
        rotation = state.snapshot()["rotation"]
        self.assertIsNotNone(rotation)
        self.assertNotEqual(rotation["action"], "NO_POSITION")
        self.assertTrue(rotation["radar_available"])

    def test_missing_teammates_is_refused(self) -> None:
        state = AppState(RotationAdvisor.load(MODEL_PATH), shadow_log=None)
        state.update(gsi_payload())
        state.update_radar(radar_payload(with_teammates=False))
        rotation = state.snapshot()["rotation"]
        self.assertEqual(rotation["action"], "NO_TEAMMATE_DATA")

    def test_outcome_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "shadow.jsonl"
            state = AppState(
                RotationAdvisor.load(MODEL_PATH),
                shadow_log=log_path,
                outcome_horizon=10.0,
            )
            state.update(gsi_payload())
            state.update_radar(radar_payload())
            self.assertTrue(state.pending_outcomes)

            # Pretend the advice is old enough, then move the player to A.
            state.pending_outcomes[0]["created"] -= 11.0
            state.update(gsi_payload(x=1095.0, y=2617.0))

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            types = {record.get("record_type") for record in records}
            self.assertIn("advice", types)
            self.assertIn("outcome", types)
            outcomes = [
                record
                for record in records
                if record.get("record_type") == "outcome"
                and record.get("status") == "ok"
            ]
            self.assertTrue(outcomes)
            self.assertIsNotNone(outcomes[0]["actual_action"])
            self.assertIn(outcomes[0]["actual_area"], {"A", "B", "Mid", "CT", "other"})

    def test_freeze_phase_uses_buy_advice(self) -> None:
        state = AppState(RotationAdvisor.load(MODEL_PATH), shadow_log=None)
        state.update(gsi_payload(phase="freezetime"))
        state.update_radar(radar_payload())
        rotation = state.snapshot()["rotation"]
        self.assertEqual(rotation["action"], "BUY_PHASE")


if __name__ == "__main__":
    unittest.main()
