"""Tests for 3 s / 5 s plan-outcome backfill."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.agent import MidroundAgent  # noqa: E402
from live.midround.route_graph import RouteGraph  # noqa: E402


class Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


MID = {"x": -346.0, "y": 1345.0}
B_SITE = {"x": -1724.0, "y": 2245.0}


def gsi_payload(*, health: int = 100, round_number: int = 5, position=None) -> dict:
    return {
        "map": {
            "name": "de_dust2",
            "round": round_number,
            "team_ct": {"score": 5},
            "team_t": {"score": 3},
        },
        "round": {"phase": "live", "bomb": "carried"},
        "player": {
            "team": "CT",
            "state": {"health": health, "armor": 100, "equipment_value": 4200},
            "position": dict(position or MID),
            "forward": {"x": 1.0, "y": 0.0},
        },
        "phase_countdowns": {"phase_ends_in": "38.0"},
    }


def radar_payload() -> dict:
    return {
        "teammates": [
            {"x": -1724.0, "y": 2245.0, "health": 80, "area": "B"},
            {"x": 1094.0, "y": 2617.0, "health": 45, "area": "A"},
        ],
        "enemies": [{"x": 1094.0, "y": 2617.0, "area": "A", "visible": True}],
        "enemy_alive_count": 5,
        "self_spots_enemy": True,
    }


def rotation() -> dict:
    return {
        "action": "ROTATE_B",
        "destination": "ROTATE_B",
        "stage1_proba": 0.64,
        "destination_proba": 0.72,
        "threshold": 0.6,
        "warnings": [],
    }


class PlanOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = RouteGraph.load()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp.name) / "shadow.jsonl"
        self.clock = Clock(1000.0)
        self.agent = MidroundAgent(
            self.graph, plan_log=self.log_path, clock=self.clock
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def update(self, **kwargs) -> None:
        self.agent.update(
            kwargs.get("gsi", gsi_payload()),
            radar_payload(),
            rotation(),
            seconds_since_freeze=77.0,
            now=self.clock.value,
        )

    def records(self, record_type: str) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [
            record
            for record in (
                json.loads(line)
                for line in self.log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if record.get("record_type") == record_type
        ]

    def test_adopted_rotation_logged_at_both_horizons(self) -> None:
        self.update()
        plan = self.agent.current_plan(hotkey=True)
        self.assertEqual(plan["target_area"], "B")
        self.assertEqual(self.agent.status()["pending_outcomes"], 1)

        self.clock.value = 1003.1
        self.update(gsi=gsi_payload(position=B_SITE))
        self.clock.value = 1005.1
        self.update(gsi=gsi_payload(position=B_SITE))

        outcomes = self.records("plan_outcome")
        self.assertEqual(len(outcomes), 2)
        self.assertEqual({record["horizon_s"] for record in outcomes}, {3.0, 5.0})
        for record in outcomes:
            self.assertEqual(record["actual_area"], "B")
            self.assertEqual(record["actual_action"], "ROTATE_B")
            self.assertTrue(record["adopted"])
            self.assertEqual(record["status"], "ok")
        status = self.agent.status()
        self.assertEqual(status["outcomes_logged"], 2)
        self.assertEqual(status["outcomes_evaluable"], 2)
        self.assertEqual(status["outcomes_adopted"], 2)
        self.assertEqual(status["adoption_rate"], 1.0)
        self.assertEqual(status["pending_outcomes"], 0)

    def test_not_adopted_when_player_stays(self) -> None:
        self.update()
        self.agent.current_plan(hotkey=True)
        self.clock.value = 1003.1
        self.update()
        outcomes = self.records("plan_outcome")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["actual_action"], "HOLD")
        self.assertFalse(outcomes[0]["adopted"])
        status = self.agent.status()
        self.assertEqual(status["outcomes_evaluable"], 1)
        self.assertEqual(status["adoption_rate"], 0.0)

    def test_plan_without_position_is_not_evaluable(self) -> None:
        no_position = gsi_payload()
        no_position["player"].pop("position")
        radar = radar_payload()
        radar.pop("self_area", None)
        self.agent.update(
            no_position,
            radar,
            {"action": "NO_POSITION", "warnings": []},
            seconds_since_freeze=77.0,
            now=1000.0,
        )
        self.agent.current_plan(hotkey=True)
        self.clock.value = 1003.1
        self.update(gsi=gsi_payload(position=B_SITE))
        outcomes = self.records("plan_outcome")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "no_position")
        self.assertFalse(outcomes[0]["adopted"])
        status = self.agent.status()
        self.assertEqual(status["outcomes_evaluable"], 0)
        self.assertIsNone(status["adoption_rate"])

    def test_death_closes_pending_outcomes(self) -> None:
        self.update()
        self.agent.current_plan(hotkey=True)
        self.clock.value = 1001.0
        self.update(gsi=gsi_payload(health=0))
        outcomes = self.records("plan_outcome")
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(record["status"] == "died" for record in outcomes))
        self.assertTrue(all(not record["adopted"] for record in outcomes))
        self.assertEqual(self.agent.status()["pending_outcomes"], 0)

    def test_round_change_closes_pending_outcomes(self) -> None:
        self.update()
        self.agent.current_plan(hotkey=True)
        self.clock.value = 1001.0
        self.update(gsi=gsi_payload(round_number=6))
        outcomes = self.records("plan_outcome")
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(
            all(record["status"] == "round_changed" for record in outcomes)
        )

    def test_duplicate_hotkey_does_not_duplicate_pending(self) -> None:
        self.update()
        self.agent.current_plan(hotkey=True)
        self.agent.current_plan(hotkey=True)
        self.assertEqual(self.agent.status()["pending_outcomes"], 1)

    def test_llm_plan_carries_action_and_target(self) -> None:
        self.update()
        plan = self.agent.current_plan()
        self.assertEqual(plan["action"], "ROTATE_B")
        self.assertEqual(plan["target_area"], "B")


if __name__ == "__main__":
    unittest.main()
