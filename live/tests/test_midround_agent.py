"""Tests for the M2 fast-loop agent and template plan logging."""

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


def gsi_payload(*, health: int = 100, phase: str = "live") -> dict:
    return {
        "map": {
            "name": "de_dust2",
            "round": 5,
            "team_ct": {"score": 5},
            "team_t": {"score": 3},
        },
        "round": {"phase": phase, "bomb": "carried"},
        "player": {
            "team": "CT",
            "state": {"health": health, "armor": 100, "equipment_value": 4200},
            "position": {"x": -346.0, "y": 1345.0},
            "forward": {"x": 1.0, "y": 0.0},
        },
        "phase_countdowns": {"phase_ends_in": "38.0"},
    }


def radar_payload(*, with_enemies: bool = True) -> dict:
    return {
        "teammates": [
            {"x": -1724.0, "y": 2245.0, "health": 80, "area": "B"},
            {"x": 1094.0, "y": 2617.0, "health": 45, "area": "A"},
        ],
        "enemies": (
            [{"x": 1094.0, "y": 2617.0, "area": "A", "visible": True}]
            if with_enemies
            else []
        ),
        "enemy_alive_count": 5,
        "self_spots_enemy": True,
    }


def rotation(action: str = "ROTATE_B") -> dict:
    return {
        "action": action,
        "destination": action,
        "stage1_proba": 0.64,
        "destination_proba": 0.72,
        "threshold": 0.6,
        "warnings": [],
    }


class MidroundAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = RouteGraph.load()

    def make_agent(self, log_path=None, clock_value: float = 1000.0):
        clock = Clock(clock_value)
        agent = MidroundAgent(self.graph, plan_log=log_path, clock=clock)
        return agent, clock

    def test_update_builds_template_plan(self) -> None:
        agent, _ = self.make_agent()
        plan = agent.update(
            gsi_payload(),
            radar_payload(),
            rotation(),
            radar_age_s=0.1,
            seconds_since_freeze=77.0,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan["source"], "template")
        self.assertEqual(plan["validation"], "ok")
        self.assertIsNone(plan["llm_latency_s"])
        self.assertEqual(plan["plan_id"], "5:15:safe:template")
        candidates = agent.plan_snapshot()["candidates"]
        self.assertEqual(plan["selected_id"], candidates[0]["id"])
        self.assertEqual(plan["call"], candidates[0]["call_hint"])
        self.assertTrue(plan["grounded_facts"])

    def test_plan_snapshot_reports_age(self) -> None:
        agent, clock = self.make_agent()
        agent.update(
            gsi_payload(),
            radar_payload(),
            rotation(),
            seconds_since_freeze=77.0,
            now=1000.0,
        )
        clock.value = 1001.0
        snapshot = agent.plan_snapshot()
        self.assertIsNotNone(snapshot["plan"])
        self.assertAlmostEqual(snapshot["age_s"], 1.0)
        self.assertTrue(snapshot["fresh"])
        self.assertFalse(snapshot["stale"])
        self.assertEqual(snapshot["source"], "template")

    def test_hotkey_returns_cached_plan_immediately(self) -> None:
        agent, clock = self.make_agent()
        agent.update(gsi_payload(), radar_payload(), rotation(), seconds_since_freeze=77.0)
        clock.value = 1000.5
        plan = agent.current_plan(hotkey=True)
        self.assertIsNotNone(plan)
        self.assertTrue(plan["hotkey"])
        self.assertLessEqual(plan["cache_age_s"], 1.0)
        self.assertEqual(agent.status()["hotkeys"], 1)

    def test_hotkey_sequence_increments(self) -> None:
        agent, _ = self.make_agent()
        agent.update(gsi_payload(), radar_payload(), rotation(), seconds_since_freeze=77.0)
        self.assertEqual(agent.plan_snapshot()["hotkey_seq"], 0)
        first = agent.current_plan(hotkey=True)
        second = agent.current_plan(hotkey=True)
        self.assertEqual(first["hotkey_seq"], 1)
        self.assertEqual(second["hotkey_seq"], 2)
        self.assertEqual(agent.plan_snapshot()["hotkey_seq"], 2)

    def test_stale_plan_is_rebuilt(self) -> None:
        agent, clock = self.make_agent()
        first = agent.update(
            gsi_payload(), radar_payload(), rotation(), seconds_since_freeze=77.0
        )
        self.assertIsNotNone(first)
        clock.value = 1006.0
        rebuilt = agent.current_plan()
        self.assertIsNotNone(rebuilt)
        self.assertEqual(rebuilt["created_at"], 1006.0)
        self.assertTrue(rebuilt["fresh"])
        self.assertAlmostEqual(rebuilt["cache_age_s"], 0.0)

    def test_state_change_changes_fingerprint(self) -> None:
        agent, _ = self.make_agent()
        first = agent.update(
            gsi_payload(), radar_payload(), rotation(), seconds_since_freeze=77.0
        )
        second = agent.update(
            gsi_payload(health=25),
            radar_payload(),
            rotation(),
            seconds_since_freeze=77.0,
            now=1000.1,
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first["fingerprint"]["self_health"], 100)
        self.assertEqual(second["fingerprint"]["self_health"], 25)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_no_plan_outside_live_round(self) -> None:
        agent, _ = self.make_agent()
        plan = agent.update(
            gsi_payload(phase="freezetime"),
            radar_payload(),
            rotation("BUY_PHASE"),
            seconds_since_freeze=0.0,
        )
        self.assertIsNone(plan)
        self.assertIsNone(agent.plan_snapshot()["plan"])

    def test_refresh_rebuilds_plan(self) -> None:
        agent, clock = self.make_agent()
        agent.update(gsi_payload(), radar_payload(), rotation(), seconds_since_freeze=77.0)
        clock.value = 1002.0
        refreshed = agent.refresh()
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed["created_at"], 1002.0)

    def test_plan_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "shadow.jsonl"
            agent, clock = self.make_agent(log_path)
            agent.update(
                gsi_payload(), radar_payload(), rotation(), seconds_since_freeze=77.0
            )
            clock.value = 1001.0
            agent.current_plan(hotkey=True)
            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["record_type"] == "plan" for record in records))
            self.assertFalse(records[0]["requested"])
            self.assertTrue(records[1]["requested"])
            self.assertIn("state_card", records[0])
            self.assertIn("candidates", records[0])

    def test_hotkey_path_p95_under_100ms(self) -> None:
        import statistics
        import time

        agent, _ = self.make_agent()
        agent.update(gsi_payload(), radar_payload(), rotation(), seconds_since_freeze=77.0)
        samples = []
        for _ in range(200):
            started = time.perf_counter()
            agent.current_plan()
            samples.append((time.perf_counter() - started) * 1000.0)
        samples.sort()
        p95 = samples[int(len(samples) * 0.95)]
        self.assertLess(p95, 100.0)

    def test_status(self) -> None:
        agent, _ = self.make_agent()
        agent.update(gsi_payload(), radar_payload(), rotation(), seconds_since_freeze=77.0)
        status = agent.status()
        self.assertTrue(status["has_plan"])
        self.assertEqual(status["source"], "template")
        self.assertEqual(status["updates"], 1)
        self.assertGreaterEqual(status["candidate_count"], 2)
        self.assertIsNone(status["last_error"])


if __name__ == "__main__":
    unittest.main()
