"""Tests for the agent's background LLM slow path."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.agent import MidroundAgent  # noqa: E402
from live.midround.llm_planner import LLMConfig  # noqa: E402
from live.midround.route_graph import RouteGraph  # noqa: E402


class Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakePlanner:
    def __init__(
        self,
        *,
        fail: bool = False,
        block: threading.Event | None = None,
        delay: float = 0.0,
    ) -> None:
        self.fail = fail
        self.block = block
        self.delay = delay
        self.calls = 0

    def plan(self, card, candidates):
        self.calls += 1
        if self.block is not None:
            self.block.wait(3.0)
        if self.delay:
            time.sleep(self.delay)
        if self.fail or not candidates:
            return {
                "plan": None,
                "error": "connection_failed",
                "validation": "connection_failed",
                "latency_s": 0.01,
            }
        chosen = candidates[0]
        return {
            "plan": {
                "selected_id": chosen["id"],
                "call": chosen["call_hint"],
                "reason_short": "模型选择理由",
                "confidence": "high",
                "grounded_facts": list(chosen.get("grounded_facts", []))[:1],
            },
            "error": None,
            "validation": "ok",
            "latency_s": 0.01,
        }


def gsi_payload(*, health: int = 100) -> dict:
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
            "state": {"health": health, "armor": 100, "equipment_value": 4200},
            "position": {"x": -346.0, "y": 1345.0},
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


class AgentLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = RouteGraph.load()

    def make_agent(self, planner: FakePlanner) -> tuple[MidroundAgent, Clock]:
        clock = Clock(1000.0)
        config = LLMConfig(
            backend="ollama", model="fake", base_url="http://127.0.0.1:9"
        )
        agent = MidroundAgent(
            self.graph,
            planner=planner,  # type: ignore[arg-type]
            llm_config=config,
            clock=clock,
        )
        return agent, clock

    def update(self, agent, **kwargs) -> None:
        agent.update(
            kwargs.get("gsi", gsi_payload()),
            kwargs.get("radar", radar_payload()),
            kwargs.get("rotation", rotation()),
            seconds_since_freeze=77.0,
            now=kwargs.get("now"),
        )

    def test_llm_plan_replaces_template(self) -> None:
        planner = FakePlanner()
        agent, _ = self.make_agent(planner)
        self.update(agent)
        agent.flush()
        snapshot = agent.plan_snapshot()
        self.assertEqual(planner.calls, 1)
        self.assertEqual(snapshot["plan"]["source"], "llm")
        self.assertEqual(snapshot["plan"]["confidence"], "high")
        self.assertEqual(snapshot["plan"]["reason_short"], "模型选择理由")
        self.assertEqual(snapshot["llm"]["successes"], 1)
        self.assertIsNotNone(snapshot["plan"]["llm_latency_s"])

    def test_llm_failure_keeps_template(self) -> None:
        planner = FakePlanner(fail=True)
        agent, _ = self.make_agent(planner)
        self.update(agent)
        agent.flush()
        snapshot = agent.plan_snapshot()
        self.assertEqual(snapshot["plan"]["source"], "template")
        self.assertEqual(snapshot["llm"]["failures"], 1)
        self.assertEqual(snapshot["llm"]["last_error"], "connection_failed")

    def test_stale_llm_result_is_discarded(self) -> None:
        gate = threading.Event()
        planner = FakePlanner(block=gate)
        agent, clock = self.make_agent(planner)
        self.update(agent)
        # State changes while the slow call is in flight.
        clock.value = 1001.0
        self.update(agent, gsi=gsi_payload(health=25), rotation=rotation())
        gate.set()
        agent.flush()
        snapshot = agent.plan_snapshot()
        self.assertEqual(snapshot["plan"]["source"], "template")
        self.assertGreaterEqual(snapshot["llm"]["discards"], 1)

    def test_no_radar_does_not_call_llm(self) -> None:
        planner = FakePlanner()
        agent, _ = self.make_agent(planner)
        agent.update(gsi_payload(), None, {"action": "NO_RADAR", "warnings": []})
        agent.flush(0.2)
        self.assertEqual(planner.calls, 0)

    def test_hotkey_does_not_wait_for_llm(self) -> None:
        gate = threading.Event()
        planner = FakePlanner(block=gate)
        agent, _ = self.make_agent(planner)
        self.update(agent)
        started = time.perf_counter()
        plan = agent.current_plan(hotkey=True)
        elapsed = time.perf_counter() - started
        gate.set()
        agent.flush()
        self.assertIsNotNone(plan)
        self.assertLess(elapsed, 0.1)
        self.assertEqual(plan["source"], "template")

    def test_configure_llm_can_disable(self) -> None:
        planner = FakePlanner()
        agent, _ = self.make_agent(planner)
        snapshot = agent.configure_llm_from_mapping(
            {
                "backend": "openai",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9/v1",
                "enabled": False,
            }
        )
        self.assertFalse(snapshot["status"]["enabled"])
        self.assertEqual(snapshot["config"]["backend"], "openai")
        self.assertEqual(snapshot["config"]["model"], "test-model")
        self.update(agent)
        agent.flush(0.2)
        self.assertEqual(planner.calls, 0)

    def test_configure_llm_rejects_unknown_backend(self) -> None:
        agent, _ = self.make_agent(FakePlanner())
        with self.assertRaises(ValueError):
            agent.configure_llm_from_mapping({"backend": "magic", "enabled": True})

    def test_llm_status_reports_config(self) -> None:
        agent, _ = self.make_agent(FakePlanner())
        status = agent.llm_status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["backend"], "ollama")
        self.assertEqual(status["model"], "fake")
        self.assertFalse(status["inflight"])


if __name__ == "__main__":
    unittest.main()
