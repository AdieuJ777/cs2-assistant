"""Tests for the LLM planner using a fake chat backend (no Ollama needed)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.llm_planner import (  # noqa: E402
    LLMConfig,
    LLMError,
    LLMPlanner,
    build_messages,
    parse_json_object,
)


class FakeBackend:
    name = "fake"

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.last_user = ""

    def chat(self, system: str, user: str, *, timeout: float) -> str:
        self.calls += 1
        self.last_user = user
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def state_card() -> dict:
    return {
        "round": {"number": 5, "phase": "live", "time_left": 38.0},
        "self": {"area": "Mid", "health": 100},
        "team": {"alive_count": 3},
        "enemies": {"visible": [{"area": "A"}], "alive_count": 5},
        "bomb": {"state": "carried"},
        "model": {"action": "ROTATE_B", "available": True},
        "perception": {"warnings": [], "teammates_source": "radar"},
    }


def candidates() -> list[dict]:
    return [
        {
            "id": "safe",
            "action": "ROTATE_B",
            "target_area": "B",
            "call_hint": "转 B，走安全路线",
            "grounded_facts": ["剩余 38 秒", "B 点有 1 名队友"],
        },
        {
            "id": "hold",
            "action": "HOLD",
            "target_area": "Mid",
            "call_hint": "守 中路，别推",
            "grounded_facts": ["剩余 38 秒"],
        },
    ]


def valid_output(**overrides) -> dict:
    payload = {
        "selected_id": "safe",
        "call": "转 B，走安全路线",
        "reason_short": "B 点有队友",
        "confidence": "high",
        "grounded_facts": ["B 点有 1 名队友"],
    }
    payload.update(overrides)
    return payload


def planner(responses: list) -> tuple[LLMPlanner, FakeBackend]:
    backend = FakeBackend(responses)
    config = LLMConfig(backend="ollama", timeout_s=0.5)
    return LLMPlanner(config, backend=backend), backend


class LLMPlannerTests(unittest.TestCase):
    def test_successful_selection(self) -> None:
        llm, backend = planner([json.dumps(valid_output())])
        result = llm.plan(state_card(), candidates())
        self.assertIsNotNone(result["plan"])
        self.assertEqual(result["plan"]["selected_id"], "safe")
        self.assertEqual(result["validation"], "ok")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(backend.calls, 1)
        self.assertIn("候选", backend.last_user)
        self.assertIn("safe", backend.last_user)

    def test_invalid_json_retries_once(self) -> None:
        llm, backend = planner(["not json", json.dumps(valid_output())])
        result = llm.plan(state_card(), candidates())
        self.assertIsNotNone(result["plan"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(backend.calls, 2)

    def test_invalid_json_twice_falls_back(self) -> None:
        llm, backend = planner(["nope", "still nope"])
        result = llm.plan(state_card(), candidates())
        self.assertIsNone(result["plan"])
        self.assertEqual(result["error"], "invalid_json")
        self.assertEqual(backend.calls, 2)

    def test_connection_failure_does_not_retry(self) -> None:
        llm, backend = planner([LLMError("connection_failed: timed out")])
        result = llm.plan(state_card(), candidates())
        self.assertIsNone(result["plan"])
        self.assertIn("connection_failed", result["error"])
        self.assertEqual(backend.calls, 1)

    def test_validation_failure_falls_back(self) -> None:
        llm, _ = planner([json.dumps(valid_output(selected_id="missing"))])
        result = llm.plan(state_card(), candidates())
        self.assertIsNone(result["plan"])
        self.assertEqual(result["error"], "unknown_selected_id")

    def test_markdown_fence_is_accepted(self) -> None:
        text = "```json\n" + json.dumps(valid_output()) + "\n```"
        llm, _ = planner([text])
        result = llm.plan(state_card(), candidates())
        self.assertIsNotNone(result["plan"])

    def test_no_candidates_skips_backend(self) -> None:
        llm, backend = planner([])
        result = llm.plan(state_card(), [])
        self.assertIsNone(result["plan"])
        self.assertEqual(result["error"], "no_candidates")
        self.assertEqual(backend.calls, 0)

    def test_prompt_includes_facts_and_schema(self) -> None:
        system, user = build_messages(state_card(), candidates())
        self.assertIn("只能从候选里选择一条", system)
        self.assertIn("grounded_facts", user)
        self.assertIn("B 点有 1 名队友", user)


class JsonParsingTests(unittest.TestCase):
    def test_fenced_and_noisy_json(self) -> None:
        parsed = parse_json_object('prefix ```json\n{"a": 1}\n``` suffix')
        self.assertEqual(parsed, {"a": 1})

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_json_object("[1, 2]")

    def test_rejects_missing_braces(self) -> None:
        with self.assertRaises(ValueError):
            parse_json_object("nothing here")


if __name__ == "__main__":
    unittest.main()
