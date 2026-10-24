"""Tests for the LLM output validator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.validator import validate_llm_output  # noqa: E402


def card(*, visible: list | None = None) -> dict:
    return {
        "self": {"area": "Mid"},
        "enemies": {"visible": visible or []},
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


def output(**overrides) -> dict:
    payload = {
        "selected_id": "safe",
        "call": "转 B，走安全路线",
        "reason_short": "B 点有队友",
        "confidence": "high",
        "grounded_facts": ["B 点有 1 名队友"],
    }
    payload.update(overrides)
    return payload


class ValidatorTests(unittest.TestCase):
    def test_valid_output(self) -> None:
        result = validate_llm_output(output(), card(), candidates())
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.output["selected_id"], "safe")
        self.assertEqual(result.output["confidence"], "high")

    def test_unknown_selected_id(self) -> None:
        result = validate_llm_output(output(selected_id="fast"), card(), candidates())
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "unknown_selected_id")

    def test_call_too_long(self) -> None:
        result = validate_llm_output(output(call="转" * 21), card(), candidates())
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "call_too_long")

    def test_reason_too_long(self) -> None:
        result = validate_llm_output(
            output(reason_short="理" * 31), card(), candidates()
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "reason_too_long")

    def test_banned_phrase(self) -> None:
        result = validate_llm_output(
            output(call="敌人一定在 A"), card(visible=[{"area": "A"}]), candidates()
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.reason.startswith("banned_phrase"))

    def test_ungrounded_fact(self) -> None:
        result = validate_llm_output(
            output(grounded_facts=["A 点有 3 名敌人"]), card(), candidates()
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "ungrounded_fact")

    def test_unknown_area_in_call(self) -> None:
        result = validate_llm_output(
            output(call="转 A，走安全路线"), card(), candidates()
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "call_mentions_unknown_area:A")

    def test_visible_enemy_area_only_is_not_a_destination(self) -> None:
        # A visible enemy at A does not make "转 A" legal when no candidate
        # actually rotates to A.
        result = validate_llm_output(
            output(call="A 点有敌人，先别转"),
            card(visible=[{"area": "A"}]),
            candidates(),
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "call_mentions_unknown_area:A")

    def test_candidate_destination_area_is_allowed(self) -> None:
        with_a = candidates() + [
            {
                "id": "rotate_a",
                "action": "ROTATE_A",
                "target_area": "A",
                "call_hint": "转 A，走安全路线",
                "grounded_facts": ["剩余 38 秒"],
            }
        ]
        result = validate_llm_output(
            output(call="转 A，走安全路线"), card(visible=[{"area": "A"}]), with_a
        )
        self.assertTrue(result.ok, result.reason)

    def test_missing_facts(self) -> None:
        result = validate_llm_output(
            output(grounded_facts=[]), card(), candidates()
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "missing_grounded_facts")

    def test_unknown_confidence_defaults_to_medium(self) -> None:
        result = validate_llm_output(
            output(confidence="certain"), card(), candidates()
        )
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.output["confidence"], "medium")

    def test_not_an_object(self) -> None:
        result = validate_llm_output(["safe"], card(), candidates())
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "output_not_object")


if __name__ == "__main__":
    unittest.main()
