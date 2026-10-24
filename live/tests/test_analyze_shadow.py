"""Tests for the plan section of the shadow-log report."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.analyze_shadow import plan_report  # noqa: E402


def records() -> list[dict]:
    return [
        {
            "record_type": "plan",
            "source": "llm",
            "selected_id": "safe",
            "call": "转 B，走安全路线",
            "grounded_facts": ["剩余 38 秒", "B 点有 1 名队友"],
            "llm_latency_s": 3.2,
        },
        {
            "record_type": "plan",
            "source": "template",
            "selected_id": "hold",
            "call": "守 中路，别推",
            "grounded_facts": ["剩余 38 秒"],
            "llm_latency_s": None,
        },
        {
            "record_type": "plan_llm",
            "ok": True,
            "error": None,
            "validation": "ok",
            "latency_s": 3.2,
            "attempts": 1,
        },
        {
            "record_type": "plan_llm",
            "ok": False,
            "error": "connection_failed",
            "validation": "connection_failed",
            "latency_s": 0.1,
            "attempts": 1,
        },
        {
            "record_type": "plan_outcome",
            "plan_source": "llm",
            "horizon_s": 3.0,
            "status": "ok",
            "adopted": True,
            "target_area": "B",
            "actual_area": "B",
        },
        {
            "record_type": "plan_outcome",
            "plan_source": "llm",
            "horizon_s": 5.0,
            "status": "ok",
            "adopted": True,
            "target_area": "B",
            "actual_area": "B",
        },
        {
            "record_type": "plan_outcome",
            "plan_source": "template",
            "horizon_s": 3.0,
            "status": "ok",
            "adopted": False,
            "target_area": "Mid",
            "actual_area": "A",
        },
        {
            "record_type": "plan_outcome",
            "plan_source": "template",
            "horizon_s": 3.0,
            "status": "died",
            "adopted": False,
            "target_area": "Mid",
            "actual_area": None,
        },
    ]


class PlanReportTests(unittest.TestCase):
    def test_summary_counts(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary = plan_report(records())
        self.assertEqual(summary["plans"]["total"], 2)
        self.assertEqual(summary["plans"]["by_source"], {"llm": 1, "template": 1})
        self.assertEqual(summary["attempts"]["ok"], 1)
        self.assertEqual(summary["attempts"]["errors"], {"connection_failed": 1})
        self.assertEqual(summary["outcomes"]["status"]["ok"], 3)
        self.assertIn(3.0, summary["outcomes"]["adoption_by_horizon"])
        self.assertEqual(
            summary["outcomes"]["adoption_by_horizon"][3.0]["adopted"], 1
        )
        self.assertEqual(summary["outcomes"]["adoption_by_horizon"][3.0]["n"], 2)
        output = buffer.getvalue()
        self.assertIn("adoption @3s: 1/2 (50.0%)", output)
        self.assertIn("LLM attempts: ok=1/2", output)

    def test_empty_records(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary = plan_report([{"record_type": "advice"}])
        self.assertEqual(summary["plans"]["total"], 0)
        self.assertEqual(buffer.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
