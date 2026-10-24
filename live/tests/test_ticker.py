"""Tests for the ticker text formatting (no Tk window is created)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.ticker import format_dashboard, format_plan  # noqa: E402


class TickerFormatTests(unittest.TestCase):
    def test_no_plan(self) -> None:
        call, meta, seq = format_plan({"plan": None, "hotkey_seq": 3})
        self.assertEqual(call, "暂无指令")
        self.assertIn("等待", meta)
        self.assertEqual(seq, 3)

    def test_llm_plan(self) -> None:
        payload = {
            "plan": {
                "call": "转 B，走安全路线",
                "source": "llm",
                "grounded_facts": ["剩余 38 秒"],
            },
            "age_s": 0.4,
            "hotkey_seq": 7,
        }
        call, meta, seq = format_plan(payload)
        self.assertEqual(call, "转 B，走安全路线")
        self.assertIn("模型建议", meta)
        self.assertIn("0.4s", meta)
        self.assertIn("剩余 38 秒", meta)
        self.assertEqual(seq, 7)

    def test_template_stale(self) -> None:
        payload = {
            "plan": {
                "call": "守 中路，别推",
                "source": "template",
                "grounded_facts": [],
            },
            "age_s": 4.0,
            "stale": True,
            "hotkey_seq": 2,
        }
        _call, meta, _seq = format_plan(payload)
        self.assertIn("快速建议", meta)
        self.assertIn("已过期", meta)

    def test_empty_payload(self) -> None:
        call, meta, seq = format_plan(None)
        self.assertEqual(call, "暂无指令")
        self.assertEqual(seq, 0)


class DashboardTests(unittest.TestCase):
    def freeze_state(self) -> dict:
        return {
            "connected": True,
            "advice": {
                "round": 3,
                "score": {"ct": 1, "t": 2},
                "phase": "freezetime",
                "seconds_left": 15,
                "money": 5000,
                "recommendation": "FULL_BUY",
                "estimated_cost": 4200,
                "reserve": 800,
                "shopping_list": [
                    {"name": "Kevlar + Helmet", "cost": 1000},
                    {"name": "Smoke", "cost": 300},
                ],
            },
            "rotation": None,
            "radar_age_s": 0.2,
        }

    def test_disconnected(self) -> None:
        dashboard = format_dashboard({"connected": False}, None)
        self.assertEqual(dashboard["headline"], "等待 GSI")
        self.assertEqual(dashboard["details"], [])

    def test_freeze_shows_buy_advice(self) -> None:
        dashboard = format_dashboard(self.freeze_state(), None)
        self.assertEqual(dashboard["headline"], "全买")
        self.assertIn("4200", dashboard["meta"])
        self.assertIn("800", dashboard["meta"])
        joined = "\n".join(dashboard["details"])
        self.assertIn("回合 3", joined)
        self.assertIn("冻结期", joined)
        self.assertIn("雷达 0.2s", joined)
        self.assertIn("购买 护甲+头盔 $1000", joined)
        self.assertIn("烟雾弹 $300", joined)

    def test_live_shows_plan_and_llm(self) -> None:
        state = self.freeze_state()
        state["advice"]["phase"] = "live"
        state["advice"]["recommendation"] = "BUY_PHASE_OVER"
        state["rotation"] = {"action": "ROTATE_B", "confidence": "high"}
        plan_payload = {
            "plan": {
                "call": "转 B，走安全路线",
                "source": "llm",
                "reason_short": "B 点有队友",
                "grounded_facts": ["剩余 38 秒"],
            },
            "age_s": 0.5,
            "hotkey_seq": 4,
            "llm": {
                "enabled": True,
                "backend": "ollama",
                "model": "qwen3:8b",
                "last_latency_s": 3.2,
            },
        }
        dashboard = format_dashboard(state, plan_payload)
        self.assertEqual(dashboard["headline"], "转 B，走安全路线")
        self.assertIn("模型建议", dashboard["meta"])
        self.assertEqual(dashboard["hotkey_seq"], 4)
        joined = "\n".join(dashboard["details"])
        self.assertIn("旋转 ROTATE_B", joined)
        self.assertIn("LLM ollama/qwen3:8b", joined)

    def test_live_without_plan(self) -> None:
        state = self.freeze_state()
        state["advice"]["phase"] = "live"
        state["rotation"] = {"action": "NO_POSITION", "reason": "无法定位"}
        dashboard = format_dashboard(state, {"plan": None})
        self.assertEqual(dashboard["headline"], "暂无指令")
        self.assertIn("无法定位", dashboard["meta"])

    def test_template_plan_source(self) -> None:
        state = self.freeze_state()
        state["advice"]["phase"] = "live"
        plan_payload = {
            "plan": {
                "call": "守 中路，别推",
                "source": "template",
                "grounded_facts": [],
            },
            "age_s": 1.0,
        }
        dashboard = format_dashboard(state, plan_payload)
        self.assertIn("快速建议", dashboard["meta"])


if __name__ == "__main__":
    unittest.main()
