"""End-to-end tests for the /plan and /agent/* HTTP routes."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "live" / "models" / "rotation_policy.json"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "live"))

try:
    from live.gsi_server import AppState, Handler  # noqa: E402
    from live.rotation_policy import RotationAdvisor  # noqa: E402
except ImportError:  # pragma: no cover
    AppState = None  # type: ignore
    Handler = None  # type: ignore
    RotationAdvisor = None  # type: ignore

TOKEN = "test-token"


def gsi_payload() -> dict:
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
            "position": {"x": -346.0, "y": 1345.0},
            "forward": {"x": 1.0, "y": 0.0},
        },
        "phase_countdowns": {"phase_ends_in": "38.0"},
        "auth": {"token": TOKEN},
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
        "auth": {"token": TOKEN},
    }


@unittest.skipUnless(
    MODEL_PATH.exists() and AppState is not None and Handler is not None,
    "rotation model or server missing",
)
class MidroundServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log_path = Path(cls.tmp.name) / "shadow.jsonl"
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.state = AppState(  # type: ignore[attr-defined]
            RotationAdvisor.load(MODEL_PATH),
            shadow_log=cls.log_path,
        )
        cls.server.token = TOKEN  # type: ignore[attr-defined]
        cls.server.verbose = False  # type: ignore[attr-defined]
        cls.base = "http://127.0.0.1:%d" % cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

        cls.post("/", gsi_payload())
        cls.post("/radar", radar_payload())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)
        cls.tmp.cleanup()

    @classmethod
    def post(cls, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            cls.base + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    @classmethod
    def get(cls, path: str) -> dict:
        with urllib.request.urlopen(cls.base + path, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_plan_endpoint(self) -> None:
        data = self.get("/plan")
        self.assertIsNotNone(data["plan"])
        self.assertEqual(data["source"], "template")
        self.assertIsInstance(data["age_s"], (int, float))
        self.assertLessEqual(len(data["plan"]["call"]), 20)
        self.assertGreaterEqual(len(data["candidates"]), 1)

    def test_hotkey_endpoint_returns_plan(self) -> None:
        data = self.post("/agent/hotkey", {"auth": {"token": TOKEN}})
        self.assertTrue(data["ok"])
        self.assertIsNotNone(data["plan"])
        self.assertTrue(data["plan"]["hotkey"])

    def test_hotkey_sequence_is_visible_to_the_page(self) -> None:
        before = self.get("/plan")["hotkey_seq"]
        self.post("/agent/hotkey", {"auth": {"token": TOKEN}})
        after = self.get("/plan")["hotkey_seq"]
        self.assertGreater(after, before)

    def test_agent_status_endpoint(self) -> None:
        data = self.get("/agent/status")
        self.assertTrue(data["enabled"])
        self.assertTrue(data["has_plan"])
        self.assertIn("hotkeys", data)
        self.assertIn("cache", data)

    def test_refresh_endpoint(self) -> None:
        data = self.post("/agent/refresh", {"auth": {"token": TOKEN}})
        self.assertTrue(data["ok"])
        self.assertIsNotNone(data["plan"])
        self.assertEqual(data["plan"]["source"], "template")

    def test_page_contains_token_and_plan_card(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=2.0) as response:
            html = response.read().decode("utf-8")
        self.assertNotIn("__TOKEN_JSON__", html)
        self.assertIn(f'const TOKEN = "{TOKEN}"', html)
        self.assertIn('id="plan"', html)
        self.assertIn('id="hotkeybtn"', html)
        self.assertIn('id="llmbackend"', html)
        self.assertIn('id="llmapply"', html)

    def test_llm_config_endpoint(self) -> None:
        data = self.get("/agent/config")
        self.assertIn("config", data)
        backends = {backend["id"] for backend in data["available_backends"]}
        self.assertEqual(backends, {"ollama", "openai"})
        self.assertIn("status", data)

    def test_llm_config_can_switch_and_disable(self) -> None:
        data = self.post(
            "/agent/config",
            {
                "auth": {"token": TOKEN},
                "backend": "openai",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9/v1",
                "timeout_s": 12,
                "enabled": False,
            },
        )
        self.assertTrue(data["ok"])
        self.assertFalse(data["status"]["enabled"])
        self.assertEqual(data["config"]["backend"], "openai")
        self.assertEqual(data["config"]["model"], "test-model")
        self.assertEqual(data["config"]["timeout_s"], 12)

    def test_llm_config_rejects_unknown_backend(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post(
                "/agent/config",
                {"auth": {"token": TOKEN}, "backend": "magic", "enabled": True},
            )
        self.assertEqual(context.exception.code, 400)
        context.exception.close()

    def test_bad_token_is_rejected(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post("/agent/hotkey", {"auth": {"token": "wrong"}})
        self.assertEqual(context.exception.code, 403)
        context.exception.close()

    def test_plan_records_are_logged(self) -> None:
        records = [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        plan_records = [r for r in records if r.get("record_type") == "plan"]
        self.assertTrue(plan_records)
        self.assertIn("state_card", plan_records[0])
        self.assertIn("candidates", plan_records[0])


if __name__ == "__main__":
    unittest.main()
