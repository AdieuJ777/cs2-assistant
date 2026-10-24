"""Tests for the pluggable LLM backends and config mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import live.midround.llm_planner as llm_planner  # noqa: E402
from live.midround.llm_planner import (  # noqa: E402
    DEFAULT_OLLAMA_URL,
    LLMConfig,
    OllamaBackend,
    OpenAICompatibleBackend,
    available_backends,
    build_backend,
)


class OllamaBackendTests(unittest.TestCase):
    def test_chat_body_uses_json_mode_and_thinking_off(self) -> None:
        config = LLMConfig(backend="ollama", model="qwen3:8b")
        backend = OllamaBackend(config)
        captured: dict = {}

        def fake_post(url, body, *, timeout, headers=None):
            captured.update(url=url, body=body, timeout=timeout, headers=headers)
            return {"message": {"content": "{}"}}

        with patch.object(llm_planner, "_post_json", side_effect=fake_post):
            content = backend.chat("system", "user", timeout=2.0)

        self.assertEqual(content, "{}")
        self.assertTrue(captured["url"].endswith("/api/chat"))
        self.assertFalse(captured["body"]["stream"])
        self.assertFalse(captured["body"]["think"])
        self.assertEqual(captured["body"]["format"], "json")
        self.assertEqual(captured["body"]["model"], "qwen3:8b")


class OpenAICompatibleBackendTests(unittest.TestCase):
    def test_base_url_gets_v1_suffix(self) -> None:
        config = LLMConfig(
            backend="openai", model="local", base_url="http://127.0.0.1:1234"
        )
        backend = OpenAICompatibleBackend(config)
        captured: dict = {}

        def fake_post(url, body, *, timeout, headers=None):
            captured.update(url=url, body=body, headers=headers)
            return {"choices": [{"message": {"content": "{}"}}]}

        with patch.object(llm_planner, "_post_json", side_effect=fake_post):
            content = backend.chat("s", "u", timeout=2.0)

        self.assertEqual(content, "{}")
        self.assertEqual(captured["url"], "http://127.0.0.1:1234/v1/chat/completions")
        self.assertNotIn("think", captured["body"])

    def test_api_key_becomes_bearer_header(self) -> None:
        config = LLMConfig(
            backend="openai",
            model="local",
            base_url="http://127.0.0.1:1234/v1",
            api_key="secret",
        )
        backend = OpenAICompatibleBackend(config)
        captured: dict = {}

        def fake_post(url, body, *, timeout, headers=None):
            captured.update(headers=headers)
            return {"choices": [{"message": {"content": "{}"}}]}

        with patch.object(llm_planner, "_post_json", side_effect=fake_post):
            backend.chat("s", "u", timeout=2.0)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")


class LLMConfigTests(unittest.TestCase):
    def test_public_hides_api_key(self) -> None:
        config = LLMConfig(api_key="secret")
        public = config.public()
        self.assertNotIn("api_key", public)
        self.assertTrue(public["has_api_key"])

    def test_from_mapping_keeps_existing_key_when_blank(self) -> None:
        base = LLMConfig(api_key="secret")
        config = LLMConfig.from_mapping({"api_key": "", "model": "m2"}, base)
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.model, "m2")

    def test_from_mapping_can_disable(self) -> None:
        config = LLMConfig.from_mapping({"enabled": False}, LLMConfig())
        self.assertFalse(config.enabled)

    def test_backend_registry(self) -> None:
        ids = {backend["id"] for backend in available_backends()}
        self.assertEqual(ids, {"ollama", "openai"})
        self.assertIsInstance(
            build_backend(LLMConfig(backend="ollama")), OllamaBackend
        )
        self.assertIsInstance(
            build_backend(LLMConfig(backend="openai")), OpenAICompatibleBackend
        )

    def test_default_ollama_url(self) -> None:
        self.assertEqual(LLMConfig().base_url, DEFAULT_OLLAMA_URL)


if __name__ == "__main__":
    unittest.main()
