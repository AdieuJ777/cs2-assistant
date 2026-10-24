"""LLM selection layer: prompt, HTTP backends, timeout and retry.

The planner is deliberately backend-agnostic. Ollama is the default local
backend; any OpenAI-compatible endpoint (LM Studio, vLLM, llama.cpp server,
OpenAI, etc.) can be plugged in through ``LLMConfig`` and appears in the web
UI. The LLM only picks one candidate and rephrases it; the validator enforces
that it never invents an action or a fact.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Protocol

from .validator import validate_llm_output

MAX_RETRIES = 1
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OPENAI_URL = "http://127.0.0.1:1234/v1"

SYSTEM_PROMPT = """你是 CS2 Dust2 CT 中期指挥。
只能从候选里选择一条。
候选已按内核优先级排序，第一条是内核首选。
默认选择第一条；只有状态卡事实明确表明第一条不可执行或明显更危险时，才选择其他候选。
只能使用状态卡和候选中的事实。
不能声称敌人一定在某处。
输出 JSON。
call 不超过 20 个汉字。
reason_short 不超过 30 个汉字。
grounded_facts 必须逐字复制候选里的原句，不能改写。"""

OUTPUT_SCHEMA = """{
  "selected_id": "候选 id",
  "call": "一句可执行指令，<=20 字",
  "reason_short": "一句理由，<=30 字",
  "confidence": "high|medium|low",
  "grounded_facts": ["从候选 grounded_facts 原样复制的句子"]
}"""


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    backend: str = "ollama"
    model: str = "qwen3:8b"
    base_url: str = DEFAULT_OLLAMA_URL
    api_key: str = ""
    timeout_s: float = 5.0
    temperature: float = 0.2
    enabled: bool = True

    def public(self) -> dict[str, Any]:
        """Config safe to expose to the web UI (never the API key itself)."""
        payload = asdict(self)
        payload["has_api_key"] = bool(self.api_key)
        payload.pop("api_key", None)
        return payload

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], base: "LLMConfig | None" = None
    ) -> "LLMConfig":
        config = replace(base) if base is not None else cls()
        if "backend" in payload:
            config.backend = str(payload["backend"])
        if "model" in payload:
            config.model = str(payload["model"])
        if "base_url" in payload:
            config.base_url = str(payload["base_url"])
        if "api_key" in payload and payload["api_key"] != "":
            config.api_key = str(payload["api_key"])
        if "timeout_s" in payload:
            config.timeout_s = max(0.5, float(payload["timeout_s"]))
        if "temperature" in payload:
            config.temperature = float(payload["temperature"])
        if "enabled" in payload:
            config.enabled = bool(payload["enabled"])
        return config


class ChatBackend(Protocol):
    name: str

    def chat(self, system: str, user: str, *, timeout: float) -> str: ...

    def list_models(self, *, timeout: float = 1.0) -> list[str]: ...


def _post_json(url: str, body: Mapping[str, Any], *, timeout: float, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"connection_failed:{exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid_response_json:{exc}") from exc


class OllamaBackend:
    name = "ollama"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + path

    def chat(self, system: str, user: str, *, timeout: float) -> str:
        data = _post_json(
            self._url("/api/chat"),
            {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": self.config.temperature},
            },
            timeout=timeout,
        )
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("ollama_missing_message") from exc
        if not isinstance(content, str):
            raise LLMError("ollama_non_string_content")
        return content

    def list_models(self, *, timeout: float = 1.0) -> list[str]:
        try:
            with urllib.request.urlopen(self._url("/api/tags"), timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return []
        return sorted(
            str(model.get("name"))
            for model in data.get("models", [])
            if model.get("name")
        )


class OpenAICompatibleBackend:
    name = "openai"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _base(self) -> str:
        base = self.config.base_url.rstrip("/") or DEFAULT_OPENAI_URL
        return base if base.endswith("/v1") else base + "/v1"

    def _headers(self) -> dict[str, str]:
        return (
            {"Authorization": f"Bearer {self.config.api_key}"}
            if self.config.api_key
            else {}
        )

    def chat(self, system: str, user: str, *, timeout: float) -> str:
        data = _post_json(
            self._base() + "/chat/completions",
            {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.config.temperature,
                "stream": False,
            },
            timeout=timeout,
            headers=self._headers(),
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("openai_missing_message") from exc
        if not isinstance(content, str):
            raise LLMError("openai_non_string_content")
        return content

    def list_models(self, *, timeout: float = 1.0) -> list[str]:
        request = urllib.request.Request(
            self._base() + "/models", headers=self._headers()
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return []
        return sorted(
            str(model.get("id"))
            for model in data.get("data", [])
            if model.get("id")
        )


BACKEND_FACTORIES = {
    "ollama": OllamaBackend,
    "openai": OpenAICompatibleBackend,
}


def available_backends() -> list[dict[str, Any]]:
    return [
        {
            "id": "ollama",
            "label": "Ollama（本地）",
            "default_url": DEFAULT_OLLAMA_URL,
            "description": "本地 Ollama /api/chat，默认 qwen3:8b。",
        },
        {
            "id": "openai",
            "label": "OpenAI 兼容接口",
            "default_url": DEFAULT_OPENAI_URL,
            "description": "LM Studio、vLLM、llama.cpp、OpenAI 等 /v1/chat/completions。",
        },
    ]


def build_backend(config: LLMConfig) -> ChatBackend:
    factory = BACKEND_FACTORIES.get(config.backend)
    if factory is None:
        raise ValueError(f"unknown LLM backend {config.backend!r}")
    return factory(config)


def list_backend_models(config: LLMConfig, *, timeout: float = 1.0) -> list[str]:
    try:
        return build_backend(config).list_models(timeout=timeout)
    except (ValueError, LLMError):
        return []


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no json object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("json is not an object")
    return parsed


def build_messages(
    state_card: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[str, str]:
    card_subset = {
        key: state_card.get(key)
        for key in ("round", "self", "team", "enemies", "bomb", "model")
    }
    user = (
        "状态卡:\n"
        + json.dumps(card_subset, ensure_ascii=False)
        + "\n\n候选:\n"
        + json.dumps(list(candidates), ensure_ascii=False)
        + "\n\n输出 JSON，格式:\n"
        + OUTPUT_SCHEMA
    )
    return SYSTEM_PROMPT, user


class LLMPlanner:
    def __init__(self, config: LLMConfig, *, backend: ChatBackend | None = None) -> None:
        self.config = config
        self.backend = backend or build_backend(config)

    def plan(
        self,
        state_card: Mapping[str, Any],
        candidates: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return a plan outcome dict; never raises for LLM/validation failures."""
        candidate_list = list(candidates)
        started = time.perf_counter()
        if not candidate_list:
            return {
                "plan": None,
                "error": "no_candidates",
                "validation": "no_candidates",
                "latency_s": 0.0,
                "attempts": 0,
                "raw": None,
            }

        system, user = build_messages(state_card, candidate_list)
        raw: str | None = None
        last_error = "invalid_json"
        attempts = 0
        for attempt in range(MAX_RETRIES + 1):
            attempts += 1
            try:
                raw = self.backend.chat(
                    system, user, timeout=self.config.timeout_s
                )
            except LLMError as exc:
                return self._outcome(
                    started, None, str(exc), str(exc), attempts, raw
                )
            try:
                parsed = parse_json_object(raw)
            except (ValueError, json.JSONDecodeError):
                last_error = "invalid_json"
                if attempt < MAX_RETRIES:
                    continue
                return self._outcome(
                    started, None, last_error, last_error, attempts, raw
                )

            result = validate_llm_output(parsed, state_card, candidate_list)
            if not result.ok:
                return self._outcome(
                    started, None, result.reason, result.reason, attempts, raw
                )
            return self._outcome(
                started, result.output, None, "ok", attempts, raw
            )

        return self._outcome(started, None, last_error, last_error, attempts, raw)

    @staticmethod
    def _outcome(
        started: float,
        plan: dict[str, Any] | None,
        error: str | None,
        validation: str,
        attempts: int,
        raw: str | None,
    ) -> dict[str, Any]:
        return {
            "plan": plan,
            "error": error,
            "validation": validation,
            "latency_s": round(time.perf_counter() - started, 3),
            "attempts": attempts,
            "raw": raw,
        }
