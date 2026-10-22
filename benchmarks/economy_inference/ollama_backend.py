"""Native Ollama backend for STP.

Ollama's OpenAI-compatible endpoint does not reliably honour ``think: false`` for
Qwen3, so the model spends its whole completion budget on hidden reasoning and
returns an empty ``content``. The native ``/api/chat`` endpoint supports
``think: false`` and ``format: json``, which is what this small adapter uses.

It implements the same two methods STP expects from an LLM backend:
``chat`` and ``structured_chat``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from kkk.errors import ModelCallError, StructuredOutputError
from kkk.llm import LLMResponse, _extract_json_text
from kkk.schema import validate_instance


class OllamaNativeBackend:
    name = "ollama-native"

    def __init__(
        self,
        *,
        model: str = "qwen3:8b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 90.0,
        num_predict: int = 400,
        keep_alive: str = "10m",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_predict = num_predict
        self.keep_alive = keep_alive

    def _post_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelCallError(
                f"Ollama HTTP {exc.code}: {body[:500]}",
                details={"model": self.model, "status": exc.code},
                cause=exc,
            ) from exc
        except Exception as exc:
            raise ModelCallError(
                f"Ollama request failed: {exc}",
                details={"model": self.model, "base_url": self.base_url},
                cause=exc,
            ) from exc

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        # urllib is blocking; keep the STP event loop free.
        return await asyncio.to_thread(self._post_sync, payload)

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
            for message in messages
        ]

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(messages),
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": float(kwargs.get("temperature", 0.0)),
                "num_predict": int(kwargs.get("max_tokens", self.num_predict)),
            },
        }
        data = await self._post(payload)
        content = str(data.get("message", {}).get("content", ""))
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        return LLMResponse(
            content=content,
            model_name=str(data.get("model", self.model)),
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            raw=data,
        )

    async def structured_chat(
        self,
        messages: list[dict[str, Any]],
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        schema_note = (
            "Return exactly one JSON object matching this JSON Schema. "
            "Do not wrap it in markdown and do not add any other text.\n"
            + json.dumps(output_schema, ensure_ascii=False)
        )
        payload_messages = self._messages(messages) + [
            {"role": "system", "content": schema_note}
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": float(kwargs.get("temperature", 0.0)),
                "num_predict": int(kwargs.get("max_tokens", self.num_predict)),
            },
        }
        data = await self._post(payload)
        content = str(data.get("message", {}).get("content", ""))
        try:
            parsed = json.loads(_extract_json_text(content))
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"model returned invalid JSON: {exc}",
                details={"content": content[:1000]},
                cause=exc,
            ) from exc
        if not isinstance(parsed, dict):
            raise StructuredOutputError(
                "model structured output must be a JSON object",
                details={"content": content[:1000]},
            )
        validate_instance(
            output_schema,
            parsed,
            error_cls=StructuredOutputError,
            context="ollama structured output",
        )
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        return LLMResponse(
            content=content,
            model_name=str(data.get("model", self.model)),
            structured=parsed,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            raw=data,
        )
