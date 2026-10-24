"""Append-only JSONL records for mid-round plans and their outcomes.

The format matches the existing ``live/shadow_log.jsonl`` convention: one
JSON object per line, ``record_type`` first. Failures are swallowed so a bad
disk path can never take the live advisor down.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class PlanLogger:
    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path is not None and str(path).lower() != "none" else None
        self._lock = threading.Lock()
        self._write_failures = 0

    def enabled(self) -> bool:
        return self.path is not None

    def log_plan(
        self,
        plan: Mapping[str, Any],
        state_card: Mapping[str, Any],
        candidates: list[Mapping[str, Any]],
        *,
        requested: bool = False,
    ) -> bool:
        record = {
            "record_type": "plan",
            "ts": datetime.now(timezone.utc).isoformat(),
            "plan_id": plan.get("plan_id"),
            "round": state_card.get("round", {}).get("number"),
            "fingerprint": plan.get("fingerprint"),
            "source": plan.get("source"),
            "selected_id": plan.get("selected_id"),
            "action": plan.get("action"),
            "target_area": plan.get("target_area"),
            "call": plan.get("call"),
            "reason_short": plan.get("reason_short"),
            "confidence": plan.get("confidence"),
            "grounded_facts": plan.get("grounded_facts"),
            "validation": plan.get("validation"),
            "cache_age_s": plan.get("cache_age_s"),
            "llm_latency_s": plan.get("llm_latency_s"),
            "requested": bool(requested),
            "candidates": [dict(candidate) for candidate in candidates],
            "state_card": dict(state_card),
        }
        return self._write(record)

    def log_llm_attempt(
        self,
        outcome: Mapping[str, Any],
        *,
        backend: str | None,
        model: str | None,
        round_number: int | None,
    ) -> bool:
        record = {
            "record_type": "plan_llm",
            "ts": datetime.now(timezone.utc).isoformat(),
            "round": round_number,
            "backend": backend,
            "model": model,
            "ok": outcome.get("plan") is not None,
            "error": outcome.get("error"),
            "validation": outcome.get("validation"),
            "latency_s": outcome.get("latency_s"),
            "attempts": outcome.get("attempts"),
        }
        return self._write(record)

    def log_outcome(
        self,
        plan: Mapping[str, Any],
        *,
        horizon_s: float,
        actual_action: str | None,
        actual_area: str | None,
        adopted: bool,
        status: str = "ok",
        elapsed_s: float | None = None,
    ) -> bool:
        record = {
            "record_type": "plan_outcome",
            "ts": datetime.now(timezone.utc).isoformat(),
            "plan_id": plan.get("plan_id"),
            "plan_source": plan.get("source"),
            "selected_id": plan.get("selected_id"),
            "call": plan.get("call"),
            "action": plan.get("action"),
            "target_area": plan.get("target_area"),
            "horizon_s": horizon_s,
            "actual_action": actual_action,
            "actual_area": actual_area,
            "adopted": bool(adopted),
            "status": status,
            "elapsed_s": round(elapsed_s, 2) if elapsed_s is not None else None,
        }
        return self._write(record)

    def _write(self, record: Mapping[str, Any]) -> bool:
        if self.path is None:
            return False
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError:
            with self._lock:
                self._write_failures += 1
            return False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self.path) if self.path is not None else None,
                "write_failures": self._write_failures,
            }
