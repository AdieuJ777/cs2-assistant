"""Plan cache with a state fingerprint and a short freshness window.

The hotkey path must never wait for an LLM. It reads this cache; the agent
writes a plan here whenever the game state changes. Thread-safe and standard
library only.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

FRESH_TTL_S = 2.0
MAX_AGE_S = 4.0
TIME_BUCKET_S = 5.0


def _time_bucket(card: Mapping[str, Any]) -> int:
    """Bucket of seconds-since-freeze; used to refresh the plan periodically."""
    seconds = card.get("round", {}).get("seconds_since_freeze")
    if seconds is None:
        time_left = card.get("round", {}).get("time_left")
        if time_left is not None:
            seconds = 115.0 - float(time_left)
    if seconds is None:
        return -1
    return int(float(seconds) // TIME_BUCKET_S)


def fingerprint(card: Mapping[str, Any]) -> dict[str, Any]:
    """The exact state signature that invalidates a cached plan."""
    visible = sorted(
        {
            str(enemy["area"])
            for enemy in card.get("enemies", {}).get("visible", [])
            if enemy.get("area")
        }
    )
    teammates = sorted(
        f"{member.get('area')}:{member.get('health')}"
        for member in card.get("team", {}).get("members", [])
        if member.get("alive", True)
    )
    return {
        "round": card.get("round", {}).get("number"),
        "phase": card.get("round", {}).get("phase"),
        "time_bucket": _time_bucket(card),
        "self_area": card.get("self", {}).get("area"),
        "self_health": card.get("self", {}).get("health"),
        "visible_enemies": visible,
        "teammates": teammates,
        "bomb": card.get("bomb", {}).get("state"),
        "model_action": card.get("model", {}).get("action"),
    }


def state_key(card: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint without the periodic time bucket.

    Used to decide whether a slow LLM answer still describes the same situation
    when it arrives. The time bucket intentionally does not invalidate it.
    """
    return {
        key: value
        for key, value in fingerprint(card).items()
        if key != "time_bucket"
    }


class PlanCache:
    """One-entry cache: a plan is valid only for its exact fingerprint."""

    def __init__(self, *, fresh_ttl_s: float = FRESH_TTL_S, max_age_s: float = MAX_AGE_S) -> None:
        self.fresh_ttl_s = float(fresh_ttl_s)
        self.max_age_s = float(max_age_s)
        self._lock = threading.Lock()
        self._plan: dict[str, Any] | None = None
        self._put_count = 0
        self._hit_count = 0
        self._miss_count = 0

    def put(self, plan: Mapping[str, Any]) -> None:
        entry = dict(plan)
        entry.setdefault("fingerprint", {})
        with self._lock:
            self._plan = entry
            self._put_count += 1

    def get(
        self, current_fingerprint: Mapping[str, Any], *, now: float | None = None
    ) -> tuple[dict[str, Any] | None, float | None]:
        """Return ``(plan, age)`` for a fresh or stale-but-valid entry.

        A missing entry, a changed fingerprint, or an entry older than
        ``max_age_s`` returns ``(None, age_or_None)``.
        """
        now = time.time() if now is None else float(now)
        with self._lock:
            plan = self._plan
            if plan is None:
                self._miss_count += 1
                return None, None
            age = max(0.0, now - float(plan.get("created_at", now)))
            if age > self.max_age_s:
                self._plan = None
                self._miss_count += 1
                return None, age
            if plan.get("fingerprint") != dict(current_fingerprint):
                self._miss_count += 1
                return None, age
            self._hit_count += 1
            return dict(plan), age

    def latest(self, *, now: float | None = None) -> tuple[dict[str, Any] | None, float | None]:
        """Return the cached plan regardless of fingerprint, if it is not too old."""
        now = time.time() if now is None else float(now)
        with self._lock:
            plan = self._plan
            if plan is None:
                return None, None
            age = max(0.0, now - float(plan.get("created_at", now)))
            if age > self.max_age_s:
                self._plan = None
                return None, age
            return dict(plan), age

    def invalidate(self) -> None:
        with self._lock:
            self._plan = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "has_plan": self._plan is not None,
                "puts": self._put_count,
                "hits": self._hit_count,
                "misses": self._miss_count,
                "fresh_ttl_s": self.fresh_ttl_s,
                "max_age_s": self.max_age_s,
            }
