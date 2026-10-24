"""Mid-round agent: fast template path plus a background LLM slow path.

Fast loop (every GSI/radar update): state card -> legal candidates -> template
plan in the cache. The hotkey path only reads that cache and never waits for a
model.

Slow loop (background thread): the LLM picks one candidate and rephrases it.
A result is applied only if the game state still matches when it arrives; on
timeout, invalid JSON or validation failure the template plan stays in place.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

from .candidate_builder import build_candidates
from .llm_planner import (
    LLMConfig,
    LLMPlanner,
    available_backends,
    list_backend_models,
)
from .logging import PlanLogger
from .plan_cache import FRESH_TTL_S, PlanCache, fingerprint, state_key
from .route_graph import RouteGraph
from .state_card import AREA_LABELS, build_state_card

MIN_LLM_INTERVAL_S = 1.5
LLM_FALLBACK_INTERVAL_S = 2.0
OUTCOME_HORIZONS = (3.0, 5.0)
MAX_PENDING_OUTCOMES = 50
ROTATE_AREAS = {"A", "B", "Mid", "CT"}


def _actual_action(area: str | None, origin_area: str | None) -> str | None:
    if not area:
        return None
    if area == origin_area:
        return "HOLD"
    if area == "Mid":
        return "ROTATE_MID"
    if area in ROTATE_AREAS:
        return f"ROTATE_{area}"
    return "OTHER"


def _area_label(area: str | None) -> str:
    if not area:
        return ""
    return AREA_LABELS.get(area, area)


def _reason_short(card: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    if not card.get("model", {}).get("available"):
        return "信息不足，先保持"
    action = str(candidate.get("action", ""))
    if action == "HOLD":
        return "保持当前站位，等队友"
    if action == "FALL_BACK":
        return "人少或残血，先退保枪"
    if action == "REGROUP":
        return "和队友汇合更安全"
    if action == "WAIT":
        return "先别动，等雷达信息"
    if action.startswith("ROTATE_"):
        return f"转 {_area_label(candidate.get('target_area'))}，避开已知敌人"
    return "按当前状态执行"


def _event_key(card: Mapping[str, Any]) -> tuple[Any, ...]:
    """Cheap event signature used to decide when the LLM is worth calling."""
    enemies = tuple(
        sorted(
            {
                enemy.get("area")
                for enemy in card.get("enemies", {}).get("visible", [])
                if enemy.get("area")
            }
        )
    )
    teammates = tuple(
        sorted(
            f"{member.get('area')}:{member.get('health')}"
            for member in card.get("team", {}).get("members", [])
            if member.get("alive", True)
        )
    )
    return (
        card.get("round", {}).get("number"),
        card.get("round", {}).get("phase"),
        card.get("self", {}).get("area"),
        card.get("self", {}).get("health"),
        enemies,
        teammates,
        card.get("bomb", {}).get("state"),
        card.get("model", {}).get("action"),
    )


class MidroundAgent:
    def __init__(
        self,
        route_graph: RouteGraph | None = None,
        *,
        plan_log: Any = None,
        cache: PlanCache | None = None,
        planner: LLMPlanner | None = None,
        llm_config: LLMConfig | None = None,
        clock: Callable[[], float] = time.time,
        outcome_horizons: tuple[float, ...] = OUTCOME_HORIZONS,
    ) -> None:
        self.graph = route_graph or RouteGraph.load()
        self.cache = cache or PlanCache()
        self.logger = (
            plan_log if isinstance(plan_log, PlanLogger) else PlanLogger(plan_log)
        )
        self.clock = clock

        self._lock = threading.RLock()
        self._card: dict[str, Any] | None = None
        self._candidates: list[dict[str, Any]] = []
        self._plan: dict[str, Any] | None = None
        self._updated_at: float | None = None
        self._last_logged_plan_id: str | None = None
        self._updates = 0
        self._hotkeys = 0
        self._hotkey_seq = 0
        self._last_error: str | None = None

        self.outcome_horizons = tuple(sorted(float(h) for h in outcome_horizons))
        self._pending_outcomes: list[dict[str, Any]] = []
        self._outcomes_logged = 0
        self._outcomes_adopted = 0
        self._outcomes_evaluable = 0

        self.planner = planner
        self.llm_config = llm_config
        self._llm_generation = 0
        self._llm_inflight = False
        self._llm_thread: threading.Thread | None = None
        self._last_llm_at = 0.0
        self._last_llm_event: tuple[Any, ...] | None = None
        self._llm_stats: dict[str, Any] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "discards": 0,
            "last_latency_s": None,
            "last_error": None,
            "last_success_at": None,
        }

    # --------------------------------------------------------------- update
    def update(
        self,
        gsi: Mapping[str, Any] | None,
        radar: Mapping[str, Any] | None,
        rotation: Mapping[str, Any] | None = None,
        *,
        radar_age_s: float | None = None,
        seconds_since_freeze: float | None = None,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Run the fast loop once. Never blocks on the LLM."""
        now = self.clock() if now is None else float(now)
        try:
            card = build_state_card(
                gsi,
                radar,
                rotation,
                radar_age_s=radar_age_s,
                seconds_since_freeze=seconds_since_freeze,
                now=now,
            )
            candidates = build_candidates(card, self.graph)
        except (KeyError, TypeError, ValueError) as exc:  # defensive: never kill the server
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            return None

        current_fingerprint = fingerprint(card)
        with self._lock:
            self._updates += 1
            self._card = card
            self._candidates = candidates
            self._updated_at = now

            cached, age = self.cache.get(current_fingerprint, now=now)
            if cached is not None and age is not None and age <= FRESH_TTL_S:
                plan = cached
            else:
                plan = self._build_template_plan(card, candidates, now)
                if plan is not None:
                    self.cache.put(plan)
                    self._maybe_log_plan(plan, requested=False)
            self._plan = plan
            self._resolve_outcomes_locked(card, now)
            self._maybe_schedule_llm(card, candidates, now)
            return dict(plan) if plan is not None else None

    # ---------------------------------------------------------------- read
    def current_plan(
        self, *, now: float | None = None, hotkey: bool = False
    ) -> dict[str, Any] | None:
        """Return the latest plan immediately; rebuild from the last card if stale."""
        now = self.clock() if now is None else float(now)
        with self._lock:
            if self._card is None:
                return None
            current_fingerprint = fingerprint(self._card)
            cached, age = self.cache.get(current_fingerprint, now=now)
            if cached is None:
                plan = self._build_template_plan(self._card, self._candidates, now)
                if plan is not None:
                    self.cache.put(plan)
                    age = 0.0
            else:
                plan = cached
            if plan is None:
                return None

            plan = dict(plan)
            age = float(age if age is not None else 0.0)
            plan["cache_age_s"] = round(age, 2)
            plan["fresh"] = age <= FRESH_TTL_S
            plan["stale"] = age > FRESH_TTL_S
            if hotkey:
                self._hotkeys += 1
                self._hotkey_seq += 1
                plan["hotkey"] = True
                plan["hotkey_seq"] = self._hotkey_seq
                # A hotkey is a deliberate request, so always log it even if
                # the underlying plan id did not change.
                self.logger.log_plan(
                    plan, self._card, self._candidates, requested=True
                )
                self._track_outcome_locked(plan, now)
            self._plan = dict(plan)
            return plan

    def plan_snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        now = self.clock() if now is None else float(now)
        plan = self.current_plan(now=now, hotkey=False)
        with self._lock:
            return {
                "plan": plan,
                "age_s": plan.get("cache_age_s") if plan else None,
                "source": plan.get("source") if plan else None,
                "fresh": bool(plan.get("fresh")) if plan else False,
                "stale": bool(plan.get("stale")) if plan else False,
                "candidates": [dict(candidate) for candidate in self._candidates],
                "updated_at": self._updated_at,
                "hotkey_seq": self._hotkey_seq,
                "llm": self._llm_status_locked(),
            }

    def refresh(self, *, now: float | None = None) -> dict[str, Any] | None:
        """Debug hook: rebuild the template plan from the latest card immediately."""
        now = self.clock() if now is None else float(now)
        with self._lock:
            if self._card is None:
                return None
            plan = self._build_template_plan(self._card, self._candidates, now)
            if plan is None:
                return None
            self.cache.put(plan)
            self._plan = dict(plan)
            self.logger.log_plan(plan, self._card, self._candidates, requested=True)
            plan["cache_age_s"] = 0.0
            plan["fresh"] = True
            plan["stale"] = False
            return plan

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            age = (
                round(max(0.0, now - self._updated_at), 2)
                if self._updated_at is not None
                else None
            )
            return {
                "has_plan": self._plan is not None,
                "plan_age_s": age,
                "source": self._plan.get("source") if self._plan else None,
                "selected_id": self._plan.get("selected_id") if self._plan else None,
                "candidate_count": len(self._candidates),
                "updates": self._updates,
                "hotkeys": self._hotkeys,
                "hotkey_seq": self._hotkey_seq,
                "last_error": self._last_error,
                "cache": self.cache.status(),
                "logger": self.logger.status(),
                "llm": self._llm_status_locked(),
                "pending_outcomes": len(self._pending_outcomes),
                "outcomes_logged": self._outcomes_logged,
                "outcomes_evaluable": self._outcomes_evaluable,
                "outcomes_adopted": self._outcomes_adopted,
                "adoption_rate": (
                    round(self._outcomes_adopted / self._outcomes_evaluable, 3)
                    if self._outcomes_evaluable
                    else None
                ),
            }

    # ------------------------------------------------------------ LLM control
    def configure_llm(self, config: LLMConfig | None) -> dict[str, Any]:
        """Swap the LLM backend at runtime (used by the web settings panel)."""
        with self._lock:
            self._llm_generation += 1
            self.llm_config = config
            self.planner = (
                LLMPlanner(config) if config is not None and config.enabled else None
            )
            self._llm_inflight = False
            self._last_llm_at = 0.0
            self._last_llm_event = None
            self._llm_stats["last_error"] = None
        # Model discovery can take a moment; never do it under the agent lock
        # because the hotkey path shares that lock.
        return self.llm_config_snapshot()

    def configure_llm_from_mapping(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        config = LLMConfig.from_mapping(payload, self.llm_config)
        if config.backend not in {backend["id"] for backend in available_backends()}:
            raise ValueError(f"unknown backend {config.backend!r}")
        return self.configure_llm(config)

    def llm_config_snapshot(self) -> dict[str, Any]:
        with self._lock:
            config = self.llm_config or (
                getattr(self.planner, "config", None) if self.planner else None
            ) or LLMConfig(enabled=False)
            public = config.public()
            status = self._llm_status_locked()
        return {
            "config": public,
            "available_backends": available_backends(),
            "models": list_backend_models(config, timeout=1.0),
            "status": status,
        }

    def llm_status(self) -> dict[str, Any]:
        with self._lock:
            return self._llm_status_locked()

    def flush(self, timeout: float = 10.0) -> None:
        """Wait for the in-flight LLM call (tests and shutdown only)."""
        with self._lock:
            thread = self._llm_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    # -------------------------------------------------------------- internal
    def _llm_status_locked(self) -> dict[str, Any]:
        config = self.llm_config or (
            getattr(self.planner, "config", None) if self.planner else None
        )
        return {
            "enabled": self.planner is not None,
            "backend": config.backend if config else None,
            "model": config.model if config else None,
            "base_url": config.base_url if config else None,
            "has_api_key": bool(config.api_key) if config else False,
            "inflight": self._llm_inflight,
            **{key: value for key, value in self._llm_stats.items()},
        }

    def _maybe_schedule_llm(
        self,
        card: Mapping[str, Any],
        candidates: list[Mapping[str, Any]],
        now: float,
    ) -> None:
        if self.planner is None or not candidates:
            return
        if not card.get("perception", {}).get("teammates_source"):
            # No radar means the candidate list is only WAIT/HOLD; no reason
            # to spend an LLM call.
            return
        if self._llm_inflight:
            return
        if now - self._last_llm_at < MIN_LLM_INTERVAL_S:
            return

        event = _event_key(card)
        if event == self._last_llm_event and now - self._last_llm_at < LLM_FALLBACK_INTERVAL_S:
            return

        self._last_llm_event = event
        self._last_llm_at = now
        self._llm_inflight = True
        self._llm_stats["calls"] += 1
        generation = self._llm_generation
        stable = state_key(card)
        thread = threading.Thread(
            target=self._run_llm,
            args=(dict(card), [dict(candidate) for candidate in candidates], stable, generation),
            name="midround-llm",
            daemon=True,
        )
        self._llm_thread = thread
        thread.start()

    def _run_llm(
        self,
        card: dict[str, Any],
        candidates: list[dict[str, Any]],
        stable: dict[str, Any],
        generation: int,
    ) -> None:
        try:
            outcome = self.planner.plan(card, candidates)
        except Exception as exc:  # never let a backend bug escape the thread
            outcome = {
                "plan": None,
                "error": f"{type(exc).__name__}: {exc}",
                "validation": "exception",
                "latency_s": 0.0,
            }
        finished = self.clock()
        config = self.llm_config or getattr(self.planner, "config", None)
        self.logger.log_llm_attempt(
            outcome,
            backend=getattr(config, "backend", None),
            model=getattr(config, "model", None),
            round_number=card.get("round", {}).get("number"),
        )

        with self._lock:
            if generation != self._llm_generation:
                # A newer backend/config replaced this call; discard silently
                # and leave _llm_inflight alone for the new generation.
                self._llm_stats["discards"] += 1
                return
            self._llm_inflight = False
            self._llm_stats["last_latency_s"] = outcome.get("latency_s")
            self._llm_stats["last_error"] = outcome.get("error")

            selected = outcome.get("plan")
            if selected is None:
                self._llm_stats["failures"] += 1
                return

            current_card = self._card
            if current_card is None or state_key(current_card) != stable:
                self._llm_stats["discards"] += 1
                self._llm_stats["last_error"] = "state_changed"
                return

            known_ids = {str(candidate.get("id")) for candidate in self._candidates}
            if str(selected.get("selected_id")) not in known_ids:
                self._llm_stats["discards"] += 1
                self._llm_stats["last_error"] = "candidate_gone"
                return

            plan = self._build_llm_plan(current_card, selected, finished)
            if plan is None:
                self._llm_stats["failures"] += 1
                return
            self.cache.put(plan)
            self._plan = dict(plan)
            self._llm_stats["successes"] += 1
            self._llm_stats["last_success_at"] = round(finished, 3)
            self._maybe_log_plan(plan, requested=False)

    def _build_llm_plan(
        self,
        card: Mapping[str, Any],
        selected: Mapping[str, Any],
        now: float,
    ) -> dict[str, Any] | None:
        current_fingerprint = fingerprint(card)
        round_number = card.get("round", {}).get("number")
        bucket = current_fingerprint.get("time_bucket")
        candidate = next(
            (
                candidate
                for candidate in self._candidates
                if str(candidate.get("id")) == str(selected["selected_id"])
            ),
            {},
        )
        return {
            "plan_id": f"{round_number}:{bucket}:{selected['selected_id']}:llm",
            "created_at": round(now, 3),
            "fingerprint": current_fingerprint,
            "source": "llm",
            "selected_id": selected["selected_id"],
            "action": candidate.get("action"),
            "target_area": candidate.get("target_area"),
            "call": selected["call"],
            "reason_short": selected["reason_short"],
            "confidence": selected.get("confidence", "medium"),
            "grounded_facts": list(selected.get("grounded_facts", [])),
            "validation": "ok",
            "cache_age_s": 0.0,
            "llm_latency_s": self._llm_stats.get("last_latency_s"),
        }

    def _build_template_plan(
        self,
        card: Mapping[str, Any],
        candidates: list[Mapping[str, Any]],
        now: float,
    ) -> dict[str, Any] | None:
        if not candidates:
            return None
        selected = dict(candidates[0])
        current_fingerprint = fingerprint(card)
        round_number = card.get("round", {}).get("number")
        bucket = current_fingerprint.get("time_bucket")
        return {
            "plan_id": f"{round_number}:{bucket}:{selected['id']}:template",
            "created_at": round(now, 3),
            "fingerprint": current_fingerprint,
            "source": "template",
            "selected_id": selected["id"],
            "action": selected.get("action"),
            "target_area": selected.get("target_area"),
            "call": selected["call_hint"],
            "reason_short": _reason_short(card, selected),
            "confidence": (
                "medium" if card.get("model", {}).get("available") else "low"
            ),
            "grounded_facts": list(selected.get("grounded_facts", [])),
            "validation": "ok",
            "cache_age_s": 0.0,
            "llm_latency_s": None,
        }

    def _track_outcome_locked(self, plan: Mapping[str, Any], now: float) -> None:
        """Remember a plan the player actually requested via the hotkey."""
        if self._card is None:
            return
        plan_id = str(plan.get("plan_id"))
        if any(pending["plan_id"] == plan_id for pending in self._pending_outcomes):
            return
        pending = {
            "plan_id": plan_id,
            "plan": dict(plan),
            "pressed_at": now,
            "round": self._card.get("round", {}).get("number"),
            "origin_area": self._card.get("self", {}).get("area"),
            "target_area": plan.get("target_area"),
            "horizons": {horizon: False for horizon in self.outcome_horizons},
        }
        self._pending_outcomes.append(pending)
        if len(self._pending_outcomes) > MAX_PENDING_OUTCOMES:
            self._pending_outcomes = self._pending_outcomes[-MAX_PENDING_OUTCOMES:]

    def _resolve_outcomes_locked(
        self, card: Mapping[str, Any], now: float
    ) -> None:
        if not self._pending_outcomes:
            return
        round_number = card.get("round", {}).get("number")
        phase = card.get("round", {}).get("phase")
        alive = bool(card.get("self", {}).get("alive"))
        area = card.get("self", {}).get("area")

        remaining: list[dict[str, Any]] = []
        for pending in self._pending_outcomes:
            elapsed = now - float(pending["pressed_at"])
            if pending["round"] != round_number:
                self._finish_pending_locked(pending, "round_changed", elapsed)
                continue
            if phase != "live":
                self._finish_pending_locked(pending, "not_live", elapsed)
                continue
            if not alive:
                self._finish_pending_locked(pending, "died", elapsed)
                continue

            for horizon, done in list(pending["horizons"].items()):
                if done or elapsed < horizon:
                    continue
                if pending["target_area"] is None or area is None:
                    # The plan was made without a usable position/target, so
                    # adoption cannot be measured. Keep the sample out of the
                    # adoption denominator instead of counting it as a miss.
                    self._write_outcome_locked(
                        pending,
                        horizon=horizon,
                        actual_action=None,
                        actual_area=area,
                        adopted=False,
                        status="no_position",
                        elapsed=elapsed,
                    )
                else:
                    actual_action = _actual_action(area, pending["origin_area"])
                    adopted = area == pending["target_area"]
                    self._write_outcome_locked(
                        pending,
                        horizon=horizon,
                        actual_action=actual_action,
                        actual_area=area,
                        adopted=adopted,
                        status="ok",
                        elapsed=elapsed,
                    )
                pending["horizons"][horizon] = True
            if not all(pending["horizons"].values()):
                remaining.append(pending)
        self._pending_outcomes = remaining

    def _finish_pending_locked(
        self, pending: dict[str, Any], status: str, elapsed: float
    ) -> None:
        for horizon, done in list(pending["horizons"].items()):
            if done:
                continue
            self._write_outcome_locked(
                pending,
                horizon=horizon,
                actual_action=None,
                actual_area=None,
                adopted=False,
                status=status,
                elapsed=elapsed,
            )
            pending["horizons"][horizon] = True

    def _write_outcome_locked(
        self,
        pending: Mapping[str, Any],
        *,
        horizon: float,
        actual_action: str | None,
        actual_area: str | None,
        adopted: bool,
        status: str,
        elapsed: float,
    ) -> None:
        self._outcomes_logged += 1
        if (
            status == "ok"
            and actual_area is not None
            and pending.get("target_area") is not None
        ):
            self._outcomes_evaluable += 1
            if adopted:
                self._outcomes_adopted += 1
        self.logger.log_outcome(
            pending["plan"],
            horizon_s=horizon,
            actual_action=actual_action,
            actual_area=actual_area,
            adopted=adopted,
            status=status,
            elapsed_s=elapsed,
        )

    def _maybe_log_plan(self, plan: Mapping[str, Any], *, requested: bool) -> None:
        plan_id = str(plan.get("plan_id"))
        if not requested and plan_id == self._last_logged_plan_id:
            return
        if self._card is None:
            return
        if self.logger.log_plan(
            plan, self._card, self._candidates, requested=requested
        ):
            self._last_logged_plan_id = plan_id
