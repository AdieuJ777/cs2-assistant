"""Validate the LLM's choice and wording against the candidate list.

The LLM may only pick an existing candidate and copy facts that the state card
already grounded. Anything else is rejected so the agent can fall back to the
kernel template instead of showing an invented call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

MAX_CALL_CHARS = 20
MAX_REASON_CHARS = 30
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
BANNED_PHRASES = (
    "敌人一定在",
    "一定在",
    "肯定是",
    "肯定在",
    "百分之",
    "100%",
    "必定",
    "绝对会",
)
_AREA_WORDS = {
    "A": ("A点", "A 点", "A包点", "A 包点", "A大", "A 大", "A小", "A 小", "A坡", "A 坡"),
    "B": ("B点", "B 点", "B包点", "B 包点", "B门", "B 门", "B洞", "B 洞"),
    "Mid": ("中路", "中门", "猫道", "中路上"),
    "CT": ("警家",),
}


@dataclass
class ValidationResult:
    ok: bool
    reason: str
    output: dict[str, Any] | None = None


def _candidate_facts(candidates: Iterable[Mapping[str, Any]]) -> set[str]:
    facts: set[str] = set()
    for candidate in candidates:
        facts.update(str(fact) for fact in candidate.get("grounded_facts", []))
    return facts


def _allowed_areas(
    card: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]
) -> set[str]:
    """Areas a call may name: the current area plus candidate destinations.

    A visible enemy area is *not* enough on its own: a call saying "转 A" must
    have a candidate that actually rotates to A. The reason string may still
    mention where the enemy was seen.
    """
    allowed: set[str] = set()
    self_area = card.get("self", {}).get("area")
    if self_area:
        allowed.add(str(self_area))
    for candidate in candidates:
        target = candidate.get("target_area")
        if target:
            allowed.add(str(target))
    return allowed


def _mentions_area(text: str, area: str) -> bool:
    if any(word in text for word in _AREA_WORDS.get(area, ())):
        return True
    if area in {"A", "B"}:
        return re.search(rf"(?<![A-Za-z]){area}(?![A-Za-z])", text) is not None
    if area == "CT":
        return "CT" in text.upper()
    if area == "Mid":
        return "Mid" in text
    return False


def _banned_phrase(text: str) -> str | None:
    for phrase in BANNED_PHRASES:
        if phrase in text:
            return phrase
    return None


def validate_llm_output(
    output: Any,
    card: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
) -> ValidationResult:
    """Return ``ValidationResult(ok=True, output=normalised)`` or a reason."""
    candidate_list = list(candidates)
    if not isinstance(output, Mapping):
        return ValidationResult(False, "output_not_object")

    known_ids = {str(candidate.get("id")) for candidate in candidate_list}
    selected_id = str(output.get("selected_id", ""))
    if selected_id not in known_ids:
        return ValidationResult(False, "unknown_selected_id")

    call = str(output.get("call", "")).strip()
    if not call:
        return ValidationResult(False, "empty_call")
    if len(call) > MAX_CALL_CHARS:
        return ValidationResult(False, "call_too_long")

    reason = str(output.get("reason_short", "")).strip()
    if not reason:
        return ValidationResult(False, "empty_reason")
    if len(reason) > MAX_REASON_CHARS:
        return ValidationResult(False, "reason_too_long")

    banned = _banned_phrase(call) or _banned_phrase(reason)
    if banned is not None:
        return ValidationResult(False, f"banned_phrase:{banned}")

    allowed_areas = _allowed_areas(card, candidate_list)
    for area in _AREA_WORDS:
        if area not in allowed_areas and _mentions_area(call, area):
            return ValidationResult(False, f"call_mentions_unknown_area:{area}")

    facts = output.get("grounded_facts")
    if not isinstance(facts, list) or not facts:
        return ValidationResult(False, "missing_grounded_facts")
    allowed_facts = _candidate_facts(candidate_list)
    for fact in facts:
        if not isinstance(fact, str) or fact not in allowed_facts:
            return ValidationResult(False, "ungrounded_fact")

    confidence = str(output.get("confidence", "medium")).lower()
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "medium"

    return ValidationResult(
        True,
        "ok",
        {
            "selected_id": selected_id,
            "call": call,
            "reason_short": reason,
            "confidence": confidence,
            "grounded_facts": list(facts),
        },
    )
