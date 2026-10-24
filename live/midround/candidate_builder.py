"""Turn a state card into 2-4 legal, grounded action candidates.

The candidate list is the only thing the LLM is ever allowed to choose from.
Every candidate must be executable now: its route exists on the graph, its
estimated travel time fits in the remaining round time, and every grounded
fact comes from the state card.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .route_graph import RouteGraph
from .state_card import AREA_LABELS

ALLOWED_ACTIONS = (
    "HOLD",
    "ROTATE_A",
    "ROTATE_B",
    "ROTATE_MID",
    "ROTATE_CT",
    "FALL_BACK",
    "REGROUP",
    "WAIT",
)
DEFAULT_EXPIRES_S = 3
MAX_CALL_CHARS = 20
ROUTE_TIME_MARGIN_S = 2.0


def _label(area: str | None) -> str:
    if not area:
        return ""
    return AREA_LABELS.get(area, area)


def _point(area: str | None) -> str:
    """Chinese place phrase used inside grounded facts."""
    if area in {"A", "B"}:
        return f"{area} 点"
    if area == "CT":
        return "CT"
    return _label(area)


def _team_counts(card: Mapping[str, Any]) -> dict[str, int]:
    counts = {area: 0 for area in AREA_LABELS}
    for member in card.get("team", {}).get("members", []):
        area = member.get("area")
        if area in counts and member.get("alive", True):
            counts[area] += 1
    return counts


def _visible_enemy_counts(card: Mapping[str, Any]) -> dict[str, int]:
    counts = {area: 0 for area in AREA_LABELS}
    for enemy in card.get("enemies", {}).get("visible", []):
        area = enemy.get("area")
        if area in counts:
            counts[area] += 1
    return counts


def _global_facts(card: Mapping[str, Any]) -> list[str]:
    facts: list[str] = []
    time_left = card.get("round", {}).get("time_left")
    if time_left is not None:
        facts.append(f"剩余 {int(round(float(time_left)))} 秒")

    for area, count in _team_counts(card).items():
        if count:
            facts.append(f"{_point(area)}有 {count} 名队友")
    for area, count in _visible_enemy_counts(card).items():
        if count:
            facts.append(f"{_point(area)}有 {count} 名可见敌人")

    enemies = card.get("enemies", {})
    if enemies.get("alive_count_source") != "default_5":
        facts.append(f"敌方存活 {enemies.get('alive_count', 0)}")

    if card.get("self", {}).get("alive"):
        facts.append(f"自己血量 {card['self']['health']}")

    bomb = card.get("bomb", {})
    if bomb.get("state") == "planted":
        site = _label(bomb.get("site"))
        facts.append(f"炸弹已下在 {site}" if site else "炸弹已下包")
    elif bomb.get("state") == "carried":
        facts.append("炸弹在敌方手上")
    return facts


def _call_for_route(
    graph: RouteGraph, route: Iterable[str], target_area: str, style: str
) -> str:
    path = list(route)
    label = _label(target_area)
    if style == "fast":
        next_hop = graph.callout(path[1]) if len(path) > 1 else ""
        call = f"快转 {label}，走{next_hop}" if next_hop else f"快转 {label}"
        return call if len(call) <= MAX_CALL_CHARS else f"快转 {label}"
    if target_area != "CT" and "CTSpawn" in path:
        return f"转 {label}，走警家"
    return f"转 {label}，走安全路线"


def _make_candidate(
    *,
    candidate_id: str,
    action: str,
    target_area: str | None,
    route: list[str],
    route_style: str,
    eta_s: float,
    exposure: str,
    call_hint: str,
    facts: list[str],
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "action": action,
        "target_area": target_area,
        "route": route,
        "route_style": route_style,
        "eta_s": round(float(eta_s), 1),
        "exposure": exposure,
        "call_hint": call_hint,
        "grounded_facts": list(dict.fromkeys(facts)),
        "expires_s": DEFAULT_EXPIRES_S,
    }


def _route_candidates(
    card: Mapping[str, Any],
    graph: RouteGraph,
    start: str,
    target_area: str,
    action: str,
    facts: list[str],
) -> list[dict[str, Any]]:
    time_left = card.get("round", {}).get("time_left")
    enemy_areas = {
        enemy.get("area")
        for enemy in card.get("enemies", {}).get("visible", [])
        if enemy.get("area")
    }
    teammate_areas = {
        member.get("area")
        for member in card.get("team", {}).get("members", [])
        if member.get("alive", True) and member.get("area")
    }
    routes = graph.find_routes(
        start,
        target_area,
        enemy_areas=enemy_areas,
        teammate_areas=teammate_areas,
        time_left=float(time_left) if time_left is not None else None,
    )

    candidates: list[dict[str, Any]] = []
    for style in ("safe", "fast"):
        route = routes.get(style)
        if route is None:
            continue
        candidates.append(
            _make_candidate(
                candidate_id=style,
                action=action,
                target_area=target_area,
                route=list(route["route"]),
                route_style=style,
                eta_s=route["eta_s"],
                exposure=route["exposure"],
                call_hint=_call_for_route(graph, route["route"], target_area, style),
                facts=facts,
            )
        )
    return candidates


def _hold_candidate(
    card: Mapping[str, Any], graph: RouteGraph, start: str, facts: list[str]
) -> dict[str, Any]:
    area = card.get("self", {}).get("area")
    label = _label(area)
    call = f"守 {label}，别推" if label else "先守住，别推"
    return _make_candidate(
        candidate_id="hold",
        action="HOLD",
        target_area=area or graph.node_area(start),
        route=[start],
        route_style="hold",
        eta_s=0.0,
        exposure="low",
        call_hint=call,
        facts=facts,
    )


def _fallback_candidate(
    card: Mapping[str, Any],
    graph: RouteGraph,
    start: str,
    facts: list[str],
) -> dict[str, Any] | None:
    if graph.node_area(start) == "CT":
        return None
    route = graph.shortest_path(
        start,
        "CT",
        style="safe",
        time_left=card.get("round", {}).get("time_left"),
    )
    if route is None:
        return None
    return _make_candidate(
        candidate_id="fallback",
        action="FALL_BACK",
        target_area="CT",
        route=list(route["route"]),
        route_style="safe",
        eta_s=route["eta_s"],
        exposure=route["exposure"],
        call_hint="退回 CT，保枪",
        facts=facts,
    )


def _wait_candidate(
    card: Mapping[str, Any], graph: RouteGraph, start: str | None, facts: list[str]
) -> dict[str, Any]:
    area = card.get("self", {}).get("area")
    route = [start] if start is not None else []
    return _make_candidate(
        candidate_id="wait",
        action="WAIT",
        target_area=area,
        route=route,
        route_style="hold",
        eta_s=0.0,
        exposure="low",
        call_hint="先别动，等信息",
        facts=facts,
    )


def _regroup_candidate(
    card: Mapping[str, Any],
    graph: RouteGraph,
    start: str,
    facts: list[str],
) -> dict[str, Any] | None:
    teammates = _team_counts(card)
    own_area = card.get("self", {}).get("area")
    options = [
        (count, area) for area, count in teammates.items() if area != own_area and count
    ]
    if not options:
        return None
    _, target = max(options)
    route = graph.shortest_path(
        start,
        target,
        style="safe",
        teammate_areas={target},
        time_left=card.get("round", {}).get("time_left"),
    )
    if route is None:
        return None
    return _make_candidate(
        candidate_id=f"regroup_{target.lower()}",
        action="REGROUP",
        target_area=target,
        route=list(route["route"]),
        route_style="safe",
        eta_s=route["eta_s"],
        exposure=route["exposure"],
        call_hint=f"去 {_label(target)}，和队友汇合",
        facts=facts,
    )


def validate_candidate(
    candidate: Mapping[str, Any], card: Mapping[str, Any], graph: RouteGraph
) -> tuple[bool, str]:
    """Check schema, graph legality and time feasibility for one candidate."""
    action = candidate.get("action")
    if action not in ALLOWED_ACTIONS:
        return False, "unknown_action"
    target_area = candidate.get("target_area")
    if target_area is not None and target_area not in AREA_LABELS:
        return False, "unknown_target_area"
    route = candidate.get("route")
    if not isinstance(route, list) or not route:
        return False, "empty_route"
    if not graph.is_path(route):
        return False, "invalid_route"
    call = str(candidate.get("call_hint", ""))
    if not call or len(call) > MAX_CALL_CHARS:
        return False, "call_too_long"
    if candidate.get("expires_s") != DEFAULT_EXPIRES_S:
        return False, "bad_expiry"

    if action not in {"HOLD", "WAIT", "REGROUP"}:
        time_left = card.get("round", {}).get("time_left")
        eta = float(candidate.get("eta_s", 0.0))
        if time_left is not None and eta + ROUTE_TIME_MARGIN_S > float(time_left):
            return False, "not_enough_time"
    return True, "ok"


def build_candidates(
    card: Mapping[str, Any],
    graph: RouteGraph,
    *,
    max_candidates: int = 4,
) -> list[dict[str, Any]]:
    """Generate the ordered candidate list for one state card.

    Ordering: kernel primary action first, then its alternative route, then at
    most one conservative option, then one regroup option.
    """
    if card.get("round", {}).get("phase") != "live":
        return []
    self_state = card.get("self", {})
    if not self_state.get("alive"):
        return []

    start = graph.resolve_start(
        self_state.get("area"),
        self_state.get("x"),
        self_state.get("y"),
    )
    facts = _global_facts(card)
    if start is None:
        return [_wait_candidate(card, graph, None, facts)][:max_candidates]

    perception = card.get("perception", {})
    model = card.get("model", {})
    radar_available = not (
        "no_radar" in (perception.get("warnings") or []) or not perception.get("teammates_source")
    )
    model_available = bool(model.get("available"))

    candidates: list[dict[str, Any]] = []

    if not radar_available or not model_available:
        # Without a usable model the safe kernel decision is to wait for
        # information, with a hold as the explicit alternative.
        candidates.append(_wait_candidate(card, graph, start, facts))
        candidates.append(_hold_candidate(card, graph, start, facts))
    else:
        action = str(model.get("action"))
        own_area = self_state.get("area")

        if action == "HOLD":
            candidates.append(_hold_candidate(card, graph, start, facts))
        else:
            target_area = action.split("_", 1)[1]
            target_area = "Mid" if target_area == "MID" else target_area
            if target_area == own_area:
                candidates.append(_hold_candidate(card, graph, start, facts))
            else:
                candidates.extend(
                    _route_candidates(
                        card, graph, start, target_area, action, facts
                    )
                )

        health = int(self_state.get("health", 100))
        outnumbered = int(card.get("team", {}).get("alive_count", 5)) < int(
            card.get("enemies", {}).get("alive_count", 5)
        )
        conservative_added = False
        if health <= 35 or outnumbered:
            fallback = _fallback_candidate(card, graph, start, facts)
            if fallback is not None:
                candidates.append(fallback)
                conservative_added = True

        if not candidates:
            candidates.append(_wait_candidate(card, graph, start, facts))
            candidates.append(_hold_candidate(card, graph, start, facts))
        elif not conservative_added:
            candidates.append(_hold_candidate(card, graph, start, facts))

        if len(candidates) < max_candidates:
            regroup = _regroup_candidate(card, graph, start, facts)
            if regroup is not None:
                candidates.append(regroup)

    # The design contract is 2-4 candidates; add the do-nothing option when
    # only the primary survived.
    if len(candidates) < 2:
        candidates.append(_wait_candidate(card, graph, start, facts))

    # Deduplicate by id, keep the first occurrence, then enforce legality.
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id"))
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        ok, _ = validate_candidate(candidate, card, graph)
        if ok:
            result.append(candidate)
        if len(result) >= max_candidates:
            break
    return result
