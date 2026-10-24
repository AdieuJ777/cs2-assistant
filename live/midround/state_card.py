"""Assemble the mid-round state card from raw GSI + radar payloads.

This module has no decision logic. It only normalises the two live inputs into
the single JSON-serialisable document described in ``MIDROUND_AGENT_DESIGN.md``
and marks every field whose source is a default rather than a measurement.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

try:  # package import (tests, `python -m live...`)
    from ..area_lookup import AreaLookup
except ImportError:  # script import (`python live/gsi_server.py`)
    from area_lookup import AreaLookup  # type: ignore

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_AREA_GRID = MODELS_DIR / "area_grid.json"
DEFAULT_PLACES = MODELS_DIR / "place_centroids.json"

SCHEMA_VERSION = 1
ROUND_SECONDS = 115.0
MODEL_ACTIONS = ("HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT")
BOMB_STATES = ("carried", "planted", "dropped", "defused", "exploded")
AREA_LABELS = {"A": "A", "B": "B", "Mid": "中路", "CT": "CT"}

_PHASE_ALIASES = {
    "freezetime": "freeze",
    "freeze": "freeze",
    "live": "live",
    "over": "over",
    "intermission": "intermission",
    "timeout": "timeout",
    "gameover": "over",
}


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def _as_int(value: Any, default: int = 0) -> int:
    number = _as_float(value)
    return int(number) if number is not None else default


def _normalise_phase(payload: Mapping[str, Any]) -> str:
    raw = str((payload.get("round") or {}).get("phase", "")).lower()
    return _PHASE_ALIASES.get(raw, raw or "unknown")


def _active_weapon(player: Mapping[str, Any]) -> str | None:
    weapons = player.get("weapons") or {}
    for weapon in weapons.values():
        if isinstance(weapon, Mapping) and weapon.get("state") == "active":
            return str(weapon.get("name") or "") or None
    return None


def _utility(player_state: Mapping[str, Any]) -> dict[str, int]:
    # A player carries at most one of molotov / incendiary, so take the max
    # rather than summing aliases of the same grenade.
    molotov = max(
        _as_int(player_state.get("molotov", 0)),
        _as_int(player_state.get("incgrenade", 0)),
    )
    return {
        "smoke": _as_int(player_state.get("smoke", 0)),
        "flash": _as_int(player_state.get("flashbang", player_state.get("flash", 0))),
        "he": _as_int(player_state.get("hegrenade", player_state.get("he", 0))),
        "molotov": molotov,
    }


@lru_cache(maxsize=1)
def _default_lookup() -> AreaLookup | None:
    try:
        return AreaLookup.load(DEFAULT_AREA_GRID, DEFAULT_PLACES)
    except (OSError, ValueError, KeyError):
        return None


def _macro_area(value: Any) -> str | None:
    text = str(value) if value is not None else ""
    return text if text in AREA_LABELS else None


def _area_for(
    area_lookup: AreaLookup | None,
    explicit: Any,
    x: float | None,
    y: float | None,
) -> str | None:
    area = _macro_area(explicit)
    if area is not None:
        return area
    if area_lookup is not None and x is not None and y is not None:
        return _macro_area(area_lookup.area(x, y))
    return None


def _self_side(payload: Mapping[str, Any]) -> str:
    player = payload.get("player") or {}
    team = player.get("team") or (payload.get("map") or {}).get("team")
    return str(team or "").lower()


def build_state_card(
    gsi: Mapping[str, Any] | None,
    radar: Mapping[str, Any] | None,
    rotation: Mapping[str, Any] | None = None,
    *,
    area_lookup: AreaLookup | None = None,
    now: float | None = None,
    radar_age_s: float | None = None,
    seconds_since_freeze: float | None = None,
    round_seconds: float = ROUND_SECONDS,
) -> dict[str, Any]:
    """Build one state card.

    ``gsi`` is the raw GSI payload, ``radar`` the payload posted to ``/radar``
    (or ``None`` when perception is unavailable), and ``rotation`` the output
    of ``RotationAdvisor.advise``. Missing data is reported through defaults
    and ``perception.warnings``; the function never raises for missing keys.
    """
    now = time.time() if now is None else float(now)
    lookup = area_lookup if area_lookup is not None else _default_lookup()
    payload: Mapping[str, Any] = gsi or {}
    radar_data: Mapping[str, Any] = radar or {}
    rotation_data: Mapping[str, Any] = rotation or {}

    map_data = payload.get("map") or {}
    round_data = payload.get("round") or {}
    player = payload.get("player") or {}
    player_state = player.get("state") or {}
    position = player.get("position") or {}
    forward = player.get("forward") or {}
    countdowns = payload.get("phase_countdowns") or {}
    phase = _normalise_phase(payload)
    warnings: list[str] = []

    # ------------------------------------------------------------- self pos
    x = _as_float(position.get("x"))
    y = _as_float(position.get("y"))
    has_gsi_position = x is not None and y is not None and not (x == 0.0 and y == 0.0)
    self_source: str | None = "gsi" if has_gsi_position else None
    if not has_gsi_position:
        radar_x = _as_float(radar_data.get("self_x"))
        radar_y = _as_float(radar_data.get("self_y"))
        if radar_x is not None and radar_y is not None:
            x, y = radar_x, radar_y
            self_source = "radar_labels"
        else:
            x, y = None, None
            warnings.append("self_position_missing")
    self_area = _area_for(lookup, radar_data.get("self_area"), x, y)
    if self_area is None:
        warnings.append("self_area_unknown")

    health = _as_int(player_state.get("health", 0))
    armor = _as_int(player_state.get("armor", 0))
    equip_value = _as_float(
        player_state.get(
            "equipment_value",
            player_state.get("equip_value", player_state.get("money")),
        ),
        0.0,
    )
    yaw = math.degrees(
        math.atan2(
            _as_float(forward.get("y"), 0.0) or 0.0,
            _as_float(forward.get("x"), 1.0) or 1.0,
        )
    )

    # ---------------------------------------------------------------- radar
    radar_available = bool(radar_data)
    if not radar_available:
        warnings.append("no_radar")
    if radar_age_s is not None and radar_age_s > 1.5:
        warnings.append("radar_stale")

    # --------------------------------------------------------------- team
    raw_teammates = radar_data.get("teammates")
    teammates = list(raw_teammates or [])
    members: list[dict[str, Any]] = []
    teammate_health_measured = bool(teammates)
    for mate in teammates:
        if not isinstance(mate, Mapping):
            continue
        mate_health = mate.get("health")
        if mate_health is None:
            teammate_health_measured = False
        mate_x = _as_float(mate.get("x"))
        mate_y = _as_float(mate.get("y"))
        members.append(
            {
                "area": _area_for(lookup, mate.get("area"), mate_x, mate_y),
                "health": _as_int(mate_health, 100),
                "alive": _as_int(mate_health, 100) > 0,
            }
        )
    health_source = "radar" if teammate_health_measured else "default_100"
    if teammates and not teammate_health_measured:
        warnings.append("teammate_health_default")
    if raw_teammates is None:
        warnings.append("teammate_positions_missing")
        health_source = "default_100"
    team_alive = (1 if health > 0 else 0) + sum(
        1 for member in members if member["alive"]
    )

    # ------------------------------------------------------------- enemies
    enemy_alive_raw = radar_data.get("enemy_alive_count")
    enemy_alive_count = _as_int(enemy_alive_raw, 5)
    enemy_alive_source = "radar" if enemy_alive_raw is not None else "default_5"
    if enemy_alive_raw is None:
        warnings.append("enemy_alive_count_default")

    visible_enemies: list[dict[str, Any]] = []
    for enemy in radar_data.get("enemies") or []:
        if not isinstance(enemy, Mapping):
            continue
        if enemy.get("visible", True) is False:
            continue
        if _as_int(enemy.get("health", 1), 1) <= 0:
            continue
        enemy_x = _as_float(enemy.get("x"))
        enemy_y = _as_float(enemy.get("y"))
        enemy_area = _area_for(lookup, enemy.get("area"), enemy_x, enemy_y)
        distance = None
        bearing = None
        if x is not None and y is not None and enemy_x is not None and enemy_y is not None:
            distance = round(math.hypot(enemy_x - x, enemy_y - y), 1)
            bearing = round(math.degrees(math.atan2(enemy_y - y, enemy_x - x)) - yaw, 1)
        visible_enemies.append(
            {
                "area": enemy_area,
                "distance": distance,
                "bearing": bearing,
                "age_s": round(float(radar_age_s), 2) if radar_age_s is not None else 0.0,
            }
        )

    visible_count = len(visible_enemies)
    nearest_enemy = (
        min(
            (enemy for enemy in visible_enemies if enemy["distance"] is not None),
            key=lambda enemy: enemy["distance"],
            default=None,
        )
        if visible_enemies
        else None
    )
    unknown_count = max(enemy_alive_count - visible_count, 0)

    # ---------------------------------------------------------------- bomb
    bomb_raw = str(round_data.get("bomb", "")).lower()
    bomb_state = bomb_raw if bomb_raw in BOMB_STATES else "unknown"
    if bomb_raw and bomb_state == "unknown":
        warnings.append("unknown_bomb_state")
    elif not bomb_raw:
        warnings.append("bomb_state_unknown")
    bomb_site = _macro_area(round_data.get("bomb_site"))
    phase_ends_in = _as_float(countdowns.get("phase_ends_in"))

    # --------------------------------------------------------------- timing
    if seconds_since_freeze is not None:
        seconds_since_freeze = max(0.0, float(seconds_since_freeze))
        if phase == "live":
            time_left = max(0.0, round_seconds - seconds_since_freeze)
        else:
            time_left = phase_ends_in
    elif phase == "live" and phase_ends_in is not None and 0.0 < phase_ends_in <= round_seconds:
        time_left = phase_ends_in
        seconds_since_freeze = max(0.0, round_seconds - phase_ends_in)
    else:
        time_left = phase_ends_in if phase_ends_in and phase_ends_in > 0 else None
        seconds_since_freeze = (
            max(0.0, round_seconds - time_left) if time_left is not None else None
        )
    if time_left is None:
        warnings.append("time_left_missing")

    # ---------------------------------------------------------------- model
    rotation_action = rotation_data.get("action")
    model_available = rotation_action in MODEL_ACTIONS
    model = {
        "action": rotation_action if model_available else None,
        "available": model_available,
        "status": rotation_action,
        "stage1_proba": rotation_data.get("stage1_proba"),
        "destination": rotation_data.get("destination"),
        "destination_proba": rotation_data.get("destination_proba"),
        "threshold": rotation_data.get("threshold"),
    }
    for warning in rotation_data.get("warnings") or []:
        if warning not in warnings:
            warnings.append(str(warning))

    # ------------------------------------------------------------- events
    recent_events: list[dict[str, Any]] = []
    if nearest_enemy is not None:
        recent_events.append(
            {
                "type": "spot",
                "area": nearest_enemy["area"],
                "t": -float(nearest_enemy["age_s"]),
            }
        )

    round_number = _as_int(map_data.get("round", round_data.get("round", 0)))
    score_ct = _as_int((map_data.get("team_ct") or {}).get("score", 0))
    score_t = _as_int((map_data.get("team_t") or {}).get("score", 0))

    return {
        "schema_version": SCHEMA_VERSION,
        "ts": now,
        "map": map_data.get("name"),
        "side": _self_side(payload),
        "round": {
            "number": round_number,
            "phase": phase,
            "time_left": round(time_left, 1) if time_left is not None else None,
            "seconds_since_freeze": (
                round(seconds_since_freeze, 1)
                if seconds_since_freeze is not None
                else None
            ),
            "score_ct": score_ct,
            "score_t": score_t,
            "pistol": round_number in {1, 13},
        },
        "self": {
            "area": self_area,
            "x": x,
            "y": y,
            "health": health,
            "armor": armor,
            "equip_value": int(equip_value or 0),
            "weapon": _active_weapon(player),
            "defuse_kit": bool(
                player_state.get("defusekit", player_state.get("defuse_kit", False))
            ),
            "utility": _utility(player_state),
            "alive": health > 0,
        },
        "team": {
            "alive_count": team_alive,
            "members": members,
            "health_source": health_source,
        },
        "enemies": {
            "alive_count": enemy_alive_count,
            "alive_count_source": enemy_alive_source,
            "visible": visible_enemies,
            "visible_count": visible_count,
            "unknown_count": unknown_count,
            "nearest_area": nearest_enemy["area"] if nearest_enemy else None,
            "nearest_distance": nearest_enemy["distance"] if nearest_enemy else None,
        },
        "bomb": {
            "state": bomb_state,
            "site": bomb_site,
            "timer": (
                round(phase_ends_in, 1)
                if bomb_state == "planted" and phase_ends_in is not None
                else None
            ),
        },
        "model": model,
        "recent_events": recent_events,
        "perception": {
            "radar_age_s": round(float(radar_age_s), 2) if radar_age_s is not None else None,
            "self_source": self_source,
            "teammates_source": "radar" if raw_teammates is not None else None,
            "warnings": warnings,
        },
    }
