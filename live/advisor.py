"""Deterministic freeze-time buy/save advisor for Dust2 CT.

This is the narrow live v1: it uses only what Counter-Strike 2's Game State
Integration (GSI) exposes to a live player:

- own money, health, armor, helmet, defuse kit and weapon inventory;
- map, side, round number, score and round phase;
- bomb state.

It does NOT see teammate money, teammate positions, enemy positions, enemy
economy, the kill feed or voice comms. The advice is therefore self-centric and
explicitly says what it does not know.

Prices are approximate and live in ``PRICES`` below; edit them when the game's
economy changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PRICES = {
    # rifles / primary
    "m4a1_silencer": 2900,
    "m4a1": 3100,
    "awp": 4750,
    "ssg08": 1700,
    "mp9": 1250,
    "ump45": 1200,
    "p90": 2350,
    # pistols
    "deagle": 700,
    "fiveseven": 500,
    "p250": 300,
    "usp_silencer": 0,
    "glock": 0,
    # armor / kit
    "armor_pack": 1000,  # kevlar + helmet when starting from no armor
    "kevlar": 650,
    "helmet": 350,
    "defuse": 400,
    # utility
    "smoke": 300,
    "flash": 200,
    "he": 300,
    "incendiary": 600,
    "molotov": 400,
    "decoy": 50,
}

RIFLES = {"m4a1_silencer", "m4a1"}
SMGS = {"mp9", "ump45", "p90"}
FULL_BUY_RIFLE = "m4a1_silencer"

GRENADE_NAMES = {
    "weapon_smokegrenade": "smoke",
    "weapon_flashbang": "flash",
    "weapon_hegrenade": "he",
    "weapon_incgrenade": "incendiary",
    "weapon_molotov": "molotov",
    "weapon_decoy": "decoy",
}


@dataclass
class RoundHistory:
    rounds: list[dict[str, Any]] = field(default_factory=list)

    def update(self, payload: dict[str, Any]) -> None:
        round_data = payload.get("round", {}) or {}
        map_data = payload.get("map", {}) or {}
        phase = str(round_data.get("phase", "")).lower()
        if phase not in {"over", "gameover"}:
            return
        round_num = _int(map_data.get("round"), 0)
        winner = str(round_data.get("win_team", "")).lower()
        if winner not in {"ct", "t"}:
            return
        if any(int(item.get("round", -1)) == round_num for item in self.rounds):
            return
        self.rounds.append({"round": round_num, "winner": winner})
        self.rounds = self.rounds[-30:]

    def t_loss_streak(self) -> int:
        streak = 0
        for item in reversed(self.rounds):
            if item["winner"] == "ct":
                streak += 1
            elif item["winner"] == "t":
                break
        return streak

    def last_winner(self) -> str | None:
        return self.rounds[-1]["winner"] if self.rounds else None

    def to_dict(self) -> dict[str, Any]:
        return {"rounds": self.rounds}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weapon_names(player: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for weapon in (player.get("weapons") or {}).values():
        if isinstance(weapon, dict):
            name = str(weapon.get("name", ""))
            if name:
                names.add(name.removeprefix("weapon_"))
    return names


def _grenade_counts(names: set[str]) -> dict[str, int]:
    counts = {"smoke": 0, "flash": 0, "he": 0, "incendiary": 0, "molotov": 0, "decoy": 0}
    for name in names:
        key = GRENADE_NAMES.get(f"weapon_{name}")
        if key:
            counts[key] += 1
    return counts


def _armor_cost(armor: int, helmet: bool) -> tuple[int, str | None]:
    if armor <= 0:
        return PRICES["armor_pack"], "Kevlar + Helmet"
    if not helmet:
        return PRICES["helmet"], "Helmet"
    return 0, None


def _add_item(
    items: list[dict[str, Any]],
    budget: int,
    name: str,
    cost: int,
    *,
    optional: bool = False,
) -> int:
    """Append an item if it fits; returns the remaining budget."""
    if budget < cost:
        return budget
    items.append({"name": name, "cost": cost, "optional": optional})
    return budget - cost


def _loadout(names: set[str], state: dict[str, Any]) -> dict[str, Any]:
    armor = _int(state.get("armor"), 0)
    helmet = bool(state.get("helmet", False))
    defuse = bool(state.get("defusekit", False))
    return {
        "primary": sorted(name for name in names if name in RIFLES | SMGS | {"awp", "ssg08"}),
        "pistol": sorted(
            name
            for name in names
            if name in {"deagle", "fiveseven", "p250", "usp_silencer", "glock"}
        ),
        "grenades": _grenade_counts(names),
        "armor": armor,
        "helmet": helmet,
        "defuse_kit": defuse,
    }


def _enemy_prior(history: RoundHistory, round_num: int) -> dict[str, Any]:
    if round_num in {1, 13}:
        return {
            "label": "pistol",
            "confidence": "high",
            "reason": "Pistol round: both teams are on pistols.",
        }
    streak = history.t_loss_streak()
    last = history.last_winner()
    if streak >= 2:
        return {
            "label": "eco or force",
            "confidence": "low",
            "reason": f"T has lost {streak} rounds in a row; loss bonus is high, but save/force behavior is unknown.",
        }
    if last == "t":
        return {
            "label": "half buy or full buy",
            "confidence": "low",
            "reason": "T won the previous round; they can afford a stronger buy, but their exact money is unknown.",
        }
    return {
        "label": "unknown",
        "confidence": "low",
        "reason": "No reliable enemy economy signal from GSI alone.",
    }


def advise(payload: dict[str, Any], history: RoundHistory | None = None) -> dict[str, Any]:
    history = history or RoundHistory()
    map_data = payload.get("map", {}) or {}
    round_data = payload.get("round", {}) or {}
    player = payload.get("player", {}) or {}
    state = player.get("state", {}) or {}
    phase = str(round_data.get("phase", "")).lower()
    round_num = _int(map_data.get("round"), 0)
    team = str(player.get("team", "")).upper()
    money = _int(state.get("money"), 0)
    health = _int(state.get("health"), 0)
    armor = _int(state.get("armor"), 0)
    helmet = bool(state.get("helmet", False))
    defuse = bool(state.get("defusekit", False))
    names = _weapon_names(player)
    grenades = _grenade_counts(names)
    score_ct = _int((map_data.get("team_ct") or {}).get("score"), 0)
    score_t = _int((map_data.get("team_t") or {}).get("score"), 0)
    map_name = str(map_data.get("name", ""))
    bomb_state = str((payload.get("round", {}) or {}).get("bomb", "")).lower()
    seconds_left = _float((payload.get("phase_countdowns") or {}).get("phase_ends_in"), 0.0)

    base = {
        "round": round_num,
        "phase": phase,
        "team": team,
        "map": map_name,
        "score": {"ct": score_ct, "t": score_t},
        "money": money,
        "health": health,
        "armor": armor,
        "helmet": helmet,
        "defuse_kit": defuse,
        "loadout": _loadout(names, state),
        "bomb": bomb_state,
        "seconds_left": round(seconds_left, 1),
        "enemy_prior": _enemy_prior(history, round_num),
        "warnings": [],
    }
    if map_name and map_name != "de_dust2":
        base["warnings"].append(
            f"Map is {map_name}; this v1 is tuned for Dust2."
        )

    if team != "CT":
        base.update(
            {
                "recommendation": "NOT_SUPPORTED",
                "confidence": "high",
                "reason": "This v1 only supports the CT side.",
                "shopping_list": [],
                "estimated_cost": 0,
                "reserve": money,
            }
        )
        return base

    if phase != "freezetime":
        base.update(
            {
                "recommendation": "BUY_PHASE_OVER",
                "confidence": "high",
                "reason": "The buy phase is over. Play the current loadout.",
                "shopping_list": [],
                "estimated_cost": 0,
                "reserve": money,
            }
        )
        if bomb_state == "planted":
            base["warnings"].append("Bomb is planted; hold or retake with the current loadout.")
        return base

    items: list[dict[str, Any]] = []
    budget = money
    pistol_round = round_num in {1, 13}

    if pistol_round:
        if armor <= 0:
            if money >= PRICES["armor_pack"]:
                armor_name, armor_cost = "Kevlar + Helmet", PRICES["armor_pack"]
            elif money >= PRICES["kevlar"]:
                armor_name, armor_cost = "Kevlar", PRICES["kevlar"]
            else:
                armor_name, armor_cost = None, 0
        elif not helmet:
            armor_name, armor_cost = (
                ("Helmet", PRICES["helmet"])
                if money >= PRICES["helmet"]
                else (None, 0)
            )
        else:
            armor_name, armor_cost = None, 0
        if armor_name:
            budget = _add_item(items, budget, armor_name, armor_cost)
        if not defuse:
            budget = _add_item(items, budget, "Defuse Kit", PRICES["defuse"])
        if grenades["smoke"] < 1:
            budget = _add_item(items, budget, "Smoke", PRICES["smoke"], optional=True)
        if grenades["flash"] < 1:
            budget = _add_item(items, budget, "Flash", PRICES["flash"], optional=True)
        if grenades["he"] < 1:
            budget = _add_item(items, budget, "HE", PRICES["he"], optional=True)
        spent = money - budget
        if spent > 0:
            recommendation = "PISTOL_BUY"
            confidence = "high" if budget >= 0 else "medium"
            reason = "Pistol round: buy armor, a defuse kit and utility; do not buy a rifle."
        else:
            recommendation = "PISTOL_SAVE"
            confidence = "low"
            reason = "Pistol round, but you cannot afford the core pistol buy; keep the money."
        base.update(
            {
                "recommendation": recommendation,
                "confidence": confidence,
                "reason": reason,
                "shopping_list": items,
                "estimated_cost": spent,
                "reserve": budget,
            }
        )
        return base

    has_rifle = bool(names & RIFLES)
    has_primary = bool(names & (RIFLES | SMGS | {"awp", "ssg08"}))
    armor_cost, armor_name = _armor_cost(armor, helmet)

    # Full buy: rifle (if missing) + armor + defuse + core utility.
    full_items: list[dict[str, Any]] = []
    full_budget = money
    if not has_rifle:
        full_budget = _add_item(
            full_items, full_budget, "M4A1-S", PRICES[FULL_BUY_RIFLE]
        )
    if armor_name:
        full_budget = _add_item(full_items, full_budget, armor_name, armor_cost)
    if not defuse:
        full_budget = _add_item(full_items, full_budget, "Defuse Kit", PRICES["defuse"])
    if grenades["smoke"] < 1:
        full_budget = _add_item(full_items, full_budget, "Smoke", PRICES["smoke"])
    if grenades["flash"] < 2:
        missing_flash = 2 - grenades["flash"]
        for _ in range(missing_flash):
            full_budget = _add_item(full_items, full_budget, "Flash", PRICES["flash"])
    if grenades["he"] < 1:
        full_budget = _add_item(full_items, full_budget, "HE", PRICES["he"])
    full_cost = money - full_budget

    # Core cost is the price of the essentials, without optional utility.
    required_core_cost = (
        (0 if has_rifle else PRICES[FULL_BUY_RIFLE])
        + armor_cost
        + (0 if defuse else PRICES["defuse"])
    )

    if money >= required_core_cost:
        base.update(
            {
                "recommendation": "FULL_BUY",
                "confidence": "high" if full_budget >= 300 else "medium",
                "reason": "You can afford the core full buy; the list below is what fits your budget.",
                "shopping_list": full_items,
                "estimated_cost": full_cost,
                "reserve": full_budget,
            }
        )
        if has_rifle:
            base["warnings"].append("You already have a rifle; the estimate only covers armor/utility.")
        if not any(item["name"] in {"Smoke", "Flash", "HE"} for item in full_items):
            base["warnings"].append("No utility in this buy; consider a smoke/flash if your team needs it.")
        if "Defuse Kit" not in {item["name"] for item in full_items} and not defuse:
            base["warnings"].append("No defuse kit in this buy.")
        return base

    # Half / force buy: armor + SMG (or keep pistol) + light utility.
    half_items: list[dict[str, Any]] = []
    half_budget = money
    if armor_name:
        half_budget = _add_item(half_items, half_budget, armor_name, armor_cost)
    if not has_primary:
        half_budget = _add_item(half_items, half_budget, "MP9", PRICES["mp9"])
    if not defuse:
        half_budget = _add_item(half_items, half_budget, "Defuse Kit", PRICES["defuse"])
    if grenades["smoke"] < 1:
        half_budget = _add_item(half_items, half_budget, "Smoke", PRICES["smoke"], optional=True)
    if grenades["flash"] < 1:
        half_budget = _add_item(half_items, half_budget, "Flash", PRICES["flash"], optional=True)
    half_cost = money - half_budget
    reserve = half_budget

    if half_cost > 0 and reserve >= 0 and money >= 2000:
        base.update(
            {
                "recommendation": "HALF_BUY",
                "confidence": "medium",
                "reason": (
                    "You cannot afford a full rifle buy, but a half buy keeps you competitive. "
                    "Team economy is unknown; if your team is saving, save with them."
                ),
                "shopping_list": half_items,
                "estimated_cost": half_cost,
                "reserve": reserve,
            }
        )
        base["warnings"].append("Teammate economy is unknown; coordinate the save/buy call.")
        return base

    base.update(
        {
            "recommendation": "SAVE",
            "confidence": "high" if money < 1500 else "medium",
            "reason": (
                "A full buy is out of reach and a half buy would leave you with little money. "
                "Save and keep the loss bonus for the next round."
            ),
            "shopping_list": [],
            "estimated_cost": 0,
            "reserve": money,
        }
    )
    base["warnings"].append("Teammate economy is unknown; confirm the team save.")
    if not defuse and money >= PRICES["defuse"] and not has_primary:
        base["warnings"].append("If you must force, a defuse kit is often worth more than a pistol upgrade.")
    return base
