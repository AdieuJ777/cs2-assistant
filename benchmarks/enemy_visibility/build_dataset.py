"""Build a CT decision dataset with enemy last-seen features from parsed demos.

Input: TAGNet-CS2-main/data/parsed/<match>/ticks.parquet + rounds.parquet,
produced by `scripts/parse_demos.py` from the nine Dust2 demos.

For every CT player at snapshots 10/12/15/18/20 s, we build the same
observable features as the action benchmark, plus enemy last-seen features from
the demo's `spotted` flag:

- how many T players are spotted now / within the last 5 s;
- how many are unknown;
- per-area spotted counts and last-seen ages;
- the most recent enemy area.

This is the "radar sees spotted enemies" simulation. It is not ground truth for
unseen enemies; absence of a dot is treated as unknown, not as an empty area.

Run:
    python benchmarks/enemy_visibility/build_dataset.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAGNET_ROOT = PROJECT_ROOT.parent / "TAGNet-CS2-main"
PARSED_ROOT = TAGNET_ROOT / "data" / "parsed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enemy_visibility"

SNAPSHOT_SECONDS = (10, 12, 15, 18, 20)
WINDOWS = ((10, 12), (12, 15), (15, 18), (18, 20))
AREAS = ("A", "B", "Mid", "CT")
TICK_RATE = 64
LOOKBACK_SECONDS = 5
LOOKBACK_TICKS = LOOKBACK_SECONDS * TICK_RATE

PLACE_CANONICAL = {"TopofMid": "TopMid"}
AREA_TO_PLACES = {
    "A": {
        "BombsiteA", "ARamp", "LongA", "LongDoors", "ExtendedA", "Catwalk",
        "ShortStairs", "Pit", "OutsideLong", "UnderA",
    },
    "B": {
        "BombsiteB", "BDoors", "LowerTunnel", "UpperTunnel", "OutsideTunnel",
        "TunnelStairs",
    },
    "Mid": {"MidDoors", "Middle", "TopMid", "Suicide", "Xbox", "Hole"},
    "CT": {"CTSpawn", "Side"},
}


def canonical_place(place: object) -> str:
    if place is None or (isinstance(place, float) and np.isnan(place)):
        return "UNKNOWN"
    text = str(place).strip()
    if not text:
        return "UNKNOWN"
    return PLACE_CANONICAL.get(text, text)


def map_area(place: object) -> str:
    name = canonical_place(place)
    for area, places in AREA_TO_PLACES.items():
        if name in places:
            return area
    return "other"


def area_centroids() -> dict[str, tuple[float, float]]:
    nodes = pd.read_parquet(
        TAGNET_ROOT
        / "data"
        / "processed"
        / "dust2_ct_5snap"
        / "ct_snapshot_nodes.parquet",
        columns=["area", "X", "Y"],
    )
    grouped = nodes[nodes["area"].isin(AREAS)].groupby("area")[["X", "Y"]].mean()
    return {area: (float(row.X), float(row.Y)) for area, row in grouped.iterrows()}


def economy_type(total_equip: float, pistol: bool) -> str:
    if pistol:
        return "pistol"
    if total_equip < 10000:
        return "eco"
    if total_equip < 25000:
        return "half_buy"
    return "long_gun"


def round_phase(round_num: int) -> str:
    if round_num <= 12:
        return "first_half"
    if round_num <= 24:
        return "second_half"
    return "overtime"


def attack_site(bomb_site: object) -> str:
    text = str(bomb_site).lower()
    if text == "bombsite_a":
        return "A"
    if text == "bombsite_b":
        return "B"
    return "unknown"


def enemy_last_seen(
    round_ticks: pd.DataFrame,
    tick: int,
    t_rows: pd.DataFrame,
) -> dict:
    """Compute enemy last-seen features at one snapshot."""
    alive = t_rows[t_rows["health"] > 0].copy()
    alive_count = int(len(alive))
    events = round_ticks[
        (round_ticks["side"] == "t")
        & (round_ticks["spotted"])
        & (round_ticks["health"] > 0)
    ][["steamid", "tick", "area"]]

    seen_now = int(alive["spotted"].sum())
    seen_now_area = {area: 0 for area in AREAS}
    seen5_area = {area: 0 for area in AREAS}
    age_min_area = {area: 99.0 for area in AREAS}
    seen5 = 0
    ages: list[float] = []
    most_recent_area = "unknown"
    most_recent_age = 99.0

    for row in alive.itertuples(index=False):
        if bool(row.spotted) and row.area in AREAS:
            seen_now_area[row.area] += 1
        player_events = events[
            (events["steamid"] == row.steamid)
            & (events["tick"] <= tick)
            & (events["tick"] >= tick - LOOKBACK_TICKS)
        ]
        if player_events.empty:
            continue
        last = player_events.iloc[-1]
        age = (tick - int(last["tick"])) / TICK_RATE
        seen5 += 1
        ages.append(float(age))
        if age < most_recent_age:
            most_recent_age = float(age)
            most_recent_area = str(last["area"])
        area = str(last["area"])
        if area in AREAS:
            seen5_area[area] += 1
            age_min_area[area] = min(age_min_area[area], float(age))

    return {
        "enemy_alive": alive_count,
        "enemy_seen_now": seen_now,
        "enemy_seen_5s": seen5,
        "enemy_unknown": max(alive_count - seen5, 0),
        "enemy_seen5_ratio": seen5 / alive_count if alive_count else 0.0,
        "enemy_last_seen_min_age": min(ages) if ages else 99.0,
        "enemy_most_recent_area": most_recent_area,
        "enemy_most_recent_age": most_recent_age,
        **{f"enemy_seen_now_{area}": seen_now_area[area] for area in AREAS},
        **{f"enemy_seen5_{area}": seen5_area[area] for area in AREAS},
        **{f"enemy_age_min_{area}": age_min_area[area] for area in AREAS},
    }


def build_match(folder: Path, centroids: dict[str, tuple[float, float]]) -> list[dict]:
    rounds = pd.read_parquet(folder / "rounds.parquet").sort_values("round_num")
    ticks = pd.read_parquet(
        folder / "ticks.parquet",
        columns=[
            "round_num",
            "tick",
            "steamid",
            "side",
            "place",
            "X",
            "Y",
            "health",
            "armor",
            "current_equip_value",
            "spotted",
        ],
    )
    ticks["area"] = ticks["place"].map(map_area)
    ticks["round_num"] = ticks["round_num"].astype(int)
    ticks["tick"] = ticks["tick"].astype(int)

    scores: defaultdict[frozenset, int] = defaultdict(int)
    previous_winner: str | None = None
    previous_site = "unknown"
    decisions: list[dict] = []

    for round_row in rounds.itertuples(index=False):
        round_num = int(round_row.round_num)
        round_ticks = ticks[ticks["round_num"] == round_num]
        ct_ids = set(round_ticks.loc[round_ticks["side"] == "ct", "steamid"].unique())
        t_ids = set(round_ticks.loc[round_ticks["side"] == "t", "steamid"].unique())
        if len(ct_ids) != 5 or len(t_ids) != 5:
            continue
        ct_key = frozenset(ct_ids)
        t_key = frozenset(t_ids)
        score_ct = scores[ct_key]
        score_t = scores[t_key]

        snapshot_ticks = {
            second: int(round_row.freeze_end) + second * TICK_RATE
            for second in SNAPSHOT_SECONDS
        }
        if any(tick > int(round_row.end) for tick in snapshot_ticks.values()):
            # The round ended before all snapshots; keep score tracking, skip decisions.
            if round_row.winner == "ct":
                scores[ct_key] += 1
            elif round_row.winner == "t":
                scores[t_key] += 1
            previous_winner = str(round_row.winner)
            previous_site = attack_site(round_row.bomb_site)
            continue

        snapshots: dict[int, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
        valid = True
        for second, tick in snapshot_ticks.items():
            rows = round_ticks[round_ticks["tick"] == tick]
            ct_rows = rows[rows["side"] == "ct"]
            t_rows = rows[rows["side"] == "t"]
            if ct_rows["steamid"].nunique() != 5 or t_rows["steamid"].nunique() != 5:
                valid = False
                break
            snapshots[second] = (
                ct_rows.copy(),
                t_rows.copy(),
                enemy_last_seen(round_ticks, tick, t_rows),
            )
        if not valid:
            if round_row.winner == "ct":
                scores[ct_key] += 1
            elif round_row.winner == "t":
                scores[t_key] += 1
            previous_winner = str(round_row.winner)
            previous_site = attack_site(round_row.bomb_site)
            continue

        for second_from, second_to in WINDOWS:
            ct0, _, enemy0 = snapshots[second_from]
            ct1, _, _ = snapshots[second_to]
            alive_ct = ct0[ct0["health"] > 0]
            if len(alive_ct) == 0:
                continue
            counts = {
                area: int((alive_ct["area"] == area).sum()) for area in AREAS
            }
            team_health_mean = float(alive_ct["health"].mean())
            team_equip_mean = float(alive_ct["current_equip_value"].mean())
            total_equip = float(ct0["current_equip_value"].sum())
            pistol = round_num in {1, 13}
            econ = economy_type(total_equip, pistol)
            next_lookup = ct1.set_index("steamid")

            for row in ct0.itertuples(index=False):
                if row.health <= 0 or row.area not in AREAS:
                    continue
                if row.steamid not in next_lookup.index:
                    continue
                next_row = next_lookup.loc[row.steamid]
                next_area = str(next_row["area"])
                if next_row["health"] <= 0 or next_area not in AREAS:
                    continue
                action = "HOLD" if next_area == row.area else f"ROTATE_{next_area.upper()}"
                distances = {
                    area: float(
                        np.hypot(row.X - centroids[area][0], row.Y - centroids[area][1])
                    )
                    / 1000.0
                    for area in AREAS
                }
                nearest = min(distances, key=distances.get)
                decisions.append(
                    {
                        "match_id": folder.name,
                        "round_num": round_num,
                        "player_id": str(row.steamid),
                        "window": f"{second_from}->{second_to}",
                        "snapshot_from": second_from,
                        "snapshot_to": second_to,
                        "action": action,
                        "area": row.area,
                        "health": int(row.health),
                        "armor": int(row.armor),
                        "current_equip_value": int(row.current_equip_value),
                        "count_A": counts["A"],
                        "count_B": counts["B"],
                        "count_Mid": counts["Mid"],
                        "count_CT": counts["CT"],
                        "dist_A": distances["A"],
                        "dist_B": distances["B"],
                        "dist_Mid": distances["Mid"],
                        "dist_CT": distances["CT"],
                        "nearest_area": nearest,
                        "team_health_mean": team_health_mean,
                        "team_equip_mean": team_equip_mean,
                        "ct_econ_type": econ,
                        "ct_total_equip": total_equip,
                        "score_diff_ct_minus_t_before": score_ct - score_t,
                        "round_num": round_num,
                        "pistol_round_flag": int(pistol),
                        "prev_round_winner_side": previous_winner or "none",
                        "prev_round_attack_site": previous_site,
                        "round_phase": round_phase(round_num),
                        **enemy0,
                    }
                )

        if round_row.winner == "ct":
            scores[ct_key] += 1
        elif round_row.winner == "t":
            scores[t_key] += 1
        previous_winner = str(round_row.winner)
        previous_site = attack_site(round_row.bomb_site)

    return decisions


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    centroids = area_centroids()
    match_dirs = sorted(path for path in PARSED_ROOT.iterdir() if path.is_dir())
    all_rows: list[dict] = []
    for folder in match_dirs:
        if not (folder / "ticks.parquet").exists():
            continue
        print("building", folder.name)
        rows = build_match(folder, centroids)
        print("  decisions:", len(rows))
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("no decisions built")
    df.to_parquet(OUTPUT_DIR / "decisions.parquet", index=False)

    enemy_columns = [column for column in df.columns if column.startswith("enemy_")]
    enemy_numeric = [
        column
        for column in enemy_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    audit = {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "rounds": int(df.groupby("match_id")["round_num"].nunique().sum()),
        "action_distribution": {
            str(k): int(v) for k, v in df["action"].value_counts().items()
        },
        "windows": {str(k): int(v) for k, v in df["window"].value_counts().items()},
        "enemy_feature_means": {
            column: float(df[column].mean()) for column in enemy_numeric
        },
        "enemy_most_recent_area_distribution": {
            str(k): int(v)
            for k, v in df["enemy_most_recent_area"].value_counts().items()
        },
        "enemy_seen5_distribution": {
            str(k): int(v) for k, v in df["enemy_seen_5s"].value_counts().items()
        },
        "enemy_unknown_distribution": {
            str(k): int(v) for k, v in df["enemy_unknown"].value_counts().items()
        },
        "note": (
            "Enemy features use the demo's `spotted` flag within a 5 s lookback. "
            "Absence of a spot is unknown, not an empty area."
        ),
    }
    (OUTPUT_DIR / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"wrote {OUTPUT_DIR / 'decisions.parquet'}")
    print(f"wrote {OUTPUT_DIR / 'audit.json'}")


if __name__ == "__main__":
    main()
