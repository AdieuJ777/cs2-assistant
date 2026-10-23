"""Build an event-driven radar dataset from the nine parsed Dust2 demos.

The fixed-snapshot experiment could not test radar information: at 10-20 s only
1-10% of T players are visible, so the enemy features were almost always zero.
This script changes the unit of analysis from a timestamp to a **spot event**:

1. Sample every 1/10 s of live round time.
2. Compute map-geometry visibility (FOV + LOS + smoke) for every alive CT/T pair.
3. When at least one T becomes visible to the CT team, create one event and a
   decision row for every alive CT (the radar is shared).
4. Label = the CT's macro area 3 s after the event (HOLD or ROTATE_<area>).

Both arms (base vs base + enemy) see the same rows; the base arm only gets CT
state, the enemy arm also gets the visible-enemy summary. This isolates the
value of radar information from the event-selection effect.

Run:
    uv run --with numpy --with pandas --with pyarrow \
        python benchmarks/enemy_visibility/build_event_dataset.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TAGNET_ROOT = PROJECT_ROOT.parent / "TAGNet-CS2-main"
PARSED_ROOT = TAGNET_ROOT / "data" / "parsed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enemy_visibility"

_spec_bd = importlib.util.spec_from_file_location(
    "enemy_build_dataset", HERE / "build_dataset.py"
)
bd = importlib.util.module_from_spec(_spec_bd)
assert _spec_bd.loader is not None
_spec_bd.loader.exec_module(bd)

_spec_vis = importlib.util.spec_from_file_location("dust2_visibility", HERE / "visibility.py")
vis_module = importlib.util.module_from_spec(_spec_vis)
assert _spec_vis.loader is not None
_spec_vis.loader.exec_module(vis_module)

AREAS = bd.AREAS
TICK_RATE = bd.TICK_RATE
SAMPLE_INTERVAL = 6  # 10.7 Hz
START_OFFSET_TICKS = TICK_RATE  # ignore the first second after freeze time
DEFAULT_HORIZON_SECONDS = 3
HORIZON_TICKS = DEFAULT_HORIZON_SECONDS * TICK_RATE
SPOT_COOLDOWN_TICKS = 2 * TICK_RATE


def active_smokes(round_smokes: pd.DataFrame | None, tick: int) -> np.ndarray | None:
    if round_smokes is None or round_smokes.empty:
        return None
    end = round_smokes["end_tick"].fillna(np.inf)
    active = (round_smokes["start_tick"] <= tick) & (end >= tick)
    return round_smokes.loc[active, ["X", "Y"]].to_numpy(dtype=float)


def load_smokes(folder: Path) -> pd.DataFrame | None:
    path = folder / "smokes.parquet"
    if not path.exists():
        return None
    smokes = pd.read_parquet(
        path, columns=["round_num", "start_tick", "end_tick", "X", "Y"]
    ).dropna(subset=["X", "Y"])
    smokes["round_num"] = smokes["round_num"].astype(int)
    smokes["start_tick"] = smokes["start_tick"].astype(int)
    return smokes


def load_ticks(folder: Path) -> pd.DataFrame:
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
            "yaw",
            "health",
            "armor",
            "current_equip_value",
        ],
    )
    ticks["round_num"] = ticks["round_num"].astype(int)
    ticks["tick"] = ticks["tick"].astype(int)
    ticks["area"] = ticks["place"].map(bd.map_area)
    return ticks


def build_match(
    folder: Path,
    centroids: dict[str, tuple[float, float]],
    visibility: vis_module.Dust2Visibility,
    horizon_ticks: int = HORIZON_TICKS,
) -> tuple[list[dict], dict]:
    rounds = pd.read_parquet(folder / "rounds.parquet").sort_values("round_num")
    ticks = load_ticks(folder)
    smokes = load_smokes(folder)

    scores: defaultdict[frozenset, int] = defaultdict(int)
    previous_winner: str | None = None
    previous_site = "unknown"
    decisions: list[dict] = []
    event_count = 0

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

        # Winner bookkeeping must happen even if the round has no events.
        round_winner = str(round_row.winner)
        round_site = bd.attack_site(round_row.bomb_site)

        indexed = round_ticks.set_index(["tick", "steamid"])
        x_df = indexed["X"].unstack("steamid")
        y_df = indexed["Y"].unstack("steamid")
        yaw_df = indexed["yaw"].unstack("steamid")
        health_df = indexed["health"].unstack("steamid")
        armor_df = indexed["armor"].unstack("steamid")
        equip_df = indexed["current_equip_value"].unstack("steamid")
        area_df = indexed["area"].unstack("steamid")

        players = list(x_df.columns)
        player_index = {player: index for index, player in enumerate(players)}
        side = round_ticks.groupby("steamid")["side"].first()
        ct_columns = [player_index[p] for p in players if side[p] == "ct"]
        t_columns = [player_index[p] for p in players if side[p] == "t"]
        ct_local = {global_index: local for local, global_index in enumerate(ct_columns)}
        t_local = {global_index: local for local, global_index in enumerate(t_columns)}

        tick_values = x_df.index.to_numpy()
        x = x_df.to_numpy(dtype=float)
        y = y_df.to_numpy(dtype=float)
        yaw = np.nan_to_num(yaw_df.to_numpy(dtype=float))
        health = health_df.to_numpy(dtype=float)
        armor = np.nan_to_num(armor_df.to_numpy(dtype=float))
        equip = np.nan_to_num(equip_df.to_numpy(dtype=float))
        area_values = area_df.to_numpy(dtype=object)
        xy = np.stack([x, y], axis=2)

        round_smokes = None
        if smokes is not None:
            round_smokes = smokes[smokes["round_num"] == round_num]

        first_tick = int(round_row.freeze_end) + START_OFFSET_TICKS
        last_event = int(round_row.end) - horizon_ticks
        first_tick = max(first_tick, int(tick_values[0]))
        last_event = min(last_event, int(tick_values[-1]) - horizon_ticks)
        if first_tick > last_event:
            if round_winner == "ct":
                scores[ct_key] += 1
            elif round_winner == "t":
                scores[t_key] += 1
            previous_winner, previous_site = round_winner, round_site
            continue

        sample_ticks = np.arange(first_tick, last_event + 1, SAMPLE_INTERVAL)
        last_spot = np.full(len(t_columns), -10**9, dtype=np.int64)
        last_event_tick = -10**9
        pistol = round_num in {1, 13}
        previous_window_winner = previous_winner
        previous_window_site = previous_site

        for tick in sample_ticks:
            position = int(np.searchsorted(tick_values, tick))
            if position >= len(tick_values) or int(tick_values[position]) != tick:
                continue
            health_now = health[position]
            alive_ct_global = [g for g in ct_columns if health_now[g] > 0]
            alive_t_global = [g for g in t_columns if health_now[g] > 0]
            if not alive_ct_global or not alive_t_global:
                continue

            smokes_now = active_smokes(round_smokes, int(tick))
            visible_pairs = visibility.visibility_matrix(
                xy[position, alive_ct_global],
                yaw[position, alive_ct_global],
                xy[position, alive_t_global],
                smokes_now,
            )
            visible = np.zeros((len(ct_columns), len(t_columns)), dtype=bool)
            visible[
                np.ix_(
                    [ct_local[global_index] for global_index in alive_ct_global],
                    [t_local[global_index] for global_index in alive_t_global],
                )
            ] = visible_pairs

            visible_any_t = visible.any(axis=0)
            newly_visible = visible_any_t & (
                (int(tick) - last_spot) >= SPOT_COOLDOWN_TICKS
            )
            last_spot[visible_any_t] = int(tick)
            if not newly_visible.any():
                continue

            event_count += 1
            visible_t_local = np.nonzero(visible_any_t)[0]
            visible_t_global = [t_columns[index] for index in visible_t_local]
            visible_t_xy = xy[position, visible_t_global]
            visible_t_area = area_values[position, visible_t_global]
            visible_count = len(visible_t_global)
            enemy_alive_count = len(alive_t_global)

            ct_xy_now = xy[position, alive_ct_global]
            pair_distance = np.linalg.norm(
                ct_xy_now[:, None, :] - visible_t_xy[None, :, :], axis=2
            )
            team_nearest_distance = float(pair_distance.min())

            area_counts = {
                area_name: int(np.sum(visible_t_area == area_name))
                for area_name in (*AREAS, "other")
            }
            seconds_since_previous_event = (
                99.0
                if last_event_tick < 0
                else (int(tick) - last_event_tick) / TICK_RATE
            )
            last_event_tick = int(tick)

            alive_ct_position = {
                global_index: local_index
                for local_index, global_index in enumerate(alive_ct_global)
            }
            ct_area_now = area_values[position, alive_ct_global]
            area_counts_ct = {
                area_name: int(np.sum(ct_area_now == area_name)) for area_name in AREAS
            }
            team_health_mean = float(health_now[alive_ct_global].mean())
            team_equip_mean = float(equip[position, alive_ct_global].mean())
            total_equip = float(equip[position, ct_columns].sum())
            econ = bd.economy_type(total_equip, pistol)

            for ct_global in alive_ct_global:
                if area_values[position, ct_global] not in AREAS:
                    continue
                horizon_position = position + horizon_ticks
                if horizon_position >= len(tick_values):
                    continue
                if int(tick_values[horizon_position]) != int(tick) + horizon_ticks:
                    continue
                if health[horizon_position, ct_global] <= 0:
                    continue
                next_area = area_values[horizon_position, ct_global]
                if next_area not in AREAS:
                    continue
                current_area = area_values[position, ct_global]
                action = (
                    "HOLD"
                    if next_area == current_area
                    else f"ROTATE_{str(next_area).upper()}"
                )

                self_xy = xy[position, ct_global]
                next_xy = xy[horizon_position, ct_global]
                displacement_units = float(
                    np.hypot(next_xy[0] - self_xy[0], next_xy[1] - self_xy[1])
                )
                distances = {
                    area_name: float(
                        np.hypot(
                            self_xy[0] - centroids[area_name][0],
                            self_xy[1] - centroids[area_name][1],
                        )
                    )
                    / 1000.0
                    for area_name in AREAS
                }
                nearest_area = min(distances, key=distances.get)

                ct_local_index = ct_local[ct_global]
                self_sees = visible[ct_local_index, visible_t_local]
                self_distances = pair_distance[alive_ct_position[ct_global]]
                nearest_visible = int(np.argmin(self_distances))
                enemy_nearest_area = str(visible_t_area[nearest_visible])
                enemy_nearest_distance = float(self_distances[nearest_visible])
                bearing = np.degrees(
                    np.arctan2(
                        visible_t_xy[nearest_visible][1] - self_xy[1],
                        visible_t_xy[nearest_visible][0] - self_xy[0],
                    )
                )
                relative_bearing = float(
                    vis_module.wrap_degrees(bearing - yaw[position, ct_global])
                )

                decisions.append(
                    {
                        "match_id": folder.name,
                        "round_num": round_num,
                        "event_id": f"{folder.name}:{round_num}:{int(tick)}",
                        "event_tick": int(tick),
                        "player_id": str(players[ct_global]),
                        "action": action,
                        "next_area": next_area,
                        "horizon_seconds": horizon_ticks / TICK_RATE,
                        "displacement_units": displacement_units,
                        "area": current_area,
                        "health": int(health[position, ct_global]),
                        "armor": int(armor[position, ct_global]),
                        "current_equip_value": int(equip[position, ct_global]),
                        "count_A": area_counts_ct["A"],
                        "count_B": area_counts_ct["B"],
                        "count_Mid": area_counts_ct["Mid"],
                        "count_CT": area_counts_ct["CT"],
                        "dist_A": distances["A"],
                        "dist_B": distances["B"],
                        "dist_Mid": distances["Mid"],
                        "dist_CT": distances["CT"],
                        "nearest_area": nearest_area,
                        "team_health_mean": team_health_mean,
                        "team_equip_mean": team_equip_mean,
                        "ct_total_equip": total_equip,
                        "ct_econ_type": econ,
                        "ct_alive_count": len(alive_ct_global),
                        "score_diff_ct_minus_t_before": score_ct - score_t,
                        "pistol_round_flag": int(pistol),
                        "prev_round_winner_side": previous_window_winner or "none",
                        "prev_round_attack_site": previous_window_site,
                        "round_phase": bd.round_phase(round_num),
                        "seconds_since_freeze": (int(tick) - int(round_row.freeze_end))
                        / TICK_RATE,
                        "seconds_since_previous_event": seconds_since_previous_event,
                        # --- enemy / radar features ---
                        "enemy_alive_count": enemy_alive_count,
                        "enemy_visible_count": visible_count,
                        "enemy_unknown_count": enemy_alive_count - visible_count,
                        "enemy_visible_ratio": visible_count / enemy_alive_count,
                        "newly_visible_count": int(newly_visible.sum()),
                        "enemy_nearest_distance": enemy_nearest_distance,
                        "enemy_mean_distance": float(self_distances.mean()),
                        "enemy_nearest_area": enemy_nearest_area,
                        "enemy_nearest_relative_bearing": relative_bearing,
                        "enemy_team_nearest_distance": team_nearest_distance,
                        "self_spots_enemy": int(self_sees.any()),
                        "self_spot_count": int(self_sees.sum()),
                        **{
                            f"enemy_visible_{area_name}": area_counts[area_name]
                            for area_name in (*AREAS, "other")
                        },
                    }
                )

        if round_winner == "ct":
            scores[ct_key] += 1
        elif round_winner == "t":
            scores[t_key] += 1
        previous_winner, previous_site = round_winner, round_site

    summary = {"events": event_count, "decisions": len(decisions)}
    return decisions, summary


def validate_visibility(
    visibility: vis_module.Dust2Visibility, folders: list[Path]
) -> dict:
    """Check the proxy against cross-side kills: at the kill tick the attacker
    should usually see the victim (except through smoke or wallbangs)."""
    rows: list[dict] = []
    for folder in folders:
        kills_path = folder / "kills.parquet"
        if not kills_path.exists():
            continue
        kills = pd.read_parquet(
            kills_path,
            columns=[
                "round_num",
                "tick",
                "attacker_X",
                "attacker_Y",
                "attacker_yaw",
                "attacker_side",
                "victim_X",
                "victim_Y",
                "victim_side",
                "thrusmoke",
                "penetrated",
            ],
        )
        kills = kills[
            (kills["attacker_side"] != kills["victim_side"])
            & kills["attacker_X"].notna()
            & kills["victim_X"].notna()
        ]
        smokes = load_smokes(folder)
        for row in kills.itertuples(index=False):
            round_smokes = (
                smokes[smokes["round_num"] == int(row.round_num)]
                if smokes is not None
                else None
            )
            smoke_xy = active_smokes(round_smokes, int(row.tick))
            start = np.array([[float(row.attacker_X), float(row.attacker_Y)]])
            end = np.array([[float(row.victim_X), float(row.victim_Y)]])
            los = bool(visibility.line_of_sight(start, end, smoke_xy)[0])
            bearing = float(
                np.degrees(
                    np.arctan2(
                        float(row.victim_Y) - float(row.attacker_Y),
                        float(row.victim_X) - float(row.attacker_X),
                    )
                )
            )
            angle_diff = float(
                abs(vis_module.wrap_degrees(float(row.attacker_yaw) - bearing))
            )
            rows.append(
                {
                    "seen": los and angle_diff <= vis_module.HALF_FOV,
                    "thrusmoke": bool(row.thrusmoke),
                    "penetrated": bool(row.penetrated),
                    "angle_diff": angle_diff,
                }
            )

    frame = pd.DataFrame(rows)
    clean = frame[~frame["thrusmoke"] & ~frame["penetrated"]]
    return {
        "cross_side_kills": int(len(frame)),
        "seen_rate_all": float(frame["seen"].mean()) if len(frame) else None,
        "seen_rate_clean": float(clean["seen"].mean()) if len(clean) else None,
        "clean_kills": int(len(clean)),
        "seen_rate_through_smoke": (
            float(frame.loc[frame["thrusmoke"], "seen"].mean())
            if frame["thrusmoke"].any()
            else None
        ),
        "seen_rate_through_wall": (
            float(frame.loc[frame["penetrated"], "seen"].mean())
            if frame["penetrated"].any()
            else None
        ),
        "median_angle_diff_clean": (
            float(clean["angle_diff"].median()) if len(clean) else None
        ),
        "note": (
            "The proxy is a 2D lower bound: no elevation, no wall penetration, "
            "smoke is a 144-unit circle."
        ),
    }


def main() -> None:
    global SAMPLE_INTERVAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", action="append", help="Only build these match ids.")
    parser.add_argument("--sample-every", type=int, default=SAMPLE_INTERVAL)
    parser.add_argument(
        "--horizon-seconds",
        type=int,
        default=DEFAULT_HORIZON_SECONDS,
        help="Prediction horizon in seconds (default 3).",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Output suffix; defaults to '' for 3 s and '_<n>s' otherwise.",
    )
    args = parser.parse_args()

    SAMPLE_INTERVAL = args.sample_every
    horizon_ticks = args.horizon_seconds * TICK_RATE
    suffix = args.suffix
    if suffix is None:
        suffix = "" if args.horizon_seconds == DEFAULT_HORIZON_SECONDS else f"_{args.horizon_seconds}s"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    centroids = bd.area_centroids()
    visibility = vis_module.Dust2Visibility()
    folders = sorted(path for path in PARSED_ROOT.iterdir() if path.is_dir())
    if args.match:
        wanted = set(args.match)
        folders = [folder for folder in folders if folder.name in wanted]
    folders = [folder for folder in folders if (folder / "ticks.parquet").exists()]
    if not folders:
        raise SystemExit("no parsed matches found")

    all_rows: list[dict] = []
    match_summaries: dict[str, dict] = {}
    for folder in folders:
        print("building", folder.name)
        rows, summary = build_match(folder, centroids, visibility, horizon_ticks)
        print(f"  events={summary['events']} decisions={summary['decisions']}")
        match_summaries[folder.name] = summary
        all_rows.extend(rows)

    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise SystemExit("no decisions built")
    decisions_path = OUTPUT_DIR / f"event_decisions{suffix}.parquet"
    audit_path = OUTPUT_DIR / f"event_audit{suffix}.json"
    frame.to_parquet(decisions_path, index=False)

    numeric_enemy = [
        column
        for column in frame.columns
        if column.startswith("enemy_") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    audit = {
        "rows": int(len(frame)),
        "events": int(frame["event_id"].nunique()),
        "matches": int(frame["match_id"].nunique()),
        "rounds": int(
            frame.groupby("match_id")["round_num"].nunique().sum()
        ),
        "sample_hz": round(TICK_RATE / SAMPLE_INTERVAL, 2),
        "horizon_seconds": horizon_ticks / TICK_RATE,
        "spot_cooldown_seconds": SPOT_COOLDOWN_TICKS / TICK_RATE,
        "per_match": match_summaries,
        "action_distribution": {
            str(k): int(v) for k, v in frame["action"].value_counts().items()
        },
        "affected_counts": {
            "all": int(len(frame)),
            "distance_le_2000": int((frame["enemy_nearest_distance"] <= 2000).sum()),
            "distance_le_1200": int((frame["enemy_nearest_distance"] <= 1200).sum()),
            "self_spots_enemy": int(frame["self_spots_enemy"].sum()),
        },
        "enemy_feature_means": {
            column: float(frame[column].mean()) for column in numeric_enemy
        },
        "enemy_nearest_distance_quantiles": {
            str(q): float(frame["enemy_nearest_distance"].quantile(q))
            for q in (0.25, 0.5, 0.75, 0.9)
        },
        "enemy_nearest_area_distribution": {
            str(k): int(v)
            for k, v in frame["enemy_nearest_area"].value_counts().items()
        },
        "visibility_validation": validate_visibility(visibility, folders),
        "note": (
            "Event-driven dataset. Rows are decisions for every alive CT at the "
            "moment the team first sees a T after a >=2 s gap. Label is the CT's "
            f"macro area {args.horizon_seconds} s later. Enemy features are not "
            "available to the base arm."
        ),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"wrote {decisions_path}")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
