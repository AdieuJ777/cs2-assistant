"""Replay a parsed demo round into the live server as GSI + radar input.

This is the end-to-end harness for the rotation advisor. It builds the same
payloads a live perception module must produce:

- GSI: map/round/score, self position/forward/health/armor/equipment;
- radar: alive teammates and *visible* enemies from the Dust2 geometry proxy.

Run the server first, then:

    uv run --with numpy --with pandas --with pyarrow \
        python live/demo_radar_feeder.py --match falcons-vs-vitality-m3-dust2 \
        --round 5 --speed 8

Open http://127.0.0.1:3000 while it runs. The feeder is an evaluation/demo
tool, not a live radar source; a real client needs screen/observer perception.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TAGNET_ROOT = PROJECT_ROOT.parent / "TAGNet-CS2-main"
PARSED_ROOT = TAGNET_ROOT / "data" / "parsed"

_spec_vis = importlib.util.spec_from_file_location(
    "dust2_visibility", PROJECT_ROOT / "benchmarks" / "enemy_visibility" / "visibility.py"
)
visibility = importlib.util.module_from_spec(_spec_vis)
assert _spec_vis.loader is not None
_spec_vis.loader.exec_module(visibility)

_spec_bd = importlib.util.spec_from_file_location(
    "enemy_build_dataset",
    PROJECT_ROOT / "benchmarks" / "enemy_visibility" / "build_dataset.py",
)
build_dataset = importlib.util.module_from_spec(_spec_bd)
assert _spec_bd.loader is not None
_spec_bd.loader.exec_module(build_dataset)


def post(url: str, payload: dict, token: str) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2.0) as response:
        response.read()


def running_scores(rounds: pd.DataFrame, target_round: int) -> tuple[int, int]:
    ct = t = 0
    for row in rounds.itertuples(index=False):
        if int(row.round_num) >= target_round:
            break
        if str(row.winner) == "ct":
            ct += 1
        elif str(row.winner) == "t":
            t += 1
    return ct, t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--player", default=None, help="CT steamid; default first CT.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--token", default="change-me")
    parser.add_argument("--speed", type=float, default=8.0, help="Replay speed multiplier.")
    parser.add_argument("--interval", type=float, default=0.5, help="Wall-clock seconds between updates.")
    args = parser.parse_args()

    folder = PARSED_ROOT / args.match
    rounds = pd.read_parquet(folder / "rounds.parquet").sort_values("round_num")
    round_row = rounds[rounds["round_num"] == args.round].iloc[0]
    ticks = pd.read_parquet(
        folder / "ticks.parquet",
        columns=[
            "round_num",
            "tick",
            "steamid",
            "side",
            "name",
            "X",
            "Y",
            "Z",
            "yaw",
            "health",
            "armor",
            "current_equip_value",
            "place",
        ],
    )
    ticks["steamid"] = ticks["steamid"].astype(str)
    round_ticks = ticks[ticks["round_num"] == args.round]

    ct_players = sorted(round_ticks.loc[round_ticks["side"] == "ct", "steamid"].unique())
    t_players = sorted(round_ticks.loc[round_ticks["side"] == "t", "steamid"].unique())
    if args.player:
        self_id = args.player
    else:
        self_id = ct_players[0]
    if self_id not in ct_players:
        raise SystemExit(f"{self_id} is not a CT player in round {args.round}")

    smokes = None
    smoke_path = folder / "smokes.parquet"
    if smoke_path.exists():
        smokes = pd.read_parquet(
            smoke_path, columns=["round_num", "start_tick", "end_tick", "X", "Y"]
        ).dropna(subset=["X", "Y"])
        smokes = smokes[smokes["round_num"] == args.round]

    by_tick = round_ticks.set_index(["tick", "steamid"]).sort_index()
    tick_values = sorted(round_ticks["tick"].unique())
    freeze_end = int(round_row.freeze_end)
    end_tick = int(round_row.end)
    score_ct, score_t = running_scores(rounds, args.round)
    vis_model = visibility.Dust2Visibility()

    stem = f"http://{args.host}:{args.port}"
    print(f"replaying {args.match} round {args.round} as {self_id} at {args.speed}x")

    start = freeze_end
    tick = start
    step = max(1, int(64 * args.interval * args.speed))
    while tick <= end_tick:
        try:
            roster = by_tick.loc[tick]
        except KeyError:
            tick += step
            continue
        roster = roster.reset_index()
        self_row = roster[roster["steamid"] == self_id]
        if self_row.empty or float(self_row.iloc[0]["health"]) <= 0:
            print("self died; stopping")
            break
        self_row = self_row.iloc[0]

        alive_ct = roster[(roster["side"] == "ct") & (roster["health"] > 0)]
        alive_t = roster[(roster["side"] == "t") & (roster["health"] > 0)]
        if alive_t.empty:
            break

        active_smokes = None
        if smokes is not None and not smokes.empty:
            ends = smokes["end_tick"].fillna(np.inf)
            mask = (smokes["start_tick"] <= tick) & (ends >= tick)
            active_smokes = smokes.loc[mask, ["X", "Y"]].to_numpy(dtype=float)

        ct_xy = alive_ct[["X", "Y"]].to_numpy(dtype=float)
        ct_yaw = alive_ct["yaw"].to_numpy(dtype=float)
        t_xy = alive_t[["X", "Y"]].to_numpy(dtype=float)
        team_visible = vis_model.visibility_matrix(
            ct_xy, ct_yaw, t_xy, active_smokes
        ).any(axis=0)
        visible_t = alive_t[team_visible]
        self_positions = np.nonzero(
            (alive_ct["steamid"] == self_id).to_numpy()
        )[0]
        self_sees = bool(
            vis_model.visibility_matrix(
                ct_xy[self_positions],
                ct_yaw[self_positions],
                t_xy,
                active_smokes,
            ).any()
        ) if len(self_positions) else False

        phase = (
            "freezetime"
            if tick < freeze_end
            else ("live" if tick <= end_tick else "over")
        )
        seconds_to_end = max(0.0, (end_tick - tick) / 64.0)
        if phase == "freezetime":
            seconds_to_end = max(0.0, (freeze_end - tick) / 64.0)

        yaw_rad = math.radians(float(self_row["yaw"]))
        gsi = {
            "auth": {"token": args.token},
            "map": {
                "name": "de_dust2",
                "round": int(args.round),
                "team_ct": {"score": score_ct},
                "team_t": {"score": score_t},
            },
            "round": {"phase": phase},
            "player": {
                "steamid": str(self_id),
                "name": str(self_row["name"]),
                "team": "CT",
                "state": {
                    "health": int(self_row["health"]),
                    "armor": int(self_row["armor"]),
                    "money": int(self_row["current_equip_value"]),
                    "equipment_value": int(self_row["current_equip_value"]),
                },
                "position": {
                    "x": float(self_row["X"]),
                    "y": float(self_row["Y"]),
                    "z": float(self_row["Z"]),
                },
                "forward": {"x": math.cos(yaw_rad), "y": math.sin(yaw_rad), "z": 0.0},
            },
            "phase_countdowns": {"phase_ends_in": str(round(seconds_to_end, 2))},
        }
        radar = {
            "auth": {"token": args.token},
            "self_area": build_dataset.map_area(self_row["place"]),
            "teammates": [
                {
                    "x": float(row["X"]),
                    "y": float(row["Y"]),
                    "health": int(row["health"]),
                    "area": build_dataset.map_area(row["place"]),
                }
                for _, row in alive_ct[alive_ct["steamid"] != self_id].iterrows()
            ],
            "enemies": [
                {
                    "x": float(row["X"]),
                    "y": float(row["Y"]),
                    "health": int(row["health"]),
                    "visible": True,
                }
                for _, row in visible_t.iterrows()
            ],
            "enemy_alive_count": int(len(alive_t)),
            "self_spots_enemy": self_sees,
            "self_spot_count": int(
                vis_model.visibility_matrix(
                    ct_xy[self_positions],
                    ct_yaw[self_positions],
                    t_xy,
                    active_smokes,
                ).sum()
            )
            if len(self_positions)
            else 0,
        }

        try:
            post(f"{stem}/", gsi, args.token)
            post(f"{stem}/radar", radar, args.token)
        except (urllib.error.URLError, ConnectionError) as exc:
            print(f"server not reachable: {exc}")
            return

        if tick % (step * 10) < step:
            print(
                f"t={tick} phase={phase} live_enemies={len(visible_t)} "
                f"ct_alive={len(alive_ct)}"
            )
        tick += step
        time.sleep(args.interval)

    print("replay finished")


if __name__ == "__main__":
    main()
