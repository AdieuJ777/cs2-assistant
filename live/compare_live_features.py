"""Compare live shadow features with the training event dataset.

The rotation model was trained on the 5 s event dataset. If live radar/HUD
features drift (e.g. fewer detected teammates, constant enemy alive count), the
model's probabilities compress and it collapses to HOLD. This script prints the
mean of each informative feature on both sides.

Run:
    uv run --with pandas --with pyarrow \
        python live/compare_live_features.py live/shadow_log.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    PROJECT_ROOT / "data" / "enemy_visibility" / "event_decisions_5s.parquet"
)

KEYS = [
    "ct_alive_count",
    "enemy_alive_count",
    "enemy_visible_count",
    "enemy_unknown_count",
    "enemy_visible_ratio",
    "enemy_nearest_distance",
    "team_health_mean",
    "seconds_since_freeze",
    "count_A",
    "count_B",
    "count_Mid",
    "count_CT",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shadow", type=Path, nargs="?", default=Path("live/shadow_log.jsonl"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    if not args.shadow.exists():
        raise SystemExit(f"no shadow log at {args.shadow}")
    records = [
        json.loads(line)
        for line in args.shadow.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    live_rows = [
        record["features"]
        for record in records
        if record.get("record_type", "advice") == "advice"
        and isinstance(record.get("features"), dict)
    ]
    if not live_rows:
        raise SystemExit(
            "no feature rows in the shadow log; restart the server and collect a "
            "new session"
        )

    training = pd.read_parquet(args.data)
    print(f"live advice rows: {len(live_rows)}")
    print(f"training rows:    {len(training)}")
    print(f"{'feature':>24}  {'live':>12}  {'train':>12}  {'ratio':>8}")
    for key in KEYS:
        live_values = [
            float(row[key]) for row in live_rows if row.get(key) is not None
        ]
        if not live_values or key not in training.columns:
            continue
        live_mean = sum(live_values) / len(live_values)
        train_mean = float(pd.to_numeric(training[key], errors="coerce").mean())
        ratio = live_mean / train_mean if train_mean else float("nan")
        print(f"{key:>24}  {live_mean:>12.3f}  {train_mean:>12.3f}  {ratio:>8.2f}")

    live_areas = pd.Series([row.get("area") for row in live_rows]).value_counts()
    train_areas = training["area"].value_counts()
    print("\nself area distribution (live):", live_areas.to_dict())
    print("self area distribution (train):", train_areas.to_dict())


if __name__ == "__main__":
    main()
