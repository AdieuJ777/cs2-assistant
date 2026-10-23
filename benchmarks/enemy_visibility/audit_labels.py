"""Audit what the event dataset's `ROTATE_*` labels actually mean.

A macro-area change 3 s later can be a strategic rotation or a small
reposition. This script measures the real displacement in map units for HOLD vs
ROTATE labels, so we can see how much of the 15.5% rotation rate is meaningful.

Run:
    uv run --with numpy --with pandas --with pyarrow \
        python benchmarks/enemy_visibility/audit_labels.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "enemy_visibility" / "event_decisions.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "enemy_visibility"

THRESHOLDS = (200, 400, 600, 800, 1000, 1500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_OUT / "label_audit.json"
    )
    args = parser.parse_args()
    df = pd.read_parquet(args.data)
    rotation = df["action"] != "HOLD"
    displacement = df["displacement_units"]

    quantiles = {
        str(q): {
            "hold": float(displacement[~rotation].quantile(q)),
            "rotate": float(displacement[rotation].quantile(q)),
            "all": float(displacement.quantile(q)),
        }
        for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    }

    rotation_counts = {
        str(threshold): {
            "rotations": int((rotation & (displacement < threshold)).sum()),
            "share_of_rotations": float(
                (displacement[rotation] < threshold).mean()
            ),
            "share_of_all_rows": float((displacement < threshold).mean()),
        }
        for threshold in THRESHOLDS
    }

    per_action = {}
    for action, group in df.groupby("action"):
        per_action[str(action)] = {
            "n": int(len(group)),
            "mean_displacement": float(group["displacement_units"].mean()),
            "median_displacement": float(group["displacement_units"].median()),
            "p90_displacement": float(group["displacement_units"].quantile(0.9)),
            "share_under_400": float((group["displacement_units"] < 400).mean()),
            "share_under_800": float((group["displacement_units"] < 800).mean()),
        }

    # Rotations split by whether the player actually changed macro area.
    changed = df["area"] != df["next_area"]
    audit = {
        "rows": int(len(df)),
        "rotations": int(rotation.sum()),
        "rotation_share": float(rotation.mean()),
        "rows_with_area_change": int(changed.sum()),
        "displacement_quantiles": quantiles,
        "rotation_micro_movements": rotation_counts,
        "per_action": per_action,
        "interpretation": (
            "A 3 s window at ~250 units/s allows ~750 units of running. "
            "Rotations below ~400 units are usually repositioning, not a "
            "strategic site change; 400-800 is ambiguous; above 800 is a real "
            "rotation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
