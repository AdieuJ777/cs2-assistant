"""Negative controls for the event-driven radar result.

The base_plus_enemy arm beats base. Two alternative explanations must be ruled
out before calling that "radar value":

1. **Extra capacity**: adding any 17 features could improve the model. Control:
   base + 17 random Gaussian features.
2. **Non-linked enemy block**: the enemy block's marginal distribution, not its
   row linkage, carries the gain. Control: permute the whole enemy block across
   rows within each match (and globally), then re-run.

If the real enemy block clearly beats both controls, the gain is tied to the
enemy information at that specific decision.

Run:
    uv run --with numpy --with pandas --with pyarrow --with scikit-learn \
        python benchmarks/enemy_visibility/run_controls.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "enemy_visibility" / "event_decisions.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "enemy_visibility"

_spec = importlib.util.spec_from_file_location(
    "event_policy", HERE / "run_event_policy.py"
)
policy = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(policy)

ENEMY_COLUMNS = policy.ENEMY_CAT + policy.ENEMY_NUM
NOISE_COLUMNS = [f"noise_{index:02d}" for index in range(len(ENEMY_COLUMNS))]


def summary(result: dict) -> dict:
    swing = result["threshold_sweep"]["0.6"]
    return {
        "stage1_auc": result["stage1_ranking"]["roc_auc"],
        "stage1_ap": result["stage1_ranking"]["average_precision"],
        "stage2_accuracy": result["stage2_on_true_rotations"]["accuracy"],
        "stage2_top2": result["stage2_on_true_rotations"]["top2_coverage"],
        "at_0.6_precision": swing["rotation"]["precision"],
        "at_0.6_recall": swing["rotation"]["recall"],
        "at_0.6_false_rotation": swing["rotation"]["false_rotation_rate"],
        "at_0.6_exact_rotation": swing["end_to_end"]["rotation_exact_accuracy"],
        "at_0.6_overall_accuracy": swing["end_to_end"]["accuracy"],
    }


def shuffled_enemy(df: pd.DataFrame, within_match: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    if within_match:
        for _, index in df.groupby("match_id").groups.items():
            order = rng.permutation(len(index))
            out.loc[index, ENEMY_COLUMNS] = df.loc[index, ENEMY_COLUMNS].to_numpy()[
                order
            ]
    else:
        order = rng.permutation(len(df))
        out.loc[:, ENEMY_COLUMNS] = df.loc[:, ENEMY_COLUMNS].to_numpy()[order]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_OUT / "controls.json"
    )
    args = parser.parse_args()
    df = pd.read_parquet(args.data)
    rng = np.random.default_rng(0)
    noise = df.copy()
    for column in NOISE_COLUMNS:
        noise[column] = rng.normal(size=len(noise))

    arms = {
        "base": (
            df,
            policy.BASE_CAT,
            policy.BASE_NUM,
        ),
        "base_plus_enemy": (
            df,
            policy.BASE_CAT + policy.ENEMY_CAT,
            policy.BASE_NUM + policy.ENEMY_NUM,
        ),
        "base_plus_enemy_shuffled_within_match": (
            shuffled_enemy(df, within_match=True, seed=1),
            policy.BASE_CAT + policy.ENEMY_CAT,
            policy.BASE_NUM + policy.ENEMY_NUM,
        ),
        "base_plus_enemy_shuffled_global": (
            shuffled_enemy(df, within_match=False, seed=2),
            policy.BASE_CAT + policy.ENEMY_CAT,
            policy.BASE_NUM + policy.ENEMY_NUM,
        ),
        "base_plus_noise": (
            noise,
            policy.BASE_CAT,
            policy.BASE_NUM + NOISE_COLUMNS,
        ),
    }

    results: dict = {"rows": int(len(df)), "arms": {}}
    for name, (frame, cat_features, num_features) in arms.items():
        print(f"\n=== {name} ===")
        result = policy.run_policy(frame, cat_features, num_features)
        results["arms"][name] = summary(result)
        print(json.dumps(results["arms"][name], indent=2))

    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
