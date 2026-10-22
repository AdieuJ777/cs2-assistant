"""Temporal features for the two-stage action policy.

The current features are single-snapshot. A rotation is a temporal event, so we
add the previous decision window for the same player:

- previous area / previous action / previous team shape;
- movement distance and speed between the two snapshots;
- health, armor and equipment deltas;
- teammate count deltas;
- window duration.

This is historical information: at the current snapshot the player already knows
where they were and how they moved. No future rows are used.

Run with an environment that has pandas/sklearn:
    python benchmarks/action_selection/run_temporal.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "action_selection" / "decisions.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "action_selection"

_spec_ab = importlib.util.spec_from_file_location("action_ablation", HERE / "run_ablation.py")
ablation = importlib.util.module_from_spec(_spec_ab)
assert _spec_ab.loader is not None
_spec_ab.loader.exec_module(ablation)

_spec_ts = importlib.util.spec_from_file_location("action_two_stage", HERE / "run_two_stage.py")
two_stage = importlib.util.module_from_spec(_spec_ts)
assert _spec_ts.loader is not None
_spec_ts.loader.exec_module(two_stage)

ACTIONS = ablation.ACTIONS
HOLD_INDEX = ACTIONS.index("HOLD")
BASE_CAT, BASE_NUM = ablation.FEATURE_SETS["gsi_radar_hud"]

PREV_START = {
    "10->12": None,
    "12->15": 10,
    "15->18": 12,
    "18->20": 15,
}
TEMPORAL_CAT = ["prev_area", "prev_action", "prev_nearest_area"]
TEMPORAL_NUM = [
    "prev_health",
    "prev_armor",
    "prev_equip",
    "prev_count_A",
    "prev_count_B",
    "prev_count_Mid",
    "prev_count_CT",
    "prev_dist_A",
    "prev_dist_B",
    "prev_dist_Mid",
    "prev_dist_CT",
    "prev_team_health_mean",
    "prev_team_equip_mean",
    "distance_moved",
    "speed",
    "health_delta",
    "armor_delta",
    "equip_delta",
    "count_delta_A",
    "count_delta_B",
    "count_delta_Mid",
    "count_delta_CT",
    "window_duration",
    "prev_start_second",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_second"] = df["second"].astype(int)
    end_seconds = df["window"].str.split("->").str[1].astype(int)
    df["end_second"] = end_seconds
    df["prev_start_second"] = df["window"].map(PREV_START)
    df["window_duration"] = df["end_second"] - df["start_second"]

    prev_columns = [
        "sample_id",
        "player_id",
        "second",
        "area",
        "X",
        "Y",
        "health",
        "armor",
        "current_equip_value",
        "count_A",
        "count_B",
        "count_Mid",
        "count_CT",
        "dist_A",
        "dist_B",
        "dist_Mid",
        "dist_CT",
        "team_health_mean",
        "team_equip_mean",
        "nearest_area",
        "action",
    ]
    previous = df[prev_columns].rename(
        columns={
            "second": "prev_second",
            "area": "prev_area",
            "X": "prev_X",
            "Y": "prev_Y",
            "health": "prev_health",
            "armor": "prev_armor",
            "current_equip_value": "prev_equip",
            "count_A": "prev_count_A",
            "count_B": "prev_count_B",
            "count_Mid": "prev_count_Mid",
            "count_CT": "prev_count_CT",
            "dist_A": "prev_dist_A",
            "dist_B": "prev_dist_B",
            "dist_Mid": "prev_dist_Mid",
            "dist_CT": "prev_dist_CT",
            "team_health_mean": "prev_team_health_mean",
            "team_equip_mean": "prev_team_equip_mean",
            "nearest_area": "prev_nearest_area",
            "action": "prev_action",
        }
    )
    merged = df.merge(
        previous,
        left_on=["sample_id", "player_id", "prev_start_second"],
        right_on=["sample_id", "player_id", "prev_second"],
        how="left",
        suffixes=("", "_prevrow"),
    )
    merged["has_prev"] = merged["prev_second"].notna()

    merged["distance_moved"] = np.hypot(
        merged["X"] - merged["prev_X"], merged["Y"] - merged["prev_Y"]
    )
    elapsed = (merged["start_second"] - merged["prev_second"]).replace(0, np.nan)
    merged["speed"] = merged["distance_moved"] / elapsed
    merged["health_delta"] = merged["health"] - merged["prev_health"]
    merged["armor_delta"] = merged["armor"] - merged["prev_armor"]
    merged["equip_delta"] = merged["current_equip_value"] - merged["prev_equip"]
    for area in ["A", "B", "Mid", "CT"]:
        merged[f"count_delta_{area}"] = merged[f"count_{area}"] - merged[f"prev_count_{area}"]

    merged["prev_area"] = merged["prev_area"].fillna("unknown")
    merged["prev_action"] = merged["prev_action"].fillna("unknown")
    merged["prev_nearest_area"] = merged["prev_nearest_area"].fillna("unknown")
    for column in TEMPORAL_NUM:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def design_matrix(df: pd.DataFrame, cat_features: list[str], num_features: list[str]):
    X = df[cat_features + num_features].copy()
    for column in cat_features:
        X[column] = X[column].astype(str).astype("category").cat.codes
    return X


def run_policy(df: pd.DataFrame, cat_features: list[str], num_features: list[str]) -> dict:
    X = design_matrix(df, cat_features, num_features)
    y_action = df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    y_rotate = (y_action != HOLD_INDEX).astype(int)
    groups = df["match_id"].to_numpy()
    cv = GroupKFold(n_splits=5)

    rotate_proba = np.zeros(len(df))
    dest_pred = np.full(len(df), HOLD_INDEX)
    dest_proba = np.zeros((len(df), len(ACTIONS)))

    for train_idx, test_idx in cv.split(X, y_action, groups):
        stage1 = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        stage1.fit(X.iloc[train_idx], y_rotate[train_idx])
        proba = stage1.predict_proba(X.iloc[test_idx])
        rotate_proba[test_idx] = proba[:, list(stage1.classes_).index(1)]

        train_rot = train_idx[y_rotate[train_idx] == 1]
        stage2 = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        stage2.fit(X.iloc[train_rot], y_action[train_rot])
        dest_proba_test = stage2.predict_proba(X.iloc[test_idx])
        for column, action_index in enumerate(stage2.classes_):
            dest_proba[test_idx, action_index] = dest_proba_test[:, column]
        dest_pred[test_idx] = stage2.classes_[dest_proba_test.argmax(axis=1)]

    result = {
        "n": int(len(df)),
        "rotation_share": float(np.mean(y_rotate)),
        "stage1_at_0_5": two_stage._rotation_metrics(
            y_rotate, (rotate_proba >= 0.5).astype(int), rotate_proba
        ),
        "stage2_on_true_rotations": two_stage._destination_metrics(
            y_action, dest_pred, dest_proba
        ),
        "threshold_sweep": {},
    }
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        rotate_pred = (rotate_proba >= threshold).astype(int)
        result["threshold_sweep"][str(threshold)] = {
            "rotation": two_stage._rotation_metrics(y_rotate, rotate_pred, rotate_proba),
            "end_to_end": two_stage._end_to_end(y_action, rotate_pred, dest_pred),
        }
    return result


def main() -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    df = add_temporal_features(pd.read_parquet(DATA_PATH))
    temporal = df[df["has_prev"]].copy().reset_index(drop=True)
    print(f"all rows={len(df)} rows with previous={len(temporal)}")

    feature_sets = {
        "base_same_rows": (BASE_CAT, BASE_NUM),
        "temporal": (BASE_CAT + TEMPORAL_CAT, BASE_NUM + TEMPORAL_NUM),
    }
    results: dict = {
        "all_rows": int(len(df)),
        "rows_with_previous": int(len(temporal)),
        "feature_sets": {},
    }
    for name, (cat_features, num_features) in feature_sets.items():
        result = run_policy(temporal, cat_features, num_features)
        results["feature_sets"][name] = {
            "cat_features": cat_features,
            "num_features": num_features,
            "result": result,
        }
        print(f"\n=== {name} ({len(cat_features) + len(num_features)} features) ===")
        print("stage2:", json.dumps(result["stage2_on_true_rotations"], ensure_ascii=False))
        print(
            f"{'thr':>5} {'rot_prec':>9} {'rot_rec':>8} {'false_rot':>10} "
            f"{'missed':>8} {'exact_rot':>10} {'overall_acc':>11}"
        )
        for threshold, item in result["threshold_sweep"].items():
            rot = item["rotation"]
            end = item["end_to_end"]
            print(
                f"{threshold:>5} {rot['precision']:>9.3f} {rot['recall']:>8.3f} "
                f"{rot['false_rotation_rate']:>10.3f} {rot['missed_rotation_rate']:>8.3f} "
                f"{end['rotation_exact_accuracy']:>10.3f} {end['accuracy']:>11.3f}"
            )

    (RESULTS_OUT / "temporal_policy.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {RESULTS_OUT / 'temporal_policy.json'}")


if __name__ == "__main__":
    main()
