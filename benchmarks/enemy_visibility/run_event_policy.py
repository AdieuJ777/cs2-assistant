"""Evaluate radar features on the event-driven dataset.

The same event rows are used for both arms:

- ``base``: CT state only (radar/HUD-observable features + event timing);
- ``base_plus_enemy``: base plus the visible-enemy summary (area, distance,
  bearing, how many enemies are visible, whether this CT is the spotter).

The two-stage policy is the same as the action benchmark: stage 1 ROTATE vs
HOLD, stage 2 destination among true rotations. All metrics are out-of-fold with
GroupKFold by match.

Run:
    uv run --with numpy --with pandas --with pyarrow --with scikit-learn \
        python benchmarks/enemy_visibility/run_event_policy.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "enemy_visibility" / "event_decisions.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "enemy_visibility"

_spec_ts = importlib.util.spec_from_file_location(
    "action_two_stage", HERE.parent / "action_selection" / "run_two_stage.py"
)
two_stage = importlib.util.module_from_spec(_spec_ts)
assert _spec_ts.loader is not None
_spec_ts.loader.exec_module(two_stage)

ACTIONS = two_stage.ACTIONS
HOLD_INDEX = ACTIONS.index("HOLD")

BASE_CAT = [
    "area",
    "round_phase",
    "prev_round_winner_side",
    "prev_round_attack_site",
    "nearest_area",
]
BASE_NUM = [
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
    "score_diff_ct_minus_t_before",
    "round_num",
    "pistol_round_flag",
    "ct_alive_count",
    "seconds_since_freeze",
    "seconds_since_previous_event",
]
ENEMY_CAT = ["enemy_nearest_area"]
ENEMY_NUM = [
    "enemy_alive_count",
    "enemy_visible_count",
    "enemy_unknown_count",
    "enemy_visible_ratio",
    "newly_visible_count",
    "enemy_nearest_distance",
    "enemy_mean_distance",
    "enemy_nearest_relative_bearing",
    "enemy_team_nearest_distance",
    "self_spots_enemy",
    "self_spot_count",
    "enemy_visible_A",
    "enemy_visible_B",
    "enemy_visible_Mid",
    "enemy_visible_CT",
    "enemy_visible_other",
]


def design_matrix(df: pd.DataFrame, cat_features: list[str], num_features: list[str]):
    X = df[cat_features + num_features].copy()
    for column in cat_features:
        X[column] = X[column].astype(str).astype("category").cat.codes
    return X


def run_policy(
    df: pd.DataFrame,
    cat_features: list[str],
    num_features: list[str],
    seed: int = 42,
) -> dict:
    X = design_matrix(df, cat_features, num_features)
    y_action = (
        df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    )
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
            random_state=seed,
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
            random_state=seed,
            n_jobs=-1,
        )
        stage2.fit(X.iloc[train_rot], y_action[train_rot])
        stage2_proba = stage2.predict_proba(X.iloc[test_idx])
        for column, action_index in enumerate(stage2.classes_):
            dest_proba[test_idx, action_index] = stage2_proba[:, column]
        dest_pred[test_idx] = stage2.classes_[stage2_proba.argmax(axis=1)]

    result = {
        "n": int(len(df)),
        "rotation_count": int(np.sum(y_rotate)),
        "rotation_share": float(np.mean(y_rotate)),
        "stage1_ranking": {
            "roc_auc": float(roc_auc_score(y_rotate, rotate_proba)),
            "average_precision": float(average_precision_score(y_rotate, rotate_proba)),
        },
        "stage2_on_true_rotations": two_stage._destination_metrics(
            y_action, dest_pred, dest_proba
        ),
        "threshold_sweep": {},
        "_rotate_proba": rotate_proba,
        "_dest_pred": dest_pred,
        "_y_rotate": y_rotate,
        "_y_action": y_action,
    }
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        rotate_pred = (rotate_proba >= threshold).astype(int)
        result["threshold_sweep"][str(threshold)] = {
            "rotation": two_stage._rotation_metrics(
                y_rotate, rotate_pred, rotate_proba
            ),
            "end_to_end": two_stage._end_to_end(y_action, rotate_pred, dest_pred),
        }
    return result


def subset_metrics(result: dict, mask: np.ndarray) -> dict:
    y_rotate = result["_y_rotate"][mask]
    proba = result["_rotate_proba"][mask]
    y_action = result["_y_action"][mask]
    rotate_pred = (proba >= 0.6).astype(int)
    payload = {
        "n": int(len(y_rotate)),
        "rotation_count": int(np.sum(y_rotate)),
    }
    if len(np.unique(y_rotate)) < 2:
        return payload
    payload.update(
        {
            "roc_auc": float(roc_auc_score(y_rotate, proba)),
            "average_precision": float(average_precision_score(y_rotate, proba)),
            "at_0_6": two_stage._rotation_metrics(y_rotate, rotate_pred, proba),
            "destination_at_0_6": two_stage._end_to_end(
                y_action, rotate_pred, result["_dest_pred"][mask]
            ),
        }
    )
    return payload


def clean(result: dict) -> dict:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def majority_baseline(df: pd.DataFrame) -> dict:
    y_action = (
        df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    )
    pred = np.full(len(df), HOLD_INDEX)
    return {
        "accuracy": float(accuracy_score(y_action, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_action, pred)),
        "rotation_recall": 0.0,
    }


def nearest_enemy_rule(df: pd.DataFrame) -> dict:
    """Rotate toward the nearest visible enemy area unless already there."""
    predicted = []
    for row in df.itertuples(index=False):
        enemy_area = str(row.enemy_nearest_area)
        if enemy_area in {"A", "B", "Mid", "CT"} and row.area != enemy_area:
            predicted.append(
                ACTIONS.index(f"ROTATE_{'MID' if enemy_area == 'Mid' else enemy_area}")
            )
        else:
            predicted.append(HOLD_INDEX)
    y_action = (
        df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    )
    predicted = np.asarray(predicted)
    rotation_mask = y_action != HOLD_INDEX
    return {
        "accuracy": float(accuracy_score(y_action, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_action, predicted)),
        "rotation_precision": float(
            np.mean(y_action[predicted != HOLD_INDEX] != HOLD_INDEX)
        )
        if np.any(predicted != HOLD_INDEX)
        else 0.0,
        "rotation_recall": float(
            np.mean(predicted[rotation_mask] != HOLD_INDEX)
        ),
        "false_rotation_rate": float(
            np.mean(predicted[y_action == HOLD_INDEX] != HOLD_INDEX)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_OUT / "event_policy.json"
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data)
    print(
        f"rows={len(df)} events={df['event_id'].nunique()} "
        f"matches={df['match_id'].nunique()} "
        f"rotations={int((df['action'] != 'HOLD').sum())}"
    )

    feature_sets = {
        "base": (BASE_CAT, BASE_NUM),
        "base_plus_enemy": (BASE_CAT + ENEMY_CAT, BASE_NUM + ENEMY_NUM),
    }
    margins = {
        "all": np.ones(len(df), dtype=bool),
        "distance_le_2000": (df["enemy_nearest_distance"] <= 2000).to_numpy(),
        "distance_le_1200": (df["enemy_nearest_distance"] <= 1200).to_numpy(),
        "self_spots_enemy": (df["self_spots_enemy"] == 1).to_numpy(),
    }

    results: dict = {
        "rows": int(len(df)),
        "events": int(df["event_id"].nunique()),
        "matches": int(df["match_id"].nunique()),
        "action_distribution": {
            str(k): int(v) for k, v in df["action"].value_counts().items()
        },
        "margins": {name: int(mask.sum()) for name, mask in margins.items()},
        "baselines": {
            "majority_hold": majority_baseline(df),
            "nearest_enemy_rule": nearest_enemy_rule(df),
        },
        "policies": {},
    }

    for name, (cat_features, num_features) in feature_sets.items():
        result = run_policy(df, cat_features, num_features)
        results["policies"][name] = {
            "cat_features": cat_features,
            "num_features": num_features,
            "overall": clean(result),
            "subsets": {
                subset: subset_metrics(result, mask)
                for subset, mask in margins.items()
            },
        }
        print(f"\n=== {name} ===")
        print("stage1 ranking:", json.dumps(result["stage1_ranking"], ensure_ascii=False))
        print(
            "stage2:",
            json.dumps(result["stage2_on_true_rotations"], ensure_ascii=False),
        )
        for threshold, item in result["threshold_sweep"].items():
            rotation = item["rotation"]
            end = item["end_to_end"]
            print(
                f"thr {threshold}: prec={rotation['precision']:.3f} "
                f"rec={rotation['recall']:.3f} "
                f"false={rotation['false_rotation_rate']:.3f} "
                f"exact={end['rotation_exact_accuracy']:.3f} "
                f"overall_acc={end['accuracy']:.3f}"
            )
        for subset, metrics in results["policies"][name]["subsets"].items():
            print(
                f"  {subset}: n={metrics['n']} rot={metrics['rotation_count']} "
                + (
                    f"auc={metrics.get('roc_auc', float('nan')):.3f} "
                    f"ap={metrics.get('average_precision', float('nan')):.3f} "
                    f"p@.6={metrics.get('at_0_6', {}).get('precision', float('nan')):.3f} "
                    f"r@.6={metrics.get('at_0_6', {}).get('recall', float('nan')):.3f}"
                )
            )

    # Seed robustness: does the enemy-arm improvement survive a different RF seed?
    robustness: dict = {}
    for name, (cat_features, num_features) in feature_sets.items():
        runs = []
        for seed in [42, 7, 2024]:
            result = run_policy(df, cat_features, num_features, seed=seed)
            swing = result["threshold_sweep"]["0.6"]
            runs.append(
                {
                    "seed": seed,
                    "stage1_auc": result["stage1_ranking"]["roc_auc"],
                    "stage1_ap": result["stage1_ranking"]["average_precision"],
                    "stage2_accuracy": result["stage2_on_true_rotations"]["accuracy"],
                    "stage2_top2": result["stage2_on_true_rotations"]["top2_coverage"],
                    "at_0.6_precision": swing["rotation"]["precision"],
                    "at_0.6_recall": swing["rotation"]["recall"],
                    "at_0.6_false_rotation": swing["rotation"]["false_rotation_rate"],
                    "at_0.6_exact_rotation": swing["end_to_end"][
                        "rotation_exact_accuracy"
                    ],
                    "at_0.6_overall_accuracy": swing["end_to_end"]["accuracy"],
                }
            )
        frame = pd.DataFrame(runs)
        robustness[name] = {
            "runs": runs,
            "mean": {
                column: float(frame[column].mean())
                for column in frame.columns
                if column != "seed"
            },
            "std": {
                column: float(frame[column].std(ddof=0))
                for column in frame.columns
                if column != "seed"
            },
        }
        print(f"\n{name} robustness over seeds 42/7/2024:")
        print(frame.to_string(index=False))
    results["seed_robustness"] = robustness

    # Feature importance on all rows.
    X = design_matrix(df, BASE_CAT + ENEMY_CAT, BASE_NUM + ENEMY_NUM)
    y_rotate = (
        df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
        != HOLD_INDEX
    ).astype(int)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X, y_rotate)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    results["stage1_feature_importance_top20"] = {
        str(k): float(v) for k, v in importance.head(20).items()
    }
    print("\nstage1 feature importance:")
    print(importance.head(20).to_string())

    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
