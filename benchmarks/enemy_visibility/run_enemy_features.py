"""Evaluate enemy last-seen features on the nine-demo dataset.

Compares the two-stage policy with base GSI/CT features against the same policy
plus enemy last-seen features derived from the demo `spotted` flag.

The enemy signal is sparse: only ~6.5% of decisions have any T player spotted
within 5 s. We therefore report both the full dataset and the "enemy seen"
subset.

Run:
    python benchmarks/enemy_visibility/run_enemy_features.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "enemy_visibility" / "decisions.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "enemy_visibility"

_spec_ab = importlib.util.spec_from_file_location(
    "action_ablation", HERE.parent / "action_selection" / "run_ablation.py"
)
ablation = importlib.util.module_from_spec(_spec_ab)
assert _spec_ab.loader is not None
_spec_ab.loader.exec_module(ablation)

_spec_ts = importlib.util.spec_from_file_location(
    "action_two_stage", HERE.parent / "action_selection" / "run_two_stage.py"
)
two_stage = importlib.util.module_from_spec(_spec_ts)
assert _spec_ts.loader is not None
_spec_ts.loader.exec_module(two_stage)

ACTIONS = ablation.ACTIONS
HOLD_INDEX = ACTIONS.index("HOLD")
BASE_CAT, BASE_NUM = ablation.FEATURE_SETS["gsi_radar_hud"]
ENEMY_CAT = ["enemy_most_recent_area"]
ENEMY_NUM = [
    "enemy_alive",
    "enemy_seen_now",
    "enemy_seen_5s",
    "enemy_unknown",
    "enemy_seen5_ratio",
    "enemy_last_seen_min_age",
    "enemy_most_recent_age",
    "enemy_seen_now_A",
    "enemy_seen_now_B",
    "enemy_seen_now_Mid",
    "enemy_seen_now_CT",
    "enemy_seen5_A",
    "enemy_seen5_B",
    "enemy_seen5_Mid",
    "enemy_seen5_CT",
    "enemy_age_min_A",
    "enemy_age_min_B",
    "enemy_age_min_Mid",
    "enemy_age_min_CT",
]


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
        stage2_proba = stage2.predict_proba(X.iloc[test_idx])
        for column, action_index in enumerate(stage2.classes_):
            dest_proba[test_idx, action_index] = stage2_proba[:, column]
        dest_pred[test_idx] = stage2.classes_[stage2_proba.argmax(axis=1)]

    result = {
        "n": int(len(df)),
        "rotation_share": float(np.mean(y_rotate)),
        "rotation_count": int(np.sum(y_rotate)),
        "stage1_at_0_5": two_stage._rotation_metrics(
            y_rotate, (rotate_proba >= 0.5).astype(int), rotate_proba
        ),
        "stage1_ranking": {
            "roc_auc": float(roc_auc_score(y_rotate, rotate_proba)),
            "average_precision": float(average_precision_score(y_rotate, rotate_proba)),
        },
        "stage2_on_true_rotations": two_stage._destination_metrics(
            y_action, dest_pred, dest_proba
        ),
        "threshold_sweep": {},
    }
    for threshold in [0.5, 0.6, 0.7]:
        rotate_pred = (rotate_proba >= threshold).astype(int)
        result["threshold_sweep"][str(threshold)] = {
            "rotation": two_stage._rotation_metrics(y_rotate, rotate_pred, rotate_proba),
            "end_to_end": two_stage._end_to_end(y_action, rotate_pred, dest_pred),
        }
    result["_rotate_proba"] = rotate_proba
    result["_dest_pred"] = dest_pred
    result["_y_rotate"] = y_rotate
    result["_y_action"] = y_action
    return result


def subset_metrics(result: dict, mask: np.ndarray) -> dict:
    y_rotate = result["_y_rotate"][mask]
    proba = result["_rotate_proba"][mask]
    if len(np.unique(y_rotate)) < 2:
        return {"n": int(len(y_rotate)), "rotation_count": int(np.sum(y_rotate))}
    return {
        "n": int(len(y_rotate)),
        "rotation_count": int(np.sum(y_rotate)),
        "roc_auc": float(roc_auc_score(y_rotate, proba)),
        "average_precision": float(average_precision_score(y_rotate, proba)),
        "at_0_6_precision": float(
            two_stage._rotation_metrics(
                y_rotate, (proba >= 0.6).astype(int), proba
            )["precision"]
        ),
        "at_0_6_recall": float(
            two_stage._rotation_metrics(
                y_rotate, (proba >= 0.6).astype(int), proba
            )["recall"]
        ),
    }


def clean(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main() -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"rows={len(df)} matches={df['match_id'].nunique()} "
          f"rotations={int((df['action'] != 'HOLD').sum())}")
    print("enemy_seen_5s distribution:", df["enemy_seen_5s"].value_counts().sort_index().to_dict())

    feature_sets = {
        "base": (BASE_CAT, BASE_NUM),
        "base_plus_enemy": (BASE_CAT + ENEMY_CAT, BASE_NUM + ENEMY_NUM),
    }
    results: dict = {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "action_distribution": {
            str(k): int(v) for k, v in df["action"].value_counts().items()
        },
        "enemy_seen5_distribution": {
            str(k): int(v) for k, v in df["enemy_seen_5s"].value_counts().sort_index().items()
        },
        "policies": {},
    }
    seen_mask = (df["enemy_seen_5s"] > 0).to_numpy()

    for name, (cat_features, num_features) in feature_sets.items():
        result = run_policy(df, cat_features, num_features)
        results["policies"][name] = {
            "cat_features": cat_features,
            "num_features": num_features,
            "overall": clean(result),
            "enemy_seen_subset": subset_metrics(result, seen_mask),
        }
        print(f"\n=== {name} ===")
        print("stage1:", json.dumps(result["stage1_ranking"], ensure_ascii=False))
        print("stage2:", json.dumps(result["stage2_on_true_rotations"], ensure_ascii=False))
        for threshold, item in result["threshold_sweep"].items():
            rot = item["rotation"]
            end = item["end_to_end"]
            print(
                f"thr {threshold}: prec={rot['precision']:.3f} rec={rot['recall']:.3f} "
                f"false={rot['false_rotation_rate']:.3f} exact={end['rotation_exact_accuracy']:.3f} "
                f"overall_acc={end['accuracy']:.3f}"
            )
        print("enemy-seen subset:", json.dumps(results["policies"][name]["enemy_seen_subset"], ensure_ascii=False))

    # Feature importance for the full stage-1 model (all rows).
    X = design_matrix(df, BASE_CAT + ENEMY_CAT, BASE_NUM + ENEMY_NUM)
    y_action = df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    y_rotate = (y_action != HOLD_INDEX).astype(int)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X, y_rotate)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    results["stage1_feature_importance_top15"] = {
        str(k): float(v) for k, v in importance.head(15).items()
    }
    print("\nstage1 feature importance:")
    print(importance.head(15).to_string())

    (RESULTS_OUT / "enemy_visibility_policy.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {RESULTS_OUT / 'enemy_visibility_policy.json'}")


if __name__ == "__main__":
    main()
