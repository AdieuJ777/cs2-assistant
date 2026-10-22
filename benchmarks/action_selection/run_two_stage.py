"""Two-stage action policy for the live GSI + radar/HUD plan.

Stage 1: ROTATE vs HOLD, with class weights to counter the 92% HOLD base rate.
Stage 2: among rotations, choose A / B / MID / CT, trained only on true rotations.

The natural-distribution single multi-class model collapses to HOLD (rotation
recall ~15%). This script tests whether a two-stage policy recovers useful
rotation recall without an unacceptable false-rotation rate.

Run with the cached environment:
    python benchmarks/action_selection/run_two_stage.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "action_selection" / "decisions.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "action_selection"

_spec = importlib.util.spec_from_file_location("action_ablation", HERE / "run_ablation.py")
ablation = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ablation)

ACTIONS = ablation.ACTIONS
HOLD_INDEX = ACTIONS.index("HOLD")
ROTATION_ACTIONS = [action for action in ACTIONS if action != "HOLD"]
CAT_FEATURES, NUM_FEATURES = ablation.FEATURE_SETS["gsi_radar_hud"]


def _rotation_metrics(y_rotate: np.ndarray, pred_rotate: np.ndarray, proba: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_rotate, pred_rotate, average="binary", zero_division=0
    )
    true_negative = int(np.sum((y_rotate == 0) & (pred_rotate == 0)))
    false_positive = int(np.sum((y_rotate == 0) & (pred_rotate == 1)))
    false_negative = int(np.sum((y_rotate == 1) & (pred_rotate == 0)))
    true_positive = int(np.sum((y_rotate == 1) & (pred_rotate == 1)))
    return {
        "n": int(len(y_rotate)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_rotation_rate": float(false_positive / max(int(np.sum(y_rotate == 0)), 1)),
        "missed_rotation_rate": float(false_negative / max(int(np.sum(y_rotate == 1)), 1)),
        "roc_auc": float(roc_auc_score(y_rotate, proba)),
        "average_precision": float(average_precision_score(y_rotate, proba)),
    }


def _destination_metrics(y_action: np.ndarray, dest_pred: np.ndarray, dest_proba: np.ndarray) -> dict:
    mask = y_action != HOLD_INDEX
    y_dest = y_action[mask]
    pred = dest_pred[mask]
    labels = list(range(len(ACTIONS)))
    top2 = np.argsort(dest_proba[mask], axis=1)[:, -2:]
    return {
        "n": int(len(y_dest)),
        "accuracy": float(accuracy_score(y_dest, pred)),
        "macro_f1": float(
            f1_score(y_dest, pred, average="macro", labels=labels, zero_division=0)
        ),
        "top2_coverage": float(
            np.mean([target in row for target, row in zip(y_dest, top2)])
        ),
        "per_class_recall": {
            action: float(value)
            for action, value in zip(
                ACTIONS,
                recall_score(
                    y_dest, pred, average=None, labels=labels, zero_division=0
                ),
            )
        },
        "confusion_matrix": confusion_matrix(y_dest, pred, labels=labels).tolist(),
    }


def _end_to_end(y_action: np.ndarray, rotate_pred: np.ndarray, dest_pred: np.ndarray) -> dict:
    pred_action = np.where(rotate_pred, dest_pred, HOLD_INDEX)
    rotation_mask = y_action != HOLD_INDEX
    exact_rotation = (rotate_pred & (pred_action == y_action))
    labels = list(range(len(ACTIONS)))
    return {
        "accuracy": float(accuracy_score(y_action, pred_action)),
        "balanced_accuracy": float(balanced_accuracy_score(y_action, pred_action)),
        "macro_f1": float(
            f1_score(y_action, pred_action, average="macro", labels=labels, zero_division=0)
        ),
        "rotation_exact_accuracy": float(np.mean(exact_rotation[rotation_mask])),
        "rotation_detected_and_correct": int(np.sum(exact_rotation)),
        "rotation_total": int(np.sum(rotation_mask)),
    }


def main() -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    X = ablation.design_matrix(df, CAT_FEATURES, NUM_FEATURES)
    y_action = df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    y_rotate = (y_action != HOLD_INDEX).astype(int)
    groups = df["match_id"].to_numpy()
    cv = GroupKFold(n_splits=5)

    rotate_proba = np.zeros(len(df))
    dest_pred = np.full(len(df), HOLD_INDEX)
    dest_proba = np.zeros((len(df), len(ACTIONS)))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_action, groups), start=1):
        stage1 = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        stage1.fit(X.iloc[train_idx], y_rotate[train_idx])
        stage1_proba = stage1.predict_proba(X.iloc[test_idx])
        rotate_proba[test_idx] = stage1_proba[:, list(stage1.classes_).index(1)]

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
        print(f"fold {fold}: stage2 classes={list(stage2.classes_)}")

    results: dict = {
        "rows": int(len(df)),
        "features": {"cat": CAT_FEATURES, "num": NUM_FEATURES},
        "rotation_share": float(np.mean(y_rotate)),
        "stage1_at_0_5": _rotation_metrics(
            y_rotate, (rotate_proba >= 0.5).astype(int), rotate_proba
        ),
        "stage2_on_true_rotations": _destination_metrics(y_action, dest_pred, dest_proba),
        "threshold_sweep": {},
    }

    # Single multi-class model with class weights, for comparison.
    weighted_multi = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    weighted_pred = cross_val_predict(
        weighted_multi, X, y_action, groups=groups, cv=cv
    )
    labels = list(range(len(ACTIONS)))
    results["weighted_multi_class"] = {
        "accuracy": float(accuracy_score(y_action, weighted_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_action, weighted_pred)),
        "macro_f1": float(
            f1_score(y_action, weighted_pred, average="macro", labels=labels, zero_division=0)
        ),
        "per_class_recall": {
            action: float(value)
            for action, value in zip(
                ACTIONS,
                recall_score(
                    y_action, weighted_pred, average=None, labels=labels, zero_division=0
                ),
            )
        },
    }

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rotate_pred = (rotate_proba >= threshold).astype(int)
        key = f"{threshold:.1f}"
        results["threshold_sweep"][key] = {
            "rotation": _rotation_metrics(y_rotate, rotate_pred, rotate_proba),
            "end_to_end": _end_to_end(y_action, rotate_pred, dest_pred),
        }

    (RESULTS_OUT / "two_stage_policy.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nrows={len(df)} matches={df['match_id'].nunique()} "
          f"rotation share={results['rotation_share']:.3f}")
    print("single weighted multi-class:", json.dumps(results["weighted_multi_class"], ensure_ascii=False))
    print("\nstage1 @0.5:", json.dumps(results["stage1_at_0_5"], ensure_ascii=False))
    print("stage2 on true rotations:", json.dumps(results["stage2_on_true_rotations"], ensure_ascii=False))
    print("\nthreshold sweep:")
    print(
        f"{'thr':>5} {'rot_prec':>9} {'rot_rec':>8} {'false_rot':>10} "
        f"{'missed':>8} {'exact_rot':>10} {'overall_acc':>11}"
    )
    for threshold, item in results["threshold_sweep"].items():
        rotation = item["rotation"]
        end = item["end_to_end"]
        print(
            f"{threshold:>5} {rotation['precision']:>9.3f} "
            f"{rotation['recall']:>8.3f} {rotation['false_rotation_rate']:>10.3f} "
            f"{rotation['missed_rotation_rate']:>8.3f} "
            f"{end['rotation_exact_accuracy']:>10.3f} "
            f"{end['accuracy']:>11.3f}"
        )
    print(f"\nwrote {RESULTS_OUT / 'two_stage_policy.json'}")


if __name__ == "__main__":
    main()
