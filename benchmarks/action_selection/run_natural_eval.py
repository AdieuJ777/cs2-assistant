"""Natural-distribution evaluation and rotation threshold sweep.

The balanced 100-row sample is useful for per-class comparison, but live play is
dominated by HOLD (about 92% of decisions). This script evaluates the
GSI+radar/HUD feature set on all 13,560 decisions, and sweeps a HOLD threshold
to show the trade-off between false rotations and missed rotations.

Run with the cached environment or any environment with pandas/sklearn:
    python benchmarks/action_selection/run_natural_eval.py
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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
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
CAT_FEATURES, NUM_FEATURES = ablation.FEATURE_SETS["gsi_radar_hud"]
HOLD_INDEX = ACTIONS.index("HOLD")


def compute_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    labels = list(range(len(ACTIONS)))
    true_rotation = y != HOLD_INDEX
    pred_rotation = pred != HOLD_INDEX
    true_positive = int(np.sum(true_rotation & pred_rotation))
    false_positive = int(np.sum(~true_rotation & pred_rotation))
    false_negative = int(np.sum(true_rotation & ~pred_rotation))
    rotation_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else None
    )
    rotation_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else None
    )
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, average="macro", labels=labels, zero_division=0)
        ),
        "per_class_recall": {
            action: float(value)
            for action, value in zip(
                ACTIONS,
                recall_score(
                    y, pred, average=None, labels=labels, zero_division=0
                ),
            )
        },
        "hold_recall": float(np.mean(pred[~true_rotation] == HOLD_INDEX)),
        "rotation_precision": rotation_precision,
        "rotation_recall": rotation_recall,
        "false_rotation_rate": float(false_positive / max(int(np.sum(~true_rotation)), 1)),
        "missed_rotation_rate": float(false_negative / max(int(np.sum(true_rotation)), 1)),
        "true_rotation_share": float(np.mean(true_rotation)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def main() -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    X = ablation.design_matrix(df, CAT_FEATURES, NUM_FEATURES)
    y = df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    groups = df["match_id"].to_numpy()
    cv = GroupKFold(n_splits=5)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    )
    probabilities = cross_val_predict(
        model, X, y, groups=groups, cv=cv, method="predict_proba"
    )
    argmax = probabilities.argmax(axis=1)

    results: dict = {
        "n": int(len(df)),
        "features": {"cat": CAT_FEATURES, "num": NUM_FEATURES},
        "action_distribution": {
            action: int(np.sum(y == index))
            for index, action in enumerate(ACTIONS)
        },
        "argmax": compute_metrics(y, argmax),
        "threshold_sweep": {},
    }

    # A rotation is only emitted when the HOLD probability is below the
    # threshold; otherwise the advice is HOLD. This trades recall for fewer
    # false rotations.
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = argmax.copy()
        pred[probabilities[:, HOLD_INDEX] >= threshold] = HOLD_INDEX
        results["threshold_sweep"][str(threshold)] = compute_metrics(y, pred)

    (RESULTS_OUT / "natural_distribution_eval.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"rows={len(df)} matches={df['match_id'].nunique()}")
    print("action distribution:", results["action_distribution"])
    print("\n=== argmax ===")
    print(json.dumps(results["argmax"], indent=2, ensure_ascii=False))
    print("\n=== HOLD threshold sweep ===")
    print(
        f"{'thr':>5} {'acc':>7} {'bal_acc':>8} {'hold_rec':>9} "
        f"{'rot_prec':>9} {'rot_rec':>8} {'false_rot':>10}"
    )
    for threshold, metrics in results["threshold_sweep"].items():
        print(
            f"{threshold:>5} {metrics['accuracy']:>7.3f} "
            f"{metrics['balanced_accuracy']:>8.3f} "
            f"{metrics['hold_recall']:>9.3f} "
            f"{metrics['rotation_precision']:>9.3f} "
            f"{metrics['rotation_recall']:>8.3f} "
            f"{metrics['false_rotation_rate']:>10.3f}"
        )
    print(f"\nwrote {RESULTS_OUT / 'natural_distribution_eval.json'}")


if __name__ == "__main__":
    main()
