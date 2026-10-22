"""Feature ablation for the action-selection model.

The live plan is GSI + radar/HUD perception. That gives:

- self: area, health, armor, equipment value, position/distance;
- teammates: area, health (radar/HUD), but NOT equipment value or money;
- match: score, round, phase, previous-round result;
- no team total economy.

The benchmark model currently uses extra features that are not available live:
`ct_econ_type`, `team_equip_mean`, `ct_total_equip`. This script measures how
much performance is lost when those are removed.

Run:
    uv run --offline --with pandas --with pyarrow --with scikit-learn \
        python benchmarks/action_selection/run_ablation.py
"""

from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "action_selection" / "sample_100.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "action_selection"
ACTIONS = ["HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT"]
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}

FULL_CAT = [
    "area",
    "ct_econ_type",
    "round_phase",
    "prev_round_winner_side",
    "prev_round_attack_site",
    "nearest_area",
]
FULL_NUM = [
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
    "score_diff_ct_minus_t_before",
    "round_num",
    "pistol_round_flag",
    "ct_total_equip",
]

FEATURE_SETS = {
    "full_benchmark": (FULL_CAT, FULL_NUM),
    # Drop team economy and teammate equipment; keep GSI/radar/HUD observables.
    "gsi_radar_hud": (
        [c for c in FULL_CAT if c != "ct_econ_type"],
        [
            c
            for c in FULL_NUM
            if c not in {"team_equip_mean", "ct_total_equip"}
        ],
    ),
    # Also drop teammate health (worst-case HUD OCR failure).
    "gsi_radar_no_hud": (
        [c for c in FULL_CAT if c != "ct_econ_type"],
        [
            c
            for c in FULL_NUM
            if c not in {"team_equip_mean", "ct_total_equip", "team_health_mean"}
        ],
    ),
    # Keep only self state + team shape; drop all economy/equipment features.
    "shape_only": (
        ["area", "round_phase", "prev_round_winner_side", "prev_round_attack_site", "nearest_area"],
        [
            "health",
            "armor",
            "count_A",
            "count_B",
            "count_Mid",
            "count_CT",
            "dist_A",
            "dist_B",
            "dist_Mid",
            "dist_CT",
            "score_diff_ct_minus_t_before",
            "round_num",
            "pistol_round_flag",
        ],
    ),
}


def design_matrix(df: pd.DataFrame, cat_features: list[str], num_features: list[str]):
    X = df[cat_features + num_features].copy()
    for column in cat_features:
        X[column] = X[column].astype(str).astype("category").cat.codes
    return X


def evaluate(df: pd.DataFrame, cat_features: list[str], num_features: list[str]) -> dict:
    X = design_matrix(df, cat_features, num_features)
    y = df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    groups = df["match_id"].to_numpy()
    cv = GroupKFold(n_splits=5)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    )
    predictions = cross_val_predict(model, X, y, groups=groups, cv=cv)
    probabilities = cross_val_predict(
        model, X, y, groups=groups, cv=cv, method="predict_proba"
    )
    labels = list(range(len(ACTIONS)))
    top2 = np.argsort(probabilities, axis=1)[:, -2:]
    return {
        "n": int(len(df)),
        "n_features": int(len(cat_features) + len(num_features)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(y, predictions, average="macro", labels=labels, zero_division=0)
        ),
        "top2_coverage": float(np.mean([target in row for target, row in zip(y, top2)])),
        "per_class_recall": {
            action: float(value)
            for action, value in zip(
                ACTIONS,
                recall_score(
                    y, predictions, average=None, labels=labels, zero_division=0
                ),
            )
        },
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
    }


def main() -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    results = {"n": int(len(df)), "feature_sets": {}}
    for name, (cat_features, num_features) in FEATURE_SETS.items():
        result = evaluate(df, cat_features, num_features)
        results["feature_sets"][name] = {
            "cat_features": cat_features,
            "num_features": num_features,
            "metrics": result,
        }
        print(
            f"{name:>18}: features={result['n_features']:>2} "
            f"acc={result['accuracy']:.3f} bal_acc={result['balanced_accuracy']:.3f} "
            f"macroF1={result['macro_f1']:.3f} top2={result['top2_coverage']:.3f}"
        )
        print("   recall:", {k: round(v, 2) for k, v in result["per_class_recall"].items()})

    (RESULTS_OUT / "feature_ablation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {RESULTS_OUT / 'feature_ablation.json'}")


if __name__ == "__main__":
    main()
