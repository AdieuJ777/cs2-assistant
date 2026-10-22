"""Benchmark: infer the enemy (T) buy type before a round from CT-observable info.

This is the first de-risking benchmark for the CS2 decision assistant. It uses the
released TAGNet Dust2 CT dataset. The benchmark is deliberately conservative:

- one row per round; the first round of each match is dropped because lagged
  features are unavailable;
- only pre-round observable features are used: score, round number, previous-round
  outcome/economy, loss streaks, pistol flag;
- the current round's T economy label is never used as a feature;
- evaluation is grouped by match to avoid leakage across rounds of the same match.

It does NOT benchmark individual buy/save or enemy positions. The released dataset
has no per-player money, weapon inventory, utility inventory, T-player positions,
or kill feed. See README.md for the data audit.

Run:
    uv run --with pandas --with pyarrow --with scikit-learn \
        python benchmarks/economy_inference/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    PROJECT_ROOT.parent
    / "TAGNet-CS2-main"
    / "data"
    / "processed"
    / "dust2_ct_5snap"
)
DATA_OUT = PROJECT_ROOT / "data" / "economy_inference"
RESULTS_OUT = PROJECT_ROOT / "results" / "economy_inference"

CAT_FEATURES = [
    "prev_t_econ_type",
    "prev_ct_econ_type",
    "prev_winner_side",
    "prev_round_winner_side",
    "prev_round_attack_site",
    "round_phase",
]
NUM_FEATURES = [
    "round_num",
    "score_diff_ct_minus_t_before",
    "ct_team_score_before",
    "t_team_score_before",
    "prev_score_diff",
    "prev_t_total_equip",
    "prev_ct_total_equip",
    "t_loss_streak_before",
    "ct_loss_streak_before",
    "pistol_round_flag",
]
LABEL = "t_econ_type"


def _loss_streaks(winner_sides: pd.Series) -> tuple[list[int], list[int]]:
    """Return (T loss streak before, CT loss streak before) for a match sequence."""
    t_streak: list[int] = []
    ct_streak: list[int] = []
    t_current = 0
    ct_current = 0
    for winner in winner_sides.tolist():
        t_streak.append(t_current)
        ct_streak.append(ct_current)
        if winner == "ct":
            t_current += 1
            ct_current = 0
        elif winner == "t":
            ct_current += 1
            t_current = 0
        else:
            t_current = 0
            ct_current = 0
    return t_streak, ct_streak


def build_round_table(data_dir: Path) -> pd.DataFrame:
    rounds = pd.read_parquet(data_dir / "ct_round_samples.parquet")
    context = pd.read_parquet(data_dir / "ct_round_context.parquet")
    context_10 = context[context["second"] == 10][
        [
            "sample_id",
            "t_econ_type",
            "ct_econ_type",
            "t_total_equip",
            "ct_total_equip",
        ]
    ].copy()

    df = rounds.merge(context_10, on="sample_id", how="left")
    df = df.sort_values(["match_id", "round_num"]).reset_index(drop=True)

    # Lagged previous-round features: available before the current round starts.
    lag_map = {
        "t_econ_type": "prev_t_econ_type",
        "ct_econ_type": "prev_ct_econ_type",
        "winner_side": "prev_winner_side",
        "score_diff_ct_minus_t_before": "prev_score_diff",
        "t_total_equip": "prev_t_total_equip",
        "ct_total_equip": "prev_ct_total_equip",
    }
    for source, target in lag_map.items():
        df[target] = df.groupby("match_id")[source].shift(1)

    t_streaks: list[int] = []
    ct_streaks: list[int] = []
    for _, group in df.groupby("match_id", sort=False):
        t_s, ct_s = _loss_streaks(group["winner_side"])
        t_streaks.extend(t_s)
        ct_streaks.extend(ct_s)
    df["t_loss_streak_before"] = t_streaks
    df["ct_loss_streak_before"] = ct_streaks

    df["round_phase"] = np.where(
        df["round_num"] <= 12,
        "first_half",
        np.where(df["round_num"] <= 24, "second_half", "overtime"),
    )
    df["pistol_round_flag"] = df["pistol_round_flag"].astype(int)

    # The first round of a match has no lag features; drop it.
    df = df[df["prev_t_econ_type"].notna()].reset_index(drop=True)
    return df


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CAT_FEATURES + NUM_FEATURES].copy()
    for column in CAT_FEATURES:
        X[column] = X[column].astype("category").cat.codes
    return X


def hand_rule(row: pd.Series) -> str:
    """A deliberately simple CS2 economy prior, used as a rule baseline."""
    if row["pistol_round_flag"]:
        return "pistol"
    if row["prev_winner_side"] == "t":
        return "long_gun"
    if row["prev_t_econ_type"] == "eco":
        return "eco"
    return "half_buy"


def evaluate_classifier(name: str, y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class_recall": {
            cls: float(value)
            for cls, value in zip(
                classes,
                recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def brier_score(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    one_hot = np.zeros((len(y_true), n_classes))
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    df = build_round_table(DEFAULT_DATA_DIR)
    X = design_matrix(df)
    y = df[LABEL].astype("category")
    classes = list(y.cat.categories)
    y_codes = y.cat.codes.to_numpy()
    groups = df["match_id"].to_numpy()

    df.to_parquet(DATA_OUT / "rounds.parquet", index=False)

    audit = {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "label_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "majority_share": float(y.value_counts(normalize=True).max()),
        "splits": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "features_categorical": CAT_FEATURES,
        "features_numeric": NUM_FEATURES,
        "label": LABEL,
        "note": (
            "Pre-round observable features only. The current round's T economy, "
            "T positions, per-player money, weapons and utility are not available "
            "in the released TAGNet dataset."
        ),
    }
    (DATA_OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    results: dict = {
        "audit": audit,
        "classifiers": [],
        "regression": {},
    }

    rule_predictions = df.apply(hand_rule, axis=1).map({cls: idx for idx, cls in enumerate(classes)}).to_numpy()
    results["classifiers"].append(evaluate_classifier("hand_rule", y_codes, rule_predictions, classes))

    cv = GroupKFold(n_splits=5)
    classifiers = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "tree_depth3": DecisionTreeClassifier(max_depth=3, random_state=42),
        "tree_depth6": DecisionTreeClassifier(max_depth=6, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
        ),
    }
    for name, model in classifiers.items():
        predictions = cross_val_predict(model, X, y_codes, groups=groups, cv=cv)
        entry = evaluate_classifier(name, y_codes, predictions, classes)
        if hasattr(model, "predict_proba"):
            probabilities = cross_val_predict(
                model, X, y_codes, groups=groups, cv=cv, method="predict_proba"
            )
            entry["brier_score"] = brier_score(y_codes, probabilities, len(classes))
        results["classifiers"].append(entry)

    regressor = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    equipment_predictions = cross_val_predict(
        regressor, X, df["t_total_equip"].to_numpy(), groups=groups, cv=cv
    )
    results["regression"]["t_total_equip_mae"] = float(
        mean_absolute_error(df["t_total_equip"], equipment_predictions)
    )
    results["regression"]["t_total_equip_mean"] = float(df["t_total_equip"].mean())

    (RESULTS_OUT / "baselines.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"rows={len(df)} matches={df['match_id'].nunique()} classes={classes}")
    print(f"label distribution: {audit['label_distribution']} majority={audit['majority_share']:.3f}")
    print()
    for entry in results["classifiers"]:
        brier = f" brier={entry['brier_score']:.3f}" if "brier_score" in entry else ""
        print(
            f"{entry['name']:>14}: acc={entry['accuracy']:.3f} "
            f"macroF1={entry['macro_f1']:.3f}{brier}"
        )
    print()
    print(f"t_total_equip MAE={results['regression']['t_total_equip_mae']:.0f} "
          f"(mean={results['regression']['t_total_equip_mean']:.0f})")
    print(f"wrote {DATA_OUT / 'rounds.parquet'}")
    print(f"wrote {RESULTS_OUT / 'baselines.json'}")


if __name__ == "__main__":
    main()
