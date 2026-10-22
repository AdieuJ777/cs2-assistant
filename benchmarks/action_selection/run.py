"""Action-selection benchmark: CT next-move imitation on Dust2.

Given the observable CT team state at snapshot t, predict where a chosen CT
player will be at snapshot t+1. This is an imitation benchmark, not an
optimal-policy benchmark: professional movement is driven by information the
agent does not have (enemy positions, voice comms, kill feed), so accuracy is
only one signal. We also care about top-2 coverage, rotation recall, abstention,
and hallucination.

Actions:
- HOLD
- ROTATE_A, ROTATE_B, ROTATE_MID, ROTATE_CT

The released TAGNet dataset is CT-only, which is enough for this task: we use
CT teammate state, score, economy and round context. No T-player data is needed.

Run:
    uv run --with pandas --with pyarrow --with scikit-learn \
        python benchmarks/action_selection/run.py
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
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    PROJECT_ROOT.parent
    / "TAGNet-CS2-main"
    / "data"
    / "processed"
    / "dust2_ct_5snap"
)
DATA_OUT = PROJECT_ROOT / "data" / "action_selection"
RESULTS_OUT = PROJECT_ROOT / "results" / "action_selection"

WINDOWS = ((10, 12), (12, 15), (15, 18), (18, 20))
ACTIONS = ["HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT"]
AREAS = ["A", "B", "Mid", "CT"]
CAT_FEATURES = [
    "area",
    "ct_econ_type",
    "round_phase",
    "prev_round_winner_side",
    "prev_round_attack_site",
    "nearest_area",
]
NUM_FEATURES = [
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
LABEL = "action"


def _round_phase(round_num: int) -> str:
    if round_num <= 12:
        return "first_half"
    if round_num <= 24:
        return "second_half"
    return "overtime"


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_parquet(DATA_DIR / "ct_snapshot_nodes.parquet")
    rounds = pd.read_parquet(DATA_DIR / "ct_round_samples.parquet")
    context = pd.read_parquet(DATA_DIR / "ct_round_context.parquet")
    return nodes, rounds, context


def area_centroids(nodes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    centroids = nodes.groupby("area")[["X", "Y"]].mean()
    return {
        str(area): (float(row.X), float(row.Y))
        for area, row in centroids.iterrows()
        if area in AREAS
    }


def build_window(
    nodes: pd.DataFrame,
    rounds: pd.DataFrame,
    context: pd.DataFrame,
    centroids: dict[str, tuple[float, float]],
    second_from: int,
    second_to: int,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "match_id",
        "round_num",
        "second",
        "player_id",
        "area",
        "X",
        "Y",
        "health",
        "armor",
        "current_equip_value",
        "has_defuser",
        "has_helmet",
        "split",
    ]
    current = nodes[nodes["second"] == second_from][columns].copy()
    following = nodes[nodes["second"] == second_to][
        ["sample_id", "player_id", "area", "health"]
    ].rename(columns={"area": "next_area", "health": "next_health"})

    merged = current.merge(following, on=["sample_id", "player_id"], how="inner")
    merged = merged[
        (merged["health"] > 0)
        & (merged["next_health"] > 0)
        & (merged["area"].isin(AREAS))
        & (merged["next_area"].isin(AREAS))
    ].copy()
    merged["action"] = np.where(
        merged["area"] == merged["next_area"],
        "HOLD",
        "ROTATE_" + merged["next_area"].astype(str).str.upper(),
    )
    merged["window"] = f"{second_from}->{second_to}"

    # Round-level context at the decision snapshot.
    context_snapshot = context[context["second"] == second_from][
        ["sample_id", "ct_econ_type", "ct_total_equip"]
    ]
    merged = merged.merge(context_snapshot, on="sample_id", how="left")

    round_meta = rounds[
        [
            "sample_id",
            "ct_team_score_before",
            "t_team_score_before",
            "score_diff_ct_minus_t_before",
            "pistol_round_flag",
            "prev_round_winner_side",
            "prev_round_attack_site",
        ]
    ].copy()
    round_meta["prev_round_winner_side"] = round_meta[
        "prev_round_winner_side"
    ].fillna("none")
    merged = merged.merge(round_meta, on="sample_id", how="left")
    merged["round_phase"] = merged["round_num"].map(_round_phase)

    # Team state at t: all CT players alive at t, including those who die by t+1.
    alive_current = current[current["health"] > 0]
    team_stats = (
        alive_current.groupby("sample_id", sort=False)
        .agg(
            count_A=("area", lambda values: int((values == "A").sum())),
            count_B=("area", lambda values: int((values == "B").sum())),
            count_Mid=("area", lambda values: int((values == "Mid").sum())),
            count_CT=("area", lambda values: int((values == "CT").sum())),
            team_health_mean=("health", "mean"),
            team_equip_mean=("current_equip_value", "mean"),
        )
        .reset_index()
    )
    merged = merged.merge(team_stats, on="sample_id", how="left")

    for area in AREAS:
        cx, cy = centroids[area]
        merged[f"dist_{area}"] = np.hypot(merged["X"] - cx, merged["Y"] - cy) / 1000.0
    merged["nearest_area"] = merged[[f"dist_{area}" for area in AREAS]].idxmin(axis=1).str.replace("dist_", "", regex=False)

    return merged.reset_index(drop=True)


def build_decision_table() -> tuple[pd.DataFrame, dict]:
    nodes, rounds, context = load_frames()
    centroids = area_centroids(nodes)
    frames = [
        build_window(nodes, rounds, context, centroids, start, end)
        for start, end in WINDOWS
    ]
    df = pd.concat(frames, ignore_index=True)
    df["round_key"] = df["sample_id"]
    df["split"] = df["split"].astype(str)

    audit = {
        "rows": int(len(df)),
        "rounds": int(df["round_key"].nunique()),
        "matches": int(df["match_id"].nunique()),
        "action_distribution": {
            str(k): int(v) for k, v in df[LABEL].value_counts().items()
        },
        "windows": {str(k): int(v) for k, v in df["window"].value_counts().items()},
        "area_centroids": {
            area: [round(value, 2) for value in xy]
            for area, xy in centroids.items()
        },
        "note": (
            "CT-only next-move imitation. Enemy positions, per-player money, "
            "weapons, utility and kill feed are not in the released dataset."
        ),
    }
    return df, audit


def sample_balanced(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """Sample per-class decisions, at most one per round."""
    selected: list[int] = []
    used_rounds: set[str] = set()
    for action in ACTIONS:
        candidates = df[df[LABEL] == action].sample(frac=1.0, random_state=seed)
        taken = 0
        for index, row in candidates.iterrows():
            if row["round_key"] in used_rounds:
                continue
            selected.append(index)
            used_rounds.add(row["round_key"])
            taken += 1
            if taken >= per_class:
                break
        if taken < per_class:
            print(f"warning: only {taken}/{per_class} available for {action}")
    return df.loc[selected].sort_values(["match_id", "round_num"]).reset_index(drop=True)


def design_matrix(df: pd.DataFrame, categories: dict[str, list[str]] | None = None):
    X = df[CAT_FEATURES + NUM_FEATURES].copy()
    categories = categories or {}
    for column in CAT_FEATURES:
        X[column] = X[column].astype(str)
        if column not in categories:
            categories[column] = sorted(X[column].unique().tolist())
        mapping = {value: index for index, value in enumerate(categories[column])}
        X[column] = X[column].map(mapping).fillna(-1).astype(int)
    return X, categories


def rule_prediction(row: pd.Series) -> str:
    own_count = int(row[f"count_{row['area']}"])
    if own_count >= 2:
        return "HOLD"
    candidates = [
        area
        for area in AREAS
        if area != row["area"] and int(row[f"count_{area}"]) >= 2
    ]
    if not candidates:
        return "HOLD"
    nearest = min(candidates, key=lambda area: float(row[f"dist_{area}"]))
    return f"ROTATE_{nearest.upper()}"


def metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None) -> dict:
    labels = list(range(len(ACTIONS)))
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "per_class_recall": {
            action: float(value)
            for action, value in zip(
                ACTIONS,
                recall_score(
                    y_true, y_pred, average=None, labels=labels, zero_division=0
                ),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if probabilities is not None:
        one_hot = np.zeros((len(y_true), len(ACTIONS)))
        one_hot[np.arange(len(y_true)), y_true] = 1.0
        result["brier"] = float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        )
        top2 = np.argsort(probabilities, axis=1)[:, -2:]
        result["top2_coverage"] = float(np.mean([y in row for y, row in zip(y_true, top2)]))
    return result


def baseline_results(df: pd.DataFrame) -> dict:
    y = df[LABEL].astype(pd.CategoricalDtype(ACTIONS, ordered=True))
    y_codes = y.cat.codes.to_numpy()
    X, _ = design_matrix(df)
    groups = df["match_id"].to_numpy()
    cv = GroupKFold(n_splits=5)

    results: dict[str, dict] = {}
    results["majority_hold"] = metrics(
        y_codes, np.full(len(df), ACTIONS.index("HOLD")), None
    )
    rule_pred = df.apply(rule_prediction, axis=1).map(
        {action: index for index, action in enumerate(ACTIONS)}
    ).to_numpy()
    results["team_need_rule"] = metrics(y_codes, rule_pred, None)

    tree = DecisionTreeClassifier(max_depth=6, random_state=42)
    tree_pred = cross_val_predict(tree, X, y_codes, groups=groups, cv=cv)
    tree_proba = cross_val_predict(
        tree, X, y_codes, groups=groups, cv=cv, method="predict_proba"
    )
    results["tree_depth6"] = metrics(y_codes, tree_pred, tree_proba)

    forest = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    )
    forest_pred = cross_val_predict(forest, X, y_codes, groups=groups, cv=cv)
    forest_proba = cross_val_predict(
        forest, X, y_codes, groups=groups, cv=cv, method="predict_proba"
    )
    results["random_forest"] = metrics(y_codes, forest_pred, forest_proba)
    return results


def build_card(row: pd.Series, teammates: pd.DataFrame) -> dict:
    teammate_rows = []
    for _, mate in teammates.iterrows():
        if mate["player_id"] == row["player_id"] or mate["health"] <= 0:
            continue
        teammate_rows.append(
            {
                "area": str(mate["area"]),
                "health": int(mate["health"]),
                "equip_value": int(mate["current_equip_value"]),
                "distance_to_self_units": int(
                    round(
                        np.hypot(
                            float(mate["X"]) - float(row["X"]),
                            float(mate["Y"]) - float(row["Y"]),
                        )
                    )
                ),
            }
        )
    return {
        "map": "de_dust2",
        "side": "ct",
        "window": str(row["window"]),
        "round_num": int(row["round_num"]),
        "round_phase": str(row["round_phase"]),
        "score": {
            "ct": int(row["ct_team_score_before"]),
            "t": int(row["t_team_score_before"]),
            "diff_ct_minus_t": int(row["score_diff_ct_minus_t_before"]),
        },
        "pistol_round": bool(row["pistol_round_flag"]),
        "ct_econ_type": str(row["ct_econ_type"]),
        "ct_total_equip": int(row["ct_total_equip"]),
        "self": {
            "area": str(row["area"]),
            "health": int(row["health"]),
            "armor": int(row["armor"]),
            "equip_value": int(row["current_equip_value"]),
            "has_defuser": bool(row["has_defuser"]),
            "distance_to_A_units": int(round(float(row["dist_A"]) * 1000)),
            "distance_to_B_units": int(round(float(row["dist_B"]) * 1000)),
            "distance_to_Mid_units": int(round(float(row["dist_Mid"]) * 1000)),
            "distance_to_CT_units": int(round(float(row["dist_CT"]) * 1000)),
        },
        "team": {
            "area_counts": {
                area: int(row[f"count_{area}"]) for area in AREAS
            },
            "teammates": teammate_rows,
        },
        "prev_round": {
            "winner_side": str(row["prev_round_winner_side"]),
            "attack_site": str(row["prev_round_attack_site"]),
        },
        "known_info": {
            "enemy_positions": "unknown",
            "enemy_economy": "unknown",
            "kill_feed": "unknown",
            "voice_comms": "unknown",
        },
    }


def attach_cards(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Attach a JSON card to each sampled decision."""
    by_key = {
        (str(sample_id), int(second)): group
        for (sample_id, second), group in nodes.groupby(
            ["sample_id", "second"], sort=False
        )
    }

    cards = []
    for _, row in df.iterrows():
        second = int(str(row["window"]).split("->")[0])
        teammates = by_key[(str(row["sample_id"]), second)]
        cards.append(json.dumps(build_card(row, teammates), ensure_ascii=False))
    out = df.copy()
    out["card_text"] = cards
    return out


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    df, audit = build_decision_table()
    df.to_parquet(DATA_OUT / "decisions.parquet", index=False)

    sample = sample_balanced(df, per_class=20)
    sample = attach_cards(sample, pd.read_parquet(DATA_DIR / "ct_snapshot_nodes.parquet"))
    sample.to_parquet(DATA_OUT / "sample_100.parquet", index=False)

    baselines = baseline_results(sample)
    payload = {
        "audit": audit,
        "sample": {
            "size": int(len(sample)),
            "action_distribution": {
                str(k): int(v) for k, v in sample[LABEL].value_counts().items()
            },
            "matches": int(sample["match_id"].nunique()),
            "rounds": int(sample["sample_id"].nunique()),
        },
        "baselines_on_sample": baselines,
    }
    (RESULTS_OUT / "baselines.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"decisions={len(df)} rounds={df['sample_id'].nunique()} matches={df['match_id'].nunique()}")
    print("action distribution:", audit["action_distribution"])
    print(f"sample={len(sample)} matches={sample['match_id'].nunique()} rounds={sample['sample_id'].nunique()}")
    print("sample actions:", payload["sample"]["action_distribution"])
    print()
    for name, result in baselines.items():
        top2 = f" top2={result['top2_coverage']:.3f}" if "top2_coverage" in result else ""
        print(
            f"{name:>16}: acc={result['accuracy']:.3f} bal_acc={result['balanced_accuracy']:.3f} "
            f"macroF1={result['macro_f1']:.3f}{top2}"
        )
    print(f"wrote {DATA_OUT / 'decisions.parquet'}")
    print(f"wrote {DATA_OUT / 'sample_100.parquet'}")
    print(f"wrote {RESULTS_OUT / 'baselines.json'}")


if __name__ == "__main__":
    main()
