"""Train the 5 s event policy on all demos and export a pure-Python artifact.

The live server must run without scikit-learn, so the two random forests are
serialised tree-by-tree into JSON. `live/rotation_policy.py` implements the
same traversal in the standard library.

The artifact contains:
- feature order, categorical mappings and numeric medians;
- stage 1 (ROTATE vs HOLD) trees and threshold;
- stage 2 (destination) trees and class order;
- metadata used by the live page and the shadow log.

Run:
    uv run --with numpy --with pandas --with pyarrow --with scikit-learn \
        python benchmarks/enemy_visibility/export_event_policy.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "enemy_visibility" / "event_decisions_5s.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "live" / "models" / "rotation_policy.json"

_spec = importlib.util.spec_from_file_location(
    "event_policy", HERE / "run_event_policy.py"
)
policy = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(policy)

_spec_bd = importlib.util.spec_from_file_location(
    "enemy_build_dataset", HERE / "build_dataset.py"
)
build_dataset = importlib.util.module_from_spec(_spec_bd)
assert _spec_bd.loader is not None
_spec_bd.loader.exec_module(build_dataset)

ACTIONS = policy.ACTIONS
HOLD_INDEX = policy.HOLD_INDEX
CAT_FEATURES = policy.BASE_CAT + policy.ENEMY_CAT
NUM_FEATURES = policy.BASE_NUM + policy.ENEMY_NUM


def export_tree(estimator) -> dict:
    tree = estimator.tree_
    return {
        "feature": tree.feature.tolist(),
        "threshold": [float(value) for value in tree.threshold],
        "left": tree.children_left.tolist(),
        "right": tree.children_right.tolist(),
        "value": tree.value[:, 0, :].tolist(),
    }


def export_forest(estimator) -> list[dict]:
    return [export_tree(step) for step in estimator.estimators_]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    X = policy.design_matrix(df, CAT_FEATURES, NUM_FEATURES)
    y_action = (
        df["action"].astype(pd.CategoricalDtype(ACTIONS, ordered=True)).cat.codes.to_numpy()
    )
    y_rotate = (y_action != HOLD_INDEX).astype(int)

    stage1 = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ).fit(X, y_rotate)

    rotation_mask = y_rotate == 1
    stage2 = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ).fit(X[rotation_mask], y_action[rotation_mask])

    cat_mappings = {
        column: sorted(df[column].astype(str).unique().tolist()) for column in CAT_FEATURES
    }
    num_medians = {
        column: float(pd.to_numeric(df[column], errors="coerce").median())
        for column in NUM_FEATURES
    }

    # A handful of encoded rows with sklearn's outputs, used by the stdlib test
    # to prove the JSON traversal matches the trained forests.
    sample = X.sample(min(20, len(X)), random_state=0)
    stage1_sample = stage1.predict_proba(sample)[:, list(stage1.classes_).index(1)]
    stage2_sample = stage2.predict_proba(sample)
    stage2_actions = stage2.classes_[stage2_sample.argmax(axis=1)]
    parity_examples = [
        {
            "encoded": {column: float(value) for column, value in row.items()},
            "stage1_proba": float(stage1_sample[index]),
            "stage2_action": int(stage2_actions[index]),
            "stage2_proba": [float(value) for value in stage2_sample[index]],
        }
        for index, (_, row) in enumerate(sample.iterrows())
    ]

    artifact = {
        "version": 1,
        "trained_on": str(args.data),
        "trained_at": str(date.today()),
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "horizon_seconds": float(df["horizon_seconds"].iloc[0]),
        "feature_order": CAT_FEATURES + NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "cat_mappings": cat_mappings,
        "num_medians": num_medians,
        "area_centroids": {
            area: [float(value) for value in centre]
            for area, centre in build_dataset.area_centroids().items()
        },
        "stage1": {
            "classes": [int(value) for value in stage1.classes_],
            "trees": export_forest(stage1),
        },
        "stage2": {
            "classes": [int(value) for value in stage2.classes_],
            "trees": export_forest(stage2),
        },
        "threshold": args.threshold,
        "actions": ACTIONS,
        "parity_examples": parity_examples,
        "notes": (
            "Stage 1 predicts ROTATE vs HOLD; stage 2 predicts the destination "
            "conditional on removing. Enemy features are radar-derived. The "
            "policy must not be used without fresh radar/teammate input."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"rows={len(df)} trees={args.trees} threshold={args.threshold} "
        f"size_mb={args.output.stat().st_size / 1e6:.2f}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
