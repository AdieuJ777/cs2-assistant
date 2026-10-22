"""LLM arm of the enemy-economy benchmark, executed through STP.

The LLM receives the same pre-round information as the tree/forest baselines,
but as a structured situation card. It must return a probability distribution
over the four enemy buy categories plus a short rationale.

The task is built with the STP framework so this also serves as a fit check:
contract-first input/output, structured LLM output, retries, timeouts, and
event logging.

Run (small smoke test first):
    uv run --with pandas --with pyarrow --with scikit-learn \
        --with pydantic --with jsonschema \
        python benchmarks/economy_inference/run_llm.py --per-class 1

Full pilot (80 rows, ~5-15 min on a local GPU):
    ... run_llm.py --per-class 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
FRAMEWORK_ROOT = (
    PROJECT_ROOT.parent / "STP-Agent-Framework-main" / "STP-Agent-Framework-main"
)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FRAMEWORK_ROOT))

from kkk import RetryPolicy, TaskRuntime, TaskSpec, ToolRegistry, task  # noqa: E402
from kkk.errors import FrameworkError  # noqa: E402

from ollama_backend import OllamaNativeBackend  # noqa: E402
from run import LABEL, design_matrix, hand_rule  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "economy_inference" / "rounds.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "economy_inference"
CLASSES = ["eco", "half_buy", "long_gun", "pistol"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}

SYSTEM_PROMPT = """You are a Counter-Strike 2 economy analyst helping a CT player in a solo-queue match on Dust2.

Facts:
- CT defends the bomb sites; T attacks them.
- Buy categories used by this benchmark:
  - pistol: a pistol round.
  - eco: a low-spend / save round (roughly under $10k of team equipment).
  - half_buy: a partial buy (roughly $10k-$25k of team equipment).
  - long_gun: a full buy (roughly $25k or more of team equipment).
- A team's loss bonus grows with consecutive losses. A team on a losing streak may force-buy or eco.
- The card describes the state BEFORE the coming round. Fields starting with "prev_" describe the previous round.

Task: infer the enemy (T) buy category for the coming round using ONLY the card.
- Do not claim exact money.
- Do not assume enemy positions or any hidden information.
- Probabilities must be calibrated; if unsure, spread them instead of putting 1.0 on one class.
- Keep the rationale short and cite card fields.

Field legend:
- score_ct / score_t / score_diff: current score.
- pistol_round: whether the coming round is a pistol round.
- prev_winner_side: who won the previous round ("ct" or "t").
- prev_attack_site: where T attacked in the previous round ("A", "B", "Mid", "unknown").
- prev_t_econ / prev_ct_econ: previous-round buy category of T and CT.
- prev_t_equip / prev_ct_equip: previous-round team equipment value.
- t_loss_streak / ct_loss_streak: consecutive losses before this round.
"""

RULES_FEW_SHOT_PROMPT = SYSTEM_PROMPT + """

Decision checklist (follow it in order):
1. If pistol_round is true, the answer is pistol. A pistol round is NOT an eco round; both teams buy pistols.
2. If T won the previous round, prev_t_econ is long_gun, and prev_t_equip is at least 25000, long_gun is most likely.
3. If T lost the previous round, t_loss_streak is at least 2, and prev_t_econ is eco or half_buy, the choice is usually eco or half_buy; do not default to long_gun.
4. If prev_t_econ is long_gun and T lost the previous round, T may save or half-buy; spread probability between half_buy and eco.
5. If prev_t_econ is half_buy and T won, half_buy or long_gun are both plausible.
6. Use prev_t_equip and prev_ct_equip only as ranges. Never write an exact dollar amount.

Examples:

Example 1 (pistol):
Card: {"round_num":13,"score_ct":6,"score_t":6,"pistol_round":true,"prev_winner_side":"ct","prev_attack_site":"A","prev_t_econ":"long_gun","prev_ct_econ":"long_gun","prev_t_equip":24000,"prev_ct_equip":26000,"t_loss_streak":0,"ct_loss_streak":0,"round_phase":"second_half"}
Output: {"enemy_buy":"pistol","probabilities":{"eco":0.02,"half_buy":0.03,"long_gun":0.0,"pistol":0.95},"confidence":0.95,"rationale":"pistol_round is true; both teams are on pistols."}

Example 2 (eco):
Card: {"round_num":5,"score_ct":3,"score_t":1,"pistol_round":false,"prev_winner_side":"ct","prev_attack_site":"A","prev_t_econ":"eco","prev_ct_econ":"long_gun","prev_t_equip":8500,"prev_ct_equip":24000,"t_loss_streak":3,"ct_loss_streak":0,"round_phase":"first_half"}
Output: {"enemy_buy":"eco","probabilities":{"eco":0.55,"half_buy":0.35,"long_gun":0.08,"pistol":0.02},"confidence":0.55,"rationale":"T has a 3-loss streak and a low previous equipment value; eco or half-buy are most likely."}

Example 3 (long_gun):
Card: {"round_num":8,"score_ct":3,"score_t":4,"pistol_round":false,"prev_winner_side":"t","prev_attack_site":"B","prev_t_econ":"long_gun","prev_ct_econ":"half_buy","prev_t_equip":26500,"prev_ct_equip":15000,"t_loss_streak":0,"ct_loss_streak":1,"round_phase":"first_half"}
Output: {"enemy_buy":"long_gun","probabilities":{"eco":0.02,"half_buy":0.08,"long_gun":0.88,"pistol":0.02},"confidence":0.88,"rationale":"T won the previous round with a long-gun buy and a high equipment value; they can afford a full buy."}

Example 4 (half_buy):
Card: {"round_num":11,"score_ct":6,"score_t":4,"pistol_round":false,"prev_winner_side":"ct","prev_attack_site":"Mid","prev_t_econ":"half_buy","prev_ct_econ":"long_gun","prev_t_equip":18000,"prev_ct_equip":27000,"t_loss_streak":1,"ct_loss_streak":0,"round_phase":"first_half"}
Output: {"enemy_buy":"half_buy","probabilities":{"eco":0.10,"half_buy":0.65,"long_gun":0.23,"pistol":0.02},"confidence":0.65,"rationale":"T had a half-buy and lost; with a small loss streak they are more likely to repeat a partial buy than commit to a full buy."}
"""

HYBRID_PROMPT = SYSTEM_PROMPT + """

You will also receive a `model_prior` field: probabilities from a statistical
model trained on historical rounds. This prior is usually strong. Use it as your
starting point. Adjust it only when the card gives a clear, explicit reason
(for example, pistol_round is true, or the equipment values strongly contradict
the prior). If you adjust it, name the card field that caused the adjustment.
Return a final calibrated distribution. Do not ignore the prior without a
card-based justification.
"""

PROMPT_VARIANTS = {
    "zero_shot": SYSTEM_PROMPT,
    "rules_few_shot": RULES_FEW_SHOT_PROMPT,
    "hybrid": HYBRID_PROMPT,
}

PROBABILITIES_SCHEMA = {
    "type": "object",
    "properties": {
        name: {"type": "number", "minimum": 0, "maximum": 1} for name in CLASSES
    },
    "required": CLASSES,
    "additionalProperties": False,
}

LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "enemy_buy": {"type": "string", "enum": CLASSES},
        "probabilities": PROBABILITIES_SCHEMA,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 800},
    },
    "required": ["enemy_buy", "probabilities", "confidence", "rationale"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "enemy_buy": {"type": "string", "enum": CLASSES},
        "llm_label": {"type": "string", "enum": CLASSES},
        "probabilities": PROBABILITIES_SCHEMA,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "normalized": {"type": "boolean"},
    },
    "required": [
        "enemy_buy",
        "llm_label",
        "probabilities",
        "confidence",
        "rationale",
        "normalized",
    ],
    "additionalProperties": False,
}

SPEC = TaskSpec(
    task_id="TSK-CS2-ECON-LLM",
    name="cs2_enemy_economy_inference_llm",
    description="Infer the enemy T buy category from a pre-round CS2 situation card.",
    input_schema={
        "type": "object",
        "properties": {"card_text": {"type": "string", "minLength": 1}},
        "required": ["card_text"],
        "additionalProperties": False,
    },
    output_schema=FINAL_SCHEMA,
    max_steps=4,
    max_model_calls=2,
    timeout_seconds=90.0,
    retry_policy=RetryPolicy(
        max_attempts=2,
        initial_delay_seconds=0.2,
        max_delay_seconds=1.5,
    ),
)


def normalize_decision(input_data: dict) -> dict:
    """Normalize the model's probabilities and make the label consistent."""
    raw = input_data["raw"]
    raw_probabilities = raw["probabilities"]
    probabilities = {
        name: max(0.0, float(raw_probabilities.get(name, 0.0))) for name in CLASSES
    }
    total = sum(probabilities.values())
    normalized = abs(total - 1.0) > 1e-6
    if total <= 1e-9:
        probabilities = {name: 1.0 / len(CLASSES) for name in CLASSES}
    else:
        probabilities = {name: value / total for name, value in probabilities.items()}
    predicted = max(CLASSES, key=lambda name: probabilities[name])
    return {
        "enemy_buy": predicted,
        "llm_label": raw["enemy_buy"],
        "probabilities": probabilities,
        "confidence": float(probabilities[predicted]),
        "rationale": raw["rationale"],
        "normalized": normalized,
    }


def make_task(system_prompt: str):
    @task(SPEC)
    def build(ctx) -> None:
        ctx.step(
            "infer_buy",
            llm=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ctx.input["card_text"]},
            ],
            output_schema=LLM_OUTPUT_SCHEMA,
            output_key="raw_decision",
            timeout_seconds=60.0,
        )
        ctx.step(
            "normalize",
            func=normalize_decision,
            input={"raw": ctx.state["raw_decision"]},
            output_schema=FINAL_SCHEMA,
            output_key="decision",
            timeout_seconds=5.0,
        )
        ctx.end(
            {
                "enemy_buy": ctx.state["decision"]["enemy_buy"],
                "llm_label": ctx.state["decision"]["llm_label"],
                "probabilities": ctx.state["decision"]["probabilities"],
                "confidence": ctx.state["decision"]["confidence"],
                "rationale": ctx.state["decision"]["rationale"],
                "normalized": ctx.state["decision"]["normalized"],
            }
        )

    return build


def build_card(row: pd.Series, prior: dict | None = None) -> dict:
    card = {
        "map": "de_dust2",
        "side": "ct",
        "round_num": int(row["round_num"]),
        "score_ct": int(row["ct_team_score_before"]),
        "score_t": int(row["t_team_score_before"]),
        "score_diff_ct_minus_t": int(row["score_diff_ct_minus_t_before"]),
        "pistol_round": bool(row["pistol_round_flag"]),
        "round_phase": str(row["round_phase"]),
        "prev_winner_side": None
        if pd.isna(row["prev_winner_side"])
        else str(row["prev_winner_side"]),
        "prev_attack_site": str(row["prev_round_attack_site"]),
        "prev_t_econ": str(row["prev_t_econ_type"]),
        "prev_ct_econ": str(row["prev_ct_econ_type"]),
        "prev_t_equip": int(row["prev_t_total_equip"]),
        "prev_ct_equip": int(row["prev_ct_total_equip"]),
        "t_loss_streak": int(row["t_loss_streak_before"]),
        "ct_loss_streak": int(row["ct_loss_streak_before"]),
    }
    if prior is not None:
        card["model_prior"] = {
            name: round(float(prior[name]), 4) for name in CLASSES
        }
    return card


def select_subset(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=42)
        for _, group in df.groupby(LABEL, sort=True)
    ]
    return pd.concat(parts).sort_index()


def classifier_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
) -> dict:
    labels = list(range(len(CLASSES)))
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "per_class_recall": {
            name: float(value)
            for name, value in zip(
                CLASSES,
                recall_score(
                    y_true, y_pred, average=None, labels=labels, zero_division=0
                ),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if probabilities is not None:
        one_hot = np.zeros((len(y_true), len(CLASSES)))
        one_hot[np.arange(len(y_true)), y_true] = 1.0
        result["brier"] = float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        )
    return result


def baseline_metrics(df: pd.DataFrame, sample_ids: list[str]) -> dict:
    """Out-of-fold baselines on exactly the same rows as the LLM arm."""
    positions = df.index[df["sample_id"].isin(sample_ids)].to_numpy()
    positions = positions[np.argsort([sample_ids.index(df.loc[p, "sample_id"]) for p in positions])]
    subset = df.loc[positions]
    y = df[LABEL].astype("category")
    classes = list(y.cat.categories)
    assert classes == CLASSES, classes
    y_codes = y.cat.codes.to_numpy()
    X = design_matrix(df)
    groups = df["match_id"].to_numpy()
    cv = GroupKFold(n_splits=5)

    results = {}
    majority_label = int(np.bincount(y_codes[positions]).argmax())
    majority_pred = np.full(len(positions), majority_label)
    results["majority"] = classifier_metrics(
        y_codes[positions], majority_pred, None
    )

    rule_pred = (
        subset.apply(hand_rule, axis=1).map(CLASS_TO_INDEX).to_numpy()
    )
    results["hand_rule"] = classifier_metrics(y_codes[positions], rule_pred, None)

    tree = DecisionTreeClassifier(max_depth=6, random_state=42)
    tree_pred = cross_val_predict(tree, X, y_codes, groups=groups, cv=cv)[positions]
    tree_proba = cross_val_predict(
        tree, X, y_codes, groups=groups, cv=cv, method="predict_proba"
    )[positions]
    results["tree_depth6"] = classifier_metrics(
        y_codes[positions], tree_pred, tree_proba
    )

    forest = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    )
    forest_pred = cross_val_predict(forest, X, y_codes, groups=groups, cv=cv)[positions]
    forest_proba = cross_val_predict(
        forest, X, y_codes, groups=groups, cv=cv, method="predict_proba"
    )[positions]
    results["random_forest"] = classifier_metrics(
        y_codes[positions], forest_pred, forest_proba
    )
    return results


def probability_vector(output: dict) -> np.ndarray:
    return np.array([float(output["probabilities"][name]) for name in CLASSES])


async def run(args: argparse.Namespace) -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    subset = df if args.all else select_subset(df, args.per_class)
    if args.limit is not None:
        subset = subset.head(args.limit)

    prior_map: dict[str, dict[str, float]] = {}
    if args.variant == "hybrid":
        y_all = df[LABEL].astype("category")
        assert list(y_all.cat.categories) == CLASSES
        tree_prior = DecisionTreeClassifier(max_depth=6, random_state=42)
        prior_probabilities = cross_val_predict(
            tree_prior,
            design_matrix(df),
            y_all.cat.codes.to_numpy(),
            groups=df["match_id"].to_numpy(),
            cv=GroupKFold(n_splits=5),
            method="predict_proba",
        )
        prior_map = {
            str(sid): {
                name: float(prior_probabilities[index][class_index])
                for class_index, name in enumerate(CLASSES)
            }
            for index, sid in enumerate(df["sample_id"])
        }

    backend = OllamaNativeBackend(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        num_predict=args.num_predict,
    )
    runtime = TaskRuntime(backend=backend, tools=ToolRegistry([]))
    task_runner = make_task(PROMPT_VARIANTS[args.variant])

    rows: list[dict] = []
    started_all = time.perf_counter()
    for index, row in subset.iterrows():
        card = build_card(row, prior_map.get(str(row["sample_id"])))
        card_text = json.dumps(card, ensure_ascii=False)
        started = time.perf_counter()
        error = None
        output = None
        try:
            run_result = await task_runner.run({"card_text": card_text}, runtime=runtime)
            if run_result.output is None:
                raise RuntimeError(f"task status {run_result.status.value} with no output")
            output = run_result.output
        except Exception as exc:  # noqa: BLE001 - one failed row must not stop the pilot
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started

        record = {
            "sample_id": str(row["sample_id"]),
            "match_id": str(row["match_id"]),
            "round_num": int(row["round_num"]),
            "split": str(row["split"]),
            "true_label": str(row[LABEL]),
            "sample_index": int(index),
            "latency_s": float(latency),
            "error": error,
        }
        if output is not None:
            record.update(
                {
                    "pred_label": output["enemy_buy"],
                    "llm_label": output["llm_label"],
                    "confidence": float(output["confidence"]),
                    "rationale": output["rationale"],
                    "normalized": bool(output["normalized"]),
                    **{
                        f"prob_{name}": float(output["probabilities"][name])
                        for name in CLASSES
                    },
                }
            )
            record["mentions_money"] = bool(
                re.search(r"\$\s?\d", record["rationale"])
            )
        rows.append(record)
        print(
            f"[{len(rows):>3}/{len(subset)}] {record['sample_id']} "
            f"true={record['true_label']:<8} pred={record.get('pred_label', 'ERROR'):<8} "
            f"{latency:5.1f}s"
            + (f" error={error[:80]}" if error else ""),
            flush=True,
        )

    suffix = "" if args.variant == "zero_shot" else f"_{args.variant}"
    predictions_path = RESULTS_OUT / f"llm_predictions{suffix}.parquet"
    metrics_path = RESULTS_OUT / f"llm_metrics{suffix}.json"

    predictions = pd.DataFrame(rows)
    predictions.to_parquet(predictions_path, index=False)

    successful = predictions[predictions["error"].isna()].copy()
    if successful.empty:
        raise RuntimeError("all LLM calls failed; see llm_predictions.parquet")

    y_true = successful["true_label"].map(CLASS_TO_INDEX).to_numpy()
    y_pred = successful["pred_label"].map(CLASS_TO_INDEX).to_numpy()
    probabilities = successful[[f"prob_{name}" for name in CLASSES]].to_numpy()
    llm_metrics = classifier_metrics(y_true, y_pred, probabilities)
    llm_metrics["model"] = args.model
    llm_metrics["errors"] = int(len(predictions) - len(successful))
    llm_metrics["label_consistency"] = float(
        np.mean(successful["pred_label"] == successful["llm_label"])
    )
    llm_metrics["normalized_rate"] = float(successful["normalized"].mean())
    llm_metrics["mentions_money_rate"] = float(successful["mentions_money"].mean())
    llm_metrics["latency_p50_s"] = float(successful["latency_s"].quantile(0.5))
    llm_metrics["latency_p95_s"] = float(successful["latency_s"].quantile(0.95))
    llm_metrics["latency_mean_s"] = float(successful["latency_s"].mean())
    llm_metrics["subset_size"] = int(len(subset))
    llm_metrics["per_class_counts"] = {
        name: int((successful["true_label"] == name).sum()) for name in CLASSES
    }

    with_pistol = successful.merge(
        df[["sample_id", "pistol_round_flag"]], on="sample_id", how="left"
    )
    hybrid_pred = successful["pred_label"].map(CLASS_TO_INDEX).to_numpy(copy=True)
    pistol_mask = with_pistol["pistol_round_flag"].astype(bool).to_numpy()
    hybrid_pred[pistol_mask] = CLASS_TO_INDEX["pistol"]
    llm_metrics["pistol_hard_rule"] = classifier_metrics(
        y_true, hybrid_pred, probabilities
    )

    baselines = baseline_metrics(
        df, successful["sample_id"].tolist()
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "model": args.model,
        "base_url": args.base_url,
        "llm": llm_metrics,
        "baselines_on_same_rows": baselines,
        "total_elapsed_s": float(time.perf_counter() - started_all),
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== LLM ===")
    print(
        f"n={llm_metrics['n']} errors={llm_metrics['errors']} "
        f"acc={llm_metrics['accuracy']:.3f} macroF1={llm_metrics['macro_f1']:.3f} "
        f"brier={llm_metrics['brier']:.3f}"
    )
    print(
        f"latency p50={llm_metrics['latency_p50_s']:.1f}s "
        f"p95={llm_metrics['latency_p95_s']:.1f}s "
        f"label_consistency={llm_metrics['label_consistency']:.3f} "
        f"mentions_money={llm_metrics['mentions_money_rate']:.3f}"
    )
    hybrid = llm_metrics["pistol_hard_rule"]
    print(
        f"LLM + pistol hard rule: acc={hybrid['accuracy']:.3f} "
        f"macroF1={hybrid['macro_f1']:.3f} brier={hybrid['brier']:.3f}"
    )
    print("\n=== baselines on the same rows ===")
    for name, metrics in baselines.items():
        brier = f" brier={metrics['brier']:.3f}" if "brier" in metrics else ""
        print(
            f"{name:>14}: acc={metrics['accuracy']:.3f} "
            f"macroF1={metrics['macro_f1']:.3f}{brier}"
        )
    print("\n=== sample rationales ===")
    for _, row in successful.head(3).iterrows():
        print(
            f"- true={row['true_label']} pred={row['pred_label']} "
            f"conf={row['confidence']:.2f}: {row['rationale'][:300]}"
        )
    print(f"\nwrote {predictions_path}")
    print(f"wrote {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM arm of the economy benchmark.")
    parser.add_argument("--per-class", type=int, default=20, help="Rows sampled per class.")
    parser.add_argument("--all", action="store_true", help="Use all rows instead of a subset.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the total number of rows.")
    parser.add_argument(
        "--variant",
        choices=sorted(PROMPT_VARIANTS),
        default="zero_shot",
        help="Prompt variant: zero-shot or rules + few-shot.",
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--num-predict", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
