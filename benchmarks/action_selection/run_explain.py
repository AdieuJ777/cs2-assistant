"""Model-proposes / LLM-explains experiment.

The random forest chooses the top-2 next actions. The LLM receives the card and
the model's top-2 (with probabilities), and must explain both options without
changing the recommendation or inventing enemy information.

This tests the only remaining plausible real-time role for the LLM in the
current design: faithful explanation, not decision-making.

Run:
    uv run --with pandas --with pyarrow --with scikit-learn \
        --with pydantic --with jsonschema \
        python benchmarks/action_selection/run_explain.py --per-class 10
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
FRAMEWORK_ROOT = (
    PROJECT_ROOT.parent / "STP-Agent-Framework-main" / "STP-Agent-Framework-main"
)
ECON_DIR = HERE.parent / "economy_inference"
sys.path.insert(0, str(ECON_DIR))
sys.path.insert(0, str(FRAMEWORK_ROOT))

from kkk import RetryPolicy, TaskRuntime, TaskSpec, ToolRegistry, task  # noqa: E402

from ollama_backend import OllamaNativeBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location("action_run", HERE / "run.py")
action_run = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(action_run)

ACTIONS = action_run.ACTIONS
DATA_PATH = PROJECT_ROOT / "data" / "action_selection" / "sample_100.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "action_selection"

SYSTEM_PROMPT = """You are a Counter-Strike 2 analyst explaining a statistical model's recommendation to a player.

The model has already chosen the top two actions. Your job is to explain those two options, not to change them.

Rules:
- Explain the model's top-1 option first, then the top-2 option. Keep their order.
- For each option: why the card supports it, the main risk, and what information would change the decision.
- Use only the fields in the card. Enemy positions, enemy economy, kill feed and voice comms are explicitly unknown; do NOT invent them.
- If the model's probability is low, say so.
- Do not tell the player to do something outside the two options.
- Keep the language concise and actionable.

Return exactly one JSON object:
{
  "options": [
    {"action": "<top1 action>", "why": "...", "risk": "...", "information_that_would_change": ["..."]},
    {"action": "<top2 action>", "why": "...", "risk": "...", "information_that_would_change": ["..."]}
  ],
  "summary": "...",
  "confidence_language": "high | medium | low"
}
"""

OPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "why": {"type": "string", "minLength": 1, "maxLength": 500},
        "risk": {"type": "string", "minLength": 1, "maxLength": 300},
        "information_that_would_change": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
    },
    "required": ["action", "why", "risk", "information_that_would_change"],
    "additionalProperties": False,
}

RAW_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": OPTION_SCHEMA,
        },
        "summary": {"type": "string", "minLength": 1, "maxLength": 400},
        "confidence_language": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["options", "summary", "confidence_language"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    **RAW_SCHEMA,
    "properties": {
        **RAW_SCHEMA["properties"],
        "missing_options": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [*RAW_SCHEMA["required"], "missing_options"],
}

SPEC = TaskSpec(
    task_id="TSK-CS2-EXPLAIN",
    name="cs2_ct_action_explanation",
    description="Explain the random forest's top-2 CT actions without changing them.",
    input_schema={
        "type": "object",
        "properties": {
            "prompt_text": {"type": "string", "minLength": 1},
            "model_top1": {"type": "string"},
            "model_top2": {"type": "string"},
        },
        "required": ["prompt_text", "model_top1", "model_top2"],
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


def normalize_explanation(input_data: dict) -> dict:
    raw = input_data["raw"]
    expected = [input_data["top1"], input_data["top2"]]
    by_action = {}
    for option in raw["options"]:
        by_action.setdefault(option["action"], option)
    options = []
    missing = []
    for action in expected:
        option = by_action.get(action)
        if option is None:
            missing.append(action)
            options.append(
                {
                    "action": action,
                    "why": "The model did not provide an explanation for this option.",
                    "risk": "Unknown.",
                    "information_that_would_change": [],
                }
            )
        else:
            options.append(
                {
                    "action": action,
                    "why": option["why"],
                    "risk": option["risk"],
                    "information_that_would_change": option[
                        "information_that_would_change"
                    ],
                }
            )
    return {
        "options": options,
        "summary": raw["summary"],
        "confidence_language": raw["confidence_language"],
        "missing_options": missing,
    }


def make_task(system_prompt: str):
    @task(SPEC)
    def build(ctx) -> None:
        ctx.step(
            "explain",
            llm=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ctx.input["prompt_text"]},
            ],
            output_schema=RAW_SCHEMA,
            output_key="raw_explanation",
            timeout_seconds=60.0,
        )
        ctx.step(
            "normalize",
            func=normalize_explanation,
            input={
                "raw": ctx.state["raw_explanation"],
                "top1": ctx.input["model_top1"],
                "top2": ctx.input["model_top2"],
            },
            output_schema=FINAL_SCHEMA,
            output_key="explanation",
            timeout_seconds=5.0,
        )
        ctx.end(
            {
                "options": ctx.state["explanation"]["options"],
                "summary": ctx.state["explanation"]["summary"],
                "confidence_language": ctx.state["explanation"]["confidence_language"],
                "missing_options": ctx.state["explanation"]["missing_options"],
            }
        )

    return build


def model_top2(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X, _ = action_run.design_matrix(sample)
    y = sample["action"].astype(
        pd.CategoricalDtype(ACTIONS, ordered=True)
    ).cat.codes.to_numpy()
    forest = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    )
    probabilities = cross_val_predict(
        forest,
        X,
        y,
        groups=sample["match_id"].to_numpy(),
        cv=GroupKFold(n_splits=5),
        method="predict_proba",
    )
    top1_indices = probabilities.argmax(axis=1)
    order = np.argsort(probabilities, axis=1)
    top2_indices = order[:, -2]
    return top1_indices, probabilities, top2_indices


def template_explanation(row: pd.Series, top1: str, top2: str, p1: float, p2: float) -> str:
    area_map = {
        "ROTATE_A": "A",
        "ROTATE_B": "B",
        "ROTATE_MID": "Mid",
        "ROTATE_CT": "CT",
    }
    top1_area = area_map.get(top1, str(row["area"]))
    distance = 0 if top1 == "HOLD" else float(row[f"dist_{top1_area}"]) * 1000
    return (
        f"The model recommends {top1} ({p1:.0%}); the alternative is {top2} ({p2:.0%}). "
        f"Your team is spread as A={int(row['count_A'])}, B={int(row['count_B'])}, "
        f"Mid={int(row['count_Mid'])}, CT={int(row['count_CT'])}. "
        f"You are in {row['area']} with {int(row['health'])} HP, "
        f"{int(row['current_equip_value'])} equipment, about {distance:.0f} units from {top1_area}. "
        f"Enemy positions and economy are unknown."
    )


def looks_like_enemy_claim(text: str) -> bool:
    """Flag definitive enemy claims, not conditional risk language.

    "The enemy could be in A" is an acceptable uncertainty statement; "the
    enemy is in A" or "the enemy is likely defending B" is not, because the
    card says enemy information is unknown.
    """
    patterns = [
        r"\benemy (?:is|are) (?:at|in|on|near|holding)\b",
        r"\benemy (?:is|are) (?:likely|probably)\b",
        r"\bT (?:is|are) (?:at|in|on|near|holding)\b",
        r"\bthey are (?:at|in|on|near|holding)\b",
        r"\b(?:one|two|three|four|five|\d+) (?:enemies|Ts|terrorists) (?:are|is|at|in)\b",
        r"\bI (?:saw|see|heard)\b",
        r"\benemy (?:has|have) \d+\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


CARD_TERMS = re.compile(
    r"\b(area|areas|health|hp|equip|equipment|score|distance|units|teammate|teammates|"
    r"round|previous|economy|pistol|defuser|count|spread|stack)\b",
    flags=re.IGNORECASE,
)


async def run(args: argparse.Namespace) -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    full_top1, full_probabilities, full_top2 = model_top2(full)

    parts = [
        group.sample(n=min(args.per_class, len(group)), random_state=42)
        for _, group in full.groupby("action", sort=True)
    ]
    sample = pd.concat(parts).sort_index()
    if args.limit is not None:
        sample = sample.head(args.limit)
    indices = sample.index.to_numpy()
    top1_indices = full_top1[indices]
    top2_indices = full_top2[indices]
    probabilities = full_probabilities[indices]
    sample = sample.reset_index(drop=True)
    model_top1_actions = [ACTIONS[index] for index in top1_indices]
    model_top2_actions = [ACTIONS[index] for index in top2_indices]
    top1_probability = probabilities[np.arange(len(sample)), top1_indices]
    top2_probability = probabilities[np.arange(len(sample)), top2_indices]

    backend = OllamaNativeBackend(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        num_predict=args.num_predict,
    )
    runtime = TaskRuntime(backend=backend, tools=ToolRegistry([]))
    task_runner = make_task(SYSTEM_PROMPT)

    rows = []
    started_all = time.perf_counter()
    for index, row in sample.iterrows():
        p1, p2 = float(top1_probability[index]), float(top2_probability[index])
        model_json = json.dumps(
            {
                "top1": {"action": model_top1_actions[index], "probability": round(p1, 4)},
                "top2": {"action": model_top2_actions[index], "probability": round(p2, 4)},
            },
            ensure_ascii=False,
        )
        prompt_text = (
            "Situation card:\n"
            + row["card_text"]
            + "\n\nStatistical model recommendation:\n"
            + model_json
            + "\n\nExplain the top-1 option first, then the top-2 option. Do not change them."
        )

        started = time.perf_counter()
        error = None
        output = None
        try:
            result = await task_runner.run(
                {
                    "prompt_text": prompt_text,
                    "model_top1": model_top1_actions[index],
                    "model_top2": model_top2_actions[index],
                },
                runtime=runtime,
            )
            if result.output is None:
                raise RuntimeError(f"task status {result.status.value} with no output")
            output = result.output
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started

        record = {
            "sample_id": str(row["sample_id"]),
            "match_id": str(row["match_id"]),
            "round_num": int(row["round_num"]),
            "window": str(row["window"]),
            "true_action": str(row["action"]),
            "model_top1": model_top1_actions[index],
            "model_top2": model_top2_actions[index],
            "model_prob_top1": p1,
            "model_prob_top2": p2,
            "model_top1_correct": bool(model_top1_actions[index] == row["action"]),
            "latency_s": float(latency),
            "error": error,
        }
        if output is not None:
            explanation_text = " ".join(
                option["why"] + " " + option["risk"] for option in output["options"]
            ) + " " + output["summary"]
            record.update(
                {
                    "options": output["options"],
                    "summary": output["summary"],
                    "confidence_language": output["confidence_language"],
                    "missing_options": output["missing_options"],
                    "action_match": len(output["missing_options"]) == 0,
                    "enemy_claim": bool(looks_like_enemy_claim(explanation_text)),
                    "cites_card_terms": bool(CARD_TERMS.search(explanation_text)),
                    "information_needed": any(
                        option["information_that_would_change"]
                        for option in output["options"]
                    ),
                    "template_explanation": template_explanation(
                        row, model_top1_actions[index], model_top2_actions[index], p1, p2
                    ),
                }
            )
        rows.append(record)
        print(
            f"[{len(rows):>3}/{len(sample)}] {record['sample_id']} "
            f"true={record['true_action']:<10} model={record['model_top1']:<12} "
            f"{latency:4.1f}s"
            + (f" error={error[:60]}" if error else ""),
            flush=True,
        )

    predictions = pd.DataFrame(rows)
    suffix = "" if args.variant == "default" else f"_{args.variant}"
    predictions_path = RESULTS_OUT / f"llm_explanations{suffix}.parquet"
    metrics_path = RESULTS_OUT / f"llm_explain_metrics{suffix}.json"
    predictions.to_parquet(predictions_path, index=False)

    successful = predictions[predictions["error"].isna()].copy()
    if successful.empty:
        raise RuntimeError("all LLM explanation calls failed")
    metrics = {
        "n": int(len(successful)),
        "errors": int(len(predictions) - len(successful)),
        "action_match_rate": float(successful["action_match"].mean()),
        "enemy_claim_rate": float(successful["enemy_claim"].mean()),
        "cites_card_terms_rate": float(successful["cites_card_terms"].mean()),
        "information_needed_rate": float(successful["information_needed"].mean()),
        "confidence_language_distribution": {
            str(k): int(v)
            for k, v in successful["confidence_language"].value_counts().items()
        },
        "model_top1_accuracy_on_subset": float(
            successful["model_top1_correct"].mean()
        ),
        "model_prob_top1_mean": float(successful["model_prob_top1"].mean()),
        "latency_p50_s": float(successful["latency_s"].quantile(0.5)),
        "latency_p95_s": float(successful["latency_s"].quantile(0.95)),
        "mean_summary_chars": float(successful["summary"].str.len().mean()),
        "elapsed_s": float(time.perf_counter() - started_all),
        "model": args.model,
        "variant": args.variant,
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "note": (
            "Accuracy is the random forest's, not the LLM's. The LLM only explains "
            "the model's fixed top-2."
        ),
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    review_cols = [
        "sample_id",
        "true_action",
        "model_top1",
        "model_top2",
        "model_prob_top1",
        "model_prob_top2",
        "summary",
        "options",
        "template_explanation",
    ]
    review = successful[review_cols].sample(
        n=min(20, len(successful)), random_state=7
    ).copy()
    review["faithfulness_1_5"] = ""
    review["usefulness_1_5"] = ""
    review["notes"] = ""
    review.to_csv(
        RESULTS_OUT / "human_review_explanations.csv", index=False, encoding="utf-8-sig"
    )

    print("\n=== explanation metrics ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\n=== sample explanations ===")
    for _, row in successful.head(4).iterrows():
        print(
            f"- model={row['model_top1']} ({row['model_prob_top1']:.0%}) "
            f"true={row['true_action']}:\n  {row['summary'][:300]}"
        )
    print(f"\nwrote {predictions_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {RESULTS_OUT / 'human_review_explanations.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the model-proposes / LLM-explains experiment.")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--num-predict", type=int, default=700)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
