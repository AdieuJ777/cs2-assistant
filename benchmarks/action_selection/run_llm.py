"""LLM arm of the action-selection benchmark, executed through STP.

The model receives only what a CT player could observe (self state, teammate
state, score, economy, previous round). Enemy positions, enemy economy, kill
feed and voice comms are explicitly unknown. It returns 2-3 next-action options
with confidences, a recommended option, and the information it would need.

Actions (true labels): HOLD, ROTATE_A, ROTATE_B, ROTATE_MID, ROTATE_CT.
The model may also output GATHER_INFO, which is treated as abstention.

Run:
    uv run --with pandas --with pyarrow --with scikit-learn \
        --with pydantic --with jsonschema \
        python benchmarks/action_selection/run_llm.py
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
from sklearn.metrics import f1_score, recall_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
FRAMEWORK_ROOT = (
    PROJECT_ROOT.parent / "STP-Agent-Framework-main" / "STP-Agent-Framework-main"
)
ECON_DIR = HERE.parent / "economy_inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ECON_DIR))
sys.path.insert(0, str(FRAMEWORK_ROOT))

from kkk import RetryPolicy, TaskRuntime, TaskSpec, ToolRegistry, task  # noqa: E402

from ollama_backend import OllamaNativeBackend  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "action_selection" / "sample_100.parquet"
RESULTS_OUT = PROJECT_ROOT / "results" / "action_selection"
ACTIONS = ["HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT"]
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
OUTPUT_ACTIONS = ACTIONS + ["GATHER_INFO"]

SYSTEM_PROMPT = """You are a Counter-Strike 2 CT analyst helping a solo-queue player on Dust2.

The card describes the current state at a decision snapshot. "window" is the
time until the next snapshot (for example, "12->15" means the next decision
point is about 3 seconds later). You must choose 2-3 possible next actions for
this player.

Actions:
- HOLD: stay in the current area.
- ROTATE_A / ROTATE_B / ROTATE_MID / ROTATE_CT: move toward that area.
- GATHER_INFO: stay for now but seek information; do not commit.

Hard constraints:
- The card's `known_info` says enemy positions, enemy economy, kill feed and
  voice comms are unknown. Do NOT invent enemy positions, enemy counts, or
  information the player does not have.
- Use only the fields in the card. Cite the field that supports each option.
- Be calibrated: if the team state does not clearly support a rotation, say so
  and consider GATHER_INFO or HOLD.
- Keep each rationale short (one or two sentences).

Return a JSON object with:
- options: 2-3 items, each {action, confidence, rationale};
- recommended: the index of the option you recommend;
- information_needed: a list of what would most change the decision;
- confidence: overall confidence in the recommendation.
"""

OPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": OUTPUT_ACTIONS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action", "confidence", "rationale"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": OPTION_SCHEMA,
        },
        "recommended": {"type": "integer", "minimum": 0, "maximum": 2},
        "information_needed": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["options", "recommended", "information_needed", "confidence"],
    "additionalProperties": False,
}

SPEC = TaskSpec(
    task_id="TSK-CS2-ACTION-LLM",
    name="cs2_ct_next_action_llm",
    description="Choose 2-3 next actions for a CT player from observable team state.",
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


def normalize_options(input_data: dict) -> dict:
    raw = input_data["raw"]
    seen: set[str] = set()
    options = []
    for option in raw["options"]:
        action = option["action"]
        if action in seen:
            continue
        seen.add(action)
        options.append(
            {
                "action": action,
                "confidence": float(option["confidence"]),
                "rationale": option["rationale"],
            }
        )
    while len(options) < 2:
        action = "HOLD" if "HOLD" not in seen else "GATHER_INFO"
        if action in seen:
            break
        options.append(
            {
                "action": action,
                "confidence": 0.5,
                "rationale": "Fallback option due to an incomplete model response.",
            }
        )
        seen.add(action)
    options = options[:3]
    recommended = int(raw["recommended"])
    if recommended >= len(options):
        recommended = 0
    return {
        "options": options,
        "recommended": recommended,
        "information_needed": raw["information_needed"],
        "confidence": float(options[recommended]["confidence"]),
    }


def make_task(system_prompt: str):
    @task(SPEC)
    def build(ctx) -> None:
        ctx.step(
            "choose_next_action",
            llm=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ctx.input["card_text"]},
            ],
            output_schema=FINAL_SCHEMA,
            output_key="raw_decision",
            timeout_seconds=60.0,
        )
        ctx.step(
            "normalize",
            func=normalize_options,
            input={"raw": ctx.state["raw_decision"]},
            output_schema=FINAL_SCHEMA,
            output_key="decision",
            timeout_seconds=5.0,
        )
        ctx.end(
            {
                "options": ctx.state["decision"]["options"],
                "recommended": ctx.state["decision"]["recommended"],
                "information_needed": ctx.state["decision"]["information_needed"],
                "confidence": ctx.state["decision"]["confidence"],
            }
        )

    return build


def action_metrics(df: pd.DataFrame) -> dict:
    y = df["true_action"].map(ACTION_TO_INDEX).to_numpy()
    pred = df["recommended_action"].map(ACTION_TO_INDEX)
    abstain = pred.isna().to_numpy()
    pred_value = pred.fillna(-1).to_numpy()
    correct = pred_value == y

    top2_hits = []
    for _, row in df.iterrows():
        true_index = ACTION_TO_INDEX.get(row["true_action"])
        option_actions = {
            option["action"] for option in row["options"]
        } - {"GATHER_INFO"}
        top2_hits.append(
            any(
                ACTION_TO_INDEX.get(action) == true_index for action in option_actions
            )
        )

    non_abstain = ~abstain
    macro_f1 = float("nan")
    if non_abstain.any():
        macro_f1 = float(
            f1_score(
                y[non_abstain],
                pred_value[non_abstain],
                average="macro",
                labels=list(range(len(ACTIONS))),
                zero_division=0,
            )
        )
    per_class_recall = {
        action: float(value)
        for action, value in zip(
            ACTIONS,
            recall_score(
                y,
                pred_value,
                average=None,
                labels=list(range(len(ACTIONS))),
                zero_division=0,
            ),
        )
    }
    rotation_mask = y > 0
    return {
        "n": int(len(df)),
        "top1_accuracy": float(np.mean(correct)),
        "accuracy_non_abstain": float(np.mean(correct[non_abstain]))
        if non_abstain.any()
        else None,
        "top2_coverage": float(np.mean(top2_hits)),
        "macro_f1_non_abstain": macro_f1,
        "per_class_recall_with_abstain_wrong": per_class_recall,
        "rotation_recall": float(np.mean(correct[rotation_mask]))
        if rotation_mask.any()
        else None,
        "hold_recall": float(np.mean(correct[~rotation_mask])),
        "abstain_rate": float(np.mean(abstain)),
        "information_needed_rate": float(
            np.mean(df["information_needed"].map(len) > 0)
        ),
        "enemy_claim_rate": float(np.mean(df["enemy_claim"])),
        "latency_p50_s": float(df["latency_s"].quantile(0.5)),
        "latency_p95_s": float(df["latency_s"].quantile(0.95)),
        "errors": int(df["error"].notna().sum()),
        "true_distribution": {
            str(k): int(v) for k, v in df["true_action"].value_counts().items()
        },
        "recommended_distribution": {
            str(k): int(v)
            for k, v in df["recommended_action"].fillna("GATHER_INFO").value_counts().items()
        },
    }


def looks_like_enemy_claim(text: str) -> bool:
    patterns = [
        r"\benemy (?:is|are|has|have|might be|may be|probably is|probably are)\b",
        r"\bT (?:is|are) (?:at|in|on|near|holding)\b",
        r"\bthey are (?:at|in|on|near|holding)\b",
        r"\b(?:one|two|three|four|five|\d+) (?:enemies|Ts|terrorists)\b",
        r"\bI (?:saw|see|heard)\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


async def run(args: argparse.Namespace) -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH).rename(columns={"action": "true_action"})
    if args.limit is not None:
        df = df.head(args.limit)

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
    for index, row in df.iterrows():
        started = time.perf_counter()
        error = None
        output = None
        try:
            run_result = await task_runner.run(
                {"card_text": row["card_text"]}, runtime=runtime
            )
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
            "window": str(row["window"]),
            "true_action": str(row["true_action"]),
            "latency_s": float(latency),
            "error": error,
        }
        if output is not None:
            recommended_index = int(output["recommended"])
            recommended = output["options"][recommended_index]["action"]
            recommended_rationale = output["options"][recommended_index]["rationale"]
            rationale_text = " ".join(
                option["rationale"] for option in output["options"]
            )
            record.update(
                {
                    "options": output["options"],
                    "recommended_index": recommended_index,
                    "recommended_action": recommended,
                    "recommended_rationale": recommended_rationale,
                    "confidence": float(output["confidence"]),
                    "information_needed": output["information_needed"],
                    "rationale_text": rationale_text,
                    "enemy_claim": bool(
                        looks_like_enemy_claim(recommended_rationale)
                    ),
                }
            )
        rows.append(record)
        print(
            f"[{len(rows):>3}/{len(df)}] {record['sample_id']} "
            f"true={record['true_action']:<10} pred={record.get('recommended_action', 'ERROR'):<12} "
            f"{latency:4.1f}s"
            + (f" error={error[:60]}" if error else ""),
            flush=True,
        )

    predictions = pd.DataFrame(rows)
    suffix = "" if args.variant == "zero_shot" else f"_{args.variant}"
    predictions_path = RESULTS_OUT / f"llm_predictions{suffix}.parquet"
    metrics_path = RESULTS_OUT / f"llm_metrics{suffix}.json"
    predictions.to_parquet(predictions_path, index=False)

    successful = predictions[predictions["error"].isna()].copy()
    if successful.empty:
        raise RuntimeError("all LLM calls failed; see predictions file")
    metrics = action_metrics(successful)
    metrics["model"] = args.model
    metrics["variant"] = args.variant
    metrics["subset_size"] = int(len(df))
    metrics["elapsed_s"] = float(time.perf_counter() - started_all)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "variant": args.variant,
        "base_url": args.base_url,
        "llm": metrics,
        "baselines_on_same_sample": json.loads(
            (RESULTS_OUT / "baselines.json").read_text(encoding="utf-8")
        )["baselines_on_sample"],
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== LLM ===")
    print(
        f"n={metrics['n']} errors={metrics['errors']} "
        f"top1={metrics['top1_accuracy']:.3f} "
        f"top2={metrics['top2_coverage']:.3f} "
        f"rotation_recall={metrics['rotation_recall']:.3f} "
        f"abstain={metrics['abstain_rate']:.3f}"
    )
    print(
        f"non-abstain acc={metrics['accuracy_non_abstain']:.3f} "
        f"macroF1(non-abstain)={metrics['macro_f1_non_abstain']:.3f} "
        f"enemy_claim_rate={metrics['enemy_claim_rate']:.3f}"
    )
    print(
        f"latency p50={metrics['latency_p50_s']:.1f}s "
        f"p95={metrics['latency_p95_s']:.1f}s"
    )
    print("\n=== baselines on the same sample ===")
    for name, result in payload["baselines_on_same_sample"].items():
        top2 = f" top2={result['top2_coverage']:.3f}" if "top2_coverage" in result else ""
        print(
            f"{name:>16}: acc={result['accuracy']:.3f} "
            f"bal_acc={result['balanced_accuracy']:.3f} "
            f"macroF1={result['macro_f1']:.3f}{top2}"
        )
    print("\n=== sample recommendations ===")
    for _, row in successful.head(5).iterrows():
        print(
            f"- true={row['true_action']} pred={row['recommended_action']} "
            f"conf={row['confidence']:.2f}: {row['recommended_rationale'][:280]}"
        )
    print(f"\nwrote {predictions_path}")
    print(f"wrote {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM action-selection benchmark.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variant", default="zero_shot")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--num-predict", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
