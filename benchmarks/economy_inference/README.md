# Enemy economy inference benchmark

## Question

Given only pre-round information a CT player can observe, can the agent infer
the enemy (T) buy type for the coming round?

This is the first de-risking benchmark for the CS2 decision assistant. It does
not test individual buy/save, enemy positions, or teammate utility: the
released TAGNet dataset does not contain those fields.

## Data

Source: `TAGNet-CS2-main/data/processed/dust2_ct_5snap`.

- 697 rounds, 33 matches, Dust2, CT perspective.
- One row per round; the first round of each match is dropped because lagged
  features are unavailable => **664 rows**.
- Label: `t_econ_type` at 10 s, one of `eco`, `half_buy`, `long_gun`, `pistol`.
- Label distribution: long_gun 315, half_buy 244, eco 73, pistol 32
  (majority share 0.474).

### Features (pre-round observable only)

- Round number and phase (first half / second half / overtime).
- Score before the round.
- Pistol-round flag.
- Previous-round winner, attack site.
- Previous-round CT and T economy type, equipment totals, score difference.
- T and CT loss streaks before the current round.

### Explicitly excluded

- Current round `t_econ_type` and `t_total_equip` (these are labels, not inputs).
- Current round T positions, money, weapons, utility.
- Kill feed, purchase events, per-player money.

The released dataset has only CT player nodes; there are no T-player positions.
It also has no per-player money, weapon names, or utility inventory. This is why
the benchmark is team-level economy only.

## Leakage controls

- **GroupKFold by match** (5 folds). Rounds from the same match never appear in
  both train and validation.
- **Lag-only features.** The current round's T economy is never used as an input.
- No random row-level split.

## Baselines

The script compares:

1. Majority class.
2. A hand-written CS2 economy rule:
   - pistol round -> `pistol`;
   - T won the previous round -> `long_gun`;
   - previous T buy was `eco` -> `eco`;
   - otherwise -> `half_buy`.
3. Decision trees (depth 3 and 6).
4. Random forest (300 trees, depth 6).

Metrics: accuracy, macro F1, per-class recall, confusion matrix, Brier score
(multi-class), and a regression MAE for `t_total_equip`.

## Result (full 664 rows)

| Model | Accuracy | Macro F1 | Brier |
|---|---:|---:|---:|
| Majority class | 0.474 | 0.161 | 1.051 |
| Hand rule | 0.551 | 0.562 | — |
| Tree depth 3 | 0.693 | 0.575 | 0.423 |
| Tree depth 6 | **0.744** | **0.740** | 0.385 |
| Random forest | 0.735 | 0.702 | **0.333** |

`t_total_equip` MAE: ~3001 (mean ~20985, ~14%).

## LLM arm (80-row balanced subset)

To compare the LLM on the same footing, we sampled 20 rounds per class (80 rows)
and evaluated every method on those rows. Baselines are out-of-fold predictions
from the full 664-row set, grouped by match.

| Method | Accuracy | Macro F1 | Brier | Notes |
|---|---:|---:|---:|---|
| Majority | 0.250 | 0.100 | — | |
| Hand rule | 0.600 | 0.581 | — | |
| Tree depth 6 | 0.725 | 0.727 | 0.380 | strong baseline |
| Random forest | 0.725 | 0.701 | 0.350 | best calibration |
| LLM zero-shot (qwen3:8b) | 0.350 | 0.298 | 0.888 | never predicted pistol |
| LLM rules + few-shot | 0.588 | 0.591 | 0.506 | fixed pistol; ~hand rule |
| LLM hybrid (tree prior) | 0.662 | 0.674 | 0.414 | mostly copies the prior |
| LLM hybrid + pistol hard rule | 0.725 | 0.726 | 0.414 | matches tree; no gain |

Prompt variants:

- **Zero-shot**: 35.0% accuracy; 9/80 rationales cited exact money; p50 latency
  4.7 s.
- **Rules + few-shot**: 58.8%; 0/80 cited exact money; p50 latency 2.8 s. It
  finally learned "pistol round -> pistol", but still sat at the hand-rule
  level.
- **Hybrid (tree prior)**: 66.2% raw; 72.5% after a deterministic pistol-round
  override. It matches the tree's accuracy but is less calibrated
  (Brier 0.414 vs 0.380) and does not improve on it.
- 0 errors across 80 calls per variant; the STP task contract, structured output
  and retries all worked.

## Interpretation

- Enemy economy is **not random**: a simple tree reaches ~74% accuracy versus
  ~47% majority. A learned inference layer is worth building.
- It is **not precise**: roughly one in four buy types is misclassified, and the
  `eco` class is the hardest. The agent should output a probability
  distribution and a confidence level, not "they have $X".
- A hand-written economy rule is substantially weaker (~55%), which is evidence
  against a rules-only design.
- The regression error (~14% of mean equipment value) means the agent should
  avoid exact-money claims even when the class is right.
- **The LLM is not a competitive inferencer on this task.** Its best result is
  to reproduce the tree prior; it does not improve accuracy or calibration. The
  pistol fix that helped most is a deterministic rule, not reasoning.
- **Architectural implication:** use the statistical/deterministic layer for
  inference; use the LLM for explanation, qualitative synthesis, action
  tradeoffs, and asking for missing information. This is the hybrid design.
- Even when instructed not to, the hybrid LLM cited exact equipment values in
  37.5% of rationales. This is a reliability risk for a user-facing assistant.

## What this benchmark does NOT prove

- That a larger or fine-tuned LLM cannot do better. It shows that
  `qwen3:8b` with zero-shot or light prompting is not enough.
- That the result transfers to solo queue. The data is professional play.
- That individual buy/save or enemy-position advice is feasible. The data does
  not support those benchmarks.
- That the LLM cannot add value in other roles. It only shows that numeric
  economy inference should not be its job.

## Reproducing the LLM arm

```bash
# zero-shot
uv run --with pandas --with pyarrow --with scikit-learn \
  --with pydantic --with jsonschema \
  python benchmarks/economy_inference/run_llm.py --per-class 20

# rules + few-shot
... --per-class 20 --variant rules_few_shot

# hybrid: tree prior in the card
... --per-class 20 --variant hybrid
```

Outputs:

- `results/economy_inference/llm_predictions*.parquet`
- `results/economy_inference/llm_metrics*.json`

The native Ollama backend lives in `ollama_backend.py`. Ollama's
OpenAI-compatible endpoint did not honour `think: false` for Qwen3 (the model
spent its whole budget on hidden reasoning and returned empty content), so the
adapter uses the native `/api/chat` endpoint with `think: false` and
`format: json`.

## How to run

```bash
cd cs2-assistant
uv run --with pandas --with pyarrow --with scikit-learn \
  python benchmarks/economy_inference/run.py
```

Outputs:

- `data/economy_inference/rounds.parquet`
- `data/economy_inference/audit.json`
- `results/economy_inference/baselines.json`
