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

## Result (pilot)

| Model | Accuracy | Macro F1 | Brier |
|---|---:|---:|---:|
| Majority class | 0.474 | 0.161 | 1.051 |
| Hand rule | 0.551 | 0.562 | — |
| Tree depth 3 | 0.693 | 0.575 | 0.423 |
| Tree depth 6 | **0.744** | **0.740** | 0.385 |
| Random forest | 0.735 | 0.702 | **0.333** |

`t_total_equip` MAE: ~3001 (mean ~20985, ~14%).

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

## What this benchmark does NOT prove

- That an LLM adds value over the tree/forest baselines. The LLM arm is the next
  step and is not implemented yet.
- That the result transfers to solo queue. The data is professional play.
- That individual buy/save or enemy-position advice is feasible. The data does
  not support those benchmarks.

## Next: LLM arm

The LLM arm should receive the same information as the baselines, in a
structured situation card, and return:

```json
{
  "enemy_buy": "eco | half_buy | long_gun | pistol",
  "probabilities": {"eco": 0.0, "half_buy": 0.0, "long_gun": 0.0, "pistol": 0.0},
  "confidence": 0.0,
  "rationale": "short explanation that cites only card fields"
}
```

Compare against the tree/forest baselines on accuracy, macro F1, Brier score,
and a hallucination check (does the rationale cite fields that are not in the
card?). A useful result would be: the LLM is competitive with the tree on
accuracy and better calibrated or more explainable; if it is worse than the
hand rule, the "LLM reasoning" hypothesis is in trouble.

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
