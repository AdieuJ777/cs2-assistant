# Enemy economy inference — experiment summary

Question: can the agent infer the enemy (T) buy type before a round from
CT-observable information only?

Data: released TAGNet Dust2 CT dataset. 664 rounds with lag features (first round
of each match dropped), 33 matches. Labels: `t_econ_type` at 10 s. GroupKFold by
match.

## Full 664-row baselines

| Model | Accuracy | Macro F1 | Brier |
|---|---:|---:|---:|
| Majority class | 0.474 | 0.161 | 1.051 |
| Hand-written rule | 0.551 | 0.562 | — |
| Decision tree, depth 3 | 0.693 | 0.575 | 0.423 |
| Decision tree, depth 6 | **0.744** | **0.740** | 0.385 |
| Random forest | 0.735 | 0.702 | **0.333** |

Equipment-value regression MAE ≈ 3001 (mean ≈ 20985, ~14%).

## LLM comparison (80-row balanced subset, 20 per class)

Baselines are out-of-fold predictions on the same rows.

| Method | Accuracy | Macro F1 | Brier | Exact-money mentions |
|---|---:|---:|---:|---:|
| Majority | 0.250 | 0.100 | — | — |
| Hand rule | 0.600 | 0.581 | — | — |
| Tree depth 6 | 0.725 | 0.727 | 0.380 | — |
| Random forest | 0.725 | 0.701 | 0.350 | — |
| LLM zero-shot (qwen3:8b) | 0.350 | 0.298 | 0.888 | 11.3% |
| LLM rules + few-shot | 0.588 | 0.591 | 0.506 | 0.0% |
| LLM hybrid (tree prior) | 0.662 | 0.674 | 0.414 | 37.5% |
| LLM hybrid + pistol hard rule | 0.725 | 0.726 | 0.414 | 37.5% |

Latency (local `qwen3:8b`, native Ollama): p50 2.8 s with rules/few-shot,
4.7–4.8 s with zero-shot/hybrid. Zero task errors across all 80-row runs.

## Findings

1. **Enemy economy is learnable at a coarse level.** A depth-6 tree reaches
   74.4% on 664 rounds vs 47.4% majority; a hand rule reaches 55.1%.
2. **It is not precise.** About one in four buy types is wrong; `eco` is the
   hardest class. Equipment value MAE is ~14% of the mean. The assistant should
   output a distribution, not an exact money claim.
3. **Zero-shot `qwen3:8b` is a poor inferencer.** 35.0% accuracy, never predicts
   pistol, Brier 0.888. It also ignores the "no exact money" instruction in
   11.3% of rationales.
4. **Prompting fixes the obvious failure, not the capability gap.** Rules +
   few-shot raises the LLM to 58.8%, roughly the hand rule, still far below the
   tree.
5. **Giving the LLM the tree prior does not add value.** It reaches 66.2% raw,
   and 72.5% with a deterministic pistol override — exactly the tree's accuracy,
   with worse calibration (Brier 0.414 vs 0.380). The LLM mostly copies the
   prior; its adjustments do not improve it.
6. **The correct architecture is hybrid:** statistical/deterministic inference
   for probabilities; LLM for explanation, qualitative synthesis, action
   tradeoffs, and asking for missing information. Do not use the LLM as the
   numeric inference engine.

## What remains blocked

The released dataset has no T-player positions, per-player money, weapons,
utility inventory, or kill feed. Therefore this benchmark cannot test:

- individual buy/save recommendations;
- enemy position inference;
- teammate utility-aware advice;
- the LLM's value in mid-round action selection.

Those need GSI logs from real matches and/or raw demos parsed with Awpy.
