# Action-selection benchmark (C)

## Question

Given only the CT state a player can observe, can the agent choose the player's
next action over the next few seconds?

This is an **imitation** benchmark, not an optimal-policy benchmark. Professional
movement is driven by information the agent does not have (enemy positions,
voice comms, kill feed), so top-1 accuracy is only one signal. We also measure
top-2 coverage, rotation recall, abstention, and hallucination.

## Data

Source: released TAGNet Dust2 CT dataset.

- 697 rounds, 33 matches, 5 CT players per snapshot.
- Four decision windows: 10→12 s, 12→15 s, 15→18 s, 18→20 s.
- 13,560 player decisions after filtering players who are dead at either end.
- Target: the player's macro area at the next snapshot.
- Actions: `HOLD`, `ROTATE_A`, `ROTATE_B`, `ROTATE_MID`, `ROTATE_CT`.

Natural action distribution:

| Action | Rows |
|---|---:|
| HOLD | 12,556 |
| ROTATE_CT | 307 |
| ROTATE_B | 296 |
| ROTATE_MID | 267 |
| ROTATE_A | 134 |

Because HOLD is 92% of decisions, the evaluation sample is **balanced**: 20 per
class, 100 total, at most one decision per round, 32 matches.

### Card (input)

- self: area, health, armor, equipment value, defuser, distance to A/B/Mid/CT;
- team: area counts, teammate areas/health/equipment/distance to self;
- round: score, round number/phase, pistol flag, economy type, team equipment;
- previous round: winner side, attack site;
- `known_info`: enemy positions, enemy economy, kill feed and voice comms are
  explicitly unknown.

### Leakage controls

- GroupKFold by match for all trained baselines.
- No future snapshot, next area, round outcome, or enemy field in the card.
- One decision per round in the evaluation sample.

## Baselines (100-row balanced sample)

| Method | Accuracy | Balanced Acc | Macro F1 | Top-2 |
|---|---:|---:|---:|---:|
| Majority (HOLD) | 0.200 | 0.200 | 0.067 | — |
| Team-need rule | 0.240 | 0.240 | 0.166 | — |
| Decision tree, depth 6 | 0.690 | 0.690 | 0.693 | 0.760 |
| Random forest | **0.820** | **0.820** | **0.806** | **0.930** |

The structured team state is highly predictive of the next area. The random
forest is a strong baseline.

## LLM arm (qwen3:8b, zero-shot, STP)

100 calls, 0 errors, p50 latency 6.8 s, p95 9.1 s.

| Metric | LLM | Random forest |
|---|---:|---:|
| Top-1 accuracy | 0.370 | 0.820 |
| Top-2 coverage | 0.620 | 0.930 |
| Rotation recall | 0.463 | — |
| HOLD recall | **0.000** | — |
| Abstain rate | 0.000 | — |
| Enemy-claim rate | 0.190 | — |

Recommended-action distribution: ROTATE_B 46, ROTATE_A 44, ROTATE_MID 10.
It never recommends HOLD or ROTATE_CT.

### Failure modes

1. **It cannot say HOLD.** Every one of the 20 HOLD cases was predicted as a
   rotation. On the natural 92%-HOLD distribution, this would be catastrophic.
2. **It over-rotates to A/B.** 90 of 100 recommendations are ROTATE_A or
   ROTATE_B; it mostly ignores MID and never chooses CT.
3. **It never abstains.** It reports `information_needed` in 100% of cases, but
   still commits to a rotation in 100% of cases. The uncertainty is decorative.
4. **It invents enemy intent.** In 19% of recommended rationales it asserts
   things like "the enemy is likely defending B", despite `known_info` saying
   enemy positions are unknown. This is a faithfulness failure, not just an
   accuracy failure.
5. **It is slow for mid-round use.** p50 6.8 s is acceptable between rounds, but
   not for a 2-3 second decision window.

## Interpretation

- The LLM is **not a competitive action selector** at this size and prompting
  level. The structured features already contain the signal, and the random
  forest captures it.
- The LLM's output is biased toward rotations and lacks the ability to represent
  "do nothing / hold / gather information", even though it is explicitly asked.
- Its rationales are not trustworthy: it asserts enemy intent it does not have.
- **Implication:** use the statistical model for action prediction; use the LLM
  for explanation, interaction, and post-round review — or not at all for the
  decision itself. This is the same conclusion as the economy benchmark.

## Feature ablation (for the live GSI + radar/HUD plan)

The live plan cannot see teammate equipment value, team total equipment, or a
team economy label. Removing those features costs only ~3-4 points on the
balanced sample:

| Feature set | Features | Balanced Acc | Macro F1 | Top-2 | HOLD recall |
|---|---:|---:|---:|---:|---:|
| Full benchmark | 23 | 0.820 | 0.806 | 0.930 | 0.45 |
| GSI + radar/HUD | 20 | 0.780 | 0.765 | 0.910 | 0.40 |
| GSI + radar, no HUD | 19 | 0.790 | 0.775 | 0.930 | 0.40 |
| Shape only | 18 | 0.790 | 0.779 | 0.930 | 0.45 |

So the missing live features are **not** the main blocker. The model still has a
low HOLD recall even with all features.

## Natural-distribution evaluation

The balanced sample over-represents rotations. On all 13,560 decisions, where
HOLD is 92%:

| Metric | Value |
|---|---:|
| Accuracy | 0.934 |
| Balanced accuracy | 0.302 |
| Macro F1 | 0.320 |
| HOLD recall | 0.997 |
| Rotation precision | 0.795 |
| Rotation recall | 0.150 |
| False rotation rate | 0.003 |
| Missed rotation rate | 0.850 |

The model collapses to HOLD: it catches only 15% of real rotations and barely
predicts ROTATE_A/MID/CT. A HOLD-threshold sweep does not fix it. This is a
policy/calibration problem, not a perception problem.

**Implication:** do not put this model into the live action prompt yet. Before
investing in CV, fix the action policy: a two-stage rotation detector, class
weights, or a cost-sensitive decision rule, evaluated on the natural
distribution.

## Two-stage policy (fixing the HOLD collapse)

Stage 1: ROTATE vs HOLD with class weights. Stage 2: destination (A/B/MID/CT)
trained only on true rotations.

Stage 2 is strong on the 1,004 true rotations:

| Metric | Value |
|---|---:|
| Accuracy | 0.921 |
| Macro F1 | 0.732 |
| Top-2 coverage | 0.994 |
| Per-class recall | 0.91-0.93 |

Stage 1 at threshold 0.5 has rotation recall 0.944 but precision 0.235;
ROC-AUC is 0.923, so the signal exists but the operating point is the problem.

End-to-end threshold sweep:

| Threshold | Rot. precision | Rot. recall | False rot. rate | Exact rotation acc. | Overall acc. |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.235 | 0.944 | 0.245 | 0.870 | 0.763 |
| **0.6** | **0.343** | **0.825** | **0.126** | **0.770** | **0.866** |
| **0.7** | **0.458** | **0.556** | **0.053** | **0.519** | **0.916** |
| 0.8 | 0.534 | 0.433 | 0.030 | 0.401 | 0.928 |

The two-stage decomposition works: destination prediction is solved once a
rotation is detected. The remaining problem is the stage-1 operating point.
For a recall-oriented coach, 0.6 is a reasonable start; for conservative advice,
0.7. This is now a product trade-off, not a feature-availability blocker.

## Temporal features (follow-up)

Added the previous decision window for the same player: previous area/action,
movement distance/speed, health/armor/equipment deltas, teammate count deltas,
and window duration.

On the 10,117 rows that have a previous window:

| Model | Stage-1 ROC-AUC | Stage-1 AP | Stage-2 acc | Stage-2 top-2 |
|---|---:|---:|---:|---:|
| Base (same rows) | 0.9145 | 0.4486 | 0.9139 | 0.9934 |
| Temporal | 0.9147 | **0.5198** | **0.9219** | 0.9894 |

At threshold 0.6:

| Model | Rot. precision | Rot. recall | False rot. rate | Exact rotation acc. |
|---|---:|---:|---:|---:|
| Base | 0.325 | 0.819 | 0.137 | 0.754 |
| Temporal | 0.317 | 0.793 | 0.138 | 0.728 |

At threshold 0.7, temporal shifts toward recall (0.661 vs 0.544) but with a
higher false rate (0.079 vs 0.054). ROC-AUC is flat; only AP improves. This is
not a breakthrough. Stage 1 is near its ceiling with teammate-only information;
the missing signal is likely enemy last-seen / communication, not more
teammate-temporal features.

## Model proposes, LLM explains (follow-up)

The random forest chooses the top-2 actions; the LLM may only explain them, not
change them. 50 decisions (10 per action).

| Metric | Result |
|---|---:|
| Errors | 0 / 50 |
| Explains both model options | 100% |
| Changes the model recommendation | 0% |
| Definitive enemy claims | 0% |
| Cites at least one card term | 98% |
| States information it would need | 100% |
| Model top-1 accuracy (this subset) | 84% |
| Latency p50 / p95 | 6.6 s / 7.7 s |

The LLM is **safe but shallow**. It does not override the model or invent enemy
positions, but it rarely cites concrete facts: any number 8%, area counts 0%,
distance 0%, HP 0%, model probability 6%. The deterministic template contains
area counts, HP, equipment, distance and the model probability, so it is more
informative, if less natural.

Confidence language is directionally calibrated (`high` mean probability 0.79,
`medium` 0.55) but never says `low`, even at ~0.31. Of the 8 rows where the
random forest's top-1 was wrong, 7 were labeled `medium` and 1 `high`; none was
`low`. Manual inspection also found softer enemy speculation ("the enemy's likely
focus/presence") that the definitive-claim regex does not catch, so 0% is a
lower bound.

Conclusion: the LLM can be a **constrained explanation layer**, not a decision
layer. It needs a grounding requirement (at least two concrete card facts plus
the model probability), explicit low-confidence language, and a human usefulness
check against the template.

## What this benchmark does NOT prove

- That a larger or fine-tuned LLM cannot do better.
- That the actual next move is optimal; it is an imitation label.
- That the result transfers to solo queue; the data is professional play.
- That a hybrid "model proposes, LLM explains" system is useful; that is the
  next thing to test.

## Next

1. Add a **grounding prompt variant**: require at least two concrete card facts
   and the model probability in each explanation; compare grounding and human
   ratings against the template.
2. Have a player rate the 20 rows in `human_review_explanations.csv` on
   faithfulness and usefulness.
3. If the grounded LLM does not beat the template, use the template as the
   explanation and keep the LLM only for free-form questions / post-round
   review.

## Run

```bash
cd cs2-assistant
uv run --with pandas --with pyarrow --with scikit-learn \
  python benchmarks/action_selection/run.py

uv run --with pandas --with pyarrow --with scikit-learn \
  --with pydantic --with jsonschema \
  python benchmarks/action_selection/run_llm.py
```

Outputs:

- `data/action_selection/decisions.parquet`
- `data/action_selection/sample_100.parquet`
- `results/action_selection/baselines.json`
- `results/action_selection/llm_predictions.parquet`
- `results/action_selection/llm_metrics.json`
