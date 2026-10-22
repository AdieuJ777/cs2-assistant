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
