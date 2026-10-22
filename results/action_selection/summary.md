# Action-selection benchmark (C) — summary

Question: given observable CT team state, can the agent choose the player's
next action?

Data: 13,560 CT player decisions from 697 Dust2 rounds (4 snapshot windows).
Evaluation sample: 100 balanced decisions (20 per action), one per round,
32 matches. Target: actual next macro area.

## Baselines (same 100 rows)

| Method | Accuracy | Balanced Acc | Macro F1 | Top-2 |
|---|---:|---:|---:|---:|
| Majority HOLD | 0.200 | 0.200 | 0.067 | — |
| Team-need rule | 0.240 | 0.240 | 0.166 | — |
| Decision tree depth 6 | 0.690 | 0.690 | 0.693 | 0.760 |
| Random forest | **0.820** | **0.820** | **0.806** | **0.930** |

## LLM (qwen3:8b, zero-shot, STP)

100 calls, 0 errors, p50 6.8 s, p95 9.1 s.

| Metric | LLM | Random forest |
|---|---:|---:|
| Top-1 accuracy | 0.370 | 0.820 |
| Top-2 coverage | 0.620 | 0.930 |
| Rotation recall | 0.463 | — |
| HOLD recall | **0.000** | — |
| Abstain rate | 0.000 | — |
| Enemy-claim rate | 0.190 | — |

Recommended actions: ROTATE_B 46, ROTATE_A 44, ROTATE_MID 10. It never chose
HOLD or ROTATE_CT.

## Findings

1. The LLM cannot represent "do nothing": it predicted a rotation for all 20
   HOLD cases. On the natural 92%-HOLD distribution this is catastrophic.
2. It over-rotates to A/B and ignores MID/CT.
3. It reports `information_needed` in 100% of cases but never abstains; its
   uncertainty is decorative.
4. 19% of recommended rationales assert enemy intent ("the enemy is likely
   defending B") even though enemy positions are explicitly unknown.
5. The random forest is a much stronger action selector (82% vs 37%), and its
   Top-2 coverage is 93% vs 62%.
6. Latency is too high for a 2-3 second mid-round window, but acceptable between
   rounds.

## Conclusion

The LLM should not be the action selector. The structured team state already
contains the signal; the statistical model captures it. The LLM's plausible role
is explanation, interaction, and post-round review — with strict grounding, or
not in the in-game loop at all.

## Feature ablation and natural distribution (live-plan check)

Feature ablation on the 100-row balanced sample:

| Feature set | Features | Balanced Acc | Top-2 | HOLD recall |
|---|---:|---:|---:|---:|
| Full | 23 | 0.820 | 0.930 | 0.45 |
| GSI + radar/HUD | 20 | 0.780 | 0.910 | 0.40 |
| GSI + radar, no HUD | 19 | 0.790 | 0.930 | 0.40 |
| Shape only | 18 | 0.790 | 0.930 | 0.45 |

Removing the features that CV cannot provide costs only 3-4 points. But on the
natural distribution (13,560 decisions, 92% HOLD):

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

The model collapses to HOLD and misses 85% of rotations. A threshold sweep does
not fix it. **The action model is not ready for live action prompts; the
bottleneck is the policy/calibration, not the missing CV features.**

## Two-stage policy (fixing the HOLD collapse)

Stage 1 (ROTATE vs HOLD, class-weighted) + stage 2 (destination, trained only
on true rotations):

- Stage 2 on 1,004 true rotations: accuracy 0.921, top-2 0.994, per-class
  recall 0.91-0.93.
- Stage 1 at 0.5: precision 0.235, recall 0.944, false rotation 0.245,
  ROC-AUC 0.923, average precision 0.493.
- End-to-end at threshold 0.6: rotation precision 0.343, recall 0.825, false
  rotation 0.126, exact rotation accuracy 0.770, overall accuracy 0.866.
- At threshold 0.7: precision 0.458, recall 0.556, false rotation 0.053,
  exact rotation accuracy 0.519.

The destination problem is solved once a rotation is detected; the remaining
choice is the stage-1 false/missed rotation trade-off.

## Temporal features

On 10,117 rows with a previous window:

| Model | Stage-1 ROC-AUC | Stage-1 AP | Stage-2 acc | Stage-2 top-2 |
|---|---:|---:|---:|---:|
| Base | 0.9145 | 0.4486 | 0.9139 | 0.9934 |
| Temporal | 0.9147 | **0.5198** | **0.9219** | 0.9894 |

At threshold 0.6, precision/recall/false rate are essentially unchanged
(0.317/0.793/0.138 vs 0.325/0.819/0.137). At 0.7, temporal trades precision for
recall (0.661 vs 0.544) with a higher false rate (0.079 vs 0.054). ROC-AUC is
flat.

Conclusion: teammate-temporal features improve the PR ranking slightly but do
not break the stage-1 ceiling. The missing signal is enemy last-seen /
communication, not more teammate history.

## Follow-up: model proposes, LLM explains

50 decisions; random-forest top-2 fixed; LLM explains only.

- 0 errors, 100% action match, 0% definitive enemy claims, 100% states
  information needed.
- Grounding is shallow: any number 8%, area counts 0%, distance 0%, HP 0%,
  model probability 6%. The deterministic template contains all of those facts.
- Confidence language is directional (`high` mean probability 0.79, `medium`
  0.55) but never `low`.
- Latency p50 6.6 s, p95 7.7 s.

Conclusion: a constrained explanation layer is viable, but it needs a grounding
requirement and a human usefulness check. It is not a decision layer.

## Next

Run a grounding prompt variant (at least two concrete card facts + model
probability) and have a player rate 20 explanations against the template. If the
grounded LLM does not beat the template, use the template and keep the LLM for
free-form questions / post-round review.
