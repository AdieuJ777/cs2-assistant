# Model proposes, LLM explains — summary

Setup: random forest (out-of-fold) chooses the top-2 CT next actions; the LLM
receives the same situation card plus the model's top-2 and probabilities. The
LLM may only explain, not change the recommendation and not invent enemy
information.

Sample: 50 decisions (10 per action) from the action-selection benchmark.

## Automated metrics

| Metric | Result |
|---|---:|
| Errors | 0 / 50 |
| Explains both model options | 100% |
| Changes the model recommendation | 0% |
| Definitive enemy claims | 0% |
| Cites at least one card term | 98% |
| States information it would need | 100% |
| Model top-1 accuracy on this subset | 84% |
| Latency p50 / p95 | 6.6 s / 7.7 s |

Confidence language: `high` 20, `medium` 30, `low` 0. Mean model probability
was 0.79 for `high` and 0.55 for `medium`, so the language is directionally
calibrated but never expresses low confidence.

## Grounding is shallow

| Concrete grounding | Share of explanations |
|---|---:|
| Any number | 8% |
| Area counts (e.g., A=3, B=2) | 0% |
| Distance in units | 0% |
| HP | 0% |
| Equipment / economy | 52% |
| Score / round / previous round | 34% |
| Model probability (e.g., 62%) | 6% |

The LLM explanations are safe but generic: "strong economy and positioning",
"team's proximity", "potential counter-attack". The deterministic template
explanation contains the area counts, HP, equipment, distance and model
probability, so it is more informative, if less natural.

## Interpretation

- This is the first **non-negative** LLM result: with a fixed model
  recommendation, the LLM did not override it and did not make definitive enemy
  claims.
- But it did not add tactical insight either. It mostly paraphrased the model's
  choice in generic language and rarely cited the concrete facts that justify it.
- It also rarely states the model's probability, and never uses "low"
  confidence, even when the model's top-1 probability is around 0.31.
- **Wrong recommendations are still explained confidently.** Of the 8 rows where
  the random forest's top-1 was wrong, 7 were labeled `medium` and 1 `high`; none
  was `low`. Manual inspection also found softer enemy speculation ("the enemy's
  likely focus/presence") that the definitive-claim regex does not catch, so 0%
  is a lower bound.
- Latency is acceptable between rounds, not for a 2-3 second mid-round window.

## Conclusion

The LLM can be a **constrained explanation layer**, but only if the prompt forces
concrete grounding and the product accepts the latency. Otherwise a deterministic
template is more faithful and more informative. It should not be the decision
layer.

## Next

1. Add a grounding prompt variant: require at least two concrete card facts and
   the model probability in each explanation; compare its grounding rate and
   human ratings against the template.
2. Have a player rate the 20 rows in `human_review_explanations.csv` on
   faithfulness and usefulness.
3. If the grounded LLM does not beat the template on usefulness, use the
   template as the explanation and keep the LLM only for free-form questions /
   post-round review.
