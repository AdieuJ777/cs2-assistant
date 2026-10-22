# CS2 Decision Assistant (working title)

A player-view decision-support agent for Counter-Strike 2. The first target is
**Dust2, CT side, solo queue**, with **GSI-only** input in the first version.
The agent is meant to be advisory: it outputs 2-3 executable options with
confidence and rationale; it never reads hidden information and never controls
the game.

This repository is currently at the **de-risking** stage, not the product stage.

## What we are trying to prove first

The core hypothesis is not "an LLM can talk about CS2". It is:

> Given only information a live player can see, can the agent make decisions
> that are measurably better than a simple rule baseline?

Before building perception, real-time plumbing, or an overlay, we need a
benchmark that can answer this. The first benchmark is deliberately narrow:
**infer the enemy (T) buy type before a round**.

## What we learned from the data audit

The released TAGNet dataset (`TAGNet-CS2-main/data/processed/dust2_ct_5snap`)
is **CT-only**:

- 697 rounds from 33 matches; 5 CT player nodes per snapshot.
- It has CT positions, health, armor, equipment value, place labels, score,
  round metadata, team equipment totals and economy type.
- It does **not** have T-player positions, per-player money, weapon inventory,
  per-player utility inventory, kill feed, or purchase events.

Consequences:

| Planned capability | Feasible with this data? |
|---|---|
| Enemy economy / buy-type inference | **Yes**, at team level |
| Individual buy/save recommendation | **No** (no per-player money/weapons) |
| Enemy position inference | **No** (no T-player nodes) |
| Teammate utility-aware advice | **No** (no utility inventory) |
| CT setup / rotation analysis | **Yes**, at team level |

The first benchmark therefore tests the enemy-economy inference layer only.
Individual buy/save and enemy-position benchmarks need new data: raw demos
parsed with Awpy, or GSI logs from real matches.

## Layout

```
cs2-assistant/
├── benchmarks/
│   ├── economy_inference/
│   │   ├── run.py              # builds the benchmark and runs baselines
│   │   ├── run_llm.py          # LLM arm through STP (zero-shot / rules / hybrid)
│   │   ├── ollama_backend.py   # native Ollama backend for STP
│   │   └── README.md           # design, leakage controls, results
│   └── action_selection/
│       ├── run.py              # 13,560 CT next-move decisions + baselines
│       ├── run_llm.py          # LLM next-action arm through STP
│       └── README.md           # design, results, failure modes
├── live/                    # live GSI-only buy/save coach (v1)
│   ├── advisor.py           # deterministic freeze-time rules
│   ├── gsi_server.py        # local GSI listener + web page
│   ├── gsi_config/          # CS2 GSI config file
│   └── README.md            # setup and limitations
├── data/                   # generated benchmark data
└── results/                # generated baseline results
```

## Run

```bash
cd cs2-assistant
uv run --with pandas --with pyarrow --with scikit-learn \
  python benchmarks/economy_inference/run.py
```

The script reads the TAGNet processed data from
`../TAGNet-CS2-main/data/processed/dust2_ct_5snap`, writes the benchmark table
to `data/economy_inference/rounds.parquet`, and writes baseline metrics to
`results/economy_inference/baselines.json`.

## Current result (pilot)

664 rounds with usable lag features; 5-fold GroupKFold by match:

| Model | Accuracy | Macro F1 | Brier |
|---|---:|---:|---:|
| Majority class | 0.474 | 0.161 | 1.051 |
| Hand-written economy rule | 0.551 | 0.562 | — |
| Decision tree (depth 3) | 0.693 | 0.575 | 0.423 |
| Decision tree (depth 6) | **0.744** | **0.740** | 0.385 |
| Random forest | 0.735 | 0.702 | **0.333** |

`t_total_equip` regression MAE is ~3001, against a mean of ~20985 (~14%).

Interpretation: enemy economy is **learnable at a coarse level** from pre-round
observable information, but not precise. The assistant should output a
distribution and uncertainty, not a single number. A hand-written rule is
clearly weaker than a learned model, which supports building an inference layer.

### LLM arm (80-row balanced subset)

| Method | Accuracy | Macro F1 | Brier |
|---|---:|---:|---:|
| Hand rule | 0.600 | 0.581 | — |
| Tree depth 6 | 0.725 | 0.727 | 0.380 |
| Random forest | 0.725 | 0.701 | 0.350 |
| LLM zero-shot (qwen3:8b) | 0.350 | 0.298 | 0.888 |
| LLM rules + few-shot | 0.588 | 0.591 | 0.506 |
| LLM hybrid (tree prior) | 0.662 | 0.674 | 0.414 |
| LLM hybrid + pistol hard rule | 0.725 | 0.726 | 0.414 |

**Key finding: the LLM is not a competitive economy inferencer at this size.**
Its best result merely reproduces the tree prior; it does not improve accuracy
or calibration. The pistol fix that helped most is a deterministic rule, not
reasoning. Its role should be explanation, qualitative synthesis, action
tradeoffs, and asking for missing information — not numeric inference.

## Action-selection benchmark (C)

13,560 CT player decisions from 697 rounds; 100 balanced decisions (20 per
action), one per round. Target: the player's actual next macro area, with enemy
information explicitly unknown.

| Method | Accuracy | Balanced Acc | Macro F1 | Top-2 |
|---|---:|---:|---:|---:|
| Majority HOLD | 0.200 | 0.200 | 0.067 | — |
| Team-need rule | 0.240 | 0.240 | 0.166 | — |
| Decision tree depth 6 | 0.690 | 0.690 | 0.693 | 0.760 |
| Random forest | **0.820** | **0.820** | **0.806** | **0.930** |
| LLM zero-shot (qwen3:8b) | 0.370 | — | 0.256 | 0.620 |

The LLM **never predicted HOLD** (0/20), recommended only ROTATE_A/B/MID, and
never abstained. 19% of its recommended rationales asserted enemy intent
("the enemy is likely defending B") despite enemy information being explicitly
unknown. On the natural 92%-HOLD distribution, this behavior would be
catastrophic.

**Natural-distribution check (13,560 decisions, 92% HOLD):** the random forest's
balanced accuracy is only 0.302 and its rotation recall is 0.150; it collapses
to HOLD and misses 85% of real rotations. Feature ablation shows that removing
the features CV cannot provide costs only 3-4 points. So the bottleneck is the
action policy/calibration, not perception.

**Two-stage follow-up:** stage 1 (ROTATE vs HOLD, class-weighted) plus stage 2
(destination, trained only on true rotations) fixes the collapse. Stage 2 reaches
92.1% accuracy and 99.4% top-2 on true rotations. Stage 1 has ROC-AUC 0.923 but
needs an operating point: at threshold 0.6, rotation recall is 0.825, false
rotation rate 0.126, exact rotation accuracy 0.770; at 0.7, recall 0.556, false
rate 0.053, exact 0.519. The destination problem is solved; the remaining choice
is the false-rotation/missed-rotation trade-off.

**Temporal-feature follow-up:** adding the previous window (movement, deltas,
previous action) raises stage-1 average precision from 0.449 to 0.520, but
ROC-AUC is flat (0.9145 -> 0.9147) and the 0.6 operating point is unchanged.
Stage 1 is near its ceiling with teammate-only information; the missing signal is
enemy last-seen / communication, not more teammate history.

**Conclusion:** the LLM should not select actions. The two-stage policy is a
viable candidate for live rotation advice, but stage 1 needs enemy last-seen
data (demos or live radar logs) before CV is worth the investment.

### Model proposes, LLM explains (follow-up)

50 decisions; the random forest's top-2 is fixed and the LLM only explains it.
0 errors, 100% action match, 0% definitive enemy claims, 100% states what
information it needs. But grounding is shallow: any number 8%, area counts 0%,
distance 0%, HP 0%, model probability 6%. The deterministic template contains
all of those facts, so it is more informative, if less natural. Latency p50
6.6 s.

Wrong recommendations are still explained confidently: of the 8 rows where the
model's top-1 was wrong, 7 were labeled `medium` and 1 `high`, none `low`.
Manual review also found softer enemy speculation that the regex does not catch.

Conclusion: a constrained explanation layer is viable; it needs a grounding
requirement (at least two concrete card facts plus the model probability),
explicit low-confidence language, and a human usefulness check before it earns
a place in the product.

## Next steps

1. **Get enemy last-seen data before more modelling.** Temporal features improved
   stage-1 average precision but not ROC-AUC or the 0.6 operating point. The
   next useful signal is enemy last-seen from radar/demos. Provide 10-20 Dust2
   demos for visibility simulation, or log live radar + GSI for shadow mode.
2. **Choose an operating point.** Stage 2 is strong (92.1% on true rotations);
   stage 1 trades recall against false rotations (0.6 = recall 0.825 / false
   0.126; 0.7 = recall 0.556 / false 0.053).
3. **Grounding variant + human review.** Require at least two concrete card
   facts and the model probability in each explanation; have a player rate the
   20 rows in `human_review_explanations.csv` against the template.
4. If the grounded LLM beats the template on usefulness, the product
   architecture is: statistical model → recommendation; LLM → explanation /
   interaction. If not, use the template and keep the LLM for free-form
   questions / post-round review.
5. Collect GSI logs from the user's own matches for the individual buy/save
   benchmark (GSI gives exact self money/weapons/utility).
6. Parse raw demos with Awpy for T-player positions, kill feed and utility, to
   build the enemy/team masked-replay benchmark.
7. Only after those pass: shadow mode, then a narrow live decision.

## Live v1: GSI-only buy/save coach

The first in-game prompt is live under `live/`. It is deliberately narrow:
Dust2, CT, freeze-time buy/save advice from GSI-only data. It does not use the
LLM or the benchmark models, because their inputs (team economy, teammate
positions) are not available from GSI alone.

```bash
cd cs2-assistant
python live/gsi_server.py --token change-me --verbose
# open http://127.0.0.1:3000
```

Copy `live/gsi_config/gamestate_integration_cs2assistant.cfg` into the CS2
`game/cs2/cfg/` directory, change the token in both places, and start the server
before launching CS2. The page shows a buy recommendation, a shopping list, the
enemy-economy prior (low confidence), and warnings.

Limitations: no teammate/enemy positions, no teammate money, no individual
history, CT only, Dust2 only, prices are hand-written and may need updating.
Tests: `python -m unittest discover -s live/tests -t .`.

## STP fit note

The LLM arm is built as an STP task, not an ad-hoc script:

- `TaskSpec` validates the input card and the final decision; the LLM step has
  its own JSON Schema; a function step normalizes probabilities.
- Retry policy, step/task timeouts, structured output and events are handled by
  the runtime.
- 80 calls per economy variant and 100 calls for the action arm, zero errors.
  p50 latency was 2.8 s with rules/few-shot, 4.7-4.8 s with the longer economy
  prompts, and 6.8 s for the action card.
- A native Ollama backend was needed: Ollama's OpenAI-compatible endpoint did
  not honour `think: false` for Qwen3, so the model spent the whole token budget
  on hidden reasoning and returned empty content. The native `/api/chat`
  endpoint with `think: false` and `format: json` works.
- The DAG was sufficient for a two-step decision task. For the live assistant,
  the remaining gaps are reactive triggers, cross-run match state, and
  single-flight/staleness handling — not the decision workflow itself.

## Non-goals (for now)

- Reading game memory, network packets, or spectator-only `allplayers`.
- Automating input or aiming.
- A general "coach" that claims to know hidden enemy positions.
#   c s 2 - a s s i s t a n t  
 