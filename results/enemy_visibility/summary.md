# Enemy visibility experiment — summary

Data: 9 HLTV Dust2 demos, 194 usable rounds, 3,813 CT decisions, 231 true
rotations. Parsed with Awpy 2.0.2 (`ticks` + `rounds`).

## Attempt 1: the demo `spotted` field

The first enemy feature set used the ticks `spotted` flag within a 5 s lookback.
It is **not a valid radar-visibility signal** in these GOTV demos:

- A T player who was killed by a CT had `spotted=False` for all 129 ticks in the
  2 s before the kill.
- Across round times 10-50 s, the spotted rate is only 0.2-0.6% of alive T
  player-ticks.
- In the decision dataset, `enemy_seen_5s` is 0 for 3,565 / 3,813 rows; only 248
  rows have any enemy seen, and only 32 have an enemy spotted at the snapshot.

Result with the `spotted` features:

| Model | Stage-1 ROC-AUC | Stage-1 AP | Rot. precision @0.6 | Rot. recall @0.6 |
|---|---:|---:|---:|---:|
| Base | 0.9065 | 0.3541 | 0.286 | 0.727 |
| + `spotted` enemy features | 0.8954 | 0.3444 | 0.244 | 0.710 |

The enemy features did not help, but this is **inconclusive**: the input signal
is mostly zero and semantically wrong. It is not evidence that radar
information is useless.

## Attempt 2: map-based line-of-sight / FOV proxy

Using the Dust2 walkable mask, CT yaw and a 50° half-FOV:

- 73.6% of cross-side kills have the attacker seeing the victim through this
  proxy (81/110), so the geometry is roughly sane.
- At fixed snapshots, only 1-10% of alive T players are visible to any CT.
- Radar-spotted enemies are rare at the fixed 10-20 s snapshots used by the
  action benchmark; most decisions have no enemy information at all.

This explains why the enemy features are mostly zero: the experiment samples
the wrong moments. The product idea is about **event-driven advice when an enemy
appears on radar**, not a fixed-snapshot model.

## What this means

- The fixed-snapshot enemy-feature experiment cannot answer whether radar
  information helps. The correct unit is a **spot event**, not a timestamp.
- The base action model on the 9 demos is similar to the 697-round benchmark
  (stage-1 ROC-AUC 0.9065, AP 0.3541); the destination stage remains strong
  (90.5% accuracy, 99.1% top-2).
- The next experiment should build an **event-driven dataset**: at the moment a
  T player becomes visible, create a decision for nearby/affected CT players and
  predict their next-area response. Enemy features will then be non-zero.
- The LOS proxy still needs smoke/utility occlusion and a better wall model; it
  should be treated as a lower bound on visibility.

## Event-driven dataset (v2)

Fixed snapshots were replaced by spot events: sample round time at 10.7 Hz,
compute FOV + LOS + smoke visibility, and whenever a T first becomes visible
after a >=2 s gap, create a decision for every alive CT. Label is the CT's area
3 s later. Both arms use the same rows; only `base_plus_enemy` sees the
visible-enemy summary.

### Data

- 9 matches, 195 rounds, 1,145 events, 4,130 decisions.
- Actions: HOLD 3,489 (84.5%), ROTATE_MID 230, ROTATE_CT 199, ROTATE_B 118,
  ROTATE_A 94.
- Affected: 3,094 rows with an enemy within 2,000 units; 1,714 within 1,200;
  968 where the CT itself is a spotter.
- Visibility validation (1,303 cross-side kills): clean 64.4% seen, through
  smoke 25.5%, through walls 38.7%. The proxy is a lower bound.

### Results (mean of seeds 42/7/2024, GroupKFold by match)

| Metric | base | + enemy | delta |
|---|---:|---:|---:|
| Stage-1 ROC-AUC | 0.833 | **0.850** | +0.017 |
| Stage-1 AP | 0.418 | **0.459** | +0.041 |
| Stage-2 accuracy | 0.901 | **0.953** | +0.052 |
| Stage-2 top-2 | 0.991 | **0.994** | +0.003 |
| @0.6 precision | 0.394 | **0.418** | +0.024 |
| @0.6 recall | 0.669 | **0.685** | +0.017 |
| @0.6 false rotation | 0.189 | **0.175** | -0.014 |
| @0.6 exact rotation | 0.586 | **0.658** | +0.072 |
| @0.6 overall accuracy | 0.776 | **0.799** | +0.023 |

Baselines: majority HOLD 84.5% accuracy (balanced 0.200, rotation recall 0);
"rotate to nearest visible enemy" 41.2% accuracy, 0.220 rotation precision,
0.871 rotation recall, 0.567 false-rotation rate. The naive enemy rule is
badly calibrated; the learned policy is not.

Stage-1 importance is led by CT geometry (`dist_Mid` 0.159, `dist_B` 0.134,
`dist_CT` 0.119, `area` 0.106), then enemy distance signals:
`enemy_mean_distance` 0.044, `enemy_nearest_distance` 0.039,
`enemy_nearest_relative_bearing` 0.019, `enemy_visible_B` 0.015.

### Caveats

- Decisions are correlated within an event and across consecutive events in the
  same round. GroupKFold by match measures generalization to unseen matches, but
  the 4,130 rows are not independent; the seed-to-seed stability is good, the
  effective sample size is much smaller.
- The visibility proxy is a 2D lower bound (no elevation, no wall penetration,
  circular smoke), so enemy features are conservative.
- Only 9 matches. The distance ≤1,200 subset does not show the same stage-1 AUC
  gain, so the effect is not uniform across proximity bands.

### Interpretation

- Radar information is useful in the right experimental unit. The event-driven
  dataset has 1.44 visible T per decision on average (vs ~0.07 in v1).
- The destination stage gains most (+5.2 points): once a CT moves, the seen
  enemy's area is a strong prior for the destination.
- Stage 1 is still the bottleneck: at 0.6, 17.5% false rotations and 31.5%
  missed rotations. The enemy features move the curve but do not break it.
- The event trigger, not the clock, is the correct live shadow-mode design.

## Five-second horizon, controls and deployment

The 3 s label is too short: 43.8% of 3 s rotations move <400 units. The 5 s
dataset is the product-relevant one: 1,099 events, 3,792 decisions, 858
rotations (22.6%), median rotation displacement 658 units, only 21.1% <400.

| Metric (seed 42, GroupKFold by match) | base | + enemy |
|---|---:|---:|
| Stage-1 ROC-AUC | 0.800 | **0.834** |
| Stage-1 AP | 0.478 | **0.547** |
| Stage-2 accuracy | 0.740 | **0.848** |
| Stage-2 top-2 | 0.953 | **0.977** |
| @0.6 precision | 0.487 | **0.544** |
| @0.6 recall | 0.626 | **0.664** |
| @0.6 false rotation | 0.193 | **0.163** |
| @0.6 exact rotation | 0.407 | **0.541** |
| @0.6 overall accuracy | 0.717 | **0.770** |

Negative controls:

| Arm | Stage-1 AP | Stage-2 accuracy |
|---|---:|---:|
| base | 0.478 | 0.740 |
| base + enemy | **0.547** | **0.848** |
| enemy shuffled within match | 0.464 | 0.724 |
| enemy shuffled globally | 0.473 | 0.720 |
| base + random noise | 0.479 | 0.719 |

The row-linked enemy block beats every control by ~0.07 AP and ~0.12 stage-2
accuracy. The gain is real, not extra capacity or marginal distribution.

Deployment: `export_event_policy.py` writes a ~2 MB JSON artifact
(`live/models/rotation_policy.json`), and `live/rotation_policy.py` reproduces
sklearn predictions with stdlib-only tree traversal (difference <1e-15). The
live server only emits a rotation call when it has teammates and at least one
visible enemy; otherwise it returns `NO_RADAR` / `NO_TEAMMATE_DATA`.

## Next

1. Add smoke/utility occlusion beyond smoke and, if possible, height-aware LOS.
2. Model radar fade (live radar keeps a dot briefly after sight is lost).
3. Tune the stage-1 threshold to the product cost of false vs missed rotations;
   test 2 s / 5 s horizons.
4. Build the radar/HUD perception module that fills the `/radar` contract in
   `live/README.md`, then run a full shadow session and compare advice against
   the frozen policy log. This is now the only missing piece for in-game use.
