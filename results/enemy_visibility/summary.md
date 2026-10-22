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

## Next

1. Build the event-driven radar dataset (spot triggers).
2. Validate LOS/FOV against actual kills and add smoke occlusion.
3. Re-run the two-stage policy on spot events.
4. If the event-driven policy shows signal, then invest in radar capture for
   live shadow mode.
