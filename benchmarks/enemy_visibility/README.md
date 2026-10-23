# Enemy visibility benchmark

This benchmark tests whether enemy last-seen information from radar helps the
CT action policy. It uses 9 HLTV Dust2 demos, parsed with Awpy 2.0.2.

## Pipeline

1. Demos are extracted from the HLTV `.rar` archives into
   `TAGNet-CS2-main/demos/`.
2. `scripts/parse_demos.py` writes `ticks.parquet` and `rounds.parquet` per
   match into `TAGNet-CS2-main/data/parsed/`.
3. `build_dataset.py` creates CT decisions at 10/12/15/18/20 s with the same
   CT features as the action benchmark, plus enemy last-seen features.
4. `run_enemy_features.py` runs the two-stage policy with and without enemy
   features.

## Dataset

- 9 matches, 194 rounds, 3,813 decisions.
- 231 true rotations (6.1%).
- Actions: HOLD 3,582, ROTATE_B 72, ROTATE_MID 71, ROTATE_CT 66, ROTATE_A 22.

## The `spotted` field is not radar visibility

Attempt 1 used the demo `spotted` flag. It is unusable in these GOTV demos:

- A T player killed by a CT had `spotted=False` for all 129 ticks in the 2 s
  before the kill.
- The spotted rate is 0.2-0.6% across round times 10-50 s.
- 3,565 / 3,813 decisions have zero enemies seen in the last 5 s.

The enemy features did not help (stage-1 ROC-AUC 0.9065 -> 0.8954, AP 0.3541 ->
0.3444), but this is **inconclusive** because the input is invalid.

## Map-based visibility proxy

`spotted` was replaced (conceptually) by a geometry proxy:

- Dust2 walkable mask (`TAGNet-CS2-main/data/map/dust2_walkable_mask.npz`);
- line-of-sight sampling every 8 units;
- CT yaw and a 50° half-FOV.

Validation: 73.6% of cross-side kills have the attacker seeing the victim
(81/110); this is a reasonable but imperfect lower bound. At fixed snapshots,
only 1-10% of alive T players are visible to any CT.

Conclusion: fixed-snapshot enemy features are mostly zero because enemy
visibility is sparse. The product question is about **event-driven advice when
an enemy appears on radar**, not a fixed-timestamp model.

## Event-driven dataset (v2)

The fixed-snapshot design cannot answer the radar question, so `v2` changes the
unit of analysis from a timestamp to a **spot event**:

1. Sample live round time at 10.7 Hz (every 6 ticks).
2. Compute map-geometry visibility for every alive CT/T pair:
   yaw within ±50°, clear line of sight on the walkable mask (8-unit samples),
   and no active smoke within 144 units of any sample point.
3. When at least one T becomes visible to the team after a ≥2 s gap, create one
   event. Every alive CT gets a decision row, because radar information is shared.
4. Label = the CT's macro area 3 s later (HOLD / ROTATE_A / B / MID / CT).

`base` and `base_plus_enemy` use exactly the same rows. The base arm gets CT
state plus event timing; the enemy arm additionally gets the visible-enemy
summary (area, distance, bearing, visible/unknown counts, spotter flag). This
isolates the value of radar information from the event-selection effect.

### Dataset

- 9 matches, 195 rounds, 1,145 events, 4,130 decisions.
- Actions: HOLD 3,489 (84.5%), ROTATE_MID 230, ROTATE_CT 199, ROTATE_B 118,
  ROTATE_A 94.
- Affected subsets: 3,094 decisions with an enemy within 2,000 units, 1,714
  within 1,200 units, 968 where this CT is a spotter.
- Visibility validation on 1,303 cross-side kills: 64.4% seen under the proxy
  for clean kills (no smoke, no wallbang); 25.5% through smoke and 38.7% through
  walls. The proxy is a deliberate lower bound (2D, no elevation/wallbang).

### Results (GroupKFold by match, mean of seeds 42/7/2024)

| Metric | base | + enemy | delta |
|---|---:|---:|---:|
| Stage-1 ROC-AUC | 0.833 | **0.850** | +0.017 |
| Stage-1 AP | 0.418 | **0.459** | +0.041 |
| Stage-2 accuracy (true rotations) | 0.901 | **0.953** | +0.052 |
| Stage-2 top-2 | 0.991 | **0.994** | +0.003 |
| @0.6 rotation precision | 0.394 | **0.418** | +0.024 |
| @0.6 rotation recall | 0.669 | **0.685** | +0.017 |
| @0.6 false rotation | 0.189 | **0.175** | -0.014 |
| @0.6 exact rotation accuracy | 0.586 | **0.658** | +0.072 |
| @0.6 overall accuracy | 0.776 | **0.799** | +0.023 |

Baselines on the same rows: majority HOLD 84.5% accuracy / 0.200 balanced
accuracy / 0 rotation recall; "rotate to the nearest visible enemy" 41.2%
accuracy, 0.220 rotation precision, 0.871 rotation recall, 0.567 false-rotation
rate. The learned policy is far better calibrated than the naive enemy rule.

Stage-1 feature importance is dominated by distance-to-area and current area
(`dist_Mid` 0.159, `dist_B` 0.134, `dist_CT` 0.119, `area` 0.106), but the
enemy features are non-trivial: `enemy_mean_distance` 0.044,
`enemy_nearest_distance` 0.039, `enemy_nearest_relative_bearing` 0.019,
`enemy_visible_B` 0.015, `enemy_team_nearest_distance` 0.011.

### Five-second horizon and negative controls

A 3 s label is too short for a strategic rotation: 43.8% of 3 s "rotations"
move <400 units, which is mostly repositioning. The 5 s dataset (`--horizon-seconds 5`)
is the product-relevant version:

- 1,099 events, 3,792 decisions, 858 rotations (22.6%);
- rotation median displacement 658 units, only 21.1% <400 units;
- labels: HOLD 2,934, ROTATE_MID 324, ROTATE_CT 262, ROTATE_A 151, ROTATE_B 121.

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

Negative controls (`run_controls.py`) rule out extra capacity and marginal
distribution:

| Arm | Stage-1 AP | Stage-2 accuracy |
|---|---:|---:|
| base | 0.478 | 0.740 |
| base + enemy | **0.547** | **0.848** |
| enemy block shuffled within match | 0.464 | 0.724 |
| enemy block shuffled globally | 0.473 | 0.720 |
| base + random-noise features | 0.479 | 0.719 |

The real enemy block beats every control by ~0.07 AP and ~0.12 stage-2
accuracy, so the gain is tied to the enemy information at that decision, not to
feature count or the event selection alone.

### Deployment artifact

`export_event_policy.py` trains the 5 s two-stage policy on all rows and writes
`live/models/rotation_policy.json` (200 trees per stage, ~2 MB). The artifact
contains feature order, category mappings, centroids, tree arrays, threshold and
20 parity examples. `live/rotation_policy.py` reproduces sklearn's prediction
with pure-standard-library traversal (max observed difference <1e-15).

### Caveats

- Rows are correlated within an event and across consecutive events of the same
  round. GroupKFold by match measures generalization to new matches, but the
  4,130 rows are not 4,130 independent samples; treat small deltas cautiously.
- Visibility is a 2D lower bound: no elevation, no wall penetration, and a
  circular smoke approximation. Enemy features are therefore conservative.
- The dataset has 9 matches; per-match heterogeneity is visible in the margins
  (enemy distance ≤1,200 units does not improve stage-1 AUC consistently).

### What this means

- Radar information is **not useless**: the event-driven dataset has dense,
  non-zero enemy signal, and it improves every stage-1 metric.
- The largest gain is the destination stage (+5.2 points): once a CT decides to
  move, the visible enemy's area is a strong prior for where it moves.
- Stage-1 remains the product bottleneck: at the 0.6 operating point, 17.5% of
  HOLD decisions still become false rotations and 31.5% of rotations are
  missed. Enemy information narrows the gap but does not close it.
- The event-driven design is therefore the right unit for live shadow mode:
  trigger advice on a spot, not on a clock.

## Run

```bash
# Parse (already done; requires awpy==2.0.2 and the demos)
cd TAGNet-CS2-main
uv run --python 3.11 --with awpy==2.0.2 python scripts/parse_demos.py \
  --demo-dir ../demos --output-root data/parsed

cd ../cs2-assistant

# v1 fixed snapshots
uv run --with numpy --with pandas --with pyarrow python benchmarks/enemy_visibility/build_dataset.py
uv run --with numpy --with pandas --with pyarrow --with scikit-learn python benchmarks/enemy_visibility/run_enemy_features.py

# v2 event-driven, 3 s default
uv run --with numpy --with pandas --with pyarrow python benchmarks/enemy_visibility/build_event_dataset.py
uv run --with numpy --with pandas --with pyarrow --with scikit-learn python benchmarks/enemy_visibility/run_event_policy.py

# v2 event-driven, 5 s (product horizon) + label audit + controls
uv run --with numpy --with pandas --with pyarrow python benchmarks/enemy_visibility/build_event_dataset.py --horizon-seconds 5
uv run --with numpy --with pandas --with pyarrow python benchmarks/enemy_visibility/audit_labels.py --data data/enemy_visibility/event_decisions_5s.parquet --output results/enemy_visibility/label_audit_5s.json
uv run --with numpy --with pandas --with pyarrow --with scikit-learn python benchmarks/enemy_visibility/run_event_policy.py --data data/enemy_visibility/event_decisions_5s.parquet --output results/enemy_visibility/event_policy_5s.json
uv run --with numpy --with pandas --with pyarrow --with scikit-learn python benchmarks/enemy_visibility/run_controls.py --data data/enemy_visibility/event_decisions_5s.parquet --output results/enemy_visibility/controls_5s.json

# export the live rotation artifact
uv run --with numpy --with pandas --with pyarrow --with scikit-learn python benchmarks/enemy_visibility/export_event_policy.py

# unit tests
uv run --with numpy python benchmarks/enemy_visibility/test_visibility.py
```

Outputs:

- `data/enemy_visibility/decisions.parquet` (v1)
- `data/enemy_visibility/audit.json` (v1)
- `data/enemy_visibility/event_decisions.parquet` (v2, 3 s)
- `data/enemy_visibility/event_audit.json` (v2, 3 s)
- `data/enemy_visibility/event_decisions_5s.parquet` (v2, 5 s)
- `data/enemy_visibility/event_audit_5s.json` (v2, 5 s)
- `results/enemy_visibility/enemy_visibility_policy.json` (v1)
- `results/enemy_visibility/event_policy.json` (v2, 3 s)
- `results/enemy_visibility/event_policy_5s.json` (v2, 5 s)
- `results/enemy_visibility/label_audit*.json`
- `results/enemy_visibility/controls*.json`
- `live/models/rotation_policy.json` (deployment artifact)
- `results/enemy_visibility/summary.md`

## Next

- Build the radar/HUD perception module that produces the `/radar` contract in
  `live/README.md`. This is the only missing piece for in-game rotation advice.
- Add utility occlusion beyond smoke (molotov/incendiary) and a 3D/height-aware
  LOS if a better map model becomes available.
- Add a radar "fade" model: the demo only gives geometry, while live radar keeps
  a dot briefly after loss of sight.
- Calibrate the stage-1 threshold for the product cost of a false rotation vs a
  missed rotation; the current 0.6 point is not yet cost-optimised.
- Run a full shadow session with the demo feeder or a real observer feed and
  review `live/shadow_log.jsonl` before showing prompts in-game.
