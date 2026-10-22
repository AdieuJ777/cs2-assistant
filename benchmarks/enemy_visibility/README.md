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

## Run

```bash
# Parse (already done; requires awpy==2.0.2 and the demos)
cd TAGNet-CS2-main
uv run --python 3.11 --with awpy==2.0.2 python scripts/parse_demos.py \
  --demo-dir ../demos --output-root data/parsed

# Build decisions
cd ../cs2-assistant
python benchmarks/enemy_visibility/build_dataset.py

# Evaluate
python benchmarks/enemy_visibility/run_enemy_features.py
```

Outputs:

- `data/enemy_visibility/decisions.parquet`
- `data/enemy_visibility/audit.json`
- `results/enemy_visibility/enemy_visibility_policy.json`
- `results/enemy_visibility/summary.md`

## Next

- Build an **event-driven** dataset: at the moment a T player becomes visible,
  create decisions for nearby CT players and predict their next-area response.
- Add smoke/utility occlusion to the LOS proxy.
- Validate the visibility model against more kill events.
- Then re-run the two-stage policy; only if it shows signal should radar capture
  be built for live shadow mode.
