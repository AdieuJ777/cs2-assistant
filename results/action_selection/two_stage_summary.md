# Two-stage action policy — summary

Data: 13,560 decisions (697 rounds), HOLD share 92.6%, true rotation share 7.4%.
Features: the GSI + radar/HUD set (20 features), which drops team economy and
teammate equipment value.

## Architecture

1. **Stage 1 — ROTATE vs HOLD.** Random forest with `class_weight="balanced"` to
   counter the HOLD base rate. Output is a calibrated-ish rotation probability.
2. **Stage 2 — destination.** Random forest trained only on true rotations
   (`A / B / MID / CT`), evaluated on true rotations.

Both stages use GroupKFold by match.

## Stage 2: destination is strong

On the 1,004 true rotations:

| Metric | Value |
|---|---:|
| Accuracy | 0.921 |
| Macro F1 | 0.732 |
| Top-2 coverage | 0.994 |
| ROTATE_A recall | 0.925 |
| ROTATE_B recall | 0.922 |
| ROTATE_MID recall | 0.910 |
| ROTATE_CT recall | 0.928 |

If the system knows a rotation is happening, it almost always gets the
destination right (and the true destination is in the top-2 99.4% of the time).

## Stage 1: detection is the bottleneck

At threshold 0.5:

| Metric | Value |
|---|---:|
| Rotation precision | 0.235 |
| Rotation recall | 0.944 |
| False rotation rate (on HOLD) | 0.245 |
| Missed rotation rate | 0.056 |
| ROC-AUC | 0.923 |
| Average precision | 0.493 |

There is real signal (ROC-AUC 0.923), but at high recall it produces many false
rotations.

## End-to-end threshold sweep

A rotation is emitted when the stage-1 probability exceeds the threshold; the
stage-2 destination is used only then.

| Threshold | Rot. precision | Rot. recall | False rot. rate | Missed rot. rate | Exact rotation acc. | Overall acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.142 | 0.982 | 0.474 | 0.018 | 0.903 | 0.554 |
| 0.2 | 0.154 | 0.979 | 0.431 | 0.021 | 0.900 | 0.594 |
| 0.3 | 0.163 | 0.975 | 0.400 | 0.025 | 0.897 | 0.622 |
| 0.4 | 0.207 | 0.961 | 0.295 | 0.039 | 0.886 | 0.719 |
| 0.5 | 0.235 | 0.944 | 0.245 | 0.056 | 0.870 | 0.763 |
| **0.6** | **0.343** | **0.825** | **0.126** | **0.175** | **0.770** | **0.866** |
| **0.7** | **0.458** | **0.556** | **0.053** | **0.444** | **0.519** | **0.916** |
| 0.8 | 0.534 | 0.433 | 0.030 | 0.567 | 0.401 | 0.928 |
| 0.9 | 0.807 | 0.096 | 0.002 | 0.904 | 0.096 | 0.931 |

"Exact rotation accuracy" = fraction of true rotations where the system both
detected the rotation and chose the correct destination.

## Single weighted multi-class baseline

A single multi-class forest with `class_weight="balanced"`:

| Metric | Value |
|---|---:|
| Accuracy | 0.731 |
| Balanced accuracy | 0.858 |
| Macro F1 | 0.448 |
| HOLD recall | 0.718 |
| ROTATE_A recall | 0.933 |
| ROTATE_B recall | 0.909 |
| ROTATE_MID recall | 0.846 |
| ROTATE_CT recall | 0.886 |

It recovers rotation recall but at the cost of a ~28% false-rotation rate on
HOLD. The two-stage model gives better control over that trade-off and a much
better conditional destination model.

## Interpretation

- The two-stage decomposition works: **destination prediction is solved**
  (92%, top-2 99.4%) once a rotation is detected.
- **Stage 1 is the remaining problem.** ROC-AUC 0.923 shows the signal exists,
  but the operating point must trade false rotations against missed rotations.
- For a coaching tool, threshold 0.6 is a reasonable starting point: catches
  82.5% of rotations, 12.6% false-rotation rate on HOLD, 77% exact rotation
  accuracy. Threshold 0.7 is the conservative option: 55.6% recall, 5.3% false
  rate, 51.9% exact.
- This is now a **policy/operating-point** decision, not a feature-availability
  problem. The GSI + radar/HUD feature set is sufficient; the next gains should
  come from better stage-1 modelling (sequence/velocity, enemy last-seen) and a
  chosen cost trade-off.

## Recommended next steps

1. Pick an operating threshold for the product (0.6 for recall-oriented coaching,
   0.7 for conservative advice).
2. Add temporal features (previous positions/velocity) to stage 1; radar gives
   these naturally.
3. Add enemy last-seen features from radar/demos to stage 1; this is where
   visibility simulation requires raw demos with T positions.
4. Only then invest in CV and shadow mode.
