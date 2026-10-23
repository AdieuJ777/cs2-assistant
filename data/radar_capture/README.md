# Radar calibration screenshots

Source material for the screen-capture radar parser.

- `ingame_mid.jpg` — live Dust2 round, CT at mid, one red enemy dot and three
  teammate dots visible on the radar. Calibrated for 1280x960.
- `settings.jpg` — video settings used for the 1280 capture: 1280x960, 4:3,
  full screen, 164 Hz.
- `parsed_overlay3.png` — parser output at 1280: red ring = detected enemy,
  green rings = detected teammates.
- `ingame_2560_clean.jpg` — clean 2560x1440 fullscreen-windowed capture used
  for the current desktop-resolution calibration.
- `parsed_2560_overlay.png` — parser output at 2560x1440.

Fullscreen-windowed mode renders at the desktop resolution (2560x1440 here),
not the in-game 1280x960 preset.

The current calibration assumes:

- `cl_radar_always_centered 1` (player at the radar centre);
- `cl_radar_rotate 0` (fixed map orientation);
- the default top-left radar position and HUD scale.

If any of those change, re-run `live/radar_calibrate.py` with a new screenshot.

Note: the screenshots were saved as JPG by the game. PNG is preferred because
JPEG compression shifts the red/teammate colours slightly; the thresholds work
on this image, but PNG is more robust.
