# Live v1: GSI-only buy/save coach (Dust2 CT)

This is the first **live, in-game** prompt. It is deliberately narrow:

- Dust2, CT side, solo queue (v1);
- reads only the official Counter-Strike 2 Game State Integration (GSI) data;
- gives a **freeze-time buy/save recommendation** and a coarse enemy-economy
  prior;
- does **not** use the LLM, the TAGNet model, or any trained model.

It never reads game memory, never reads spectator-only `allplayers`, and never
sends input to the game. GSI is Valve's official local HTTP feed.

## Why this scope

GSI exposes the local player's state plus global match state. For a live player
it does **not** expose teammate money/positions, enemy positions, enemy economy,
the kill feed or voice comms. A buy/save decision only needs the player's own
money/loadout and the round context, so it is the one useful prompt we can ship
with GSI alone.

The benchmark models (economy tree, action random forest) are not used here
because their inputs include team-level economy and teammate positions that GSI
does not provide. Using them live would require a screen/radar perception layer
that is not built yet.

## What you see

A local page at `http://127.0.0.1:3000` with:

- recommendation: `FULL BUY`, `HALF BUY`, `SAVE`, `PISTOL BUY`, etc.;
- a shopping list with estimated prices and the money left;
- round, score, phase and timer;
- enemy prior (`eco or force`, `half/full`, `unknown`) with **low** confidence;
- warnings (no defuse kit, teammate economy unknown, non-Dust2 map, etc.).

The page auto-refreshes every 0.5 s. Put it on a second monitor; it is not an
overlay and does not inject into the game.

## Mid-round rotation advisor (v2, experimental)

The same server also serves a second card: `HOLD` or `ROTATE_A/B/MID/CT` for
the current mid-round state. It is a **two-stage random forest** trained on the
event-driven radar dataset (`benchmarks/enemy_visibility`):

- stage 1: ROTATE vs HOLD, threshold 0.6;
- stage 2: destination conditional on rotating;
- horizon: 5 s.

The model is exported to `live/models/rotation_policy.json` and evaluated by a
pure-standard-library implementation (`live/rotation_policy.py`), so the live
server does not need scikit-learn.

**It refuses to guess without radar.** GSI alone does not expose teammates or
enemy positions, so the rotation card stays at `NO_RADAR` until a local
perception module posts data to `POST /radar`. This is deliberate: the policy
was trained with radar features and is out of distribution without them.

### Radar feed contract

`POST http://127.0.0.1:3000/radar` (same token as the GSI config):

```json
{
  "auth": {"token": "change-me"},
  "self_area": "B",
  "teammates": [
    {"x": -500, "y": 200, "health": 90, "area": "B"},
    {"x": 900, "y": 1800, "health": 100, "area": "A"}
  ],
  "enemies": [
    {"x": 1200, "y": 1600, "health": 100, "area": "A", "visible": true}
  ],
  "enemy_alive_count": 3,
  "self_spots_enemy": true,
  "self_spot_count": 1
}
```

- `teammates` are the other alive CTs (self is added from GSI). Missing
  `teammates` makes the server return `NO_TEAMMATE_DATA`.
- `enemies` are radar-visible enemies only. `enemy_alive_count` is the number of
  living T players (visible + unknown).
- Radar older than 1.5 s is ignored (`NO_RADAR`).
- Positions are Awpy/GSI world units; `area` is one of `A/B/Mid/CT` or omitted
  (then the nearest area centroid is used).

### Screen radar capture (implemented)

`live/radar_capture.py` reads the radar from the screen and feeds `/radar`
automatically. It is a screen reader only: no memory access, no input injection.

Requirements:

- radar at the default top-left position and HUD scale;
- `cl_radar_rotate 0` (fixed orientation);
- `cl_radar_always_centered 1` (player at the radar centre).

Setup for 1280x960 / 4:3 (the captured resolution):

```bash
# optional but recommended: rebuild the world -> area lookup from demos
uv run --with numpy --with pandas --with pyarrow --with scipy \
  python live/build_area_grid.py

# recalibrate when resolution or HUD scale changes
uv run --with pillow --with numpy --with scipy \
  python live/radar_calibrate.py \
  --screenshot data/radar_capture/ingame_mid.jpg \
  --output live/models/radar_config_1280x960.json

# terminal 1: server
python live/gsi_server.py --token change-me --verbose

# terminal 2: radar capture
uv run --with mss --with pillow --with numpy --with scipy \
  python live/radar_capture.py --token change-me
```

Important:

- The token must match in **three** places: the installed GSI cfg, the server
  `--token`, and the capture `--token`. A mismatch returns `403 Forbidden`.
- **GSI does not expose the local player's position during normal play**; it is
  only available while spectating. The parser therefore recovers the player's
  position from the two A/B bomb-site labels on the radar every frame. The
  `"player_position" "1"` cfg section is harmless but is not required for the
  rotation card.
- If the game resolution differs from the calibration resolution, the capture
  prints a warning and detection will be wrong; either set the game to the
  calibrated resolution or recalibrate with a clean screenshot.
- Restart CS2 after changing the GSI cfg; the cfg is only read at launch.

What it reads:

- red enemy dots -> world position and macro area;
- coloured teammate dots -> world position and macro area;
- the local player position from the radar A/B labels (or GSI when spectating),
  so the map translation is updated every frame;
- if either A/B label is missing, the card returns `NO_POSITION` instead of
  guessing.

What it does **not** read yet:

- teammate health (radar does not show it; defaults to 100 with a warning);
- enemy alive count (defaults to 5 with a warning);
- whether the local player personally sees an enemy (`self_spots_enemy` false).

The parser was validated on `data/radar_capture/ingame_mid.jpg`: one enemy on
the T side and three teammates near mid, with `parsed_overlay3.png` showing the
detected rings. See `requirements-radar.txt` for the optional dependencies.

### Mid-round agent (M2: template plan)

`live/midround/` turns GSI + radar into one short, executable call and caches
it. M2 has no LLM yet: the kernel's first legal candidate becomes the template
plan, so the hotkey path stays available while the LLM work is built.

Endpoints:

- `GET /plan` — cached plan, age, `source`, `fresh`/`stale`, plus candidates;
- `POST /agent/hotkey` — return the latest plan immediately (never waits);
- `POST /agent/refresh` — debug: rebuild the template plan now;
- `GET /agent/status` — cache hits, plan age, candidate count, last error.

The web page polls `/plan` every 250 ms, shows the call in large text, turns it
gray after 3 s, and includes a "按一下获取指令" button. Optional global hotkey:

```bash
python -m pip install pynput
python -m live.midround.hotkey --token my-secret --button x1
```

It only listens for the mouse side button; it never injects input into the
game. Plans are appended to the same shadow log with `record_type: "plan"`.
The plan cache is invalidated whenever the state fingerprint changes; a plan
older than 2 s is marked stale and one older than 4 s is rebuilt.

### LLM selection (M3)

The background slow loop asks an LLM to pick one candidate and rephrase it.
Ollama is the default; any OpenAI-compatible endpoint can be used instead:

```bash
# local Ollama (default)
python live/gsi_server.py --token my-secret \
  --llm-backend ollama --llm-model qwen3:8b

# LM Studio / vLLM / llama.cpp / OpenAI-compatible
python live/gsi_server.py --token my-secret \
  --llm-backend openai --llm-url http://127.0.0.1:1234/v1 \
  --llm-model your-model --llm-api-key optional

# template only
python live/gsi_server.py --token my-secret --llm-backend none
```

The page also has a **模型接入** panel: choose the backend, model and base URL,
optionally enter an API key (memory only, never written to disk), and apply it
at runtime. `GET /agent/config` exposes the same information for other tools.
No extra Python package is needed: both backends use `urllib` directly.

Safety and failure behaviour:

- the LLM only selects from the existing candidates and copies grounded facts;
- the validator rejects unknown ids, calls >20 chars, invented facts, unknown
  areas and phrases such as "敌人一定在"/"肯定"/"百分之";
- timeout 5 s (`--llm-timeout`), one retry on invalid JSON, no retry on
  connection failure;
- any failure leaves the kernel template plan in place;
- a result that arrives after the game state changed is discarded;
- the hotkey path still only reads the cache and never waits for the model.

Plan records carry `source: "llm"` or `"template"` and `llm_latency_s`.

### Single-screen use

The page is not an overlay. With one monitor there are three options:

- **Always-on-top ticker** (recommended, no browser needed):

  ```bash
  python -m live.midround.ticker --token change-me
  # optional: --font-size 32 --opacity 0.85
  ```

  It reads `/state` and `/plan` every 250 ms and follows the round:

  - freeze time: buy/save recommendation and estimated cost;
  - live round: the mid-round call, source and freshness;
  - left-click: expand round, score, radar, rotation, facts and LLM status;
  - right-click: **模型设置**, **立即刷新**, **退出**;
  - flashes for 1.4 s when a hotkey is pressed.

  The settings window can switch backend/model/base URL, set the timeout and
  enter an API key at runtime; `--token` must match the server. The default
  position is below the top-left radar (do not drag it over the radar). It
  works over CS2 in fullscreen-windowed/borderless mode and never injects
  input.

- **Steam overlay browser**: Shift+Tab, open `http://127.0.0.1:3000`, tick
  **朗读**. Mouse4/5 navigate the overlay browser forward/back, so use a
  keyboard hotkey there:

  ```bash
  python -m live.midround.hotkey --key f8
  ```

  Keep the overlay **closed** while playing: the overlay dims the screen and
  `radar_capture.py` then detects no dots or A/B labels, so the agent reports
  `NO_POSITION`. Use the ticker for in-game display and open the overlay only
  between rounds.

- **Phone on the same LAN**: run the server with `--host 0.0.0.0` and open
  `http://<pc-ip>:3000` on the phone (local network only; do not forward the
  port to the internet).

The page watches `hotkey_seq` in `/plan`, so a press from the separate hotkey
process makes a background tab speak the new call even when the game has focus.
Background-tab polling may be throttled to about 1 s by the browser; the call
itself is already cached, so audio latency stays low.

### Other LLM providers

Any OpenAI-compatible API works, for example DeepSeek:

```powershell
$env:CS2_LLM_API_KEY = "sk-..."
python live/gsi_server.py --token my-secret --llm-backend openai `
  --llm-url https://api.deepseek.com --llm-model your-model-id
```

`--llm-api-key` defaults to `CS2_LLM_API_KEY`, so the key does not have to
appear in the command line. The same values can be applied at runtime in the
**模型接入** panel; a key entered there is memory-only.

### Demo replay (end-to-end harness)

To exercise the full chain without a live radar client, replay a parsed demo
round into the server:

```bash
# terminal 1
python live/gsi_server.py --token test --verbose

# terminal 2
uv run --with numpy --with pandas --with pyarrow \
  python live/demo_radar_feeder.py \
  --match vitality-vs-furia-m2-dust2 --round 14 \
  --player 76561197973140692 --speed 20 --interval 0.05 --token test
```

The feeder synthesises GSI and radar payloads from Awpy ticks at the requested
speed and uses the same FOV/LOS/smoke proxy as the benchmark.

### Shadow log and outcome evaluation

Every rotation call is appended to `live/shadow_log.jsonl` (disable with
`--shadow-log none`). The mid-round agent uses the same file. Record types:

- `advice` / `outcome` — the v2 rotation call and what the player did
  `--outcome-horizon` seconds later (default 5 s);
- `plan` — a mid-round plan (template or LLM), with the full state card and
  candidates;
- `plan_llm` — one LLM attempt, including timeouts, invalid JSON and
  validation failures;
- `plan_outcome` — what the player actually did 3 s and 5 s after pressing the
  hotkey (`adopted`, `actual_action`, `actual_area`, `status`; statuses are
  `ok`, `died`, `round_changed`, `not_live`, `no_position`).

Adoption means the player's area at the horizon equals the plan's target area
(so holding and rotating are both measurable). Plans made without a usable
position/target are logged as `no_position` and excluded from the adoption
denominator, because there is nothing to adopt. Duplicate hotkey presses for
the same plan do not create duplicate outcome samples.

The outcome uses the same majority-vote area grid as the benchmark, so
`HOLD`/`ROTATE_*` mean the same thing offline and live.

Review a session with:

```bash
python live/analyze_shadow.py live/shadow_log.jsonl
```

It reports advice coverage, action/confidence distributions, exact match rate,
and binary rotate precision/recall/false/missed rates. For the mid-round agent
it also reports plan source distribution, call lengths, LLM latency and
attempt/validation failures, plus adoption rate at 3 s / 5 s split by
`llm` vs `template`. The log is the intended input for the "prompt vs no
prompt" decision; it does not change the game and does not show a prompt
automatically.

Note: advice is deduplicated per round/action every 2 s, so a long hold produces
several overlapping 5 s outcomes. Treat records as samples, not independent
events.

## Setup

1. Find the CS2 cfg directory. It is usually one of:
   - `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\cs2\cfg\`
   - `<your Steam library>\steamapps\common\Counter-Strike Global Offensive\game\cs2\cfg\`
2. Copy `live/gsi_config/gamestate_integration_cs2assistant.cfg` there.
3. Change the token in **both** the cfg and the server command, e.g. `my-secret`.
4. Start the server **before** launching CS2:

   ```bash
   cd cs2-assistant
   python live/gsi_server.py --token my-secret --verbose
   ```

5. Open `http://127.0.0.1:3000`.
6. Launch CS2, join a Dust2 game on CT. GSI is picked up when CS2 starts.

If the page says "Waiting for GSI", check that:

- the server is running on port 3000;
- the cfg is in the right directory and the token matches;
- CS2 was restarted after adding the cfg;
- the game is on a server that allows GSI (official matchmaking does).

## Logic

- **Pistol rounds** (round 1/13): armor + helmet, defuse kit, then utility; no
  rifle.
- **Full buy** when the core cost fits: M4A1-S (if missing), armor/helmet,
  defuse kit, smoke, flashes, HE.
- **Half buy** when the full buy does not fit but a partial buy is affordable:
  armor, MP9 (if no primary), defuse kit, light utility.
- **Save** when even a half buy leaves too little money.
- **Enemy prior** is a coarse loss-streak heuristic, not a prediction. It is
  always labeled low confidence except on pistol rounds.

Prices are approximate and live in `live/advisor.py` (`PRICES`). Edit them when
the economy changes.

## Tests

```bash
cd cs2-assistant
python -m unittest discover -s live/tests -t .
```

The tests use synthetic GSI payloads; they do not require CS2. The rotation
tests also check that the stdlib tree traversal matches scikit-learn exactly
(`test_tree_traversal_matches_sklearn`). Current suite: 157 tests, including
the mid-round state card, route graph, candidates, plan cache, agent, validator,
LLM planner/backends and HTTP routes; the radar parser tests and the screenshot
integration test need Pillow/numpy/scipy and skip in the stdlib-only run. LLM
tests use a fake backend, so they do not need Ollama.

## Limitations

- CT only; Dust2 only.
- **Radar capture is calibrated for two layouts**: 1280x960 and 2560x1440,
  default top-left radar, `cl_radar_always_centered 1`. The capture
  auto-selects a config by screen size. Fullscreen-windowed mode uses the
  desktop resolution, not the in-game resolution. The A/B-label transform
  handles a rotating radar, but a different HUD scale/resolution still needs a
  new calibration.
- Teammate health and enemy alive count are not read from the radar yet; they
  use conservative defaults and raise warnings.
- Without a fresh `/radar` feed the rotation card stays at `NO_RADAR`; this is
  a hard safety rule, not a bug.
- The rotation policy is trained on 9 professional demos and a 2D geometry
  proxy; it has not been validated in live ranked play.
- At the 0.6 operating point the 5 s policy has ~0.54 rotation precision,
  ~0.66 recall and ~16% false-rotation rate on out-of-fold event rows. That is
  research-grade, not a finished product.
- No teammate economy, enemy economy, kill feed or voice comms.
- Enemy prior is low-confidence and can be wrong.
- Round history is in memory; restarting the server resets the loss-streak prior.
- Prices and buy rules are hand-written; they are not learned from the user's
  matches.
- Not an overlay; it is a browser page.

## Next

- Run a full shadow session and review `shadow_log.jsonl` (advice + 5 s
  outcomes) before showing any prompt in-game.
- Support `cl_radar_rotate 1` (yaw-aware transform) or keep documenting the
  fixed-radar requirement.
- Parse teammate health and alive counts from the HUD/top bar.
- Add post-plant timing and bomb-site context to the rotation state.
- Add an optional grounded explanation from the LLM after the deterministic
  recommendation is fixed.
