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

The tests use synthetic GSI payloads; they do not require CS2.

## Limitations

- CT only; Dust2 only (other maps get a warning but still use the generic rules).
- No teammate economy, teammate positions, enemy positions, enemy economy,
  kill feed or voice comms.
- Enemy prior is low-confidence and can be wrong.
- Round history is in memory; restarting the server resets the loss-streak prior.
- Prices and buy rules are hand-written; they are not learned from the user's
  matches.
- Not an overlay; it is a browser page.

## Next

- Add a **post-plant rotation timer** using the player's position and bomb timer.
- Add radar/HUD perception for teammate state; then the action random forest can
  run live.
- Add an optional grounded explanation from the LLM after the deterministic
  recommendation is fixed.
