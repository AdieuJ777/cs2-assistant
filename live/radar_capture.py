"""Capture the CS2 radar and feed teammate/enemy positions to the live server.

Pipeline:

1. read the local player's world position from the server's ``/raw_state``;
2. grab the calibrated radar ROI with ``mss``;
3. ``radar_parser`` detects red enemy dots and teammate dots and converts them
   to world coordinates;
4. POST the result to ``/radar``.

This module reads the screen only. It does not read game memory, inject input,
or modify the game in any way.

Run (server must be running):
    uv run --with mss --with pillow --with numpy --with scipy \
        python live/radar_capture.py --token change-me

Calibrate first with ``live/radar_calibrate.py``. The radar must be centred on
the player and non-rotating (``cl_radar_always_centered 1``,
``cl_radar_rotate 0``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import numpy as np
    import PIL  # noqa: F401
    import scipy  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}\n"
        "Run the capture with its dependencies, for example:\n"
        "  uv run --with mss --with pillow --with numpy --with scipy "
        "python live/radar_capture.py --token change-me\n"
        "Or install them into the current environment:\n"
        "  python -m pip install -r requirements-radar.txt"
    ) from exc

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from radar_parser import DEFAULT_CONFIG, load_config, parse_frame  # noqa: E402


def http_json(url: str, payload: dict | None = None, timeout: float = 1.5):
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--token", default="change-me")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--left", type=int, default=None, help="Override ROI left.")
    parser.add_argument("--top", type=int, default=None, help="Override ROI top.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print frames, do not POST.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Save captured frames here when self-localization fails.",
    )
    args = parser.parse_args()

    try:
        from mss import mss
    except ImportError:
        raise SystemExit(
            "mss is required: uv run --with mss --with pillow --with numpy "
            "--with scipy python live/radar_capture.py"
        )

    base = f"http://{args.host}:{args.port}"

    with mss() as screen:
        monitor = screen.monitors[1]
        config_path = args.config
        if args.config == DEFAULT_CONFIG:
            candidate = (
                PROJECT_ROOT
                / "live"
                / "models"
                / f"radar_config_{monitor['width']}x{monitor['height']}.json"
            )
            if candidate.exists():
                config_path = candidate
        config = load_config(config_path)
        roi = config["radar_roi"]
        size = int(roi.get("size", 340))
        left = roi.get("x0", 0) if args.left is None else args.left
        top = roi.get("y0", 0) if args.top is None else args.top

        print(f"capture ROI {left},{top} {size}x{size}; server {base}")
        print(f"config: {config_path}")
        print("radar must be centred and non-rotating (cl_radar_rotate 0)")

        calibrated = config.get("resolution", [None, None])
        if calibrated[0] and (
            abs(monitor["width"] - calibrated[0]) > 8
            or abs(monitor["height"] - calibrated[1]) > 8
        ):
            print(
                f"WARNING: screen is {monitor['width']}x{monitor['height']} but "
                f"the radar config was calibrated for {calibrated[0]}x{calibrated[1]}.\n"
                "         Re-run radar_calibrate.py for this resolution, or set "
                "the game to the calibrated resolution.\n"
                "         Detection will be wrong until then."
            )
        region = {"left": left, "top": top, "width": size, "height": size}
        last_print = 0.0
        last_self: tuple[float, float] | None = None
        while True:
            try:
                state = http_json(f"{base}/raw_state")
            except (urllib.error.URLError, ConnectionError) as exc:
                print(f"server not reachable: {exc}")
                if args.once:
                    return
                time.sleep(0.5)
                continue

            if not state.get("connected"):
                print("waiting for GSI")
                if args.once:
                    return
                time.sleep(0.5)
                continue
            if "dust2" not in str(state.get("map", "")).lower():
                print(f"map is {state.get('map')}; radar parser is Dust2-only")
                if args.once:
                    return
                time.sleep(1.0)
                continue

            frame = np.asarray(screen.grab(region))[:, :, :3][:, :, ::-1]
            gsi_self = (state["self"]["x"], state["self"]["y"])
            if gsi_self[0] == 0.0 and gsi_self[1] == 0.0:
                gsi_self = None  # CS2 GSI has no live player position
            parsed = parse_frame(
                frame, config, gsi_self, prior_self=last_self
            )
            parsed_self = parsed.get("self")
            if parsed_self is not None:
                last_self = (parsed_self["x"], parsed_self["y"])

            payload = {
                "auth": {"token": args.token},
                "self_area": parsed_self["area"] if parsed_self else None,
                "self_x": parsed_self["x"] if parsed_self else None,
                "self_y": parsed_self["y"] if parsed_self else None,
                "self_source": parsed_self["source"] if parsed_self else None,
                "teammates": parsed["teammates"],
                "enemies": parsed["enemies"],
                "enemy_alive_count": parsed["enemy_alive_count"],
                "self_spots_enemy": parsed["self_spots_enemy"],
                "self_spot_count": 0,
            }

            if parsed_self is None:
                if args.dry_run:
                    print(json.dumps(payload, ensure_ascii=False))
                now = time.time()
                if now - last_print > 5.0:
                    print("could not locate self from radar A/B labels; not posting")
                    if args.debug_dir is not None:
                        try:
                            from PIL import Image

                            args.debug_dir.mkdir(parents=True, exist_ok=True)
                            path = args.debug_dir / f"radar_fail_{int(now)}.png"
                            Image.fromarray(frame).save(path)
                            print(f"saved debug frame: {path}")
                        except OSError as exc:
                            print(f"could not save debug frame: {exc}")
                    last_print = now
            elif args.dry_run:
                print(json.dumps(payload, ensure_ascii=False))
            else:
                try:
                    http_json(f"{base}/radar", payload)
                except (urllib.error.URLError, ConnectionError) as exc:
                    print(f"radar POST failed: {exc}")

            now = time.time()
            if parsed_self is not None and now - last_print > 5.0:
                print(
                    f"self=({parsed_self['x']:.0f},{parsed_self['y']:.0f}) "
                    f"source={parsed_self['source']} "
                    f"enemies={len(parsed['enemies'])} "
                    f"teammates={len(parsed['teammates'])}"
                )
                last_print = now

            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
