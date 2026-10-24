"""Global hotkey -> ``POST /agent/hotkey``.

The listener only reads a button press; it never injects input into the game.
``pynput`` is optional, so the server and tests keep working without it.

Run:
    python -m live.midround.hotkey --token change-me
    python live/midround/hotkey.py --token change-me --button x2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def request_plan(
    url: str = "http://127.0.0.1:3000",
    token: str = "change-me",
    *,
    timeout: float = 1.0,
) -> dict[str, Any]:
    body = json.dumps({"auth": {"token": token}}).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/agent/hotkey",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Listen for a global hotkey and request one mid-round call."
    )
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--token", default="change-me", help="Same token as the GSI config.")
    parser.add_argument(
        "--button",
        default=None,
        choices=("x1", "x2", "middle"),
        help="Mouse button to listen for (default: x1 when --key is not set).",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Keyboard key instead of a mouse button, e.g. f8. Recommended with "
        "the Steam overlay browser because mouse4/5 navigate back/forward.",
    )
    parser.add_argument("--debounce", type=float, default=0.3)
    args = parser.parse_args(argv)
    if args.key and args.button:
        parser.error("use either --key or --button, not both")

    try:
        if args.key:
            from pynput import keyboard
        else:
            from pynput import mouse
    except ImportError:
        print(
            "pynput is not installed; the global hotkey is disabled.\n"
            "Install it with: python -m pip install pynput\n"
            "Alternatively use the button on the web page.",
            file=sys.stderr,
        )
        return 2

    last_press = [0.0]

    def trigger() -> None:
        now = time.monotonic()
        if now - last_press[0] < args.debounce:
            return
        last_press[0] = now
        try:
            result = request_plan(args.url, args.token)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            return
        plan = result.get("plan") or {}
        call = plan.get("call") or "(no plan yet)"
        print(
            f"{time.strftime('%H:%M:%S')} {call} "
            f"[{plan.get('source', '-')}, {plan.get('cache_age_s', 0)} s]"
        )

    if args.key:
        target_key = _resolve_key(args.key, keyboard)
        if target_key is None:
            print(f"unknown key {args.key!r}; try f8, f9 or a single character.", file=sys.stderr)
            return 2

        def on_press(key: Any) -> None:
            if key == target_key:
                trigger()

        listener_context = keyboard.Listener(on_press=on_press)
        target_label = f"key {args.key}"
    else:
        button_name = args.button or "x1"
        target_button = getattr(mouse.Button, button_name, None)
        if target_button is None:  # pragma: no cover - depends on pynput version
            print(f"pynput has no mouse button {button_name!r}.", file=sys.stderr)
            return 2

        def on_click(x: int, y: int, button: Any, pressed: bool) -> None:
            if pressed and button == target_button:
                trigger()

        listener_context = mouse.Listener(on_click=on_click)
        target_label = f"mouse {button_name}"

    print(
        f"Listening on {target_label} -> {args.url}/agent/hotkey "
        f"(Ctrl+C to stop). No game input is injected."
    )
    try:
        with listener_context as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nStopping.")
    return 0


def _resolve_key(name: str, keyboard: Any) -> Any:
    """Map 'f8' / 'F8' / a single character to a pynput key object."""
    normalized = name.strip().lower()
    special = getattr(keyboard.Key, normalized, None)
    if special is not None:
        return special
    if len(name.strip()) == 1:
        return keyboard.KeyCode.from_char(name.strip())
    return None


if __name__ == "__main__":
    raise SystemExit(main())
