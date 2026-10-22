"""Local GSI listener and buy/save advisor for Dust2 CT.

Run:
    python live/gsi_server.py --token change-me

Then open http://127.0.0.1:3000 in a browser (a second monitor is ideal).

Counter-Strike 2 pushes JSON to http://127.0.0.1:3000. The server keeps only
the latest state, logs round results for the loss-streak prior, and serves a
small auto-refreshing page. It never reads game memory and never sends input
to the game.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

try:  # package import (tests, `python -m live.gsi_server`)
    from .advisor import RoundHistory, advise
except ImportError:  # script import (`python live/gsi_server.py`)
    from advisor import RoundHistory, advise  # type: ignore


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CS2 Buy Coach</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; padding: 24px; background: #0f1115; color: #e9edf5;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
  }
  .wrap { max-width: 760px; margin: 0 auto; }
  .muted { color: #8b95a7; }
  .card {
    background: #171b22; border: 1px solid #2a3140; border-radius: 14px;
    padding: 20px 22px; margin: 16px 0;
  }
  .rec { font-size: 42px; font-weight: 800; letter-spacing: .5px; }
  .rec.full { color: #58d68d; }
  .rec.half { color: #f5c542; }
  .rec.save { color: #ff6b6b; }
  .rec.pistol { color: #6cb6ff; }
  .rec.over { color: #9aa4b2; }
  .row { display: flex; justify-content: space-between; gap: 16px; padding: 7px 0; border-bottom: 1px solid #232a36; }
  .row:last-child { border-bottom: 0; }
  .label { color: #8b95a7; }
  ul { margin: 8px 0 0 20px; padding: 0; }
  li { margin: 4px 0; }
  .warn { color: #f5c542; }
  .pill { display: inline-block; padding: 3px 9px; border-radius: 999px; background: #222a36; color: #b9c3d3; font-size: 13px; }
  .footer { color: #687386; font-size: 13px; margin-top: 18px; line-height: 1.5; }
</style>
</head>
<body>
<div class="wrap">
  <div class="muted" id="conn">Waiting for GSI...</div>
  <div class="card">
    <div class="rec over" id="rec">WAITING</div>
    <div class="muted" id="reason">Start CS2 with the GSI config and join a Dust2 CT game.</div>
  </div>
  <div class="card" id="details"></div>
  <div class="footer">
    GSI-only v1: own state + match state. Teammate/enemy positions and economy are unknown.
  </div>
</div>
<script>
const recEl = document.getElementById('rec');
const reasonEl = document.getElementById('reason');
const connEl = document.getElementById('conn');
const detailsEl = document.getElementById('details');

function cls(rec) {
  if (!rec) return 'over';
  if (rec.startsWith('FULL')) return 'full';
  if (rec.startsWith('HALF') || rec.startsWith('FORCE')) return 'half';
  if (rec.startsWith('SAVE')) return 'save';
  if (rec.startsWith('PISTOL')) return 'pistol';
  return 'over';
}

function row(label, value) {
  const div = document.createElement('div');
  div.className = 'row';
  const l = document.createElement('span'); l.className = 'label'; l.textContent = label;
  const v = document.createElement('span'); v.textContent = value;
  div.appendChild(l); div.appendChild(v);
  return div;
}

function render(s) {
  if (!s.connected || !s.advice) {
    connEl.textContent = 'Waiting for GSI...';
    recEl.textContent = 'WAITING';
    recEl.className = 'rec over';
    reasonEl.textContent = 'Start CS2 with the GSI config and join a Dust2 CT game.';
    detailsEl.replaceChildren();
    return;
  }
  const a = s.advice;
  connEl.textContent = 'GSI connected · ' + (s.age_s ?? 0).toFixed(1) + 's ago';
  recEl.textContent = (a.recommendation || 'UNKNOWN').replaceAll('_', ' ');
  recEl.className = 'rec ' + cls(a.recommendation);
  reasonEl.textContent = a.reason || '';

  const box = document.createDocumentFragment();
  box.appendChild(row('Round', (a.round ?? '?') + '  ·  CT ' + (a.score?.ct ?? '?') + ' : ' + (a.score?.t ?? '?') + ' T'));
  box.appendChild(row('Phase', (a.phase || '?') + '  ·  ' + (a.seconds_left ?? '?') + 's'));
  box.appendChild(row('Money', '$' + (a.money ?? '?')));
  box.appendChild(row('Reserve after buy', '$' + (a.reserve ?? '?')));
  box.appendChild(row('Estimated spend', '$' + (a.estimated_cost ?? '?')));
  box.appendChild(row('Enemy prior', (a.enemy_prior?.label || 'unknown') + '  (' + (a.enemy_prior?.confidence || 'low') + ')'));
  if (a.enemy_prior?.reason) box.appendChild(row('Prior reason', a.enemy_prior.reason));

  if (a.shopping_list && a.shopping_list.length) {
    const title = document.createElement('div');
    title.className = 'row';
    title.innerHTML = '<span class="label">Buy list</span><span></span>';
    box.appendChild(title);
    const ul = document.createElement('ul');
    for (const item of a.shopping_list) {
      const li = document.createElement('li');
      li.textContent = item.name + '  ($' + item.cost + ')' + (item.optional ? '  · optional' : '');
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }

  if (a.warnings && a.warnings.length) {
    const ul = document.createElement('ul');
    for (const w of a.warnings) {
      const li = document.createElement('li');
      li.className = 'warn';
      li.textContent = w;
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
  detailsEl.replaceChildren(box);
}

async function tick() {
  try {
    const res = await fetch('/state', { cache: 'no-store' });
    render(await res.json());
  } catch (e) {
    connEl.textContent = 'Server not reachable';
  }
}
tick();
setInterval(tick, 500);
</script>
</body>
</html>
"""


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.payload: dict[str, Any] | None = None
        self.advice: dict[str, Any] | None = None
        self.history = RoundHistory()
        self.updated_at = 0.0

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.history.update(payload)
            advice = advise(payload, self.history)
            self.payload = payload
            self.advice = advice
            self.updated_at = time.time()
            return advice

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            age = time.time() - self.updated_at if self.updated_at else None
            connected = bool(self.payload) and age is not None and age < 10.0
            return {
                "connected": connected,
                "age_s": age,
                "advice": self.advice if connected else None,
            }


class Handler(BaseHTTPRequestHandler):
    server_version = "CS2BuyCoach/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, code: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return

        token = getattr(self.server, "token", None)
        if token:
            supplied = str((payload.get("auth") or {}).get("token", ""))
            if supplied != token:
                self._send_json(403, {"error": "bad token"})
                return

        advice = self.server.state.update(payload)
        if getattr(self.server, "verbose", False):
            print(
                f"{time.strftime('%H:%M:%S')} round={advice.get('round')} "
                f"phase={advice.get('phase')} money={advice.get('money')} "
                f"rec={advice.get('recommendation')}"
            )
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(200, PAGE)
            return
        if path == "/state":
            self._send_json(200, self.server.state.snapshot())
            return
        self._send_json(404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser(description="GSI listener and Dust2 CT buy coach.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--token", default="change-me", help="Must match the GSI config token.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = AppState()  # type: ignore[attr-defined]
    server.token = args.token  # type: ignore[attr-defined]
    server.verbose = args.verbose  # type: ignore[attr-defined]

    print(f"GSI listener: http://{args.host}:{args.port}")
    print(f"Open http://{args.host}:{args.port} in a browser.")
    print("GSI config token must match:", args.token)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
