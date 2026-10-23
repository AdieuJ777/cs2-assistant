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
import math
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # package import (tests, `python -m live.gsi_server`)
    from .advisor import RoundHistory, advise
    from .area_lookup import AreaLookup
    from .rotation_policy import RotationAdvisor
except ImportError:  # script import (`python live/gsi_server.py`)
    from advisor import RoundHistory, advise  # type: ignore
    from area_lookup import AreaLookup  # type: ignore
    from rotation_policy import RotationAdvisor  # type: ignore

RADAR_TTL_SECONDS = 1.5
ROUND_SECONDS = 115.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "live" / "models" / "rotation_policy.json"
DEFAULT_SHADOW_LOG = PROJECT_ROOT / "live" / "shadow_log.jsonl"
DEFAULT_AREA_GRID = PROJECT_ROOT / "live" / "models" / "area_grid.json"
DEFAULT_PLACES = PROJECT_ROOT / "live" / "models" / "place_centroids.json"
OUTCOME_ACTIONS = {"HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT"}


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CS2 决策助手</title>
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
  .rec.rotate { color: #f5c542; }
  .rec.hold { color: #58d68d; }
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
  <div class="muted" id="conn">等待 GSI 数据...</div>
  <div class="card">
    <div class="rec over" id="rec">等待中</div>
    <div class="muted" id="reason">启动 CS2 并进入 Dust2 CT 对局。</div>
  </div>
  <div class="card">
    <div class="rec over" id="rot">暂无中期建议</div>
    <div class="muted" id="rotreason">转点建议需要本地雷达数据。</div>
  </div>
  <div class="card" id="details"></div>
  <div class="card" id="rotdetails"></div>
  <div class="footer">
    买/保建议只使用官方 GSI。中期转点建议使用 GSI + 本地雷达数据
    （<code>POST /radar</code>）；没有新鲜雷达数据时保持沉默。
  </div>
</div>
<script>
const recEl = document.getElementById('rec');
const reasonEl = document.getElementById('reason');
const connEl = document.getElementById('conn');
const detailsEl = document.getElementById('details');
const rotEl = document.getElementById('rot');
const rotReasonEl = document.getElementById('rotreason');
const rotDetailsEl = document.getElementById('rotdetails');

const REC_TEXT = {
  FULL_BUY: '全买', HALF_BUY: '半买', SAVE: '存钱',
  PISTOL_BUY: '手枪局购买', PISTOL_SAVE: '手枪局存钱',
  BUY_PHASE_OVER: '购买阶段结束', NOT_SUPPORTED: '不支持',
  BUY_PHASE: '购买阶段', HOLD: '保持站位',
  ROTATE_A: '转 A', ROTATE_B: '转 B', ROTATE_MID: '转中路', ROTATE_CT: '回 CT',
  NO_RADAR: '无雷达数据', NO_POSITION: '无法定位',
  NO_TEAMMATE_DATA: '缺少队友数据', MONITORING: '观察中', NONE: '无',
};
const ITEM_TEXT = {
  'Kevlar + Helmet': '护甲 + 头盔', Kevlar: '护甲', Helmet: '头盔',
  'Defuse Kit': '拆弹器', Smoke: '烟雾弹', Flash: '闪光弹', HE: '高爆手雷',
  'M4A1-S': 'M4A1-S', MP9: 'MP9',
};
const REASON_TEXT = {
  'Pistol round: both teams are on pistols.': '手枪局：双方都在手枪局。',
  'No reliable enemy economy signal from GSI alone.': '仅凭 GSI 无法可靠判断敌方经济。',
  'This v1 only supports the CT side.': '当前版本只支持 CT 方。',
  'The buy phase is over. Play the current loadout.': '购买阶段已结束，使用当前装备。',
  'Pistol round: buy armor, a defuse kit and utility; do not buy a rifle.':
    '手枪局：购买护甲、拆弹器和道具，不要买步枪。',
  'Pistol round, but you cannot afford the core pistol buy; keep the money.':
    '手枪局，但买不起核心装备；保留金钱。',
  'You can afford the core full buy; the list below is what fits your budget.':
    '可以完成核心全买；下面是预算内的购买清单。',
  'You cannot afford a full rifle buy, but a half buy keeps you competitive. Team economy is unknown; if your team is saving, save with them.':
    '买不起全枪全弹，但半买能保持竞争力。队友经济未知；如果队友在存钱，就跟他们一起存。',
  'A full buy is out of reach and a half buy would leave you with little money. Save and keep the loss bonus for the next round.':
    '无法全买，半买后也会所剩无几。存钱，把败者奖金留到下一回合。',
  'Bomb is planted; hold or retake with the current loadout.':
    '炸弹已下包；用当前装备架点或回防。',
  'You already have a rifle; the estimate only covers armor/utility.':
    '你已经有步枪；估算只包含护甲/道具。',
  'No utility in this buy; consider a smoke/flash if your team needs it.':
    '本次购买没有道具；如果队友需要，考虑烟雾/闪光。',
  'No defuse kit in this buy.': '本次购买没有拆弹器。',
  'Teammate economy is unknown; coordinate the save/buy call.':
    '队友经济未知；协调存钱/购买。',
  'Teammate economy is unknown; confirm the team save.':
    '队友经济未知；确认队伍是否存钱。',
  'If you must force, a defuse kit is often worth more than a pistol upgrade.':
    '如果必须强起，拆弹器通常比升级手枪更值。',
  'Teammate health is not available from the radar; using 100.':
    '雷达不提供队友血量；按 100 计算。',
  'Enemy alive count is not available from the radar; assuming 5 alive.':
    '雷达不提供敌人存活数；按 5 人计算。',
  'No visible enemy in the radar feed; not guessing from CT state.':
    '雷达没有可见敌人；不会仅凭 CT 状态猜测。',
  'Radar data is stale (>1.5 s); treat the call as degraded.':
    '雷达数据超过 1.5 秒；建议可信度下降。',
  'Teammate positions are missing; CT counts would be out of distribution, so no rotation call is made.':
    '缺少队友位置；CT 数量会偏离训练分布，因此不给出转点建议。',
  'Teammate radar data is required for a safe rotation call.':
    '安全转点建议需要队友雷达数据。',
  'No live round in progress.': '当前没有进行中的回合。',
  'Freeze time: use the buy/save advice.': '冻结期：请看买/保建议。',
  'Rotation policy is Dust2-only.': '转点策略仅支持 Dust2。',
  'Rotation policy is CT-only.': '转点策略仅支持 CT 方。',
  'Could not locate the player from the radar A/B labels; rotation advice is withheld.':
    '无法从雷达 A/B 标签定位玩家；暂不给出转点建议。',
  'self=(0,0): radar cannot be mapped to the world.':
    'self=(0,0)：雷达无法映射到世界坐标。',
  'GSI has no live player position (official limitation) and the radar A/B labels were not found. Hold still until both A and B labels are visible on the radar.':
    'GSI 实战不提供玩家坐标，且雷达上的 A/B 标签未找到。请站住，等 A、B 标签都出现在雷达上。',
  'self position unavailable: GSI omits it during live play and the radar label anchors were not detected.':
    '无法定位：GSI 实战不提供坐标，雷达标签也未识别到。',
};

function recText(code) {
  return REC_TEXT[code] || String(code || '未知').replaceAll('_', ' ');
}
function itemText(name) { return ITEM_TEXT[name] || name; }
function confidenceText(value) {
  return { high: '高', medium: '中', low: '低', none: '无' }[value] || value || '无';
}
function postureText(value) {
  return { HOLD_ANGLE: '架点', ROTATE: '转点', NONE: '无' }[value] || value || '无';
}
function phaseText(value) {
  return { freezetime: '冻结期', live: '进行中', over: '回合结束' }[value] || value || '?';
}
function areaText(value) {
  return { Mid: '中路', other: '其他' }[value] || value;
}
function destinationText(value) {
  if (!value) return '';
  return areaText(String(value).replace('ROTATE_', ''));
}
function priorText(value) {
  return {
    'eco or force': '经济/强起', 'half/full': '半买/全买',
    unknown: '未知', pistol: '手枪局',
  }[value] || value || '未知';
}
function rotationReason(r) {
  const f = r.features;
  if (f && f.enemy_nearest_area) {
    const area = areaText(f.enemy_nearest_area);
    const distance = Math.round(Number(f.enemy_nearest_distance || 0));
    const p = Number(r.stage1_proba || 0).toFixed(2);
    if (String(r.action || '').startsWith('ROTATE')) {
      return `发现${area}敌人（约 ${distance} 单位）；转点概率 ${p}。`;
    }
    return `暂不转点（转点概率 ${p}）；最近敌人在${area}。`;
  }
  return translateText(r.reason || '');
}

function translateText(text) {
  if (!text) return '';
  if (REASON_TEXT[text]) return REASON_TEXT[text];
  let match = text.match(/^T has lost (\\d+) rounds in a row;.*$/);
  if (match) return `T 已连败 ${match[1]} 回合；败者奖金较高，但存钱/强起行为未知。`;
  if (text.startsWith('T won the previous round;')) {
    return 'T 上回合获胜，可能买得更强，但具体经济未知。';
  }
  match = text.match(/^Map is (.+); this v1 is tuned for Dust2\\.$/);
  if (match) return `地图是 ${match[1]}；本版本针对 Dust2。`;
  return text;
}

function cls(rec) {
  if (!rec) return 'over';
  if (rec.startsWith('FULL')) return 'full';
  if (rec.startsWith('HALF') || rec.startsWith('FORCE')) return 'half';
  if (rec.startsWith('SAVE')) return 'save';
  if (rec.startsWith('PISTOL')) return 'pistol';
  return 'over';
}

function rotCls(action) {
  if (!action) return 'over';
  if (action.startsWith('ROTATE')) return 'rotate';
  if (action === 'HOLD') return 'hold';
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
    connEl.textContent = '等待 GSI 数据...';
    recEl.textContent = '等待中';
    recEl.className = 'rec over';
    reasonEl.textContent = '启动 CS2 并进入 Dust2 CT 对局。';
    detailsEl.replaceChildren();
    renderRotation(s.rotation || null, s.radar_age_s);
    return;
  }
  const a = s.advice;
  connEl.textContent = 'GSI 已连接 · ' + (s.age_s ?? 0).toFixed(1) + ' 秒前';
  recEl.textContent = recText(a.recommendation || 'UNKNOWN');
  recEl.className = 'rec ' + cls(a.recommendation);
  reasonEl.textContent = translateText(a.reason || '');

  const box = document.createDocumentFragment();
  box.appendChild(row('回合', (a.round ?? '?') + '  ·  CT ' + (a.score?.ct ?? '?') + ' : ' + (a.score?.t ?? '?') + ' T'));
  box.appendChild(row('阶段', phaseText(a.phase || '?') + '  ·  ' + (a.seconds_left ?? '?') + ' 秒'));
  box.appendChild(row('金钱', '$' + (a.money ?? '?')));
  box.appendChild(row('购买后余额', '$' + (a.reserve ?? '?')));
  box.appendChild(row('预计花费', '$' + (a.estimated_cost ?? '?')));
  box.appendChild(row('敌方经济先验', priorText(a.enemy_prior?.label || 'unknown') + '（' + confidenceText(a.enemy_prior?.confidence || 'low') + '置信度）'));
  if (a.enemy_prior?.reason) box.appendChild(row('先验依据', translateText(a.enemy_prior.reason)));

  if (a.shopping_list && a.shopping_list.length) {
    const title = document.createElement('div');
    title.className = 'row';
    title.innerHTML = '<span class="label">购买清单</span><span></span>';
    box.appendChild(title);
    const ul = document.createElement('ul');
    for (const item of a.shopping_list) {
      const li = document.createElement('li');
      li.textContent = itemText(item.name) + '  ($' + item.cost + ')' + (item.optional ? '  · 可选' : '');
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }

  if (a.warnings && a.warnings.length) {
    const ul = document.createElement('ul');
    for (const w of a.warnings) {
      const li = document.createElement('li');
      li.className = 'warn';
      li.textContent = translateText(w);
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
  detailsEl.replaceChildren(box);
  renderRotation(s.rotation, s.radar_age_s);
}

function renderRotation(r, radarAge) {
  if (!r) {
    rotEl.textContent = '暂无中期建议';
    rotEl.className = 'rec over';
    rotReasonEl.textContent = '未加载转点模型。';
    rotDetailsEl.replaceChildren();
    return;
  }
  rotEl.textContent = recText(r.action || 'NONE');
  rotEl.className = 'rec ' + rotCls(r.action);
  rotReasonEl.textContent = rotationReason(r);

  const box = document.createDocumentFragment();
  if (r.stage1_proba !== null && r.stage1_proba !== undefined) {
    box.appendChild(row('转点概率', Number(r.stage1_proba).toFixed(2)));
  }
  if (r.destination) {
    box.appendChild(row('目的地', destinationText(r.destination)));
    box.appendChild(row('目的地置信度', Number(r.destination_proba ?? 0).toFixed(2)));
  }
  box.appendChild(row('置信度', confidenceText(r.confidence)));
  box.appendChild(row('姿态', postureText(r.posture)));
  box.appendChild(row('雷达延迟', radarAge === null || radarAge === undefined ? '无数据' : Number(radarAge).toFixed(1) + ' 秒'));
  if (r.warnings && r.warnings.length) {
    const ul = document.createElement('ul');
    for (const w of r.warnings) {
      const li = document.createElement('li');
      li.className = 'warn';
      li.textContent = translateText(w);
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
  rotDetailsEl.replaceChildren(box);
}

async function tick() {
  try {
    const res = await fetch('/state', { cache: 'no-store' });
    render(await res.json());
  } catch (e) {
    connEl.textContent = '服务器不可达';
  }
}
tick();
setInterval(tick, 500);
</script>
</body>
</html>
"""


class AppState:
    def __init__(
        self,
        rotation_advisor: RotationAdvisor | None = None,
        shadow_log: Path | None = None,
        area_lookup: AreaLookup | None = None,
        outcome_horizon: float = 5.0,
    ) -> None:
        self.lock = threading.Lock()
        self.payload: dict[str, Any] | None = None
        self.advice: dict[str, Any] | None = None
        self.history = RoundHistory()
        self.updated_at = 0.0
        self.rotation_advisor = rotation_advisor
        self.shadow_log = shadow_log
        self.area_lookup = area_lookup or AreaLookup.load(
            DEFAULT_AREA_GRID, DEFAULT_PLACES
        )
        self.outcome_horizon = outcome_horizon
        self.pending_outcomes: list[dict[str, Any]] = []
        self.radar: dict[str, Any] | None = None
        self.radar_updated_at = 0.0
        self.last_visible_count = 0
        self.last_event_at = 0.0
        self.last_log_key: tuple[Any, ...] | None = None
        self.last_log_at = 0.0
        self.round_key: tuple[str, int] | None = None
        self.round_live_started_at = 0.0

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.history.update(payload)
            advice = advise(payload, self.history)
            self.payload = payload
            self.advice = advice
            self.updated_at = time.time()
            self._rotation_locked()
            return advice

    def update_radar(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.radar = payload
            self.radar_updated_at = time.time()
            self._rotation_locked()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            age = time.time() - self.updated_at if self.updated_at else None
            connected = bool(self.payload) and age is not None and age < 10.0
            rotation = self._rotation_locked() if connected else None
            radar_age = (
                time.time() - self.radar_updated_at
                if self.radar_updated_at
                else None
            )
            return {
                "connected": connected,
                "age_s": age,
                "advice": self.advice if connected else None,
                "rotation": rotation,
                "radar_age_s": radar_age,
            }

    def debug_payload(self) -> dict[str, Any]:
        """Expose what CS2 actually sent, without auth values."""
        with self.lock:
            if self.payload is None:
                return {"connected": False}
            player = self.payload.get("player", {}) or {}
            return {
                "connected": True,
                "map": (self.payload.get("map", {}) or {}).get("name"),
                "round_phase": (self.payload.get("round", {}) or {}).get("phase"),
                "player_keys": sorted(player.keys()),
                "has_position": "position" in player,
                "position": player.get("position"),
                "has_forward": "forward" in player,
                "forward": player.get("forward"),
                "auth_present": "auth" in self.payload,
                "phase_countdowns": self.payload.get("phase_countdowns"),
                "round_live_elapsed_s": (
                    round(time.time() - self.round_live_started_at, 1)
                    if self.round_live_started_at
                    else None
                ),
            }

    def raw_state(self) -> dict[str, Any]:
        """Minimal GSI state for the local radar capture client."""
        with self.lock:
            if self.payload is None:
                return {"connected": False}
            map_data = self.payload.get("map", {}) or {}
            player = self.payload.get("player", {}) or {}
            position = player.get("position", {}) or {}
            forward = player.get("forward", {}) or {}
            state = player.get("state", {}) or {}
            yaw = math.degrees(
                math.atan2(float(forward.get("y", 0.0)), float(forward.get("x", 1.0)))
            )
            return {
                "connected": True,
                "map": map_data.get("name"),
                "round": int(map_data.get("round", 0) or 0),
                "phase": str((self.payload.get("round", {}) or {}).get("phase", "")),
                "team": player.get("team"),
                "self": {
                    "x": float(position.get("x", 0.0)),
                    "y": float(position.get("y", 0.0)),
                    "yaw": yaw,
                    "health": int(state.get("health", 0) or 0),
                },
            }

    # ------------------------------------------------------------- rotation
    def _rotation_locked(self) -> dict[str, Any] | None:
        if self.rotation_advisor is None or self.payload is None:
            return None
        now = time.time()
        radar_age = now - self.radar_updated_at if self.radar_updated_at else None
        radar = None
        if self.radar is not None and radar_age is not None and radar_age <= RADAR_TTL_SECONDS:
            radar = self.radar

        map_data = self.payload.get("map", {}) or {}
        player = self.payload.get("player", {}) or {}
        player_state = player.get("state", {}) or {}
        position = player.get("position", {}) or {}
        forward = player.get("forward", {}) or {}
        phase = str((self.payload.get("round", {}) or {}).get("phase", "")).lower()
        phase = {"freezetime": "freeze"}.get(phase, phase)

        x = float(position.get("x", 0.0))
        y = float(position.get("y", 0.0))
        yaw = math.degrees(
            math.atan2(float(forward.get("y", 0.0)), float(forward.get("x", 1.0)))
        )
        round_num = int(map_data.get("round", 0) or 0)
        round_key = (str(map_data.get("name", "")), round_num)
        if round_key != self.round_key:
            self.round_key = round_key
            self.round_live_started_at = now if phase == "live" else 0.0
        elif phase == "live" and self.round_live_started_at == 0.0:
            self.round_live_started_at = now

        has_position = bool(position) and not (x == 0.0 and y == 0.0)
        position_source = "gsi"
        if not has_position and radar is not None:
            radar_x = radar.get("self_x")
            radar_y = radar.get("self_y")
            if radar_x is not None and radar_y is not None:
                x, y = float(radar_x), float(radar_y)
                has_position = True
                position_source = "radar_labels"
        if not has_position:
            return {
                "action": "NO_POSITION",
                "destination": None,
                "confidence": "none",
                "confidence_value": None,
                "stage1_proba": None,
                "destination_proba": None,
                "radar_available": radar is not None,
                "reason": (
                    "GSI has no live player position (official limitation) and "
                    "the radar A/B labels were not found. Hold still until both "
                    "A and B labels are visible on the radar."
                ),
                "posture": "NONE",
                "warnings": [
                    "self position unavailable: GSI omits it during live play "
                    "and the radar label anchors were not detected."
                ],
                "features": None,
            }
        raw_seconds_left = (
            (self.payload.get("phase_countdowns") or {}).get("phase_ends_in", 0.0)
        )
        try:
            seconds_left = float(raw_seconds_left)
        except (TypeError, ValueError):
            seconds_left = 0.0
        if phase != "live":
            seconds_since_freeze = 0.0
        elif 0.0 < seconds_left <= ROUND_SECONDS:
            seconds_since_freeze = max(0.0, ROUND_SECONDS - seconds_left)
        else:
            # GSI often omits phase_ends_in during live play. Track the live
            # phase locally so the feature does not collapse to 115.
            seconds_since_freeze = (
                max(0.0, now - self.round_live_started_at)
                if self.round_live_started_at
                else 0.0
            )

        enemies_raw = (radar or {}).get("enemies", [])
        enemies = [
            enemy
            for enemy in enemies_raw
            if enemy.get("visible", True) and enemy.get("health", 1) > 0
        ]
        visible_count = len(enemies)
        event_now = visible_count > 0 and (
            self.last_visible_count == 0
            or now - self.last_event_at >= 2.0
        )
        if event_now:
            self.last_event_at = now
        newly_visible = (
            visible_count
            if self.last_visible_count == 0
            else max(visible_count - self.last_visible_count, 0)
        )
        if event_now and newly_visible == 0:
            newly_visible = 1
        seconds_since_event = (
            99.0 if self.last_event_at == 0 else now - self.last_event_at
        )
        self.last_visible_count = visible_count

        teammates = (radar or {}).get("teammates")
        state = {
            "map": map_data.get("name", ""),
            "side": str(player.get("team", "")).lower(),
            "phase": phase,
            "round_num": round_num,
            "round_phase": "first_half"
            if round_num <= 12
            else ("second_half" if round_num <= 24 else "overtime"),
            "score_diff": int((map_data.get("team_ct") or {}).get("score", 0))
            - int((map_data.get("team_t") or {}).get("score", 0)),
            "seconds_since_freeze": seconds_since_freeze,
            "seconds_since_previous_event": seconds_since_event,
            "prev_round_winner_side": self.history.last_winner() or "none",
            "prev_round_attack_site": "unknown",
            "self": {
                "x": x,
                "y": y,
                "yaw": yaw,
                "area": (radar or {}).get("self_area"),
                "source": position_source,
                "health": int(player_state.get("health", 0) or 0),
                "armor": int(player_state.get("armor", 0) or 0),
                "equip": float(
                    player_state.get(
                        "equipment_value",
                        player_state.get("equip_value", player_state.get("money", 0)),
                    )
                    or 0
                ),
            },
            "teammates": teammates or [],
            "teammates_perception": teammates is not None,
            "enemies": enemies,
            "enemy_alive_count": (radar or {}).get("enemy_alive_count"),
            "self_spots_enemy": bool((radar or {}).get("self_spots_enemy", False)),
            "self_spot_count": int((radar or {}).get("self_spot_count", 0) or 0),
            "newly_visible_count": newly_visible,
        }
        rotation = self.rotation_advisor.advise(state)
        logged = self._maybe_log(rotation, map_data, round_num, phase, radar_age)
        if logged and rotation.get("action") in OUTCOME_ACTIONS:
            features = rotation.get("features") or {}
            self.pending_outcomes.append(
                {
                    "created": now,
                    "round": round_num,
                    "advice_action": rotation.get("action"),
                    "advice_area": features.get("area"),
                    "confidence": rotation.get("confidence"),
                    "stage1_proba": rotation.get("stage1_proba"),
                    "radar_available": rotation.get("radar_available"),
                    "newly_visible_count": features.get("newly_visible_count"),
                    "ct_alive_count": features.get("ct_alive_count"),
                    "enemy_visible_count": features.get("enemy_visible_count"),
                    "enemy_unknown_count": features.get("enemy_unknown_count"),
                }
            )
            if len(self.pending_outcomes) > 50:
                self.pending_outcomes = self.pending_outcomes[-50:]
        self._resolve_pending_locked(now, round_num)
        return rotation

    def _maybe_log(
        self,
        rotation: dict[str, Any],
        map_data: dict[str, Any],
        round_num: int,
        phase: str,
        radar_age: float | None,
    ) -> bool:
        if self.shadow_log is None:
            return False
        action = rotation.get("action")
        if action in {
            "BUY_PHASE",
            "NOT_DUST2",
            "NOT_CT",
            "NO_MID_ROUND",
            "NO_RADAR",
            "NO_TEAMMATE_DATA",
            "NO_POSITION",
        }:
            return False
        now = time.time()
        key = (round_num, action, rotation.get("destination"))
        if key == self.last_log_key and now - self.last_log_at < 2.0:
            return False
        self.last_log_key = key
        self.last_log_at = now
        record = {
            "record_type": "advice",
            "ts": datetime.now(timezone.utc).isoformat(),
            "match_map": map_data.get("name"),
            "round": round_num,
            "phase": phase,
            "action": action,
            "destination": rotation.get("destination"),
            "confidence": rotation.get("confidence"),
            "confidence_value": rotation.get("confidence_value"),
            "stage1_proba": rotation.get("stage1_proba"),
            "destination_proba": rotation.get("destination_proba"),
            "radar_available": rotation.get("radar_available"),
            "radar_age_s": radar_age,
            "posture": rotation.get("posture"),
            "reason": rotation.get("reason"),
            "warnings": rotation.get("warnings", []),
            # Full feature row: lets us compare the live distribution with the
            # training event dataset after a session.
            "features": rotation.get("features"),
            "newly_visible_count": (rotation.get("features") or {}).get(
                "newly_visible_count"
            ),
        }
        return self._write_record(record)

    def _write_record(self, record: dict[str, Any]) -> bool:
        if self.shadow_log is None:
            return False
        try:
            self.shadow_log.parent.mkdir(parents=True, exist_ok=True)
            with self.shadow_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError:
            return False

    def _resolve_pending_locked(self, now: float, round_num: int) -> None:
        if not self.pending_outcomes:
            return
        player = (self.payload or {}).get("player", {}) or {}
        position = player.get("position", {}) or {}
        health = int((player.get("state", {}) or {}).get("health", 0) or 0)
        x = float(position.get("x", 0.0))
        y = float(position.get("y", 0.0))
        has_position = bool(position) and not (x == 0.0 and y == 0.0)
        if not has_position and self.radar is not None:
            radar_x = self.radar.get("self_x")
            radar_y = self.radar.get("self_y")
            if radar_x is not None and radar_y is not None:
                x, y = float(radar_x), float(radar_y)
                has_position = True
        current_area = (
            self.area_lookup.area(x, y)
            if health > 0 and has_position
            else None
        )

        remaining: list[dict[str, Any]] = []
        for pending in self.pending_outcomes:
            elapsed = now - pending["created"]
            if pending["round"] != round_num:
                self._write_outcome(pending, "round_changed", None, None, elapsed)
                continue
            if elapsed < self.outcome_horizon:
                remaining.append(pending)
                continue
            if health <= 0:
                self._write_outcome(pending, "died", None, None, elapsed)
                continue
            if current_area is None:
                self._write_outcome(pending, "no_position", None, None, elapsed)
                continue

            advice_area = pending.get("advice_area")
            if current_area == advice_area:
                actual_action = "HOLD"
            elif current_area in {"A", "B", "Mid", "CT"}:
                actual_action = (
                    "ROTATE_MID" if current_area == "Mid" else f"ROTATE_{current_area}"
                )
            else:
                actual_action = "OTHER"
            matched = actual_action == pending["advice_action"]
            self._write_outcome(
                pending, "ok", actual_action, current_area, elapsed, matched
            )
        self.pending_outcomes = remaining

    def _write_outcome(
        self,
        pending: dict[str, Any],
        status: str,
        actual_action: str | None,
        actual_area: str | None,
        elapsed: float,
        matched: bool = False,
    ) -> None:
        self._write_record(
            {
                "record_type": "outcome",
                "ts": datetime.now(timezone.utc).isoformat(),
                "round": pending["round"],
                "advice_action": pending["advice_action"],
                "advice_area": pending.get("advice_area"),
                "advice_confidence": pending.get("confidence"),
                "advice_stage1_proba": pending.get("stage1_proba"),
                "advice_radar_available": pending.get("radar_available"),
                "advice_newly_visible_count": pending.get("newly_visible_count"),
                "advice_ct_alive_count": pending.get("ct_alive_count"),
                "advice_enemy_visible_count": pending.get("enemy_visible_count"),
                "advice_enemy_unknown_count": pending.get("enemy_unknown_count"),
                "actual_action": actual_action,
                "actual_area": actual_area,
                "player_moved": (
                    actual_area is not None
                    and actual_area != pending.get("advice_area")
                ),
                "matched": bool(matched),
                "status": status,
                "elapsed_s": round(elapsed, 2),
            }
        )


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
        path = urlparse(self.path).path
        if path not in {"/", "/radar"}:
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
                if getattr(self.server, "verbose", False):
                    print(
                        f"{time.strftime('%H:%M:%S')} rejected {path}: "
                        f"token mismatch (server={token!r}, "
                        f"received={supplied!r})"
                    )
                self._send_json(403, {"error": "bad token"})
                return

        if path == "/radar":
            self.server.state.update_radar(payload)
            if getattr(self.server, "verbose", False):
                print(
                    f"{time.strftime('%H:%M:%S')} radar "
                    f"enemies={len(payload.get('enemies', []))} "
                    f"teammates={len(payload.get('teammates', []))}"
                )
            self._send_json(200, {"ok": True})
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
        if path == "/raw_state":
            self._send_json(200, self.server.state.raw_state())
            return
        if path == "/debug_payload":
            self._send_json(200, self.server.state.debug_payload())
            return
        self._send_json(404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser(description="GSI listener and Dust2 CT buy coach.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--token", default="change-me", help="Must match the GSI config token.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--rotation-model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Exported rotation policy JSON.",
    )
    parser.add_argument(
        "--shadow-log",
        type=Path,
        default=DEFAULT_SHADOW_LOG,
        help="Append rotation advice here; use 'none' to disable.",
    )
    parser.add_argument(
        "--outcome-horizon",
        type=float,
        default=5.0,
        help="Seconds after an advice before logging what the player actually did.",
    )
    parser.add_argument(
        "--area-grid",
        type=Path,
        default=DEFAULT_AREA_GRID,
        help="JSON area grid used to compare advice with the actual area.",
    )
    args = parser.parse_args()

    advisor = None
    if args.rotation_model.exists():
        advisor = RotationAdvisor.load(args.rotation_model)
        print(f"Rotation model loaded: {args.rotation_model}")
    else:
        print(
            f"Rotation model not found at {args.rotation_model}; "
            "mid-round advice disabled."
        )
    shadow_log: Path | None = args.shadow_log
    if str(args.shadow_log).lower() == "none":
        shadow_log = None

    area_lookup = AreaLookup.load(args.area_grid, DEFAULT_PLACES)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = AppState(  # type: ignore[attr-defined]
        advisor,
        shadow_log,
        area_lookup=area_lookup,
        outcome_horizon=args.outcome_horizon,
    )
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
