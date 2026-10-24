"""Always-on-top mid-round ticker for single-screen play.

It reads ``GET /state`` and ``GET /plan`` from localhost and never injects
input into the game. The display follows the round phase:

- freeze time: buy/save recommendation and estimated cost;
- live round: the mid-round call (LLM or template);
- otherwise: waiting state.

Left-click expands the details panel (round, radar, rotation, LLM status).
Right-click opens the menu: model settings, refresh, exit.

Run:
    python -m live.midround.ticker --token change-me
    python live/midround/ticker.py --font-size 30 --opacity 0.88

Drag with the left mouse button. Right-click (or the small ×) to close.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import ttk
from typing import Any

SOURCE_LABELS = {"llm": "模型建议", "template": "快速建议"}
ADVICE_LABELS = {
    "FULL_BUY": "全买",
    "HALF_BUY": "半买",
    "SAVE": "存钱",
    "PISTOL_BUY": "手枪局购买",
    "PISTOL_SAVE": "手枪局存钱",
    "BUY_PHASE_OVER": "购买结束",
    "NOT_SUPPORTED": "不支持",
    "MONITORING": "观察中",
}
PHASE_LABELS = {"freezetime": "冻结期", "live": "进行中", "over": "回合结束"}
ITEM_LABELS = {
    "Kevlar + Helmet": "护甲+头盔",
    "Kevlar": "护甲",
    "Helmet": "头盔",
    "Defuse Kit": "拆弹器",
    "Smoke": "烟雾弹",
    "Flash": "闪光弹",
    "HE": "高爆手雷",
    "M4A1-S": "M4A1-S",
    "MP9": "MP9",
}
BG = "#10141a"
FG = "#58d68d"
MUTED = "#8b95a7"
FLASH_BG = "#2b3a2f"
COMPACT_HEIGHT = 118
EXPANDED_HEIGHT = 310
WIDTH = 540


def format_plan(payload: dict[str, Any] | None) -> tuple[str, str, int]:
    """Return ``(call, meta_line, hotkey_seq)`` for the plan card."""
    payload = payload or {}
    plan = payload.get("plan")
    seq = int(payload.get("hotkey_seq") or 0)
    if not plan:
        return "暂无指令", "等待雷达和回合状态", seq

    call = str(plan.get("call") or "—")
    source = SOURCE_LABELS.get(str(plan.get("source")), "指令")
    age = payload.get("age_s")
    meta = source
    if age is not None:
        meta += f" · {float(age):.1f}s"
    if payload.get("stale") or (age is not None and float(age) > 3.0):
        meta += " · 已过期"
    facts = plan.get("grounded_facts") or []
    if facts:
        meta += " · " + str(facts[0])
    return call, meta, seq


def format_dashboard(
    state: dict[str, Any] | None, plan_payload: dict[str, Any] | None
) -> dict[str, Any]:
    """Build headline, meta and detail lines from ``/state`` + ``/plan``."""
    state = state or {}
    plan_payload = plan_payload or {}
    plan = plan_payload.get("plan")
    seq = int(plan_payload.get("hotkey_seq") or 0)
    advice = state.get("advice") or {}
    rotation = state.get("rotation") or {}
    connected = bool(state.get("connected"))

    details: list[str] = []
    if not connected:
        return {
            "headline": "等待 GSI",
            "meta": "启动 CS2 并进入 Dust2 CT 对局",
            "details": [],
            "hotkey_seq": seq,
            "phase": None,
        }

    round_number = advice.get("round")
    score = advice.get("score") or {}
    phase = str(advice.get("phase") or "")
    seconds = advice.get("seconds_left")
    money = advice.get("money")

    details.append(
        f"回合 {round_number or '?'} · CT {score.get('ct', '?')}:{score.get('t', '?')} T"
    )
    phase_line = PHASE_LABELS.get(phase, phase or "?")
    if seconds is not None:
        phase_line += f" · {seconds} 秒"
    if money is not None:
        phase_line += f" · ${money}"
    details.append(phase_line)

    radar_age = state.get("radar_age_s")
    details.append(
        "雷达 " + (f"{float(radar_age):.1f}s" if radar_age is not None else "无数据")
    )
    if rotation:
        details.append(
            f"旋转 {rotation.get('action', '-')} · "
            f"置信度 {rotation.get('confidence', '-')}"
        )
    if plan:
        details.append(
            f"计划 {SOURCE_LABELS.get(str(plan.get('source')), '指令')} · "
            f"{plan.get('reason_short', '')}"
        )
        facts = plan.get("grounded_facts") or []
        if facts:
            details.append("事实 " + "；".join(str(fact) for fact in facts[:3]))

    llm = plan_payload.get("llm") or {}
    if llm.get("enabled"):
        llm_line = f"LLM {llm.get('backend')}/{llm.get('model')}"
        if llm.get("inflight"):
            llm_line += " · 生成中"
        elif llm.get("last_error"):
            llm_line += f" · {llm['last_error']}"
        elif llm.get("last_latency_s") is not None:
            llm_line += f" · {float(llm['last_latency_s']):.1f}s"
        details.append(llm_line)
    else:
        details.append("LLM 未启用（模板回退）")

    if phase in {"freezetime", "freeze"}:
        recommendation = str(advice.get("recommendation") or "")
        headline = ADVICE_LABELS.get(recommendation, recommendation or "冻结期")
        meta_parts: list[str] = []
        cost = advice.get("estimated_cost")
        reserve = advice.get("reserve")
        if cost is not None:
            meta_parts.append(f"预计花费 ${cost}")
        if reserve is not None:
            meta_parts.append(f"剩余 ${reserve}")
        meta = " · ".join(meta_parts) or "冻结期购买建议"
        shopping = advice.get("shopping_list") or []
        if shopping:
            items = []
            for item in shopping:
                name = ITEM_LABELS.get(str(item.get("name")), str(item.get("name")))
                cost = item.get("cost")
                items.append(f"{name} ${cost}" if cost is not None else name)
            details.append("购买 " + "、".join(items))
        return {
            "headline": headline,
            "meta": meta,
            "details": details,
            "hotkey_seq": seq,
            "phase": phase,
        }

    if phase == "live":
        if plan:
            headline = str(plan.get("call") or "—")
            source = SOURCE_LABELS.get(str(plan.get("source")), "指令")
            age = plan_payload.get("age_s")
            meta = source
            if age is not None:
                meta += f" · {float(age):.1f}s"
            if plan_payload.get("stale") or (age is not None and float(age) > 3.0):
                meta += " · 已过期"
        else:
            headline = "暂无指令"
            meta = str(rotation.get("reason") or "等待雷达和回合状态")
        return {
            "headline": headline,
            "meta": meta,
            "details": details,
            "hotkey_seq": seq,
            "phase": phase,
        }

    recommendation = str(advice.get("recommendation") or "")
    return {
        "headline": ADVICE_LABELS.get(recommendation, recommendation or "等待对局"),
        "meta": "等待下一回合",
        "details": details,
        "hotkey_seq": seq,
        "phase": phase,
    }


class Ticker:
    def __init__(
        self,
        root: tk.Tk,
        *,
        url: str,
        token: str,
        interval: float = 0.25,
        font_size: int = 28,
        opacity: float = 0.9,
    ) -> None:
        self.root = root
        self.url = url.rstrip("/")
        self.token = token
        self.interval = max(0.1, interval)
        self.queue: queue.Queue[tuple[dict | None, dict | None]] = queue.Queue()
        self.stop = threading.Event()
        self.last_seq = 0
        self.last_ok = time.monotonic()
        self.expanded = False
        self.pos_x = 20
        self.pos_y = 500
        self._drag_offset = (0, 0)
        self._press_at = (0, 0)
        self._dragging = False
        self._flash_job: str | None = None
        self._settings: tk.Toplevel | None = None

        self.headline_var = tk.StringVar(value="等待 GSI")
        self.meta_var = tk.StringVar(value="启动 CS2 并进入 Dust2 CT 对局")
        self.detail_var = tk.StringVar(value="")

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", max(0.35, min(1.0, opacity)))
        root.configure(bg=BG)

        self.frame = tk.Frame(root, bg=BG, padx=14, pady=8)
        self.frame.pack(fill="both", expand=True)
        self.headline_label = tk.Label(
            self.frame,
            textvariable=self.headline_var,
            font=("Microsoft YaHei", font_size, "bold"),
            fg=FG,
            bg=BG,
            anchor="w",
            justify="left",
            wraplength=WIDTH - 40,
        )
        self.headline_label.pack(fill="x")
        self.meta_label = tk.Label(
            self.frame,
            textvariable=self.meta_var,
            font=("Microsoft YaHei", 11),
            fg=MUTED,
            bg=BG,
            anchor="w",
            justify="left",
            wraplength=WIDTH - 40,
        )
        self.meta_label.pack(fill="x", pady=(2, 0))
        self.detail_label = tk.Label(
            self.frame,
            textvariable=self.detail_var,
            font=("Microsoft YaHei", 10),
            fg=MUTED,
            bg=BG,
            anchor="nw",
            justify="left",
            wraplength=WIDTH - 40,
        )

        self.close_label = tk.Label(
            self.frame,
            text="×",
            font=("Segoe UI", 14, "bold"),
            fg=MUTED,
            bg=BG,
            cursor="hand2",
        )
        self.close_label.place(relx=1.0, x=-6, y=-2, anchor="ne")
        self.close_label.bind("<Button-1>", lambda _event: self.close())

        for widget in (
            root,
            self.frame,
            self.headline_label,
            self.meta_label,
            self.detail_label,
        ):
            widget.bind("<ButtonPress-1>", self._on_press)
            widget.bind("<B1-Motion>", self._on_motion)
            widget.bind("<ButtonRelease-1>", self._on_release)
            widget.bind("<Button-3>", self._show_menu)

        self.menu = tk.Menu(root, tearoff=0)
        self.menu.add_command(label="设置模型…", command=self.open_settings)
        self.menu.add_command(label="立即刷新", command=self.refresh_plan)
        self.menu.add_separator()
        self.menu.add_command(label="退出", command=self.close)

        self._place(WIDTH, COMPACT_HEIGHT)
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        root.after(100, self._drain)
        root.after(1000, self._check_alive)

    # ------------------------------------------------------------------ UI
    def _place(self, width: int, height: int) -> None:
        # Default position is below the radar (radar ROI is ~486 px at 2560).
        # The player can drag it anywhere; the ticker must not cover the radar.
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = max(0, min(self.pos_x, screen_width - width))
        y = max(0, min(self.pos_y, screen_height - height))
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _on_press(self, event: tk.Event) -> None:
        self._press_at = (event.x_root, event.y_root)
        self._drag_offset = (
            event.x_root - self.root.winfo_x(),
            event.y_root - self.root.winfo_y(),
        )
        self._dragging = False

    def _on_motion(self, event: tk.Event) -> None:
        dx = abs(event.x_root - self._press_at[0])
        dy = abs(event.y_root - self._press_at[1])
        if dx < 4 and dy < 4:
            return
        self._dragging = True
        x = event.x_root - self._drag_offset[0]
        y = event.y_root - self._drag_offset[1]
        self.pos_x, self.pos_y = x, y
        self.root.geometry(f"+{x}+{y}")

    def _on_release(self, _event: tk.Event) -> None:
        if self._dragging:
            self._dragging = False
            return
        self._toggle_details()

    def _toggle_details(self) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self.detail_label.pack(fill="both", expand=True, pady=(6, 0))
            self._place(WIDTH, EXPANDED_HEIGHT)
        else:
            self.detail_label.pack_forget()
            self._place(WIDTH, COMPACT_HEIGHT)

    def _show_menu(self, event: tk.Event) -> None:
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _flash(self) -> None:
        if self._flash_job is not None:
            self.root.after_cancel(self._flash_job)
        self._set_bg(FLASH_BG)
        self._flash_job = self.root.after(1400, self._unflash)

    def _set_bg(self, color: str) -> None:
        self.root.configure(bg=color)
        self.frame.configure(bg=color)
        for widget in (
            self.headline_label,
            self.meta_label,
            self.detail_label,
            self.close_label,
        ):
            widget.configure(bg=color)

    def _unflash(self) -> None:
        self._flash_job = None
        self._set_bg(BG)

    def _drain(self) -> None:
        updated = False
        while True:
            try:
                state, plan_payload = self.queue.get_nowait()
            except queue.Empty:
                break
            updated = True
            if state is None and plan_payload is None:
                self.meta_var.set("服务端不可达")
                continue
            self.last_ok = time.monotonic()
            dashboard = format_dashboard(state, plan_payload)
            self.headline_var.set(dashboard["headline"])
            self.meta_var.set(dashboard["meta"])
            self.detail_var.set("\n".join(dashboard["details"]))
            if dashboard["hotkey_seq"] > self.last_seq:
                self.last_seq = dashboard["hotkey_seq"]
                self._flash()
        self.root.after(100 if updated else 150, self._drain)

    def _check_alive(self) -> None:
        if time.monotonic() - self.last_ok > 3.0:
            self.meta_var.set("服务端不可达")
        self.root.after(1000, self._check_alive)

    # -------------------------------------------------------------- polling
    def _fetch(self, path: str) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(self.url + path, timeout=1.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None

    def _poll_loop(self) -> None:
        while not self.stop.is_set():
            state = self._fetch("/state")
            plan_payload = self._fetch("/plan")
            self.queue.put((state, plan_payload))
            self.stop.wait(self.interval)

    # -------------------------------------------------------------- actions
    def _post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        body = json.dumps(
            {**payload, "auth": {"token": self.token}}, ensure_ascii=False
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8.0) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                detail = {"error": str(exc)}
            return exc.code, detail
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, {"error": str(exc)}

    def refresh_plan(self) -> None:
        def worker() -> None:
            code, data = self._post("/agent/refresh", {})
            self.root.after(
                0,
                lambda: self.meta_var.set(
                    "已刷新" if code == 200 else f"刷新失败：{data.get('error', code)}"
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------- settings
    def open_settings(self) -> None:
        if self._settings is not None and self._settings.winfo_exists():
            self._settings.lift()
            return
        self._settings = SettingsWindow(self)

    def close(self) -> None:
        self.stop.set()
        self.root.destroy()


class SettingsWindow:
    def __init__(self, ticker: Ticker) -> None:
        self.ticker = ticker
        self.window = tk.Toplevel(ticker.root)
        self.window.title("模型设置")
        self.window.attributes("-topmost", True)
        self.window.configure(bg=BG)
        self.window.geometry(f"520x360+{ticker.root.winfo_x()}+{ticker.root.winfo_y() + 130}")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.backends: list[dict[str, Any]] = []
        self.backend_var = tk.StringVar(value="Ollama（本地）")
        self.model_var = tk.StringVar()
        self.url_var = tk.StringVar()
        self.key_var = tk.StringVar()
        self.timeout_var = tk.StringVar(value="5")
        self.enabled_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="正在读取配置…")

        self.backend_box = ttk.Combobox(
            self.window, textvariable=self.backend_var, state="readonly"
        )
        self._row("后端", self.backend_box)
        self.backend_box.bind("<<ComboboxSelected>>", self._on_backend)
        self._row("模型", tk.Entry(
            self.window, textvariable=self.model_var, bg="#0f1319", fg="#e6ebf3",
            insertbackground="#e6ebf3",
        ))
        self._row("接口地址", tk.Entry(
            self.window, textvariable=self.url_var, bg="#0f1319", fg="#e6ebf3",
            insertbackground="#e6ebf3",
        ))
        self._row("API Key", tk.Entry(
            self.window, textvariable=self.key_var, show="*", bg="#0f1319",
            fg="#e6ebf3", insertbackground="#e6ebf3",
        ))
        self._row("超时（秒）", tk.Entry(
            self.window, textvariable=self.timeout_var, bg="#0f1319",
            fg="#e6ebf3", insertbackground="#e6ebf3",
        ))
        tk.Checkbutton(
            self.window,
            text="启用 LLM（关闭后使用内核模板）",
            variable=self.enabled_var,
            bg=BG,
            fg=MUTED,
            selectcolor=BG,
            activebackground=BG,
            activeforeground=FG,
        ).pack(anchor="w", padx=14, pady=(4, 0))

        buttons = tk.Frame(self.window, bg=BG)
        buttons.pack(fill="x", padx=14, pady=10)
        tk.Button(buttons, text="应用", command=self.apply).pack(side="left")
        tk.Button(buttons, text="关闭", command=self.close).pack(side="left", padx=8)
        tk.Label(
            self.window,
            textvariable=self.status_var,
            bg=BG,
            fg=MUTED,
            justify="left",
            anchor="w",
            wraplength=480,
        ).pack(fill="x", padx=14, pady=(0, 10))

        threading.Thread(target=self._load_config, daemon=True).start()

    def _row(self, label: str, widget: tk.Widget) -> None:
        row = tk.Frame(self.window, bg=BG)
        row.pack(fill="x", padx=14, pady=3)
        tk.Label(row, text=label, bg=BG, fg=MUTED, width=10, anchor="w").pack(side="left")
        widget.pack(side="left", fill="x", expand=True)

    def _load_config(self) -> None:
        data = self.ticker._fetch("/agent/config")
        if data is None:
            self.ticker.root.after(
                0, lambda: self.status_var.set("无法读取配置（服务端不可达）")
            )
            return
        self.ticker.root.after(0, lambda: self._populate(data))

    def _populate(self, data: dict[str, Any]) -> None:
        config = data.get("config") or {}
        self.backends = data.get("available_backends") or []
        labels = [backend.get("label", backend.get("id")) for backend in self.backends]
        self.backend_box.configure(values=labels)
        backend_id = config.get("backend", "ollama")
        for backend in self.backends:
            if backend.get("id") == backend_id:
                self.backend_var.set(backend.get("label", backend_id))
                break
        self.model_var.set(config.get("model", ""))
        self.url_var.set(config.get("base_url", ""))
        self.timeout_var.set(str(config.get("timeout_s", 5)))
        self.enabled_var.set(bool(config.get("enabled", True)))
        models = data.get("models") or []
        hint = "可用模型：" + ", ".join(models) if models else ""
        if config.get("has_api_key"):
            hint = (hint + "  " if hint else "") + "已保存 API Key（仅内存）"
        self.status_var.set(hint or "配置读取完成")

    def _backend_id(self) -> str:
        label = self.backend_var.get()
        for backend in self.backends:
            if backend.get("label") == label:
                return str(backend.get("id"))
        return "ollama"

    def _on_backend(self, _event: tk.Event) -> None:
        for backend in self.backends:
            if backend.get("label") == self.backend_var.get():
                if backend.get("default_url"):
                    self.url_var.set(backend["default_url"])
                break

    def apply(self) -> None:
        payload = {
            "backend": self._backend_id(),
            "model": self.model_var.get(),
            "base_url": self.url_var.get(),
            "api_key": self.key_var.get(),
            "timeout_s": self.timeout_var.get(),
            "enabled": self.enabled_var.get(),
        }
        self.status_var.set("正在应用…")

        def worker() -> None:
            code, data = self.ticker._post("/agent/config", payload)
            def report() -> None:
                if code == 200:
                    self.status_var.set("已应用；下一次状态更新会调用该模型。")
                    self.key_var.set("")
                elif code == 403:
                    self.status_var.set(
                        "bad token：ticker 的 --token 与服务端不一致。"
                    )
                else:
                    self.status_var.set(f"应用失败：{data.get('error', code)}")
            self.ticker.root.after(0, report)

        threading.Thread(target=worker, daemon=True).start()

    def close(self) -> None:
        self.window.destroy()
        self.ticker._settings = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--token", default="change-me", help="Same token as the server.")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--opacity", type=float, default=0.9)
    args = parser.parse_args(argv)

    root = tk.Tk()
    root.title("CS2 Mid-round")
    Ticker(
        root,
        url=args.url,
        token=args.token,
        interval=args.interval,
        font_size=args.font_size,
        opacity=args.opacity,
    )
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
