"""Оповещения в Telegram: устойчивые спреды/funding и устойчивые лаги.

Отправка — history.alerts.send (переменные TELEGRAM_BOT_TOKEN и
TELEGRAM_CHAT_ID). Правила, чтобы чат не превратился в шум:

  * спред или funding — только если связка держится: не меньше
    ALERT_HOT_MINUTES минутных снимков из последних ALERT_HOT_WINDOW;
  * одна и та же связка повторно — не раньше чем через срок молчания
    (спреды — 4 ч, лаги — сутки), если только итог не вырос вдвое;
  * в каждом сообщении — что считалось и при каком допущении.
"""

from __future__ import annotations

import html
import json
import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from history import alerts

from .catalog import CHAINS
from .settings import LEADLAG_DIR, ScreenSettings

log = logging.getLogger(__name__)

STATE_PATH = LEADLAG_DIR / "alerts_sent.json"
ALERT_HOT_WINDOW = 15
ALERT_HOT_MINUTES = 10
MUTE_SPREAD_S = 4 * 3600
MUTE_LAG_S = 24 * 3600
MAX_PER_RUN = 5          # больше за раз — признак сбоя данных, а не удачи


def load_state() -> Dict[str, dict]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict[str, dict]) -> None:
    now = time.time()
    state = {k: v for k, v in state.items() if now - v.get("at", 0) < 7 * 86400}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def due(state: dict, key: str, value: float, mute_s: float, now: float) -> bool:
    """Слать, если не слали, срок молчания истёк или значение выросло вдвое."""
    prev = state.get(key)
    if prev is None or now - prev["at"] >= mute_s:
        return True
    return prev.get("value", 0) > 0 and value >= 2 * prev["value"]


def _a(url: Optional[str], text: str) -> str:
    text = html.escape(text)
    return f'<a href="{html.escape(url)}">{text}</a>' if url else text


# ------------------------------------------------------------------ спреды

def _avg(r) -> str:
    """«(средний за час +0,41%, за сутки +0,35%)» — если средние посчитаны."""
    parts = []
    for name, label in (("spread_avg_1h", "за час"), ("spread_avg_24h", "за сутки")):
        v = getattr(r, name, None)
        if v is not None and v == v:
            parts.append(f"{label} {v:+.2f}%")
    return f" (средний {', '.join(parts)})" if parts else ""


def _depth(r) -> str:
    v1, v5 = getattr(r, "spread_1000", None), getattr(r, "spread_5000", None)
    fmt = lambda v: f"{v:+.2f}%" if v is not None and v == v else "не хватает стакана"
    if v1 is None and v5 is None:
        return ""
    return f"Спред на объёме: $1 000 {fmt(v1)}, $5 000 {fmt(v5)}\n"


def spread_message(r, hold_h: float) -> str:
    carry = r.pnl_carry_pct >= r.pnl_converge_pct
    head = "💰 Funding-связка" if carry else "↔️ Спред фьючерсов"
    lines = [
        f"<b>{head}: {html.escape(r.coin)}</b>",
        f"Лонг {_a(r.url_l, r.venue_l.capitalize())} · шорт {_a(r.url_s, r.venue_s.capitalize())}",
        f"Спред {r.spread_pct:+.2f}%{_avg(r)} · funding {r.funding_day_pct:+.2f}% в сутки",
        f"Итог за {hold_h:g} ч: спред сошёлся <b>{r.pnl_converge_pct:+.2f}%</b>, "
        f"только funding <b>{r.pnl_carry_pct:+.2f}%</b> (после комиссий {r.fees_pct:.2f}%)",
        _depth(r) +
        f"Держится {int(r.minutes_hot)} мин из {ALERT_HOT_WINDOW} · объём меньшей ноги "
        f"${r.min_volume_usd / 1e6:.1f} млн",
    ]
    if r.reasons:
        lines.append(f"<i>{html.escape(r.reasons)}</i>")
    lines.append("<i>Ставка funding — текущая, на расчётах меняется. Цены — лучшие в стакане.</i>")
    return "\n".join(lines)


def notify_spreads(q: pd.DataFrame, hot: pd.DataFrame, s: ScreenSettings) -> int:
    """q — spreads.select(...) за текущий снимок, hot — spreads.persistence(...)."""
    if not s.alerts_enabled or not alerts.configured() or q is None or q.empty:
        return 0
    q = q.merge(hot, on=["coin", "venue_l", "venue_s"], how="left")
    q = q[(q["minutes_hot"].fillna(0) >= ALERT_HOT_MINUTES)
          & (q[["pnl_converge_pct", "pnl_carry_pct"]].max(axis=1) >= s.alert_min_pnl_pct)
          & (q["reasons"].fillna("").str.contains("разные токены") == False)]  # noqa: E712
    state, now, sent = load_state(), time.time(), 0
    for r in q.sort_values("best_pnl", ascending=False).itertuples():
        key = f"spread:{r.coin}:{r.venue_l}:{r.venue_s}"
        if not due(state, key, r.best_pnl, MUTE_SPREAD_S, now):
            continue
        if sent >= MAX_PER_RUN:
            break
        if alerts.send(spread_message(r, s.spread_hold_hours)):
            state[key] = {"at": now, "value": float(r.best_pnl)}
            sent += 1
    save_state(state)
    return sent


# -------------------------------------------------------------------- лаги

def _label(inst: str, meta: dict) -> str:
    kind, venue = inst.split(":")[:2]
    if kind == "dex":
        return f"DEX {CHAINS[venue].label if venue in CHAINS else venue}"
    return f"фьючерс {venue.capitalize()}"


def lag_message(r, meta: dict) -> str:
    leader = _a(meta.get(r.leader, {}).get("url"), _label(r.leader, meta))
    follower = _a(meta.get(r.follower, {}).get("url"), _label(r.follower, meta))
    return "\n".join([
        f"<b>⏱ Устойчивая зависимость: {html.escape(str(r.coin))}</b>",
        f"{leader} ведёт → {follower} следует через <b>{r.lag_s:g} с</b>",
        f"Корреляция на лаге {r.corr:.2f} (без сдвига {r.corr0:.2f}) · "
        f"держится {int(r.hours_stable)} ч из {int(r.hours_data)}",
        "<i>Лаг по времени сделок; у редко торгуемого ведомого он включает ожидание "
        "следующей сделки.</i>",
    ])


def notify_lags(res: pd.DataFrame, meta: dict, s: ScreenSettings) -> int:
    if not s.alerts_enabled or not alerts.configured() or res is None or res.empty:
        return 0
    q = res[(~res["synchronous"]) & (res["hours_stable"] >= s.alert_min_lag_hours)
            & (res["lag_s"].fillna(0) >= 1)]
    state, now, sent = load_state(), time.time(), 0
    for r in q.sort_values(["hours_stable", "corr"], ascending=False).itertuples():
        key = f"lag:{r.leader}:{r.follower}"
        if not due(state, key, float(r.hours_stable), MUTE_LAG_S, now):
            continue
        if sent >= MAX_PER_RUN:
            break
        if alerts.send(lag_message(r, meta)):
            state[key] = {"at": now, "value": float(r.hours_stable)}
            sent += 1
    save_state(state)
    return sent
