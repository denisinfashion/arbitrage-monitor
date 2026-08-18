"""Оповещения в Telegram о появившихся связках.

Работает как шаг после сбора: свежие данные уже в базе, по ним считаются
связки, и те, что прошли порог, уходят сообщением в чат.

Три вещи, без которых оповещения быстро становятся мусором.

**Дедупликация.** Одна и та же связка живёт десятки минут и попадёт
в несколько прогонов подряд. Слать её каждые пятнадцать минут — верный
способ добиться того, что уведомления замьютят. Отправленное запоминается
и повторно не шлётся, пока не истечёт срок молчания.

**Порог по существу, а не по красоте.** Отбираются связки, прибыльные
в последней точке, с достаточной ликвидностью и без переводов между
площадками. Красивый максимум за неделю к делу не относится: он уже
в прошлом.

**Честная подпись.** В сообщении указано, что расчёт исторический и
окно могло закрыться. Оповещение — повод посмотреть, а не сигнал
покупать не глядя.

Настройка — две переменные окружения:

    TELEGRAM_BOT_TOKEN   токен бота от @BotFather
    TELEGRAM_CHAT_ID     идентификатор чата (свой или группы)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import DATA_DIR, SETTINGS, ensure_data_dir

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/sendMessage"
STATE_FILE = "alerts_sent.json"

ENV_TOKEN = "TELEGRAM_BOT_TOKEN"
ENV_CHAT = "TELEGRAM_CHAT_ID"


@dataclass
class AlertConfig:
    """Что считать поводом для сообщения."""

    min_margin_pct: float = 0.30
    """Порог чистой маржи в последней точке. Ниже — не беспокоим."""

    min_liquidity_usd: float = 200_000.0
    """Минимальный пул в цепочке: на мелких маржа неисполнима."""

    require_single_chain: bool = True
    """Только связки без переводов между площадками."""

    max_per_run: int = 5
    """Сколько связок слать за раз, чтобы не заваливать чат."""

    mute_minutes: int = 90
    """Сколько молчать про уже отправленную связку."""

    @classmethod
    def from_env(cls) -> "AlertConfig":
        def num(name, default):
            try:
                return type(default)(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default
        return cls(
            min_margin_pct=num("ALERT_MIN_MARGIN", 0.30),
            min_liquidity_usd=num("ALERT_MIN_LIQUIDITY", 200_000.0),
            require_single_chain=os.environ.get("ALERT_SINGLE_CHAIN", "1") != "0",
            max_per_run=num("ALERT_MAX_PER_RUN", 5),
            mute_minutes=num("ALERT_MUTE_MINUTES", 90),
        )


# --------------------------------------------------------------------------
# Память об отправленном
# --------------------------------------------------------------------------


def _state_path():
    ensure_data_dir()
    return DATA_DIR / STATE_FILE


def load_sent() -> dict:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_sent(state: dict) -> None:
    try:
        _state_path().write_text(json.dumps(state, ensure_ascii=False),
                                 encoding="utf-8")
    except OSError as exc:
        log.warning("не удалось сохранить память об оповещениях: %s", exc)


def _prune_sent(state: dict, mute_minutes: int) -> dict:
    """Забывает связки, про которые молчать уже не нужно."""
    cutoff = time.time() - mute_minutes * 60
    return {k: v for k, v in state.items() if v > cutoff}


# --------------------------------------------------------------------------
# Отбор
# --------------------------------------------------------------------------


def pick(cycles: Sequence, cfg: AlertConfig, sent: dict) -> List:
    """Оставляет связки, о которых стоит сообщить."""
    import numpy as np

    out = []
    for c in cycles:
        m = c.margin_pct()
        if not len(m) or not np.isfinite(m[-1]):
            continue
        now_margin = float(m[-1])
        if now_margin < cfg.min_margin_pct:
            continue

        liq = c.bottleneck_liquidity()
        if cfg.min_liquidity_usd and (not liq or liq < cfg.min_liquidity_usd):
            continue

        if cfg.require_single_chain and c.needs_transfer():
            continue

        if c.label in sent:
            continue

        out.append((now_margin, c))

    out.sort(key=lambda x: -x[0])
    return [c for _, c in out[: cfg.max_per_run]]


# --------------------------------------------------------------------------
# Сообщение
# --------------------------------------------------------------------------


def format_message(cycle, margin: float) -> str:
    """Текст одного оповещения. Разметка HTML — она надёжнее Markdown,
    который спотыкается о символы вроде подчёркиваний в именах площадок."""
    from .links import chain_name

    def esc(s: str) -> str:
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    legs = cycle.leg_links()
    chain = cycle.single_chain()
    liq = cycle.bottleneck_liquidity()

    lines = [
        f"<b>{esc(cycle.label)}</b>",
        f"Маржа сейчас: <b>{margin:+.3f}%</b>"
        f"  ·  объём ${cycle.grid.trade_size_usd:,.0f}".replace(",", " "),
    ]
    if chain:
        lines.append(f"Сеть: {esc(chain_name(chain))} · переводы не нужны")
    if liq:
        lines.append(f"Узкое место по ликвидности: ${liq:,.0f}".replace(",", " "))

    stats = cycle.stats()
    window = stats.get("Окно средн, мин")
    if window:
        lines.append(f"Типичное окно: около {window} мин")

    lines.append("")
    for leg in legs:
        row = f"{leg['n']}. {esc(leg['from'])} → {esc(leg['to'])} · {esc(leg['venue'])}"
        if leg["swap"]:
            row += f'  <a href="{leg["swap"]}">обмен</a>'
        lines.append(row)

    note = cycle.token_note()
    if note:
        lines.append("")
        lines.append(f"<i>{esc(note)}</i>")

    lines.append("")
    lines.append("<i>Расчёт по истории котировок. Окно могло уже закрыться — "
                 "проверьте цену перед сделкой.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Отправка
# --------------------------------------------------------------------------


def configured() -> bool:
    return bool(os.environ.get(ENV_TOKEN) and os.environ.get(ENV_CHAT))


def send(text: str, token: Optional[str] = None,
         chat_id: Optional[str] = None) -> bool:
    """Отправляет сообщение. Возвращает True при успехе."""
    token = token or os.environ.get(ENV_TOKEN, "")
    chat_id = chat_id or os.environ.get(ENV_CHAT, "")
    if not token or not chat_id:
        log.debug("оповещения не настроены")
        return False

    from .http import HttpError, post_json
    try:
        r = post_json(API.format(token=token), {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        })
        if not r.get("ok"):
            log.warning("Telegram отказал: %s", str(r)[:200])
            return False
        return True
    except HttpError as exc:
        # Токен в текст ошибки не попадает: адрес с ним внутри не логируем.
        log.warning("не удалось отправить оповещение: %s",
                    str(exc).replace(token, "***"))
        return False


def notify(cycles: Sequence, cfg: Optional[AlertConfig] = None) -> int:
    """Полный цикл: отобрать, отправить, запомнить. Возвращает число сообщений."""
    if not configured():
        log.info("оповещения не настроены — пропускаю")
        return 0

    cfg = cfg or AlertConfig.from_env()
    sent = _prune_sent(load_sent(), cfg.mute_minutes)
    chosen = pick(cycles, cfg, sent)

    if not chosen:
        log.info("подходящих связок нет (порог %.2f%%, ликвидность от $%s)",
                 cfg.min_margin_pct, f"{cfg.min_liquidity_usd:,.0f}".replace(",", " "))
        save_sent(sent)
        return 0

    n = 0
    for c in chosen:
        margin = float(c.margin_pct()[-1])
        if send(format_message(c, margin)):
            sent[c.label] = time.time()
            n += 1
            log.info("оповещение: %s (%+.3f%%)", c.label, margin)

    save_sent(sent)
    return n
