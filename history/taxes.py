"""Налог на перевод: то, чего не видно в цене.

Повод написать этот модуль был конкретный. В чат ушло оповещение
о связке USDT → SPCXB → MARSCOIN → USDT с маржой +0.363%, все фильтры
пройдены: пулы крупные, оборот есть, маржа скромная, сеть одна.
На деле связка убыточна.

Причина не в наших расчётах. У таких токенов контракт удерживает
процент при покупке или продаже — иногда три, иногда десять, иногда
продать нельзя вовсе. В цене пула этого нет: пул отдаёт ровно то, что
следует из резервов, а налог списывается уже при переводе токена.
Никакой источник котировок про это не сообщает, и посчитать по ценам
правильно — значит посчитать неправильно.

Прочитать это можно только из самого контракта, причём не глядя на код,
а выполнив обмен. Что и делает honeypot.is: симулирует покупку и продажу
на форке сети и сравнивает полученное с обещанным. Разница и есть налог.
Ключ не нужен, ответ приходит за секунду-две.

Дальше налог входит в расчёт наравне с комиссией и проскальзыванием.
Именно входит, а не запрещает: токен с честным одним процентом остаётся
в игре, если спред его перекрывает. Запрет остаётся для случаев, где
торговать нельзя в принципе — honeypot, продажа заблокирована, симуляция
не прошла.

Отдельно про доверие к источнику. Ответ honeypot.is — это симуляция
на текущем состоянии сети, а не приговор навсегда: у многих контрактов
владелец может поменять налог одной транзакцией. Поэтому результат
хранится с отметкой времени и протухает; и поэтому же «налога нет»
никогда не означает «безопасно», а означает «на момент проверки
контракт вёл себя честно».
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import SETTINGS

log = logging.getLogger(__name__)

API = "https://api.honeypot.is/v2/IsHoneypot"

CHAIN_IDS = {"bsc": 56, "eth": 1, "ethereum": 1, "base": 8453,
             "arbitrum": 42161, "polygon": 137, "polygon_pos": 137,
             "avax": 43114, "optimism": 10}

TTL_SEC = 24 * 3600
"""Сколько верить проверке. Сутки — компромисс.

Меньше не нужно: контракт меняют редко. Больше опасно: владелец многих
токенов может включить налог одной транзакцией, и вчерашнее «чисто»
перестанет что-либо значить.
"""

MAX_PLAUSIBLE_TAX_PCT = 50.0
"""Выше этого считаем, что продать нельзя, каким бы числом это ни назвали."""


@dataclass
class TokenRisk:
    """Что симуляция говорит про один токен."""

    address: str
    chain: str = ""
    symbol: str = ""
    buy_pct: float = 0.0
    sell_pct: float = 0.0
    transfer_pct: float = 0.0
    honeypot: bool = False
    tradable: bool = True
    """Симуляция прошла и продать можно. False — торговать нельзя."""
    reason: str = ""
    risk_level: int = 0
    open_source: bool = True
    checked_at: int = 0

    @property
    def round_trip_pct(self) -> float:
        """Во что обходится войти и выйти — сумма обоих налогов."""
        return self.buy_pct + self.sell_pct

    @property
    def stale(self) -> bool:
        return (time.time() - self.checked_at) > TTL_SEC

    def label(self) -> str:
        if not self.tradable:
            return self.reason or "продать нельзя"
        if self.round_trip_pct <= 0:
            return "налога нет"
        return f"налог {self.buy_pct:.1f}% на вход, {self.sell_pct:.1f}% на выход"


def chain_id(chain: str) -> int:
    return CHAIN_IDS.get((chain or "").strip().lower(), 56)


def _f(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def check(address: str, chain: str = "", timeout: int = 20) -> Optional[TokenRisk]:
    """Спрашивает симуляцию про один токен. None — не удалось спросить.

    None и «токен плохой» — разные вещи, и путать их нельзя: недоступный
    источник не повод объявить токен непригодным, иначе первый же сбой
    сети выключит нам половину рынка.
    """
    from .http import HttpError, get_json

    address = (address or "").strip()
    if not address.startswith("0x") or len(address) != 42:
        return None
    chain = chain or SETTINGS.chain

    try:
        payload = get_json(API, params={"address": address,
                                        "chainID": chain_id(chain)},
                           attempts=2, timeout=timeout)
    except HttpError as exc:
        log.debug("проверка контракта %s: %s", address[:10], exc)
        return None
    if not isinstance(payload, dict):
        return None

    token = payload.get("token") or {}
    sim = payload.get("simulationResult") or {}
    hp = payload.get("honeypotResult") or {}
    summary = payload.get("summary") or {}
    code = payload.get("contractCode") or {}

    ok = bool(payload.get("simulationSuccess"))
    honeypot = bool(hp.get("isHoneypot"))
    buy = _f(sim.get("buyTax"))
    sell = _f(sim.get("sellTax"))
    transfer = _f(sim.get("transferTax"))

    reason = str(hp.get("honeypotReason") or "").strip()
    if not ok and not reason:
        reason = str(payload.get("simulationError") or "симуляция не прошла")

    tradable = ok and not honeypot and sell < MAX_PLAUSIBLE_TAX_PCT
    if ok and not honeypot and sell >= MAX_PLAUSIBLE_TAX_PCT and not reason:
        reason = f"налог на продажу {sell:.0f}% — фактически продать нельзя"

    return TokenRisk(
        address=address.lower(),
        chain=chain,
        symbol=str(token.get("symbol") or "").upper(),
        buy_pct=max(0.0, buy),
        sell_pct=max(0.0, sell),
        transfer_pct=max(0.0, transfer),
        honeypot=honeypot,
        tradable=tradable,
        reason=reason,
        risk_level=int(_f(summary.get("riskLevel"))),
        open_source=bool(code.get("openSource", True)),
        checked_at=int(time.time()),
    )


def refresh(addresses: Iterable[str], chain: str = "", budget: int = 40,
            deadline: Optional[float] = None) -> Dict[str, TokenRisk]:
    """Проверяет то, что ещё не проверено или протухло. Пишет в базу.

    Бюджет и срок обязательны по той же причине, что и везде в сборщике:
    внешний источник может отвечать медленно, а прогон ограничен по времени.
    Непроверенное просто останется непроверенным до следующего раза —
    это хуже, чем знать, но лучше, чем сорвать сбор.
    """
    from .store import read_token_risk, write_token_risk

    chain = chain or SETTINGS.chain
    known = read_token_risk(chain)
    fresh: Dict[str, TokenRisk] = {}
    done = 0

    for addr in addresses:
        addr = (addr or "").strip().lower()
        if not addr or addr in fresh:
            continue
        have = known.get(addr)
        if have is not None and not have.stale:
            continue
        if done >= budget:
            break
        if deadline is not None and time.time() >= deadline:
            log.info("проверка контрактов остановлена по времени")
            break
        risk = check(addr, chain)
        done += 1
        if risk is None:
            continue
        fresh[addr] = risk

    if fresh:
        write_token_risk(fresh.values())
        taxed = [r for r in fresh.values() if r.round_trip_pct > 0]
        blocked = [r for r in fresh.values() if not r.tradable]
        log.info("контрактов проверено: %d, с налогом: %d, торговать нельзя: %d",
                 len(fresh), len(taxed), len(blocked))
        for r in blocked[:5]:
            log.warning("%s (%s): %s", r.symbol or r.address[:10],
                        r.address[:10], r.reason or "продать нельзя")
    known.update(fresh)
    return known


# --------------------------------------------------------------------------
# Применение к расчёту
# --------------------------------------------------------------------------


def leg_factor(sell_risk: Optional[TokenRisk],
               buy_risk: Optional[TokenRisk]) -> float:
    """Множитель к курсу плеча из-за налогов на перевод.

    В обмене a -> b токен `a` уходит в пул — это продажа, действует налог
    на продажу. Токен `b` приходит из пула — это покупка, действует налог
    на покупку. Оба списываются самим контрактом, мимо цены.
    """
    factor = 1.0
    if sell_risk is not None and sell_risk.sell_pct > 0:
        factor *= max(0.0, 1.0 - sell_risk.sell_pct / 100.0)
    if buy_risk is not None and buy_risk.buy_pct > 0:
        factor *= max(0.0, 1.0 - buy_risk.buy_pct / 100.0)
    return factor


def by_symbol(risks: Dict[str, TokenRisk],
              addresses: Dict[str, str]) -> Dict[str, TokenRisk]:
    """Перекладывает проверки с адресов на тикеры.

    Сетка курсов оперирует тикерами, а проверка привязана к адресу —
    и правильно привязана: одноимённых подделок в сети сколько угодно,
    и вся суть проверки в том, чтобы говорить о конкретном контракте.
    Сопоставление идёт по справочнику адресов из пулов.
    """
    out: Dict[str, TokenRisk] = {}
    for symbol, addr in addresses.items():
        risk = risks.get((addr or "").strip().lower())
        if risk is not None:
            out[str(symbol).upper()] = risk
    return out


def untradable(risks: Dict[str, TokenRisk]) -> Dict[str, str]:
    """Тикер -> почему с ним нельзя работать."""
    return {k: (r.reason or "продать нельзя")
            for k, r in risks.items() if not r.tradable}
