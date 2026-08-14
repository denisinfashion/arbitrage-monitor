"""Монитор спредов: постоянно смотрит котировки и показывает, где
разница между площадками превышает издержки.

Устроен как набор подключаемых адаптеров с общим интерфейсом. Каждый
адаптер отвечает за свой класс инструментов и возвращает список сигналов
в едином формате, поэтому таблица в интерфейсе одна на всё.

Что работает по-настоящему на бесплатных данных:
  * CexSpread       — один актив на разных крипто-биржах;
  * StablecoinDepeg — отклонение стейблкоина от номинала;
  * PerpBasis       — базис между спотом и бессрочным фьючерсом;
  * DexCexSpread    — цена в пуле BNB Chain против цены на бирже.

Что НЕ работает и почему (адаптеры помечены как индикативные):
  * акции  — котировки в разрезе площадок это лицензируемые данные L1/L2.
    Бесплатные источники отдают ОДИН консолидированный фид с задержкой
    15 минут. Консолидированный фид по построению не может показать спред
    между площадками: он и есть усреднение по ним;
  * металлы — реальное время по COMEX/LBMA платное, бесплатно доступен
    один агрегированный спот с задержкой.

Адаптеры для них оставлены заглушками с рабочей структурой: если появится
платный фид, достаточно подставить его в fetch() — остальной код готов.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .config import (CEX_TAKER_PCT, SETTINGS, USD_LIKE, is_leveraged_token,
                     norm_asset)

log = logging.getLogger(__name__)

try:
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover
    ccxt = None


@dataclass
class Signal:
    """Один найденный спред."""

    instrument: str          # 'BTC/USDT'
    kind: str                # 'крипта' | 'стейбл' | 'базис' | 'акции' | 'металлы'
    buy_at: str
    sell_at: str
    buy_price: float
    sell_price: float
    gross_bps: float         # спред до издержек
    fees_bps: float          # суммарные комиссии обеих ног
    net_bps: float           # что остаётся
    tradable: bool = True    # False для индикативных источников
    note: str = ""
    ts: float = field(default_factory=time.time)

    def as_row(self) -> dict:
        return {
            "Инструмент": self.instrument,
            "Класс": self.kind,
            "Купить на": self.buy_at,
            "Продать на": self.sell_at,
            "Цена покупки": self.buy_price,
            "Цена продажи": self.sell_price,
            "Спред, б.п.": round(self.gross_bps, 1),
            "Комиссии, б.п.": round(self.fees_bps, 1),
            "Чистый спред, б.п.": round(self.net_bps, 1),
            "Чистый спред, %": round(self.net_bps / 100, 3),
            "Исполнимо": "да" if self.tradable else "нет — индикативно",
            "Примечание": self.note,
        }


# --------------------------------------------------------------------------
# Базовый адаптер
# --------------------------------------------------------------------------


class Adapter:
    name = "adapter"
    kind = "крипта"
    tradable = True

    def scan(self) -> List[Signal]:  # pragma: no cover - интерфейс
        raise NotImplementedError


# --------------------------------------------------------------------------
# Крипта: один актив на разных биржах
# --------------------------------------------------------------------------


class CexSpread(Adapter):
    """Классический межбиржевой спред: где купить дешевле, где продать дороже.

    Считает по последней сделке, а не по стакану, поэтому это ОРИЕНТИР.
    Прежде чем торговать, объём надо проверить по стакану — для этого
    в проекте есть основной калькулятор с реальными книгами заявок.
    """

    name = "Крипто-биржи"
    kind = "крипта"

    def __init__(self, venues: Optional[Sequence[str]] = None,
                 quote: str = "USDT", top_symbols: int = 60,
                 spot_only: Optional[bool] = None):
        if ccxt is None:
            raise RuntimeError("ccxt не установлен")
        self.venues = list(venues or SETTINGS.cex_venues)
        self.quote = quote.upper()
        self.top_symbols = top_symbols
        self.spot_only = SETTINGS.spot_only if spot_only is None else spot_only
        self._ex: Dict[str, object] = {}

    def _exchange(self, vid: str):
        if vid not in self._ex:
            self._ex[vid] = getattr(ccxt, vid)({"enableRateLimit": True, "timeout": 15000})
        return self._ex[vid]

    def _tickers(self, vid: str) -> Dict[str, float]:
        try:
            ex = self._exchange(vid)
            data = ex.fetch_tickers()
        except Exception as exc:
            log.debug("%s: тикеры недоступны (%s)", vid, exc)
            return {}
        out = {}
        for sym, t in data.items():
            if not sym.endswith("/" + self.quote):
                continue
            price = t.get("last") or t.get("close")
            if price and price > 0:
                out[sym] = float(price)
        return out

    def scan(self) -> List[Signal]:
        snaps: Dict[str, Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(self.venues))) as pool:
            futs = {pool.submit(self._tickers, v): v for v in self.venues}
            for f in as_completed(futs):
                v = futs[f]
                try:
                    snaps[v] = f.result()
                except Exception as exc:
                    log.debug("%s: %s", v, exc)

        snaps = {k: v for k, v in snaps.items() if v}
        if len(snaps) < 2:
            return []

        # символы, встречающиеся минимум на двух площадках
        counts: Dict[str, int] = {}
        for prices in snaps.values():
            for s in prices:
                counts[s] = counts.get(s, 0) + 1
        symbols = [s for s, c in counts.items() if c >= 2]
        if self.spot_only:
            known = {s.split("/")[0].upper() for s in counts}
            symbols = [s for s in symbols
                       if not is_leveraged_token(s.split("/")[0].upper(), known)]
        symbols = symbols[: self.top_symbols * 4]

        out: List[Signal] = []
        for sym in symbols:
            quotes = [(v, p[sym]) for v, p in snaps.items() if sym in p]
            if len(quotes) < 2:
                continue
            buy_v, buy_p = min(quotes, key=lambda x: x[1])
            sell_v, sell_p = max(quotes, key=lambda x: x[1])
            if buy_p <= 0 or buy_v == sell_v:
                continue
            gross = (sell_p / buy_p - 1) * 1e4
            fees = (CEX_TAKER_PCT.get(buy_v, 0.10) + CEX_TAKER_PCT.get(sell_v, 0.10)) * 100
            out.append(Signal(
                instrument=sym, kind=self.kind,
                buy_at=buy_v, sell_at=sell_v,
                buy_price=buy_p, sell_price=sell_p,
                gross_bps=gross, fees_bps=fees, net_bps=gross - fees,
                note="цена последней сделки, объём не проверен по стакану",
            ))
        return out


# --------------------------------------------------------------------------
# Стейблкоины
# --------------------------------------------------------------------------


class StablecoinDepeg(Adapter):
    """Отклонение стейблкоина от номинала.

    Самый надёжный из бесплатных сигналов: обе стороны — стейблы,
    сетевые издержки известны, риск цены минимален.
    """

    name = "Депег стейблкоинов"
    kind = "стейбл"

    def __init__(self, venues: Optional[Sequence[str]] = None):
        if ccxt is None:
            raise RuntimeError("ccxt не установлен")
        self.venues = list(venues or ["binance", "okx", "kucoin", "gate"])
        self.pairs = ["USDC/USDT", "DAI/USDT", "FDUSD/USDT", "TUSD/USDT", "BUSD/USDT"]
        self._ex: Dict[str, object] = {}

    def scan(self) -> List[Signal]:
        out: List[Signal] = []
        for vid in self.venues:
            try:
                if vid not in self._ex:
                    self._ex[vid] = getattr(ccxt, vid)({"enableRateLimit": True, "timeout": 15000})
                tickers = self._ex[vid].fetch_tickers()
            except Exception as exc:
                log.debug("%s: %s", vid, exc)
                continue
            fee_bps = CEX_TAKER_PCT.get(vid, 0.10) * 100 * 2
            for pair in self.pairs:
                t = tickers.get(pair)
                if not t:
                    continue
                price = t.get("last") or t.get("close")
                if not price or price <= 0:
                    continue
                dev = (price - 1.0) * 1e4
                if abs(dev) < 1:
                    continue
                if dev > 0:      # стейбл дороже USDT: продаём его
                    buy_at, sell_at = f"{vid} (USDT)", f"{vid} ({pair.split('/')[0]})"
                else:
                    buy_at, sell_at = f"{vid} ({pair.split('/')[0]})", f"{vid} (USDT)"
                out.append(Signal(
                    instrument=pair, kind=self.kind,
                    buy_at=buy_at, sell_at=sell_at,
                    buy_price=min(price, 1.0), sell_price=max(price, 1.0),
                    gross_bps=abs(dev), fees_bps=fee_bps, net_bps=abs(dev) - fee_bps,
                    note="возврат к номиналу не гарантирован по времени",
                ))
        return out


# --------------------------------------------------------------------------
# Базис спот - бессрочный фьючерс
# --------------------------------------------------------------------------


class PerpBasis(Adapter):
    """Разница между спотом и бессрочным фьючерсом на одной бирже.

    Позиция дельта-нейтральна: покупаем спот, шортим перп. Доход —
    базис плюс фандинг. Данные бесплатные и полные.
    """

    name = "Базис спот-перп"
    kind = "базис"

    def __init__(self, venue: str = "binance", symbols: Optional[Sequence[str]] = None):
        if ccxt is None:
            raise RuntimeError("ccxt не установлен")
        self.venue = venue
        self.symbols = list(symbols or ["BTC", "ETH", "BNB", "SOL", "XRP"])

    def scan(self) -> List[Signal]:
        try:
            spot = getattr(ccxt, self.venue)({"enableRateLimit": True, "timeout": 15000})
            swap = getattr(ccxt, self.venue)({"enableRateLimit": True, "timeout": 15000,
                                              "options": {"defaultType": "swap"}})
            spot_t = spot.fetch_tickers()
            swap_t = swap.fetch_tickers()
        except Exception as exc:
            log.debug("базис %s: %s", self.venue, exc)
            return []

        fee_bps = CEX_TAKER_PCT.get(self.venue, 0.10) * 100 * 2
        out: List[Signal] = []
        for sym in self.symbols:
            s = spot_t.get(f"{sym}/USDT") or {}
            p = swap_t.get(f"{sym}/USDT:USDT") or {}
            sp = s.get("last") or s.get("close")
            pp = p.get("last") or p.get("close")
            if not sp or not pp or sp <= 0:
                continue
            basis = (pp / sp - 1) * 1e4
            if abs(basis) < 1:
                continue
            out.append(Signal(
                instrument=f"{sym} спот/перп", kind=self.kind,
                buy_at=f"{self.venue} спот" if basis > 0 else f"{self.venue} перп",
                sell_at=f"{self.venue} перп" if basis > 0 else f"{self.venue} спот",
                buy_price=min(sp, pp), sell_price=max(sp, pp),
                gross_bps=abs(basis), fees_bps=fee_bps, net_bps=abs(basis) - fee_bps,
                note="доход дополняется фандингом; нужна маржа под шорт",
            ))
        return out


# --------------------------------------------------------------------------
# DEX против CEX
# --------------------------------------------------------------------------


class DexCexSpread(Adapter):
    """Цена токена в пуле BNB Chain против цены на бирже.

    Использует уже существующий модуль dexes.py: он умеет брать котировку
    прямо из смарт-контракта PancakeSwap, то есть цену с учётом реального
    размера свопа, а не справочный спот.
    """

    name = "DEX против CEX"
    kind = "крипта"

    def __init__(self, tokens: Optional[Sequence[str]] = None,
                 chain: str = "BNB Chain", venue_dex: str = "PancakeSwap",
                 cex: str = "binance", size_usd: float = 1000.0):
        self.tokens = list(tokens or ["BNB", "CAKE", "ETH", "BTCB", "XRP"])
        self.chain = chain
        self.venue_dex = venue_dex
        self.cex = cex
        self.size_usd = size_usd

    def scan(self) -> List[Signal]:
        try:
            import dexes  # модуль исходного проекта
        except ImportError:
            log.debug("dexes.py недоступен")
            return []
        if ccxt is None:
            return []

        try:
            ex = getattr(ccxt, self.cex)({"enableRateLimit": True, "timeout": 15000})
            tickers = ex.fetch_tickers()
        except Exception as exc:
            log.debug("%s: %s", self.cex, exc)
            return []

        dex_fee_bps = 25.0
        cex_fee_bps = CEX_TAKER_PCT.get(self.cex, 0.10) * 100
        out: List[Signal] = []

        for tok in self.tokens:
            cex_sym = f"{norm_asset(tok)}/USDT"
            t = tickers.get(cex_sym) or {}
            cex_price = t.get("last") or t.get("close")
            if not cex_price or cex_price <= 0:
                continue
            try:
                q = dexes.dex_quote(self.venue_dex, self.chain, "USDT", tok, self.size_usd)
                if not q.ok or not q.amount_out:
                    continue
                dex_price = self.size_usd / q.amount_out   # USDT за 1 токен
                gas_usd = q.gas_usd or 0.0
            except Exception as exc:
                log.debug("DEX %s: %s", tok, exc)
                continue

            gross = abs(cex_price / dex_price - 1) * 1e4
            gas_bps = (gas_usd / self.size_usd) * 1e4
            fees = dex_fee_bps + cex_fee_bps + gas_bps
            cheap_dex = dex_price < cex_price
            out.append(Signal(
                instrument=f"{tok}/USDT", kind=self.kind,
                buy_at=self.venue_dex if cheap_dex else self.cex,
                sell_at=self.cex if cheap_dex else self.venue_dex,
                buy_price=min(dex_price, cex_price),
                sell_price=max(dex_price, cex_price),
                gross_bps=gross, fees_bps=fees, net_bps=gross - fees,
                note=f"газ ≈ ${gas_usd:.2f}; нужен вывод токена с биржи в сеть",
            ))
        return out


# --------------------------------------------------------------------------
# Индикативные адаптеры: акции и металлы
# --------------------------------------------------------------------------


class DelayedInstrument(Adapter):
    """Каркас для инструментов, по которым бесплатных биржевых данных нет.

    Возвращает одну поясняющую строку вместо выдуманных сигналов.
    Чтобы включить по-настоящему, нужен платный фид с котировками
    В РАЗРЕЗЕ ПЛОЩАДОК — тогда достаточно реализовать fetch_by_venue()
    и снять флаг tradable.
    """

    tradable = False

    def __init__(self, name: str, kind: str, reason: str, what_is_needed: str):
        self.name = name
        self.kind = kind
        self.reason = reason
        self.what_is_needed = what_is_needed

    def fetch_by_venue(self) -> Dict[str, Dict[str, float]]:
        """Подставьте сюда платный источник: {инструмент: {площадка: цена}}."""
        return {}

    def scan(self) -> List[Signal]:
        data = self.fetch_by_venue()
        if not data:
            return [Signal(
                instrument=f"({self.kind})", kind=self.kind,
                buy_at="—", sell_at="—", buy_price=0.0, sell_price=0.0,
                gross_bps=0.0, fees_bps=0.0, net_bps=0.0,
                tradable=False,
                note=f"{self.reason} Нужно: {self.what_is_needed}",
            )]
        out: List[Signal] = []
        for inst, by_venue in data.items():
            if len(by_venue) < 2:
                continue
            buy_v, buy_p = min(by_venue.items(), key=lambda x: x[1])
            sell_v, sell_p = max(by_venue.items(), key=lambda x: x[1])
            gross = (sell_p / buy_p - 1) * 1e4 if buy_p else 0.0
            out.append(Signal(
                instrument=inst, kind=self.kind,
                buy_at=buy_v, sell_at=sell_v, buy_price=buy_p, sell_price=sell_p,
                gross_bps=gross, fees_bps=0.0, net_bps=gross,
                tradable=self.tradable, note="комиссии не заданы",
            ))
        return out


def equities_adapter() -> DelayedInstrument:
    return DelayedInstrument(
        name="Акции",
        kind="акции",
        reason=("Межбиржевой спред по акциям на бесплатных данных не считается: "
                "публичные источники отдают один консолидированный фид "
                "с задержкой 15 минут, а он и есть усреднение по площадкам."),
        what_is_needed="подписка на L1-данные конкретных площадок (NYSE, Nasdaq, Cboe)",
    )


def metals_adapter() -> DelayedInstrument:
    return DelayedInstrument(
        name="Металлы",
        kind="металлы",
        reason=("Реальное время по COMEX и LBMA распространяется по платной "
                "подписке; бесплатно доступен один агрегированный спот с задержкой."),
        what_is_needed="подписка на биржевые данные COMEX/LBMA",
    )


# --------------------------------------------------------------------------
# Сводный запуск
# --------------------------------------------------------------------------


def default_adapters(include_indicative: bool = True) -> List[Adapter]:
    out: List[Adapter] = []
    for factory in (CexSpread, StablecoinDepeg, PerpBasis, DexCexSpread):
        try:
            out.append(factory())
        except Exception as exc:
            log.warning("адаптер %s не создан: %s", factory.__name__, exc)
    if include_indicative:
        out += [equities_adapter(), metals_adapter()]
    return out


def scan_all(adapters: Optional[Sequence[Adapter]] = None,
             min_net_bps: float = 0.0,
             top: int = 100) -> pd.DataFrame:
    """Опрашивает все адаптеры параллельно и собирает единую таблицу."""
    adapters = list(adapters or default_adapters())
    signals: List[Signal] = []

    with ThreadPoolExecutor(max_workers=min(6, len(adapters))) as pool:
        futs = {pool.submit(a.scan): a for a in adapters}
        for f in as_completed(futs):
            a = futs[f]
            try:
                signals.extend(f.result() or [])
            except Exception as exc:
                log.warning("адаптер %s: %s", a.name, exc)

    if not signals:
        return pd.DataFrame()

    df = pd.DataFrame([s.as_row() for s in signals])
    tradable = df["Исполнимо"] == "да"
    keep = (~tradable) | (df["Чистый спред, б.п."] >= min_net_bps)
    df = df[keep]
    return df.sort_values(["Исполнимо", "Чистый спред, б.п."],
                          ascending=[True, False]).head(top).reset_index(drop=True)
