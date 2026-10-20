"""Ставки финансирования (funding) бессрочных фьючерсов по всем рынкам биржи.

Для арбитража важна не только ставка, но и интервал расчёта (1, 2, 4, 8 ч —
у бирж и даже у монет разный) и время ближайшего расчёта. Единого пакетного
запроса с этими полями в ccxt нет у половины бирж, поэтому где нужно —
прямые публичные запросы.

Ответ: id рынка биржи → Funding(rate, interval_h, next_ts).
rate — доля за один расчёт (0.0001 = 0,01%); плюс — лонг платит шорту.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import requests

TIMEOUT = 30


@dataclass(frozen=True)
class Funding:
    rate: float
    interval_h: float
    next_ts: Optional[int]   # мс UTC


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _get(url: str, **params):
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def binance(ex) -> Dict[str, Funding]:
    # интервал — только для монет, где он отличается от 8 ч
    intervals = {x["symbol"]: float(x["fundingIntervalHours"])
                 for x in _get("https://fapi.binance.com/fapi/v1/fundingInfo")}
    out = {}
    for x in _get("https://fapi.binance.com/fapi/v1/premiumIndex"):
        rate = _f(x.get("lastFundingRate"))
        if rate is not None:
            out[x["symbol"]] = Funding(rate, intervals.get(x["symbol"], 8.0),
                                       int(x["nextFundingTime"]) or None)
    return out


def bitget(ex) -> Dict[str, Funding]:
    data = _get("https://api.bitget.com/api/v2/mix/market/current-fund-rate",
                productType="USDT-FUTURES")["data"]
    return {x["symbol"]: Funding(float(x["fundingRate"]), float(x["fundingRateInterval"]),
                                 int(x["nextUpdate"]) or None)
            for x in data if _f(x.get("fundingRate")) is not None}


def mexc(ex) -> Dict[str, Funding]:
    data = _get("https://contract.mexc.com/api/v1/contract/funding_rate")["data"]
    return {x["symbol"]: Funding(float(x["fundingRate"]), float(x["collectCycle"]),
                                 int(x["nextSettleTime"]) or None)
            for x in data if _f(x.get("fundingRate")) is not None}


def kucoin(ex) -> Dict[str, Funding]:
    data = _get("https://api-futures.kucoin.com/api/v1/contracts/active")["data"]
    out = {}
    for x in data:
        rate, gran = _f(x.get("fundingFeeRate")), _f(x.get("fundingRateGranularity"))
        if rate is not None and gran:
            out[x["symbol"]] = Funding(rate, gran / 3_600_000, x.get("nextFundingRateDateTime"))
    return out


def _interval_h(text) -> Optional[float]:
    if not text:
        return None
    s = str(text).strip().lower()
    try:
        return float(s[:-1]) if s.endswith("h") else float(s)
    except ValueError:
        return None


def via_ccxt(ex) -> Dict[str, Funding]:
    """Bybit, OKX, Gate, BingX: ccxt fetch_funding_rates с интервалом
    (у BingX — в сыром поле fundingIntervalHours)."""
    out = {}
    for sym, x in ex.fetch_funding_rates().items():
        rate = _f(x.get("fundingRate"))
        interval = _interval_h(x.get("interval")) or _f((x.get("info") or {}).get("fundingIntervalHours"))
        if rate is None or not interval:  # «0h» у Bybit — рынок без funding
            continue
        market = ex.markets.get(sym)
        if market is None:  # рынок вне загруженного справочника (не USDT-бессрочный)
            continue
        nxt = x.get("nextFundingTimestamp") or x.get("fundingTimestamp")
        out[market["id"]] = Funding(rate, float(interval), int(nxt) if nxt else None)
    return out


FETCHERS: Dict[str, Callable] = {
    "binance": binance, "bitget": bitget, "mexc": mexc, "kucoin": kucoin,
    "bybit": via_ccxt, "okx": via_ccxt, "gate": via_ccxt, "bingx": via_ccxt,
}


def settlements(next_ts_ms: Optional[int], interval_h: float, now_s: float, hold_h: float) -> int:
    """Сколько расчётов funding попадёт в (now, now + hold]."""
    if not interval_h or interval_h <= 0:
        return 0
    step = interval_h * 3600
    first = next_ts_ms / 1000 if next_ts_ms else None
    end = now_s + hold_h * 3600
    if first is None:                       # время неизвестно — в среднем
        return int(hold_h // interval_h)
    while first <= now_s:                   # ответ мог устареть
        first += step
    if first > end:
        return 0
    return int((end - first) // step) + 1
