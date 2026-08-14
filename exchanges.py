"""Загрузка реальных стаканов через публичные REST API бирж.

Все эндпоинты публичные, ключи API не требуются. Каждый адаптер приводит ответ
к единому формату pandas.DataFrame с колонками level / price / size,
отсортированному «от лучшей цены» (ask по возрастанию, bid по убыванию).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

TIMEOUT = 10
HEADERS = {"User-Agent": "arb-calculator/1.0"}


# --------------------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------------------

def _get(url: str, params: Optional[dict] = None) -> dict:
    r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def _to_book(levels: List[Tuple[str, str]], side: str) -> pd.DataFrame:
    rows = [(float(p), float(s)) for p, s, *_ in levels if float(s) > 0]
    if not rows:
        raise ValueError("пустой стакан")
    rows.sort(key=lambda x: x[0], reverse=(side == "bid"))
    df = pd.DataFrame(rows, columns=["price", "size"])
    df.insert(0, "level", np.arange(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------------------
# Описание бирж
# --------------------------------------------------------------------------------------

@dataclass
class Exchange:
    name: str
    symbol_fmt: str                 # шаблон тикера: {b} база, {q} котировка
    fetch: Callable[[str, int], Tuple[pd.DataFrame, pd.DataFrame]]
    maker_pct: float                # базовая комиссия спота (без VIP-скидок)
    taker_pct: float
    upper: bool = True

    def symbol(self, base: str, quote: str) -> str:
        s = self.symbol_fmt.format(b=base, q=quote)
        return s.upper() if self.upper else s.lower()


# --- Адаптеры -------------------------------------------------------------------------

def _binance(symbol: str, limit: int):
    # data-api.binance.vision — публичное зеркало рыночных данных Binance
    d = _get("https://data-api.binance.vision/api/v3/depth",
             {"symbol": symbol, "limit": min(limit, 5000)})
    return _to_book(d["asks"], "ask"), _to_book(d["bids"], "bid")


def _okx(symbol: str, limit: int):
    d = _get("https://www.okx.com/api/v5/market/books",
             {"instId": symbol, "sz": min(limit, 400)})
    t = d["data"][0]
    return _to_book(t["asks"], "ask"), _to_book(t["bids"], "bid")


def _bybit(symbol: str, limit: int):
    d = _get("https://api.bybit.com/v5/market/orderbook",
             {"category": "spot", "symbol": symbol, "limit": min(limit, 200)})
    t = d["result"]
    return _to_book(t["a"], "ask"), _to_book(t["b"], "bid")


def _kraken(symbol: str, limit: int):
    d = _get("https://api.kraken.com/0/public/Depth",
             {"pair": symbol, "count": min(limit, 500)})
    if d.get("error"):
        raise ValueError("; ".join(d["error"]))
    t = next(iter(d["result"].values()))
    return _to_book(t["asks"], "ask"), _to_book(t["bids"], "bid")


def _kucoin(symbol: str, limit: int):
    part = "level2_100" if limit > 20 else "level2_20"
    d = _get(f"https://api.kucoin.com/api/v1/market/orderbook/{part}", {"symbol": symbol})
    t = d["data"]
    return _to_book(t["asks"], "ask"), _to_book(t["bids"], "bid")


def _gate(symbol: str, limit: int):
    d = _get("https://api.gateio.ws/api/v4/spot/order_book",
             {"currency_pair": symbol, "limit": min(limit, 100)})
    return _to_book(d["asks"], "ask"), _to_book(d["bids"], "bid")


def _mexc(symbol: str, limit: int):
    d = _get("https://api.mexc.com/api/v3/depth",
             {"symbol": symbol, "limit": min(limit, 5000)})
    return _to_book(d["asks"], "ask"), _to_book(d["bids"], "bid")


def _bitget(symbol: str, limit: int):
    d = _get("https://api.bitget.com/api/v2/spot/market/orderbook",
             {"symbol": symbol, "limit": min(limit, 150)})
    t = d["data"]
    return _to_book(t["asks"], "ask"), _to_book(t["bids"], "bid")


def _htx(symbol: str, limit: int):
    d = _get("https://api.huobi.pro/market/depth",
             {"symbol": symbol, "type": "step0", "depth": 20})
    t = d["tick"]
    ask = [(str(p), str(s)) for p, s in t["asks"]]
    bid = [(str(p), str(s)) for p, s in t["bids"]]
    return _to_book(ask, "ask"), _to_book(bid, "bid")


def _coinbase(symbol: str, limit: int):
    d = _get(f"https://api.exchange.coinbase.com/products/{symbol}/book", {"level": 2})
    return _to_book(d["asks"], "ask"), _to_book(d["bids"], "bid")


def _bingx(symbol: str, limit: int):
    d = _get("https://open-api.bingx.com/openApi/spot/v1/market/depth",
             {"symbol": symbol, "limit": min(limit, 100)})
    t = d["data"]
    return _to_book(t["asks"], "ask"), _to_book(t["bids"], "bid")


def _cryptocom(symbol: str, limit: int):
    d = _get("https://api.crypto.com/exchange/v1/public/get-book",
             {"instrument_name": symbol, "depth": min(limit, 50)})
    t = d["result"]["data"][0]
    return _to_book(t["asks"], "ask"), _to_book(t["bids"], "bid")


EXCHANGES: Dict[str, Exchange] = {
    "Binance": Exchange("Binance", "{b}{q}", _binance, 0.10, 0.10),
    "OKX": Exchange("OKX", "{b}-{q}", _okx, 0.08, 0.10),
    "Bybit": Exchange("Bybit", "{b}{q}", _bybit, 0.10, 0.10),
    "Kraken": Exchange("Kraken", "{b}{q}", _kraken, 0.25, 0.40),
    "KuCoin": Exchange("KuCoin", "{b}-{q}", _kucoin, 0.10, 0.10),
    "Gate.io": Exchange("Gate.io", "{b}_{q}", _gate, 0.09, 0.09),
    "MEXC": Exchange("MEXC", "{b}{q}", _mexc, 0.00, 0.05),
    "Bitget": Exchange("Bitget", "{b}{q}", _bitget, 0.10, 0.10),
    "HTX": Exchange("HTX", "{b}{q}", _htx, 0.20, 0.20, upper=False),
    "Coinbase": Exchange("Coinbase", "{b}-{q}", _coinbase, 0.40, 0.60),
    "BingX": Exchange("BingX", "{b}-{q}", _bingx, 0.10, 0.10),
    "Crypto.com": Exchange("Crypto.com", "{b}_{q}", _cryptocom, 0.25, 0.50),
}

# Kraken использует нестандартные тикеры для части монет
KRAKEN_ALIASES = {"BTC": "XBT", "DOGE": "XDG"}


def resolve_symbol(ex_name: str, base: str, quote: str) -> str:
    ex = EXCHANGES[ex_name]
    b, q = base.upper(), quote.upper()
    if ex_name == "Kraken":
        b = KRAKEN_ALIASES.get(b, b)
        q = KRAKEN_ALIASES.get(q, q)
    return ex.symbol(b, q)


# --------------------------------------------------------------------------------------
# Публичный интерфейс
# --------------------------------------------------------------------------------------

@dataclass
class BookSnapshot:
    exchange: str
    symbol: str
    ask: Optional[pd.DataFrame]
    bid: Optional[pd.DataFrame]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.ask is not None and self.bid is not None

    @property
    def best_ask(self) -> Optional[float]:
        return float(self.ask["price"].iloc[0]) if self.ok else None

    @property
    def best_bid(self) -> Optional[float]:
        return float(self.bid["price"].iloc[0]) if self.ok else None


def fetch_book(ex_name: str, base: str, quote: str, limit: int = 100) -> BookSnapshot:
    """Стакан одной биржи. Ошибки не выбрасываются, а возвращаются в поле error."""
    symbol = resolve_symbol(ex_name, base, quote)
    try:
        ask, bid = EXCHANGES[ex_name].fetch(symbol, limit)
        return BookSnapshot(ex_name, symbol, ask, bid)
    except Exception as exc:  # сеть, гео-блокировка, отсутствующая пара
        msg = str(exc)
        if "451" in msg:
            msg = "недоступно из текущего региона (HTTP 451)"
        elif "403" in msg:
            msg = "доступ заблокирован (HTTP 403)"
        elif "400" in msg or "404" in msg:
            msg = f"пара {symbol} не найдена"
        return BookSnapshot(ex_name, symbol, None, None, msg[:160])


def fetch_many(
    ex_names: List[str], base: str, quote: str, limit: int = 100
) -> Dict[str, BookSnapshot]:
    """Параллельная загрузка стаканов с нескольких бирж."""
    out: Dict[str, BookSnapshot] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(ex_names)))) as pool:
        futures = {pool.submit(fetch_book, n, base, quote, limit): n for n in ex_names}
        for f in as_completed(futures):
            snap = f.result()
            out[snap.exchange] = snap
    return {n: out[n] for n in ex_names if n in out}


def spread_matrix(snaps: Dict[str, BookSnapshot]) -> pd.DataFrame:
    """Матрица спредов: строка — где покупаем (ask), столбец — где продаём (bid), б.п."""
    ok = {n: s for n, s in snaps.items() if s.ok}
    names = list(ok)
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                m.loc[a, b] = np.nan
            else:
                m.loc[a, b] = (ok[b].best_bid / ok[a].best_ask - 1.0) * 10_000.0
    return m


def best_pairs(snaps: Dict[str, BookSnapshot], top: int = 10) -> pd.DataFrame:
    """Топ связок «купить на A / продать на B» по грязному спреду."""
    ok = {n: s for n, s in snaps.items() if s.ok}
    rows = []
    for a, sa in ok.items():
        for b, sb in ok.items():
            if a == b:
                continue
            gross = (sb.best_bid / sa.best_ask - 1.0) * 10_000.0
            fees = EXCHANGES[a].taker_pct * 100 + EXCHANGES[b].taker_pct * 100  # в б.п.
            rows.append(
                {
                    "Купить на": a,
                    "Продать на": b,
                    "Ask": sa.best_ask,
                    "Bid": sb.best_bid,
                    "Спред, б.п.": gross,
                    "Комиссии тейкера, б.п.": fees,
                    "Спред за вычетом комиссий, б.п.": gross - fees,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Спред за вычетом комиссий, б.п.", ascending=False).head(top)


# --------------------------------------------------------------------------------------
# Треугольный арбитраж
# --------------------------------------------------------------------------------------

QUOTE_PRIORITY = {"USDT": 0, "USDC": 1, "USD": 2, "EUR": 3, "BTC": 4, "ETH": 5, "BNB": 6}


def _preferred_orientation(asset_a: str, asset_b: str) -> Tuple[str, str]:
    """Выбирает наиболее распространённую ориентацию пары base/quote."""
    rank_a = QUOTE_PRIORITY.get(asset_a, 100)
    rank_b = QUOTE_PRIORITY.get(asset_b, 100)
    return (asset_a, asset_b) if rank_a > rank_b else (asset_b, asset_a)


def _fetch_supported_pair(ex_name: str, asset_a: str, asset_b: str, limit: int) -> Tuple[str, str, BookSnapshot]:
    base, quote = _preferred_orientation(asset_a, asset_b)
    snapshot = fetch_book(ex_name, base, quote, limit)
    if snapshot.ok:
        return base, quote, snapshot

    reverse = fetch_book(ex_name, quote, base, limit)
    return (quote, base, reverse) if reverse.ok else (base, quote, snapshot)


def fetch_triangle_books(
    ex_names: List[str], assets: List[str], limit: int = 20
) -> Tuple[Dict[str, Dict[frozenset, Tuple[str, str, BookSnapshot]]], Dict[str, Tuple[int, int]]]:
    """Загружает все нужные пары для треугольников на выбранных биржах.

    Значение статуса — число доступных пар и число запрошенных пар на биржу.
    """
    unique_assets = list(dict.fromkeys(asset.upper() for asset in assets if asset.strip()))
    asset_pairs = list(combinations(unique_assets, 2))
    books: Dict[str, Dict[frozenset, Tuple[str, str, BookSnapshot]]] = {name: {} for name in ex_names}
    status = {name: [0, len(asset_pairs)] for name in ex_names}
    if len(unique_assets) < 3:
        return books, {name: tuple(value) for name, value in status.items()}

    jobs = [(name, asset_a, asset_b) for name in ex_names for asset_a, asset_b in asset_pairs]
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(jobs)))) as pool:
        futures = {
            pool.submit(_fetch_supported_pair, name, asset_a, asset_b, limit): (name, asset_a, asset_b)
            for name, asset_a, asset_b in jobs
        }
        for future in as_completed(futures):
            name, asset_a, asset_b = futures[future]
            base, quote, snapshot = future.result()
            books[name][frozenset((asset_a, asset_b))] = (base, quote, snapshot)
            if snapshot.ok:
                status[name][0] += 1

    return books, {name: tuple(value) for name, value in status.items()}


def _convert_at_top(
    amount: float,
    source: str,
    target: str,
    pair: Tuple[str, str, BookSnapshot],
    fee_pct: float,
) -> Optional[Tuple[float, str, float]]:
    """Конвертирует сумму через верхний уровень стакана с комиссией тейкера."""
    base, quote, snapshot = pair
    if not snapshot.ok:
        return None
    fee_factor = 1.0 - fee_pct / 100.0
    if source == base and target == quote:
        return amount * snapshot.best_bid * fee_factor, f"sell {base}/{quote}", snapshot.best_bid
    if source == quote and target == base:
        return amount / snapshot.best_ask * fee_factor, f"buy {base}/{quote}", snapshot.best_ask
    return None


def best_triangles(
    books: Dict[str, Dict[frozenset, Tuple[str, str, BookSnapshot]]],
    start_asset: str,
    start_amount: float,
    top: int = 50,
) -> pd.DataFrame:
    """Ищет циклы start → asset A → asset B → start на каждой бирже.

    Расчёт основан на лучшей цене и тейкерских комиссиях. Глубина стакана и
    перевод между биржами не учитываются, потому что все три сделки совершаются
    на одной площадке.
    """
    start = start_asset.upper()
    rows = []
    for ex_name, pair_books in books.items():
        assets = set().union(*pair_books) if pair_books else set()
        other_assets = sorted(assets - {start})
        fee_pct = EXCHANGES[ex_name].taker_pct
        for first, second in permutations(other_assets, 2):
            route = (start, first, second, start)
            pairs = [pair_books.get(frozenset((route[index], route[index + 1]))) for index in range(3)]
            if any(pair is None or not pair[2].ok for pair in pairs):
                continue

            net_amount = start_amount
            gross_amount = start_amount
            actions = []
            quotes = []
            complete = True
            for index, pair in enumerate(pairs):
                net = _convert_at_top(net_amount, route[index], route[index + 1], pair, fee_pct)
                gross = _convert_at_top(gross_amount, route[index], route[index + 1], pair, 0.0)
                if net is None or gross is None:
                    complete = False
                    break
                amount_before = net_amount
                net_amount, action, price = net
                gross_amount = gross[0]
                actions.append(action)
                quotes.append(f"{amount_before:.8g} {route[index]} → {net_amount:.8g} {route[index + 1]} @ {price:.8g}")
            if not complete:
                continue

            gross_bps = (gross_amount / start_amount - 1.0) * 10_000.0
            net_bps = (net_amount / start_amount - 1.0) * 10_000.0
            rows.append(
                {
                    "Биржа": ex_name,
                    "Маршрут": " → ".join(route),
                    "Нога 1": actions[0],
                    "Котировка 1": quotes[0],
                    "Нога 2": actions[1],
                    "Котировка 2": quotes[1],
                    "Нога 3": actions[2],
                    "Котировка 3": quotes[2],
                    f"Старт, {start}": start_amount,
                    f"Финиш, {start}": net_amount,
                    f"Прибыль, {start}": net_amount - start_amount,
                    "Грязный спред, б.п.": gross_bps,
                    "Комиссии тейкера, б.п.": gross_bps - net_bps,
                    "Чистый спред, б.п.": net_bps,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("Чистый спред, б.п.", ascending=False).head(top).reset_index(drop=True)
