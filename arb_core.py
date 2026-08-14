"""Ядро расчётов межбиржевого арбитража: стакан, проскальзывание, комиссии, безубыточность."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Модель стакана
# --------------------------------------------------------------------------------------

def synthetic_book(
    best_price: float,
    side: str,
    total_depth: float,
    levels: int = 25,
    step_bps: float = 2.0,
    shape: float = 1.0,
) -> pd.DataFrame:
    """Синтетический стакан вокруг лучшей цены.

    side='ask'  — цены растут (мы покупаем и «съедаем» стакан вверх)
    side='bid'  — цены падают (мы продаём и «съедаем» стакан вниз)
    total_depth — суммарный объём в базовой монете на всех уровнях
    shape       — распределение ликвидности: <1 больше у топа, >1 больше в глубине
    """
    idx = np.arange(levels)
    sign = 1.0 if side == "ask" else -1.0
    prices = best_price * (1.0 + sign * (step_bps / 10_000.0) * idx)

    weights = (idx + 1.0) ** shape
    weights = weights / weights.sum()
    sizes = total_depth * weights

    return pd.DataFrame({"level": idx + 1, "price": prices, "size": sizes})


def parse_manual_book(text: str) -> Optional[pd.DataFrame]:
    """Разбор стакана, вставленного вручную: строки вида 'цена, объём'."""
    rows: List[Tuple[float, float]] = []
    for line in text.strip().splitlines():
        line = line.replace(";", ",").replace("\t", ",").strip()
        if not line:
            continue
        parts = [p for p in line.split(",") if p.strip()]
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["price", "size"])
    df.insert(0, "level", np.arange(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------------------
# Исполнение по стакану (VWAP + проскальзывание)
# --------------------------------------------------------------------------------------

@dataclass
class Fill:
    qty: float               # реально исполненный объём в базовой монете
    quote: float             # сумма в котируемой валюте (без комиссий)
    vwap: float              # средняя цена исполнения
    best: float              # лучшая цена в стакане
    slippage_bps: float      # проскальзывание относительно лучшей цены, б.п.
    filled_fully: bool       # хватило ли ликвидности
    levels_used: int


def walk_book(book: pd.DataFrame, qty: float, side: str) -> Fill:
    """Проход по стакану на объём qty. side='buy' идёт по ask, 'sell' — по bid."""
    best = float(book["price"].iloc[0])
    remaining = qty
    quote = 0.0
    done = 0.0
    used = 0

    for _, row in book.iterrows():
        if remaining <= 1e-12:
            break
        take = min(remaining, float(row["size"]))
        quote += take * float(row["price"])
        done += take
        remaining -= take
        used += 1

    vwap = quote / done if done > 0 else best
    if side == "buy":
        slip = (vwap / best - 1.0) * 10_000.0
    else:
        slip = (best / vwap - 1.0) * 10_000.0

    return Fill(
        qty=done,
        quote=quote,
        vwap=vwap,
        best=best,
        slippage_bps=slip,
        filled_fully=remaining <= 1e-9,
        levels_used=used,
    )


# --------------------------------------------------------------------------------------
# Параметры сделки
# --------------------------------------------------------------------------------------

@dataclass
class Leg:
    name: str
    fee_maker_pct: float
    fee_taker_pct: float
    order_type: str  # 'taker' | 'maker'

    @property
    def fee_pct(self) -> float:
        return self.fee_maker_pct if self.order_type == "maker" else self.fee_taker_pct


@dataclass
class Transfer:
    network: str
    fee_coin: float = 0.0      # комиссия сети в базовой монете (вывод монеты)
    fee_quote: float = 0.0     # комиссия за возврат котируемой валюты (напр. USDT)
    minutes: float = 10.0      # время подтверждения


@dataclass
class TradeResult:
    qty_requested: float
    buy: Fill
    sell: Fill
    buy_fee_quote: float
    sell_fee_quote: float
    withdraw_fee_quote: float
    quote_fee_quote: float
    gross_quote: float
    net_quote: float
    cost_quote: float
    spread_bps: float
    net_margin_pct: float
    liquidity_ok: bool
    extras: dict = field(default_factory=dict)


def evaluate_trade(
    qty: float,
    ask_book: pd.DataFrame,
    bid_book: pd.DataFrame,
    buy_leg: Leg,
    sell_leg: Leg,
    transfer: Transfer,
    fee_in_coin: bool = True,
) -> TradeResult:
    """Полный расчёт одной арбитражной сделки объёмом qty базовой монеты.

    Логика: покупаем qty на бирже A по ask → платим комиссию биржи →
    выводим в сети (комиссия сети в монете) → продаём остаток на бирже B по bid →
    платим комиссию биржи B → возвращаем котируемую валюту (комиссия перевода).
    """
    buy = walk_book(ask_book, qty, "buy")

    # комиссия биржи покупки
    buy_fee_quote = buy.quote * buy_leg.fee_pct / 100.0
    coin_after_buy = buy.qty
    if fee_in_coin:
        # комиссия списывается монетой — уменьшает количество, а не деньги
        coin_after_buy = buy.qty * (1.0 - buy_leg.fee_pct / 100.0)
        buy_fee_quote_effective = (buy.qty - coin_after_buy) * buy.vwap
    else:
        buy_fee_quote_effective = buy_fee_quote

    # вывод в сети
    coin_to_sell = max(coin_after_buy - transfer.fee_coin, 0.0)
    withdraw_fee_quote = transfer.fee_coin * buy.vwap

    sell = walk_book(bid_book, coin_to_sell, "sell")
    sell_fee_quote = sell.quote * sell_leg.fee_pct / 100.0

    cost_quote = buy.quote if not fee_in_coin else buy.quote
    if not fee_in_coin:
        cost_quote = buy.quote + buy_fee_quote

    proceeds = sell.quote - sell_fee_quote - transfer.fee_quote
    net = proceeds - cost_quote
    gross = sell.best * qty - buy.best * qty
    spread_bps = (sell.best / buy.best - 1.0) * 10_000.0 if buy.best else 0.0
    margin = net / cost_quote * 100.0 if cost_quote > 0 else 0.0

    return TradeResult(
        qty_requested=qty,
        buy=buy,
        sell=sell,
        buy_fee_quote=buy_fee_quote_effective,
        sell_fee_quote=sell_fee_quote,
        withdraw_fee_quote=withdraw_fee_quote,
        quote_fee_quote=transfer.fee_quote,
        gross_quote=gross,
        net_quote=net,
        cost_quote=cost_quote,
        spread_bps=spread_bps,
        net_margin_pct=margin,
        liquidity_ok=buy.filled_fully and sell.filled_fully,
        extras={"coin_to_sell": coin_to_sell, "coin_after_buy": coin_after_buy},
    )


# --------------------------------------------------------------------------------------
# Сканирование объёма и безубыточность
# --------------------------------------------------------------------------------------

def scan_volumes(
    qty_max: float,
    points: int,
    ask_book: pd.DataFrame,
    bid_book: pd.DataFrame,
    buy_leg: Leg,
    sell_leg: Leg,
    transfer: Transfer,
    fee_in_coin: bool = True,
) -> pd.DataFrame:
    qtys = np.linspace(qty_max / points, qty_max, points)
    rows = []
    for q in qtys:
        r = evaluate_trade(q, ask_book, bid_book, buy_leg, sell_leg, transfer, fee_in_coin)
        rows.append(
            {
                "Объём (монета)": q,
                "Оборот (котируемая)": r.cost_quote,
                "VWAP покупки": r.buy.vwap,
                "VWAP продажи": r.sell.vwap,
                "Проскальзывание покупки, б.п.": r.buy.slippage_bps,
                "Проскальзывание продажи, б.п.": r.sell.slippage_bps,
                "Комиссии бирж": r.buy_fee_quote + r.sell_fee_quote,
                "Комиссия сети": r.withdraw_fee_quote + r.quote_fee_quote,
                "Чистая прибыль": r.net_quote,
                "Рентабельность, %": r.net_margin_pct,
                "Ликвидности хватает": r.liquidity_ok,
            }
        )
    return pd.DataFrame(rows)


def _net_at(q: float, *args, **kwargs) -> float:
    return evaluate_trade(q, *args, **kwargs).net_quote


def breakeven_volumes(
    qty_max: float,
    ask_book: pd.DataFrame,
    bid_book: pd.DataFrame,
    buy_leg: Leg,
    sell_leg: Leg,
    transfer: Transfer,
    fee_in_coin: bool = True,
    grid: int = 200,
) -> dict:
    """Нижняя и верхняя точки безубыточности + оптимум прибыли."""
    args = (ask_book, bid_book, buy_leg, sell_leg, transfer, fee_in_coin)
    qs = np.linspace(qty_max / grid, qty_max, grid)
    nets = np.array([_net_at(q, *args) for q in qs])

    def bisect(lo: float, hi: float) -> float:
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if _net_at(lo, *args) * _net_at(mid, *args) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    lower = upper = None
    signs = np.sign(nets)
    for i in range(1, len(qs)):
        if signs[i - 1] < 0 <= signs[i] and lower is None:
            lower = bisect(qs[i - 1], qs[i])
        if signs[i - 1] > 0 >= signs[i]:
            upper = bisect(qs[i - 1], qs[i])

    best_i = int(np.argmax(nets))
    return {
        "lower": lower,
        "upper": upper,
        "q_opt": float(qs[best_i]),
        "net_opt": float(nets[best_i]),
        "curve": pd.DataFrame({"qty": qs, "net": nets}),
        "profitable_anywhere": bool(nets.max() > 0),
    }
