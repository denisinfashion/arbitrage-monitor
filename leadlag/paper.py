"""Бумажная торговля на лаге: ведущий сдвинулся — «торгуем» ведомого.

Без денег и без ключей: сервис слушает стаканы ведущего и ведомого фьючерса
и на каждом сигнале записывает, что получилось бы у бота.

  Сигнал:  середина стакана ведущего сдвинулась на ≥ MIN_MOVE_PCT за WINDOW_MS.
  Вход:    через REACT_MS — по стакану ведомого на NOTIONAL_USD (покупка
           по ask-уровням, продажа по bid), средняя цена исполнения.
  Выход:   через 1, 2 и 5 с — так же по стакану обратной стороны.
  Итог:    (выход / вход − 1) × направление − 2 × тейкер ведомого.

Стакан некоторых бирж (BingX) приходит пачками раз в 200+ мс — «отставание»
может оказаться задержкой рассылки, а не цены. Поэтому каждая сделка
проверяется по ленте сделок ведомого: прошла ли в течение CONFIRM_MS после
входа реальная сделка по цене входа или лучше (confirmed). Итоги считаются
и по всем сигналам, и только по подтверждённым.

Время — момент получения котировки этим сервисом: ровно то, что видел бы бот
на этом сервере. Пары берутся из итогов поиска лага (ведомый — фьючерс,
устойчиво 12+ часов), список обновляется каждые 30 минут.

Запуск: python -m leadlag.paper
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from . import lag
from .catalog import FUTURES, FUTURES_TAKER_PCT
from .settings import LEADLAG_DIR

log = logging.getLogger(__name__)

PAPER_DB = LEADLAG_DIR / "paper.sqlite"
NOTIONAL_USD = 1000.0
REACT_MS = 100
WINDOW_MS = 500
MIN_MOVE_PCT = 0.15
COOLDOWN_MS = 5000
HOLDS_MS = (1000, 2000, 5000)
CONFIRM_MS = 500         # окно после входа, в котором ищем реальную сделку по нашей цене
MAX_PAIRS = 20
MIN_STABLE_HOURS = 12
REFRESH_PAIRS_S = 1800
BOOK_DEPTH = 20

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    ts INTEGER NOT NULL,          -- мс, момент сигнала
    leader TEXT NOT NULL, follower TEXT NOT NULL,
    move_pct REAL,                -- движение ведущего за окно, со знаком
    hold_ms INTEGER NOT NULL,
    entry REAL, exit REAL,
    filled_usd REAL,              -- сколько удалось исполнить из NOTIONAL_USD
    book_age_ms INTEGER,          -- возраст стакана ведомого на входе
    follower_pre_pct REAL,        -- насколько ведомый уже сдвинулся к сигналу (в сторону сигнала)
    pnl_pct REAL,                 -- после комиссий
    confirmed INTEGER,            -- была ли у ведомого реальная сделка по цене входа или лучше
    PRIMARY KEY (ts, leader, follower, hold_ms)
) WITHOUT ROWID;
"""


def connect() -> sqlite3.Connection:
    LEADLAG_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(PAPER_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def vwap(levels: List[List[float]], notional: float) -> Tuple[Optional[float], float]:
    """Средняя цена исполнения на notional долларов по уровням стакана; исполнено, $."""
    cost = qty = 0.0
    for price, amount in levels:
        take = min(amount, (notional - cost) / price)
        cost += take * price
        qty += take
        if cost >= notional - 1e-9:
            break
    return (cost / qty if qty else None), cost


@dataclass
class Book:
    bids: List[List[float]]
    asks: List[List[float]]
    t: float  # мс получения

    @property
    def mid(self) -> Optional[float]:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return None


@dataclass(frozen=True)
class Pair:
    leader: str     # "binance:ARIA/USDT:USDT"
    follower: str


def pick_pairs() -> List[Pair]:
    res = lag.load_results()
    if res is None or res.empty:
        return []
    q = res[(~res["synchronous"]) & (res["hours_stable"] >= MIN_STABLE_HOURS)
            & res["leader"].str.startswith("fut:") & res["follower"].str.startswith("fut:")]
    q = q.sort_values(["hours_stable", "corr"], ascending=False).drop_duplicates("follower")
    return [Pair(r.leader[4:], r.follower[4:]) for r in q.head(MAX_PAIRS).itertuples()]


class Paper:
    def __init__(self):
        self.con = connect()
        self.books: Dict[str, Book] = {}
        self.mids: Dict[str, Deque[Tuple[float, float]]] = {}
        self.last_signal: Dict[Pair, float] = {}
        self.pairs: List[Pair] = []
        self.by_leader: Dict[str, List[Pair]] = {}
        self.exchanges: Dict[str, object] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.trades: Dict[str, Deque[Tuple[float, float]]] = {}  # ведомый → (мс получения, цена)

    def exchange(self, venue: str):
        if venue not in self.exchanges:
            import ccxt.pro as ccxtpro  # type: ignore
            v = FUTURES[venue]
            self.exchanges[venue] = getattr(ccxtpro, v.ccxt_id)({"options": dict(v.options)})
        return self.exchanges[venue]

    # ---- потоки стаканов

    async def watch(self, key: str) -> None:
        venue, symbol = key.split(":", 1)
        ex = self.exchange(venue)
        size: Optional[float] = None
        delay = 5
        while True:
            try:
                ob = await ex.watch_order_book(symbol, BOOK_DEPTH)
                if size is None:
                    # у части бирж объём в стакане — в контрактах, а не в монетах
                    size = float(ex.market(symbol).get("contractSize") or 1)
                now = time.time() * 1000
                book = Book([[l[0], l[1] * size] for l in ob["bids"][:BOOK_DEPTH]],
                            [[l[0], l[1] * size] for l in ob["asks"][:BOOK_DEPTH]], now)
                self.books[key] = book
                if book.mid:
                    self.remember(key, book.mid, now)
                    if key in self.by_leader:
                        self.on_leader(key, book.mid, now)
                delay = 5
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("%s: %s — переподключение через %d с", key, str(e)[:150], delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300)

    async def watch_trades(self, key: str) -> None:
        venue, symbol = key.split(":", 1)
        ex = self.exchange(venue)
        delay = 5
        while True:
            try:
                got = await ex.watch_trades(symbol)
                now = time.time() * 1000
                q = self.trades.setdefault(key, deque())
                for t in got:
                    if t.get("price"):
                        q.append((now, float(t["price"])))
                while q and q[0][0] < now - 10_000:
                    q.popleft()
                delay = 5
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("%s сделки: %s — переподключение через %d с", key, str(e)[:150], delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300)

    def confirmed(self, key: str, side: int, best: float, t_from: float) -> bool:
        """Была ли сделка по цене входа или лучше для нас (покупка — не дороже ask)."""
        for t, price in self.trades.get(key, ()):
            if t_from <= t <= t_from + CONFIRM_MS and (price <= best if side > 0 else price >= best):
                return True
        return False

    def remember(self, key: str, mid: float, now: float) -> None:
        hist = self.mids.setdefault(key, deque())
        hist.append((now, mid))
        while hist and hist[0][0] < now - 2 * WINDOW_MS:
            hist.popleft()

    def mid_at(self, key: str, t: float) -> Optional[float]:
        """Последняя середина стакана не позже момента t."""
        return next((m for ts, m in reversed(self.mids.get(key, ())) if ts <= t), None)

    def on_leader(self, key: str, mid: float, now: float) -> None:
        past = self.mid_at(key, now - WINDOW_MS)
        if not past:
            return
        move = (mid / past - 1) * 100
        if abs(move) < MIN_MOVE_PCT:
            return
        for pair in self.by_leader[key]:
            if now - self.last_signal.get(pair, 0) < COOLDOWN_MS:
                continue
            self.last_signal[pair] = now
            asyncio.create_task(self.trade(pair, now, move))

    # ---- сделка

    async def trade(self, pair: Pair, t0: float, move: float) -> None:
        side = 1 if move > 0 else -1
        # насколько ведомый уже сдвинулся в сторону сигнала за то же окно
        f_now, f_past = self.mid_at(pair.follower, t0), self.mid_at(pair.follower, t0 - WINDOW_MS)
        pre = (f_now / f_past - 1) * 100 * side if f_now and f_past else None
        await asyncio.sleep(REACT_MS / 1000)
        book = self.books.get(pair.follower)
        if book is None or not book.bids or not book.asks:
            return
        now = time.time() * 1000
        entry, filled = vwap(book.asks if side > 0 else book.bids, NOTIONAL_USD)
        if entry is None:
            return
        fee = 2 * FUTURES_TAKER_PCT.get(pair.follower.split(":")[0], 0.06)
        best = book.asks[0][0] if side > 0 else book.bids[0][0]
        start = now
        rows = []
        conf: Optional[bool] = None
        for hold in HOLDS_MS:
            await asyncio.sleep(max(0.0, (start + hold - time.time() * 1000) / 1000))
            b = self.books.get(pair.follower)
            if b is None:
                continue
            exit_, _ = vwap(b.bids if side > 0 else b.asks, NOTIONAL_USD)
            if exit_ is None:
                continue
            if conf is None:  # первый выход — через 1 с, окно подтверждения уже прошло
                conf = self.confirmed(pair.follower, side, best, start)
            pnl = (exit_ / entry - 1) * 100 * side - fee
            rows.append((int(t0), pair.leader, pair.follower, float(move), hold, float(entry), float(exit_),
                         float(filled), int(now - book.t), pre, float(pnl), int(conf)))
        if rows:
            self.con.executemany("INSERT OR IGNORE INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.con.commit()

    # ---- список пар

    async def manage(self) -> None:
        while True:
            pairs = pick_pairs()
            if pairs != self.pairs:
                self.pairs = pairs
                self.by_leader = {}
                for p in pairs:
                    self.by_leader.setdefault(p.leader, []).append(p)
                wanted = {p.leader for p in pairs} | {p.follower for p in pairs}
                for key in list(self.tasks):
                    if key.removeprefix("trades:") not in wanted:
                        self.tasks.pop(key).cancel()
                for key in wanted - set(self.tasks):
                    self.tasks[key] = asyncio.create_task(self.watch(key))
                for key in {p.follower for p in pairs}:
                    tkey = "trades:" + key
                    if tkey not in self.tasks:
                        self.tasks[tkey] = asyncio.create_task(self.watch_trades(key))
                log.info("пар: %d, стаканов: %d — %s", len(pairs), len(wanted),
                         ", ".join(f"{p.leader}→{p.follower.split(':')[0]}" for p in pairs[:8]))
            await asyncio.sleep(REFRESH_PAIRS_S)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(Paper().manage())


if __name__ == "__main__":
    main()
