"""Сбор сделок с метками времени источника по кандидатам в связки.

Время берётся у источника, а не у сборщика: у фьючерса — время сделки биржи
(мс), у DEX — время блока. Поэтому лаг в секунды виден даже при том, что
сами данные приходят с задержкой и пачками.

  * Фьючерсы — WebSocket-ленты сделок через ccxt.pro, по потоку на биржу.
    Сделки сворачиваются в секунды: последняя цена, объём, число сделок.
  * DEX в EVM-сетях (BNB Chain, Base, Arbitrum) — переводы токенов пула
    прямо из блокчейна через публичный RPC (evm.py), без пропусков.
  * DEX в остальных сетях (Solana, Sui…) — лента последних 300 сделок пула
    в GeckoTerminal (бесплатно, ~6 запросов в минуту). Пул опрашивается тем
    чаще, чем быстрее он набирает 300 сделок.

Список инструментов берётся из universe.parquet (шаг 1) с фильтрами
из настроек и обновляется каждые 30 минут.

Тем же процессом раз в 30 минут считается лаг по закрытым часам (lag.py).

Запуск: python -m leadlag.ticks
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from . import evm, universe
from .catalog import CHAINS, FUTURES, futures_url, norm_address
from .settings import LEADLAG_DIR, ScreenSettings

log = logging.getLogger(__name__)

TICKS_DB = LEADLAG_DIR / "ticks.sqlite"
GT_BASE = "https://api.geckoterminal.com/api/v2"
GT_PAUSE_S = 10.5           # фактический лимит бесплатного API — около 6 запросов в минуту
DEX_MIN_POLL_S = 30
DEX_MAX_POLL_S = 600
REFRESH_INSTRUMENTS_S = 1800
FLUSH_S = 5
LAG_EVERY_S = 1800


# ------------------------------------------------------------------ хранилище

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    inst  TEXT    NOT NULL,   -- fut:<биржа>:<символ> | dex:<сеть>:<пул>
    ts    INTEGER NOT NULL,   -- секунда, UTC
    price REAL    NOT NULL,   -- цена последней сделки в этой секунде, USD за 1 монету
    vol   REAL    NOT NULL,   -- объём сделок за секунду, USD
    n     INTEGER NOT NULL,   -- число сделок
    PRIMARY KEY (inst, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS dex_seen (     -- уже учтённые свопы: опросы перекрываются
    pool TEXT NOT NULL, tx TEXT NOT NULL, ts INTEGER NOT NULL,
    PRIMARY KEY (pool, tx)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS instruments (
    inst TEXT PRIMARY KEY, coin TEXT, kind TEXT, venue TEXT, symbol TEXT,
    url TEXT, chain TEXT, contract TEXT, updated_at INTEGER
);
"""


def connect(path: Path = TICKS_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def write_seconds(con: sqlite3.Connection, rows: List[Tuple[str, int, float, float, int]]) -> None:
    """Секунда может прийти частями (две пачки DEX) — объёмы складываются,
    цена остаётся от последней части."""
    con.executemany(
        "INSERT INTO prices(inst, ts, price, vol, n) VALUES (?,?,?,?,?) "
        "ON CONFLICT(inst, ts) DO UPDATE SET price=excluded.price, "
        "vol=vol+excluded.vol, n=n+excluded.n", rows)
    con.commit()


def prune(con: sqlite3.Connection, keep_days: float) -> None:
    cutoff = int(time.time() - keep_days * 86400)
    con.execute("DELETE FROM prices WHERE ts < ?", (cutoff,))
    con.execute("DELETE FROM dex_seen WHERE ts < ?", (cutoff,))
    con.commit()


# ---------------------------------------------------------------- инструменты

@dataclass(frozen=True)
class Instrument:
    inst: str
    coin: str
    kind: str           # fut | dex
    venue: str          # биржа или сеть
    symbol: str         # символ ccxt или адрес пула
    url: str
    chain: str = ""
    contract: str = ""  # адрес токена — нужен, чтобы понять сторону свопа
    mult: float = 1.0   # 1000PEPE: цена фьючерса делится на множитель


def fut_inst(venue: str, symbol: str) -> str:
    return f"fut:{venue}:{symbol}"


def dex_inst(chain: str, pool: str) -> str:
    return f"dex:{chain}:{pool}"


def select_instruments(s: ScreenSettings) -> List[Instrument]:
    df, _ = universe.load()
    if df is None or df.empty:
        return []
    f = universe.apply_filters(df, s)
    coins = (f.groupby("coin")["fut_volume_usd"].sum()
             .sort_values(ascending=False).head(s.track_max_coins).index)
    f = f[f["coin"].isin(coins)]
    out: Dict[str, Instrument] = {}
    for r in f.itertuples():
        i = fut_inst(r.fut_venue, r.fut_symbol)
        out[i] = Instrument(i, r.coin, "fut", r.fut_venue, r.fut_symbol, r.fut_url or "",
                            mult=float(r.fut_mult or 1))
        d = dex_inst(r.chain, r.dex_pool)
        out[d] = Instrument(d, r.coin, "dex", r.chain, r.dex_pool, r.dex_url or "",
                            chain=r.chain, contract=r.contract)
    # курсы BNB/ETH для пересчёта пулов к WBNB/WETH в доллары
    for sym in evm.ref_symbols({i.chain for i in out.values() if i.kind == "dex"}):
        i = fut_inst(evm.REF_VENUE, sym)
        out.setdefault(i, Instrument(i, sym.split("/")[0], "ref", evm.REF_VENUE, sym,
                                     futures_url(evm.REF_VENUE, sym.replace("/USDT:USDT", "USDT")) or ""))
    return list(out.values())


def ref_price(con: sqlite3.Connection, symbol: str, ts: int) -> Optional[float]:
    """Цена ref-фьючерса на секунду ts (последняя сделка не старше 30 с)."""
    row = con.execute("SELECT price FROM prices WHERE inst=? AND ts<=? AND ts>=? ORDER BY ts DESC LIMIT 1",
                      (fut_inst(evm.REF_VENUE, symbol), ts, ts - 30)).fetchone()
    return row[0] if row else None


def store_swaps(con: sqlite3.Connection, rows: List[Tuple[str, str, int, float, float]]) -> int:
    """(inst, tx, ts, цена USD, объём USD) → секунды, без повторов между опросами."""
    if not rows:
        return 0
    fresh = []
    for inst, tx, ts, price, vol in rows:
        cur = con.execute("INSERT OR IGNORE INTO dex_seen VALUES (?,?,?)", (inst, tx, ts))
        if cur.rowcount:
            fresh.append((tx, ts, price, vol, inst))
    by_inst: Dict[str, List[Tuple[str, int, float, float]]] = defaultdict(list)
    for tx, ts, price, vol, inst in fresh:
        by_inst[inst].append((tx, ts, price, vol))
    out = []
    for inst, swaps in by_inst.items():
        out.extend(seconds_from_swaps(inst, swaps))
    write_seconds(con, out)
    return len(fresh)


def save_instruments(con: sqlite3.Connection, items: List[Instrument]) -> None:
    now = int(time.time())
    con.executemany(
        "INSERT INTO instruments VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(inst) DO UPDATE SET "
        "coin=excluded.coin, url=excluded.url, updated_at=excluded.updated_at",
        [(i.inst, i.coin, i.kind, i.venue, i.symbol, i.url, i.chain, i.contract, now) for i in items])
    con.commit()


# ------------------------------------------------------------------ фьючерсы

class SecondBuffer:
    """Сделки → секунды. Отдаёт только закрытые секунды (с запасом 2 с)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data: Dict[Tuple[str, int], List[float]] = {}  # (inst, ts) → [ts_ms, price, vol, n]

    def add(self, inst: str, ts_ms: int, price: float, vol: float) -> None:
        key = (inst, ts_ms // 1000)
        with self.lock:
            cur = self.data.get(key)
            if cur is None:
                self.data[key] = [ts_ms, price, vol, 1]
            else:
                if ts_ms >= cur[0]:
                    cur[0], cur[1] = ts_ms, price
                cur[2] += vol
                cur[3] += 1

    def pop_closed(self, now: float) -> List[Tuple[str, int, float, float, int]]:
        edge = int(now) - 2
        out = []
        with self.lock:
            for k in [k for k in self.data if k[1] < edge]:
                v = self.data.pop(k)
                out.append((k[0], k[1], v[1], v[2], int(v[3])))
        return out


async def watch_venue(venue: str, items: List[Instrument], buf: SecondBuffer,
                      stop: asyncio.Event) -> None:
    import ccxt.pro as ccxtpro  # type: ignore

    v = FUTURES[venue]
    by_symbol = {i.symbol: i for i in items}
    ex = getattr(ccxtpro, v.ccxt_id)({"enableRateLimit": True, "options": dict(v.options)})
    multi = bool(ex.has.get("watchTradesForMultipleSymbols"))

    def take(trades) -> None:
        for t in trades or []:
            inst = by_symbol.get(t.get("symbol"))
            price, ts = t.get("price"), t.get("timestamp")
            if inst is None or not price or not ts:
                continue
            cost = t.get("cost") or price * (t.get("amount") or 0)
            buf.add(inst.inst, int(ts), float(price) / inst.mult, float(cost or 0))

    async def one(symbols: List[str]) -> None:
        delay = 5
        while not stop.is_set():
            try:
                if multi and len(symbols) > 1:
                    take(await ex.watch_trades_for_multiple_symbols(symbols))
                else:
                    take(await ex.watch_trades(symbols[0]))
                delay = 5
            except Exception as e:
                log.warning("%s %s: %s — переподключение через %d с", venue, symbols[:3], str(e)[:200], delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300)

    symbols = sorted(by_symbol)
    # у бирж ограничения на число подписок в одном соединении — порции по 20
    groups = ([symbols[i:i + 20] for i in range(0, len(symbols), 20)] if multi
              else [[s] for s in symbols])
    try:
        await asyncio.gather(*(one(g) for g in groups))
    finally:
        try:
            await ex.close()
        except Exception:
            pass


def run_futures(get_items, buf: SecondBuffer, stop_all: threading.Event) -> None:
    """Поток с asyncio: подписки пересоздаются при смене списка инструментов."""

    async def main() -> None:
        current: Dict[str, Tuple[frozenset, asyncio.Task, asyncio.Event]] = {}
        while not stop_all.is_set():
            items = [i for i in get_items() if i.kind in ("fut", "ref")]
            by_venue: Dict[str, List[Instrument]] = defaultdict(list)
            for i in items:
                by_venue[i.venue].append(i)
            for venue in list(current):
                if venue not in by_venue or frozenset(x.inst for x in by_venue[venue]) != current[venue][0]:
                    current[venue][2].set()
                    current[venue][1].cancel()
                    del current[venue]
            for venue, lst in by_venue.items():
                if venue not in current:
                    ev = asyncio.Event()
                    task = asyncio.create_task(watch_venue(venue, lst, buf, ev))
                    current[venue] = (frozenset(x.inst for x in lst), task, ev)
                    log.info("фьючерсы %s: подписка на %d рынков", venue, len(lst))
            await asyncio.sleep(60)

    asyncio.run(main())


# ----------------------------------------------------------------------- DEX

def parse_gt_trades(items: List[dict], contract: str, chain: str) -> List[Tuple[str, int, float, float]]:
    """Сделки GeckoTerminal → (tx, ts, цена USD за монету, объём USD).

    Цена считается из самой сделки: объём в USD / количество нашей монеты.
    Так она не зависит от того, как площадка пересчитала котируемый токен.
    """
    want = norm_address(chain, contract)
    out = []
    for it in items:
        a = it.get("attributes") or {}
        if a.get("is_anomalous"):
            continue
        frm = norm_address(chain, a.get("from_token_address") or "")
        to = norm_address(chain, a.get("to_token_address") or "")
        try:
            vol = float(a.get("volume_in_usd") or 0)
            if frm == want:
                amount = float(a.get("from_token_amount") or 0)
            elif to == want:
                amount = float(a.get("to_token_amount") or 0)
            else:
                continue
            ts = int(datetime.fromisoformat(a["block_timestamp"].replace("Z", "+00:00")).timestamp())
        except (TypeError, ValueError, KeyError):
            continue
        if amount <= 0 or vol <= 0:
            continue
        # tx + номер блока + сумма: в одной транзакции бывает несколько свопов в пуле
        key = f"{a.get('tx_hash')}:{a.get('block_number')}:{amount:.10g}"
        out.append((key, ts, vol / amount, vol))
    return out


def seconds_from_swaps(inst: str, swaps: List[Tuple[str, int, float, float]]
                       ) -> List[Tuple[str, int, float, float, int]]:
    agg: Dict[int, List[float]] = {}
    for _, ts, price, vol in sorted(swaps, key=lambda x: x[1]):
        cur = agg.setdefault(ts, [price, 0.0, 0])
        cur[0] = price  # порядок внутри секунды (блока) неизвестен точнее — берём последнюю
        cur[1] += vol
        cur[2] += 1
    return [(inst, ts, v[0], v[1], int(v[2])) for ts, v in agg.items()]


class DexPoller:
    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.s = requests.Session()
        self.due: Dict[str, float] = {}
        self._last = 0.0

    def fetch(self, i: Instrument) -> Optional[List[dict]]:
        net = CHAINS[i.chain].geckoterminal
        wait = GT_PAUSE_S - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        r = self.s.get(f"{GT_BASE}/networks/{net}/pools/{i.symbol}/trades", timeout=30)
        if r.status_code == 429:
            time.sleep(30)
            return None
        r.raise_for_status()
        return r.json().get("data") or []

    def poll(self, i: Instrument) -> int:
        items = self.fetch(i)
        if items is None:
            return 0
        swaps = parse_gt_trades(items, i.contract, i.chain)
        if not swaps:
            self.due[i.inst] = time.time() + DEX_MAX_POLL_S
            return 0
        fresh = store_swaps(self.con, [(i.inst, *sw) for sw in swaps])
        # следующий опрос — когда пул наберёт примерно половину ленты
        span = max(s[1] for s in swaps) - min(s[1] for s in swaps)
        full = len(items) >= 300
        interval = span / 2 if full else DEX_MAX_POLL_S
        if full and fresh == len(swaps) and i.inst in self.due:
            log.warning("%s: лента переполнена между опросами — возможна дыра", i.inst)
        self.due[i.inst] = time.time() + min(DEX_MAX_POLL_S, max(DEX_MIN_POLL_S, interval))
        return fresh

    def step(self, items: List[Instrument]) -> None:
        dex = [i for i in items if i.kind == "dex" and i.chain not in evm.EVM]
        if not dex:
            time.sleep(5)
            return
        now = time.time()
        i = min(dex, key=lambda x: self.due.get(x.inst, 0))
        if self.due.get(i.inst, 0) > now:
            time.sleep(min(5.0, self.due[i.inst] - now))
            return
        try:
            self.poll(i)
        except Exception as e:
            log.warning("%s: %s", i.inst, str(e)[:200])
            self.due[i.inst] = now + 120


# -------------------------------------------------------------------- запуск

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    con = connect()
    state = {"items": [], "loaded": 0.0}
    lock = threading.Lock()

    def refresh() -> None:
        s = ScreenSettings.load()
        items = select_instruments(s)
        with lock:
            state["items"], state["loaded"] = items, time.time()
            state["keep_days"] = s.keep_days
        save_instruments(con, items)
        log.info("инструменты: %d фьючерсов, %d пулов (%d из блокчейна), курсов %d",
                 sum(i.kind == "fut" for i in items), sum(i.kind == "dex" for i in items),
                 sum(i.kind == "dex" and i.chain in evm.EVM for i in items),
                 sum(i.kind == "ref" for i in items))

    def get_items() -> List[Instrument]:
        with lock:
            return list(state["items"])

    refresh()
    buf = SecondBuffer()
    stop = threading.Event()
    threading.Thread(target=run_futures, args=(get_items, buf, stop), daemon=True,
                     name="futures").start()

    # запись фьючерсов — отдельным соединением из своего потока
    def flusher() -> None:
        with closing(connect()) as c:
            while not stop.is_set():
                time.sleep(FLUSH_S)
                rows = buf.pop_closed(time.time())
                if rows:
                    write_seconds(c, rows)

    threading.Thread(target=flusher, daemon=True, name="flush").start()

    # EVM-сети — по потоку на сеть, у каждого своё соединение с базой
    def evm_loop(chain: str) -> None:
        with closing(connect()) as c:
            poller = evm.EvmPoller(chain, c, lambda rows: store_swaps(c, rows),
                                   lambda sym, ts: ref_price(c, sym, ts))
            pools_at = 0.0
            while not stop.is_set():
                started = time.time()
                try:
                    if started - pools_at > 300:
                        poller.set_pools([i for i in get_items() if i.kind == "dex" and i.chain == chain])
                        pools_at = started
                    poller.step()
                except Exception as e:
                    log.warning("%s: %s", chain, str(e)[:200])
                time.sleep(max(1.0, poller.c.poll_s - (time.time() - started)))

    for chain in evm.EVM:
        threading.Thread(target=evm_loop, args=(chain,), daemon=True, name=f"evm-{chain}").start()

    # поиск лага по закрытым часам — тем же процессом, раз в 30 минут
    def lag_loop() -> None:
        from . import lag
        while not stop.is_set():
            time.sleep(LAG_EVERY_S)
            try:
                lag.run_once()
            except Exception:
                log.exception("поиск лага упал")

    threading.Thread(target=lag_loop, daemon=True, name="lag").start()

    poller = DexPoller(con)
    last_prune = 0.0
    while True:
        if time.time() - state["loaded"] > REFRESH_INSTRUMENTS_S:
            try:
                refresh()
            except Exception as e:
                log.warning("список инструментов не обновлён: %s", e)
                state["loaded"] = time.time()
        if time.time() - last_prune > 3600:
            prune(con, state.get("keep_days", 3.0))
            last_prune = time.time()
        poller.step(get_items())


if __name__ == "__main__":
    main()
