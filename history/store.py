"""Хранилище временных рядов котировок.

SQLite в режиме WAL: один писатель (процесс-сборщик) и сколько угодно
читателей (вкладки Streamlit) работают одновременно без блокировок.
Для объёма «100 пулов x 7 дней x 1 мин» ~ 1 млн строк это с запасом.

Схема сознательно денормализована в одну широкую таблицу `quotes`:
все источники приводятся к общему виду «курс base->quote на площадке
в момент времени», и дальше анализ не знает, CEX это или DEX.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .config import DB_PATH, ensure_data_dir

log = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Основной ряд: одна строка = одна свеча одной пары на одной площадке.
CREATE TABLE IF NOT EXISTS quotes (
    ts            INTEGER NOT NULL,   -- unix-секунды, открытие свечи, UTC
    venue         TEXT    NOT NULL,   -- binance | pancakeswap_v2 | ...
    venue_kind    TEXT    NOT NULL,   -- cex | dex
    chain         TEXT    NOT NULL,   -- bsc | '' для CEX
    base          TEXT    NOT NULL,
    quote         TEXT    NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL    NOT NULL,   -- курс: сколько quote за 1 base
    volume        REAL,
    liquidity_usd REAL,               -- TVL пула, NULL для CEX
    pool          TEXT,               -- адрес пула, NULL для CEX
    PRIMARY KEY (ts, venue, chain, base, quote)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_quotes_pair    ON quotes (base, quote, ts);
CREATE INDEX IF NOT EXISTS idx_quotes_ts      ON quotes (ts);
CREATE INDEX IF NOT EXISTS idx_quotes_venue   ON quotes (venue, ts);

-- Справочник наблюдаемых пулов DEX.
CREATE TABLE IF NOT EXISTS pools (
    chain         TEXT NOT NULL,
    pool          TEXT NOT NULL,
    dex           TEXT,
    base          TEXT,
    quote         TEXT,
    base_addr     TEXT,
    quote_addr    TEXT,
    reserve_usd   REAL,
    volume_24h    REAL,
    fee_pct       REAL,
    updated_at    INTEGER,
    PRIMARY KEY (chain, pool)
);

-- Газ и цена нативной монеты сети.
CREATE TABLE IF NOT EXISTS gas (
    ts             INTEGER NOT NULL,
    chain          TEXT    NOT NULL,
    gas_price_gwei REAL,
    native_usd     REAL,
    PRIMARY KEY (ts, chain)
);

-- Комиссии площадок.
CREATE TABLE IF NOT EXISTS fees (
    venue      TEXT PRIMARY KEY,
    venue_kind TEXT,
    taker_pct  REAL,
    maker_pct  REAL,
    updated_at INTEGER
);

-- Состояние сборщиков: докуда дотянули, когда и с каким результатом.
CREATE TABLE IF NOT EXISTS collector_state (
    source     TEXT NOT NULL,
    key        TEXT NOT NULL,
    last_ts    INTEGER,
    last_run   INTEGER,
    ok         INTEGER,
    error      TEXT,
    rows_total INTEGER DEFAULT 0,
    PRIMARY KEY (source, key)
);
"""

_LOCAL = threading.local()


def connect(read_only: bool = False) -> sqlite3.Connection:
    """Соединение на поток. SQLite-объекты не потокобезопасны, поэтому
    каждому потоку своё."""
    key = f"conn_ro" if read_only else "conn_rw"
    conn = getattr(_LOCAL, key, None)
    if conn is not None:
        return conn

    ensure_data_dir()
    if read_only and DB_PATH.exists():
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=15, check_same_thread=False)
    else:
        conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
        conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    setattr(_LOCAL, key, conn)
    return conn


def init() -> None:
    """Создаёт схему. Вызывать один раз при старте сборщика."""
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def transaction():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --------------------------------------------------------------------------
# Запись
# --------------------------------------------------------------------------


@dataclass
class Candle:
    """Одна свеча в общем формате."""

    ts: int
    venue: str
    venue_kind: str
    chain: str
    base: str
    quote: str
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    liquidity_usd: Optional[float] = None
    pool: Optional[str] = None

    def as_row(self) -> tuple:
        return (
            int(self.ts), self.venue, self.venue_kind, self.chain,
            self.base, self.quote,
            self.open, self.high, self.low, float(self.close), self.volume,
            self.liquidity_usd, self.pool,
        )


_INSERT_QUOTE = """
INSERT INTO quotes (ts, venue, venue_kind, chain, base, quote,
                    open, high, low, close, volume, liquidity_usd, pool)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT (ts, venue, chain, base, quote) DO UPDATE SET
    open=excluded.open, high=excluded.high, low=excluded.low,
    close=excluded.close, volume=excluded.volume,
    liquidity_usd=excluded.liquidity_usd, pool=excluded.pool
"""


def write_candles(candles: Sequence[Candle]) -> int:
    """Апсерт пачки свечей. Возвращает число записанных строк."""
    if not candles:
        return 0
    rows = [c.as_row() for c in candles]
    with transaction() as conn:
        conn.executemany(_INSERT_QUOTE, rows)
    return len(rows)


def write_pools(rows: Iterable[dict]) -> int:
    sql = """
    INSERT INTO pools (chain, pool, dex, base, quote, base_addr, quote_addr,
                       reserve_usd, volume_24h, fee_pct, updated_at)
    VALUES (:chain,:pool,:dex,:base,:quote,:base_addr,:quote_addr,
            :reserve_usd,:volume_24h,:fee_pct,:updated_at)
    ON CONFLICT (chain, pool) DO UPDATE SET
        dex=excluded.dex, base=excluded.base, quote=excluded.quote,
        base_addr=excluded.base_addr, quote_addr=excluded.quote_addr,
        reserve_usd=excluded.reserve_usd, volume_24h=excluded.volume_24h,
        fee_pct=excluded.fee_pct, updated_at=excluded.updated_at
    """
    rows = list(rows)
    if not rows:
        return 0
    now = int(time.time())
    for r in rows:
        r.setdefault("updated_at", now)
        for k in ("dex", "base", "quote", "base_addr", "quote_addr",
                  "reserve_usd", "volume_24h", "fee_pct"):
            r.setdefault(k, None)
    with transaction() as conn:
        conn.executemany(sql, rows)
    return len(rows)


def write_fees(venue: str, venue_kind: str, taker_pct: float, maker_pct: float) -> None:
    with transaction() as conn:
        conn.execute(
            """INSERT INTO fees (venue, venue_kind, taker_pct, maker_pct, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT (venue) DO UPDATE SET
                 venue_kind=excluded.venue_kind, taker_pct=excluded.taker_pct,
                 maker_pct=excluded.maker_pct, updated_at=excluded.updated_at""",
            (venue, venue_kind, taker_pct, maker_pct, int(time.time())),
        )


def write_gas(chain: str, ts: int, gas_price_gwei: float, native_usd: float) -> None:
    with transaction() as conn:
        conn.execute(
            """INSERT INTO gas (ts, chain, gas_price_gwei, native_usd) VALUES (?,?,?,?)
               ON CONFLICT (ts, chain) DO UPDATE SET
                 gas_price_gwei=excluded.gas_price_gwei, native_usd=excluded.native_usd""",
            (int(ts), chain, gas_price_gwei, native_usd),
        )


def set_state(source: str, key: str, *, last_ts: Optional[int] = None,
              ok: bool = True, error: str = "", rows: int = 0) -> None:
    with transaction() as conn:
        conn.execute(
            """INSERT INTO collector_state (source, key, last_ts, last_run, ok, error, rows_total)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT (source, key) DO UPDATE SET
                 last_ts=COALESCE(excluded.last_ts, collector_state.last_ts),
                 last_run=excluded.last_run, ok=excluded.ok, error=excluded.error,
                 rows_total=collector_state.rows_total + excluded.rows_total""",
            (source, key, last_ts, int(time.time()), 1 if ok else 0, error[:400], rows),
        )


def get_last_ts(source: str, key: str) -> Optional[int]:
    conn = connect()
    row = conn.execute(
        "SELECT last_ts FROM collector_state WHERE source=? AND key=?", (source, key)
    ).fetchone()
    return row["last_ts"] if row and row["last_ts"] else None


def prune(older_than_ts: int) -> int:
    """Удаляет свечи старше границы — держит базу в заданной глубине."""
    with transaction() as conn:
        cur = conn.execute("DELETE FROM quotes WHERE ts < ?", (int(older_than_ts),))
        deleted = cur.rowcount
        conn.execute("DELETE FROM gas WHERE ts < ?", (int(older_than_ts),))
    return deleted


# --------------------------------------------------------------------------
# Чтение
# --------------------------------------------------------------------------


def read_quotes(
    since_ts: Optional[int] = None,
    until_ts: Optional[int] = None,
    venues: Optional[Sequence[str]] = None,
    venue_kinds: Optional[Sequence[str]] = None,
    assets: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Окно котировок в виде DataFrame.

    assets — если задан, оставляются только пары, где обе стороны из набора.
    """
    where, params = [], []
    if since_ts:
        where.append("ts >= ?"); params.append(int(since_ts))
    if until_ts:
        where.append("ts <= ?"); params.append(int(until_ts))
    if venues:
        where.append(f"venue IN ({','.join('?' * len(venues))})"); params += list(venues)
    if venue_kinds:
        where.append(f"venue_kind IN ({','.join('?' * len(venue_kinds))})"); params += list(venue_kinds)
    if assets:
        ph = ",".join("?" * len(assets))
        where.append(f"base IN ({ph}) AND quote IN ({ph})")
        params += list(assets) + list(assets)

    sql = "SELECT ts, venue, venue_kind, chain, base, quote, close, volume, liquidity_usd, pool FROM quotes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts"

    conn = connect(read_only=True)
    df = pd.read_sql_query(sql, conn, params=params)
    if not df.empty:
        df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def read_pools(chain: str, min_reserve_usd: float = 0.0) -> pd.DataFrame:
    conn = connect(read_only=True)
    return pd.read_sql_query(
        "SELECT * FROM pools WHERE chain=? AND COALESCE(reserve_usd,0) >= ? "
        "ORDER BY reserve_usd DESC",
        conn, params=(chain, min_reserve_usd),
    )


def read_fees() -> Dict[str, float]:
    conn = connect(read_only=True)
    rows = conn.execute("SELECT venue, taker_pct FROM fees").fetchall()
    return {r["venue"]: r["taker_pct"] for r in rows}


def read_state() -> pd.DataFrame:
    conn = connect(read_only=True)
    df = pd.read_sql_query("SELECT * FROM collector_state ORDER BY source, key", conn)
    for col in ("last_ts", "last_run"):
        if col in df.columns:
            df[col + "_dt"] = pd.to_datetime(df[col], unit="s", utc=True)
    return df


def stats() -> dict:
    """Сводка по содержимому базы — для панели состояния в UI."""
    conn = connect(read_only=True)
    row = conn.execute(
        """SELECT COUNT(*) n, MIN(ts) t0, MAX(ts) t1,
                  COUNT(DISTINCT venue) venues,
                  COUNT(DISTINCT base || '/' || quote) pairs
           FROM quotes"""
    ).fetchone()
    by_kind = conn.execute(
        "SELECT venue_kind, COUNT(*) n, COUNT(DISTINCT venue) v FROM quotes GROUP BY venue_kind"
    ).fetchall()
    return {
        "rows": row["n"] or 0,
        "t0": row["t0"],
        "t1": row["t1"],
        "venues": row["venues"] or 0,
        "pairs": row["pairs"] or 0,
        "by_kind": {r["venue_kind"]: {"rows": r["n"], "venues": r["v"]} for r in by_kind},
        "db_mb": round(DB_PATH.stat().st_size / 1e6, 1) if DB_PATH.exists() else 0.0,
    }


def db_exists() -> bool:
    return DB_PATH.exists()
