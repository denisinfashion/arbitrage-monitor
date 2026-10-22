"""Спреды одного бессрочного фьючерса между биржами — с учётом funding.

Связка: лонг там, где дешевле (покупка по ask), шорт там, где дороже
(продажа по bid). Итог за время удержания H:

    спред  −  комиссии (4 сделки тейкером)  +  funding за H

funding за H = ставка_шорта × расчётов_шорта − ставка_лонга × расчётов_лонга:
при положительной ставке лонг платит, шорт получает. Расчёты считаются
по фактическому расписанию каждой биржи (интервал 1–8 ч, время ближайшего),
а ставка берётся текущая — она меняется, поэтому длинное удержание рискованнее.

Два итога:
  * «спред сошёлся» — цены выровнялись к выходу;
  * «только funding» — спред остался тем же, доход только со ставок.

Раз в минуту — снимок цен всех бессрочных USDT-фьючерсов 8 бирж,
ставки funding — раз в 5 минут. Связки выше порога пишутся в историю,
чтобы видеть, сколько минут спред держится.

Запуск: python -m leadlag.spreads
"""

from __future__ import annotations

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import funding as fnd
from . import notify
from .catalog import FUTURES, FUTURES_TAKER_PCT, futures_url
from .settings import LEADLAG_DIR, ScreenSettings
from .universe import _exchange, _num, split_multiplier

log = logging.getLogger(__name__)

SNAPSHOT_PATH = LEADLAG_DIR / "spreads_now.parquet"
DEPTH_PATH = LEADLAG_DIR / "spreads_depth.parquet"
DEPTH_TOP_PAIRS = 40              # для скольких лучших связок смотреть стаканы
DEPTH_NOTIONALS = (1000, 5000)    # объём сделки на ногу, $
HIST_DB = LEADLAG_DIR / "spreads.sqlite"
SNAPSHOT_EVERY_S = 60
FUNDING_EVERY_S = 300
MARKETS_EVERY_S = 6 * 3600
STALE_S = 180            # котировка старше — цена могла устареть
HIST_MIN_SPREAD_PCT = 0.2
# Средний спред: пара попадает под учёт, как только спред ≥ TRACK_MIN_SPREAD_PCT при
# объёме каждой ноги ≥ TRACK_MIN_VOLUME_USD, и дальше пишется каждую минуту сутки —
# иначе среднее считалось бы только по «горячим» минутам и было бы завышено.
TRACK_MIN_SPREAD_PCT = 0.1
TRACK_MIN_VOLUME_USD = 100_000
TRACK_FOR_S = 86400
_tracked: Dict[Tuple[str, str, str], float] = {}   # пара → до какого момента учитывать

SCHEMA = """
CREATE TABLE IF NOT EXISTS spread_hist (
    ts INTEGER NOT NULL, coin TEXT NOT NULL, long TEXT NOT NULL, short TEXT NOT NULL,
    spread_pct REAL, funding_day_pct REAL,
    PRIMARY KEY (coin, long, short, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS spread_recent (      -- поминутно, последний час
    ts INTEGER NOT NULL, coin TEXT NOT NULL, long TEXT NOT NULL, short TEXT NOT NULL,
    spread_pct REAL,
    PRIMARY KEY (coin, long, short, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS spread_hourly (      -- суммы по часам, неделя
    hour INTEGER NOT NULL, coin TEXT NOT NULL, long TEXT NOT NULL, short TEXT NOT NULL,
    s REAL NOT NULL, n INTEGER NOT NULL,
    PRIMARY KEY (coin, long, short, hour)
) WITHOUT ROWID;
"""


# ------------------------------------------------------------------- данные

class Venue:
    """Биржа с загруженным справочником рынков и кэшем funding."""

    def __init__(self, key: str):
        self.key = key
        v = FUTURES[key]
        self.ex = _exchange(v.ccxt_id, v.options)
        self.markets_at = 0.0
        self.funding: Dict[str, fnd.Funding] = {}
        self.funding_at = 0.0
        self.swaps: Dict[str, dict] = {}

    def refresh_markets(self) -> None:
        markets = self.ex.load_markets(reload=True)
        self.swaps = {s: m for s, m in markets.items()
                      if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
                      and m.get("active") is not False}
        self.markets_at = time.time()

    def depth(self, symbol: str) -> Optional[Tuple[list, list]]:
        """Стакан (bids, asks) с ценой за 1 монету и объёмом в монетах."""
        m = self.swaps.get(symbol)
        if m is None:
            return None
        ob = self.ex.fetch_order_book(symbol, 20)
        size = float(m.get("contractSize") or 1)           # объём бывает в контрактах
        _, mult = split_multiplier(str(m["base"]).upper())  # 1000PEPE: цена за 1000 монет
        conv = lambda side: [[lv[0] / mult, lv[1] * size * mult] for lv in side]
        return conv(ob["bids"]), conv(ob["asks"])

    def snapshot(self) -> List[dict]:
        now = time.time()
        if now - self.markets_at > MARKETS_EVERY_S:
            self.refresh_markets()
        if now - self.funding_at > FUNDING_EVERY_S:
            try:
                self.funding = fnd.FETCHERS[self.key](self.ex)
                self.funding_at = now
            except Exception as e:
                log.warning("funding %s: %s", self.key, str(e)[:200])
        tickers = self.ex.fetch_tickers(list(self.swaps))
        rows = []
        for sym, m in self.swaps.items():
            t = tickers.get(sym)
            if not t:
                continue
            last = _num(t.get("last")) or _num(t.get("close"))
            if not last:
                continue
            coin, mult = split_multiplier(str(m["base"]).upper())
            bid, ask = _num(t.get("bid")), _num(t.get("ask"))
            qv = _num(t.get("quoteVolume"))
            if qv is None:
                bv = _num(t.get("baseVolume"))
                qv = bv * last * float(m.get("contractSize") or 1) if bv else None
            f = self.funding.get(m["id"])
            rows.append({
                "venue": self.key, "symbol": sym, "url": futures_url(self.key, m["id"]),
                "coin": coin,
                "last": last / mult,
                "bid": bid / mult if bid else None,
                "ask": ask / mult if ask else None,
                "volume_usd": qv,
                "quote_ts": t.get("timestamp"),
                "f_rate": f.rate if f else None,
                "f_interval_h": f.interval_h if f else None,
                "f_next_ts": f.next_ts if f else None,
                "taker_pct": FUTURES_TAKER_PCT.get(self.key, 0.06),
            })
        return rows


# ------------------------------------------------------------------- расчёт

def pairs(m: pd.DataFrame, hold_h: float, now: Optional[float] = None) -> pd.DataFrame:
    """Все направленные пары бирж по каждой монете: лонг L, шорт S."""
    now = now or time.time()
    m = m.copy()
    m["bid_x"] = m["bid"].fillna(m["last"])
    m["ask_x"] = m["ask"].fillna(m["last"])
    m["n_settle"] = [fnd.settlements(nx if pd.notna(nx) else None, iv if pd.notna(iv) else 0, now, hold_h)
                     for nx, iv in zip(m["f_next_ts"], m["f_interval_h"])]
    m["f_day"] = np.where(m["f_interval_h"] > 0, m["f_rate"] * 24 / m["f_interval_h"], np.nan)
    cols = ["coin", "venue", "symbol", "url", "last", "bid", "ask", "bid_x", "ask_x", "volume_usd",
            "quote_ts", "f_rate", "f_interval_h", "f_next_ts", "n_settle", "f_day", "taker_pct"]
    L = m[cols].add_suffix("_l").rename(columns={"coin_l": "coin"})
    S = m[cols].add_suffix("_s").rename(columns={"coin_s": "coin"})
    p = L.merge(S, on="coin")
    p = p[p["venue_l"] != p["venue_s"]].copy()

    p["spread_pct"] = (p["bid_x_s"] / p["ask_x_l"] - 1) * 100
    p["fees_pct"] = 2 * (p["taker_pct_l"] + p["taker_pct_s"])
    # шорт получает свою ставку, лонг платит свою (при отрицательной — наоборот)
    p["funding_hold_pct"] = (p["f_rate_s"].fillna(0) * p["n_settle_s"]
                             - p["f_rate_l"].fillna(0) * p["n_settle_l"]) * 100
    p["funding_day_pct"] = (p["f_day_s"].fillna(0) - p["f_day_l"].fillna(0)) * 100
    p["pnl_converge_pct"] = p["spread_pct"] - p["fees_pct"] + p["funding_hold_pct"]
    p["pnl_carry_pct"] = p["funding_hold_pct"] - p["fees_pct"]
    p["price_ratio"] = np.maximum(p["last_l"], p["last_s"]) / np.minimum(p["last_l"], p["last_s"])
    p["min_volume_usd"] = np.minimum(p["volume_usd_l"].fillna(0), p["volume_usd_s"].fillna(0))
    p["funding_known"] = p["f_rate_l"].notna() & p["f_rate_s"].notna()
    p["by_last"] = p["bid_s"].isna() | p["ask_l"].isna()
    p["reasons"] = [explain(r, now) for r in p.itertuples()]
    return p


def explain(r, now: float) -> str:
    """Подсказки, чем может объясняться раздвижка (задача B2 — первые проверки)."""
    out = []
    if r.price_ratio > 1.3:
        out.append("цены отличаются в разы — возможно, разные токены под одним тикером")
    if r.spread_pct > 0 and r.funding_day_pct > 0:
        out.append("дорогой фьючерс платит более высокий funding — спред отчасти его цена")
    if r.spread_pct > 0 and r.funding_day_pct < 0:
        out.append("funding против связки: шорт платит больше, чем получает лонг")
    for side in ("l", "s"):
        ts = getattr(r, f"quote_ts_{side}")
        if ts and pd.notna(ts) and now - ts / 1000 > STALE_S:
            out.append(f"котировка {getattr(r, f'venue_{side}')} старше {STALE_S // 60} мин — торги могли встать")
    if r.by_last:
        out.append("нет bid/ask — расчёт по последней сделке")
    if not r.funding_known:
        out.append("ставка funding неизвестна")
    return "; ".join(out)


def depth_spreads(q: pd.DataFrame, venues: Dict[str, "Venue"]) -> pd.DataFrame:
    """Спред по средней цене исполнения на DEPTH_NOTIONALS долларов: покупка по
    уровням ask лонга, продажа по уровням bid шорта. Тонкий стакан — спред на
    объёме меньше, чем по лучшей цене, или его нет совсем."""
    from .paper import vwap
    books: Dict[Tuple[str, str], Optional[Tuple[list, list]]] = {}

    def book(venue: str, symbol: str):
        key = (venue, symbol)
        if key not in books:
            try:
                books[key] = venues[venue].depth(symbol)
            except Exception as e:
                log.warning("стакан %s %s: %s", venue, symbol, str(e)[:120])
                books[key] = None
        return books[key]

    rows = []
    for r in q.head(DEPTH_TOP_PAIRS).itertuples():
        bl, bs = book(r.venue_l, r.symbol_l), book(r.venue_s, r.symbol_s)
        row = {"coin": r.coin, "venue_l": r.venue_l, "venue_s": r.venue_s}
        for n in DEPTH_NOTIONALS:
            buy, got_l = vwap(bl[1], n) if bl else (None, 0)
            sell, got_s = vwap(bs[0], n) if bs else (None, 0)
            full = got_l >= n * 0.999 and got_s >= n * 0.999
            row[f"spread_{n}"] = (sell / buy - 1) * 100 if buy and sell and full else None
        rows.append(row)
    out = pd.DataFrame(rows)
    out["depth_ts"] = int(time.time())
    return out


def select(p: pd.DataFrame, s: ScreenSettings) -> pd.DataFrame:
    """Пороги и галочки из настроек; из двух направлений пары — лучшее."""
    if p is None or p.empty:
        return p
    q = p[p["venue_l"].isin(s.futures_exchanges) & p["venue_s"].isin(s.futures_exchanges)]
    q = q[(q["min_volume_usd"] >= s.spread_min_volume_usd)
          & (q["price_ratio"] <= s.spread_max_price_ratio)]
    q = q.assign(best_pnl=q[["pnl_converge_pct", "pnl_carry_pct"]].max(axis=1))
    q = q[q["best_pnl"] >= s.spread_min_pnl_pct]
    q = q.assign(pair=[" ".join(sorted((a, b))) for a, b in zip(q["venue_l"], q["venue_s"])])
    q = q.sort_values("best_pnl", ascending=False).drop_duplicates(["coin", "pair"])
    return q.drop(columns="pair")


# ---------------------------------------------------------------- история

def hist_connect():
    LEADLAG_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(HIST_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def record(con, p: pd.DataFrame, ts: int) -> None:
    hot = p[(p["spread_pct"] >= HIST_MIN_SPREAD_PCT) & (p["price_ratio"] <= 1.3)]
    con.executemany("INSERT OR REPLACE INTO spread_hist VALUES (?,?,?,?,?,?)",
                    [(ts, r.coin, r.venue_l, r.venue_s, float(r.spread_pct), float(r.funding_day_pct))
                     for r in hot.itertuples()])
    con.execute("DELETE FROM spread_hist WHERE ts < ?", (ts - 7 * 86400,))

    ok = p[(p["min_volume_usd"] >= TRACK_MIN_VOLUME_USD) & (p["price_ratio"] <= 1.3)]
    for r in ok[ok["spread_pct"] >= TRACK_MIN_SPREAD_PCT].itertuples():
        _tracked[(r.coin, r.venue_l, r.venue_s)] = ts + TRACK_FOR_S
    for k in [k for k, until in _tracked.items() if until < ts]:
        del _tracked[k]
    keys = list(zip(ok["coin"], ok["venue_l"], ok["venue_s"]))
    rows = [(*k, float(v)) for k, v in zip(keys, ok["spread_pct"]) if k in _tracked]
    hour = ts // 3600 * 3600
    con.executemany("INSERT OR REPLACE INTO spread_recent VALUES (?,?,?,?,?)",
                    [(ts, c, l, sh, v) for c, l, sh, v in rows])
    con.executemany("INSERT INTO spread_hourly VALUES (?,?,?,?,?,1) ON CONFLICT(coin, long, short, hour) "
                    "DO UPDATE SET s = s + excluded.s, n = n + 1",
                    [(hour, c, l, sh, v) for c, l, sh, v in rows])
    con.execute("DELETE FROM spread_recent WHERE ts < ?", (ts - 3900,))
    con.execute("DELETE FROM spread_hourly WHERE hour < ?", (ts - 8 * 86400,))
    con.commit()


def averages(con) -> pd.DataFrame:
    """Средний спред пары за последний час (поминутно) и за сутки (по часовым суммам)."""
    now = int(time.time())
    h1 = pd.read_sql_query(
        "SELECT coin, long AS venue_l, short AS venue_s, avg(spread_pct) AS spread_avg_1h, "
        "count(*) AS minutes_1h FROM spread_recent WHERE ts >= ? GROUP BY coin, long, short",
        con, params=(now - 3600,))
    h24 = pd.read_sql_query(
        "SELECT coin, long AS venue_l, short AS venue_s, sum(s) / sum(n) AS spread_avg_24h, "
        "sum(n) AS minutes_24h FROM spread_hourly WHERE hour >= ? GROUP BY coin, long, short",
        con, params=(now // 3600 * 3600 - 23 * 3600,))
    return h1.merge(h24, on=["coin", "venue_l", "venue_s"], how="outer")


def persistence(con, window_min: int = 60) -> pd.DataFrame:
    """Сколько снимков из последних window_min минут пара была в спреде ≥ порога."""
    since = int(time.time()) - window_min * 60
    return pd.read_sql_query(
        "SELECT coin, long AS venue_l, short AS venue_s, count(*) AS minutes_hot, "
        "avg(spread_pct) AS spread_avg FROM spread_hist WHERE ts >= ? GROUP BY coin, long, short",
        con, params=(since,))


def load_snapshot() -> Optional[pd.DataFrame]:
    if not SNAPSHOT_PATH.exists():
        return None
    return pd.read_parquet(SNAPSHOT_PATH)


# ------------------------------------------------------------------- запуск

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    venues = {k: Venue(k) for k in FUTURES}
    with closing(hist_connect()) as con:
        while True:
            started = time.time()

            def one(v: Venue):
                try:
                    return v.snapshot()
                except Exception as e:
                    log.warning("%s: %s", v.key, str(e)[:200])
                    return []

            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = [r for lst in pool.map(one, venues.values()) for r in lst]
            if rows:
                m = pd.DataFrame(rows)
                m["snap_ts"] = int(started)
                tmp = SNAPSHOT_PATH.with_suffix(".tmp")
                m.to_parquet(tmp, index=False)
                tmp.replace(SNAPSHOT_PATH)
                s = ScreenSettings.load()
                p = pairs(m, s.spread_hold_hours)
                record(con, p, int(started))
                try:
                    q = select(p, s).merge(averages(con), on=["coin", "venue_l", "venue_s"], how="left")
                    try:
                        d = depth_spreads(q, venues)
                        tmp = DEPTH_PATH.with_suffix(".tmp")
                        d.to_parquet(tmp, index=False)
                        tmp.replace(DEPTH_PATH)
                        q = q.merge(d, on=["coin", "venue_l", "venue_s"], how="left")
                    except Exception:
                        log.exception("глубина стаканов не посчитана")
                    notify.notify_spreads(q, persistence(con, notify.ALERT_HOT_WINDOW), s)
                except Exception:
                    log.exception("оповещения о спредах не отправлены")
                log.info("снимок: %d рынков, %d монет на 2+ биржах, %d пар со спредом ≥ %.1f%% за %.0f с",
                         len(m), int((m.groupby("coin")["venue"].nunique() > 1).sum()),
                         int((p["spread_pct"] >= HIST_MIN_SPREAD_PCT).sum()), HIST_MIN_SPREAD_PCT,
                         time.time() - started)
            time.sleep(max(5.0, SNAPSHOT_EVERY_S - (time.time() - started)))


if __name__ == "__main__":
    main()
