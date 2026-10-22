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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import funding as fnd
from . import notify
from .catalog import FUTURES, FUTURES_TAKER_PCT, futures_url
from .settings import LEADLAG_DIR, ScreenSettings
from .universe import _exchange, _num, split_multiplier

log = logging.getLogger(__name__)

SNAPSHOT_PATH = LEADLAG_DIR / "spreads_now.parquet"
HIST_DB = LEADLAG_DIR / "spreads.sqlite"
SNAPSHOT_EVERY_S = 60
FUNDING_EVERY_S = 300
MARKETS_EVERY_S = 6 * 3600
STALE_S = 180            # котировка старше — цена могла устареть
HIST_MIN_SPREAD_PCT = 0.2

SCHEMA = """
CREATE TABLE IF NOT EXISTS spread_hist (
    ts INTEGER NOT NULL, coin TEXT NOT NULL, long TEXT NOT NULL, short TEXT NOT NULL,
    spread_pct REAL, funding_day_pct REAL,
    PRIMARY KEY (coin, long, short, ts)
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
    con.commit()


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
                    notify.notify_spreads(select(p, s), persistence(con, notify.ALERT_HOT_WINDOW), s)
                except Exception:
                    log.exception("оповещения о спредах не отправлены")
                log.info("снимок: %d рынков, %d монет на 2+ биржах, %d пар со спредом ≥ %.1f%% за %.0f с",
                         len(m), int((m.groupby("coin")["venue"].nunique() > 1).sum()),
                         int((p["spread_pct"] >= HIST_MIN_SPREAD_PCT).sum()), HIST_MIN_SPREAD_PCT,
                         time.time() - started)
            time.sleep(max(5.0, SNAPSHOT_EVERY_S - (time.time() - started)))


if __name__ == "__main__":
    main()
