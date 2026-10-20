"""Поиск лага между инструментами одного токена.

Для каждой пары инструментов (фьючерс ↔ DEX, фьючерс ↔ фьючерс на разных
биржах, DEX ↔ DEX в разных сетях) и каждого закрытого часа:

  1. Цены на сетке в 1 с; секунда без сделок наследует прошлую цену.
  2. Логарифмические доходности по секундам.
  3. Корреляция доходностей A(t) и B(t+τ) для τ от −L до +L секунд.
     Пик при τ > 0 — A ведёт B на τ секунд; при τ = 0 — движутся синхронно.

Часовые оценки копятся в базе; по ним за окно (по умолчанию сутки) пара
получает итог: кто ведёт, медианный лаг, сила связи и сколько часов
направление и лаг держались. Именно устойчивость отличает рабочую связку
от случайного совпадения.

Запуск: python -m leadlag.lag [--every-minutes 30]
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import ticks
from .settings import LEADLAG_DIR, ScreenSettings

log = logging.getLogger(__name__)

RESULTS_PATH = LEADLAG_DIR / "lag_results.parquet"
HOUR = 3600
MIN_TRADES_PER_HOUR = 30     # у более редкого инструмента: меньше — оценка — шум
LAG_TOLERANCE_S = 2          # лаг часа «тот же», если отличается от медианы не больше

SCHEMA = """
CREATE TABLE IF NOT EXISTS lag_hours (
    a TEXT NOT NULL, b TEXT NOT NULL, hour INTEGER NOT NULL,
    lag INTEGER, corr REAL, corr0 REAL, llr REAL, n_a INTEGER, n_b INTEGER,
    PRIMARY KEY (a, b, hour)
) WITHOUT ROWID;
"""


# ------------------------------------------------------------------- расчёт

def price_grid(con: sqlite3.Connection, inst: str, t0: int, t1: int) -> Tuple[np.ndarray, int]:
    """Посекундная цена на [t0, t1) с переносом последней цены; число секунд со сделками."""
    rows = con.execute("SELECT ts, price FROM prices WHERE inst=? AND ts>=? AND ts<? ORDER BY ts",
                       (inst, t0, t1)).fetchall()
    prev = con.execute("SELECT price FROM prices WHERE inst=? AND ts<? AND ts>=? ORDER BY ts DESC LIMIT 1",
                       (inst, t0, t0 - HOUR)).fetchone()
    grid = np.full(t1 - t0, np.nan)
    for ts, price in rows:
        grid[ts - t0] = price
    if prev and np.isnan(grid[0]):
        grid[0] = prev[0]
    grid = pd.Series(grid).ffill().to_numpy()
    return grid, len(rows)


def returns(grid: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(grid))
    r[~np.isfinite(r)] = 0.0
    return r


def xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """corr(a[t], b[t+τ]) для τ = −max_lag…+max_lag. Положительный τ — a раньше b."""
    out = np.full(2 * max_lag + 1, np.nan)
    for k, tau in enumerate(range(-max_lag, max_lag + 1)):
        if tau >= 0:
            x, y = a[:len(a) - tau], b[tau:]
        else:
            x, y = a[-tau:], b[:len(b) + tau]
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            out[k] = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
    return out


@dataclass
class HourResult:
    lag: int
    corr: float
    corr0: float
    llr: float


def analyse(a: np.ndarray, b: np.ndarray, max_lag: int) -> Optional[HourResult]:
    c = xcorr(a, b, max_lag)
    if np.all(np.isnan(c)):
        return None
    k = int(np.nanargmax(c))
    pos, neg = c[max_lag + 1:], c[:max_lag]
    denom = float(np.nansum(neg ** 2))
    llr = float(np.nansum(pos ** 2)) / denom if denom > 0 else float("inf")
    return HourResult(k - max_lag, float(c[k]), float(c[max_lag]), llr)


def summarize(hours: pd.DataFrame, min_corr: float) -> Optional[dict]:
    """Часовые оценки пары → итог. Направление — большинством значимых часов."""
    if hours.empty:
        return None
    data = hours[(hours[["n_a", "n_b"]].min(axis=1) >= MIN_TRADES_PER_HOUR)]
    sig = data[data["corr"] >= min_corr]
    out = {"hours_data": len(data), "hours_sig": len(sig)}
    if sig.empty:
        return {**out, "direction": 0, "lag_s": None, "hours_stable": 0,
                "corr": float(data["corr"].median()) if len(data) else None,
                "corr0": float(data["corr0"].median()) if len(data) else None,
                "llr": None}
    signs = np.sign(sig["lag"])
    direction = int(signs.value_counts().idxmax())
    same = sig[signs == direction]
    lag = float(same["lag"].median())
    stable = same[(same["lag"] - lag).abs() <= max(LAG_TOLERANCE_S, 0.3 * abs(lag))]
    return {**out, "direction": direction, "lag_s": abs(lag), "hours_stable": len(stable),
            "corr": float(same["corr"].median()), "corr0": float(same["corr0"].median()),
            "llr": float(same["llr"].replace(np.inf, np.nan).median())}


# ------------------------------------------------------------------- прогон

def pairs_by_coin(con: sqlite3.Connection, since: int) -> Dict[str, List[str]]:
    rows = con.execute("SELECT inst, coin FROM instruments WHERE kind IN ('fut','dex') "
                       "AND updated_at >= ?", (since,)).fetchall()
    out: Dict[str, List[str]] = {}
    for inst, coin in rows:
        out.setdefault(coin, []).append(inst)
    return out


def run_once(s: Optional[ScreenSettings] = None) -> pd.DataFrame:
    s = s or ScreenSettings.load()
    with closing(ticks.connect()) as con:
        con.executescript(SCHEMA)
        now = int(time.time())
        last_closed = now // HOUR * HOUR           # начало текущего (незакрытого) часа
        first = last_closed - s.lag_window_hours * HOUR
        coins = pairs_by_coin(con, now - 7 * 86400)

        done = {(a, b, h) for a, b, h in con.execute(
            "SELECT a, b, hour FROM lag_hours WHERE hour >= ?", (first,))}
        computed = 0
        for coin, insts in coins.items():
            for hour in range(first, last_closed, HOUR):
                grids: Dict[str, Tuple[np.ndarray, int]] = {}
                for a, b in combinations(sorted(insts), 2):
                    if (a, b, hour) in done:
                        continue
                    for i in (a, b):
                        if i not in grids:
                            g, n = price_grid(con, i, hour, hour + HOUR)
                            grids[i] = (returns(g), n)
                    (ra, na), (rb, nb) = grids[a], grids[b]
                    res = analyse(ra, rb, s.max_lag_s) if na and nb else None
                    con.execute("INSERT OR REPLACE INTO lag_hours VALUES (?,?,?,?,?,?,?,?,?)",
                                (a, b, hour, res.lag if res else None, res.corr if res else None,
                                 res.corr0 if res else None, res.llr if res else None, na, nb))
                    computed += 1
            con.commit()
        con.execute("DELETE FROM lag_hours WHERE hour < ?", (now - 14 * 86400,))
        con.commit()

        hours = pd.read_sql_query("SELECT * FROM lag_hours WHERE hour >= ? AND corr IS NOT NULL",
                                  con, params=(first,))
        meta = pd.read_sql_query("SELECT inst, coin, kind, venue, symbol, url FROM instruments", con)
    log.info("посчитано %d пар-часов", computed)

    info = meta.set_index("inst").to_dict("index")
    rows = []
    for (a, b), g in hours.groupby(["a", "b"]):
        res = summarize(g.sort_values("hour"), s.min_corr)
        if res is None:
            continue
        last = g.sort_values("hour").iloc[-1]
        if res["direction"] >= 0:
            leader, follower = a, b
        else:
            leader, follower = b, a
        rows.append({
            "coin": info.get(a, {}).get("coin"),
            "leader": leader, "follower": follower,
            "leader_url": info.get(leader, {}).get("url"),
            "follower_url": info.get(follower, {}).get("url"),
            "synchronous": res["direction"] == 0,
            **res,
            "last_hour": int(last["hour"]),
            "last_lag": int(last["lag"]) if pd.notna(last["lag"]) else None,
            "last_corr": float(last["corr"]),
            "pair_kind": "-".join(sorted(i.split(":")[0] for i in (a, b))),
        })
    df = pd.DataFrame(rows)
    df.attrs["built_at"] = int(time.time())
    LEADLAG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_PATH, index=False)
    return df


def load_results() -> Optional[pd.DataFrame]:
    if not RESULTS_PATH.exists():
        return None
    return pd.read_parquet(RESULTS_PATH)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Поиск лага между инструментами одного токена")
    p.add_argument("--every-minutes", type=float, default=0, help="повторять каждые N минут")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    while True:
        started = time.time()
        try:
            df = run_once()
            log.info("пар с итогом: %d, с устойчивым лагом: %d", len(df),
                     int((df.get("hours_stable", pd.Series(dtype=int)) >= 3).sum()) if len(df) else 0)
        except Exception:
            if not args.every_minutes:
                raise
            log.exception("прогон упал, следующий по расписанию")
        if not args.every_minutes:
            return
        time.sleep(max(30.0, args.every_minutes * 60 - (time.time() - started)))


if __name__ == "__main__":
    main()
