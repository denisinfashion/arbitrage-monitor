"""Проверка охвата: сколько активов и площадок тянет расчёт.

Повод. Потолок в сорок активов был выставлен по памяти, но настоящим
ограничением была скорость: полный перебор всех пар стоит T·n³, и при
двухстах активах это миллиарды операций. Поиск от якоря стоит T·n² —
на три порядка дешевле при том же результате для циклов, которые
начинаются и кончаются на USDT.

Тест проверяет обе стороны: что заложенная связка находится и на большой
сетке, и что расчёт укладывается в разумное время.

Запуск:  python -m history.tests.test_scale
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("ARB_DATA_DIR", tempfile.mkdtemp())

import numpy as np
import pandas as pd

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def synth(n_assets: int, n_venues: int, n_points: int, seed: int = 3):
    """Рынок из n_assets токенов: все торгуются к USDT на нескольких
    площадках, и в одну пару заложен устойчивый перекос."""
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000
    venues = [f"dex{v}" for v in range(n_venues)]
    names = ["USDT", "WBNB"] + [f"T{i:03d}" for i in range(n_assets - 2)]
    base_price = {a: float(rng.uniform(0.5, 500)) for a in names}
    base_price["USDT"] = 1.0
    base_price["WBNB"] = 600.0

    rows = []
    for k in range(n_points):
        ts = t0 + k * 300
        for i, a in enumerate(names[1:], start=1):
            p = base_price[a] * (1 + rng.normal(0, 0.0005))
            for v in venues:
                rows.append(dict(ts=ts, venue=v, venue_kind="dex", chain="bsc",
                                 base=a, quote="USDT", close=p, volume=1e6,
                                 liquidity_usd=8e6, pool=f"0x{v}{i}"))
            # вторая нога: часть токенов торгуется ещё и к WBNB
            if i % 3 == 0 and a != "WBNB":
                rows.append(dict(ts=ts, venue=venues[0], venue_kind="dex",
                                 chain="bsc", base=a, quote="WBNB",
                                 close=p / base_price["WBNB"], volume=1e6,
                                 liquidity_usd=8e6, pool=f"0xw{i}"))
    # заложенный перекос: T001 на последней площадке стоит на 1.2% дороже
    for r in rows:
        if r["base"] == "T001" and r["venue"] == venues[-1] and r["quote"] == "USDT":
            r["close"] *= 1.012
    return pd.DataFrame(rows), names


def main() -> int:
    from history.config import Settings
    from history.paths import (DENSE_LIMIT, _candidate_paths_anchored,
                               _candidate_paths_dense, find_cycles)
    from history.rates import build_grid, grid_bytes, max_assets_for

    print("\n1. Ёмкость считается от памяти, а не задана числом")
    check("при часовом шаге активов больше, чем при пятиминутном",
          max_assets_for(24) > max_assets_for(288),
          f"{max_assets_for(24)} против {max_assets_for(288)}")
    check("оценка памяти согласована с потолком",
          grid_bytes(288, max_assets_for(288)) <= 1_300_000_000,
          f"{grid_bytes(288, max_assets_for(288)) / 1e6:.0f} МБ")

    print("\n2. Малая сетка: две схемы поиска согласованы")
    df_small, _ = synth(12, 3, 20)
    s = Settings()
    s.analysis_timeframe = "5m"
    s.staleness_sec = 1800
    g_small = build_grid(df_small, settings=s, venue_kinds=["dex"],
                         drop_suspicious=False)
    a_idx = g_small.asset_index("USDT")
    anch = _candidate_paths_anchored(g_small, a_idx, 4, 1, 12)
    dense = set(_candidate_paths_dense(g_small, a_idx, 4, 1, 12))
    check("от якоря что-то найдено", len(anch) > 0, f"{len(anch)} маршрутов")
    check("полный перебор что-то нашёл", len(dense) > 0, f"{len(dense)} маршрутов")
    best_anch = max(
        (float(np.nanmax(_margin(g_small, p))) for p in anch), default=-99)
    best_dense = max(
        (float(np.nanmax(_margin(g_small, p))) for p in dense), default=-99)
    check("лучшая маржа совпадает у обеих схем",
          abs(best_anch - best_dense) < 1e-6,
          f"{best_anch:.6f}% против {best_dense:.6f}%")

    print("\n3. Большая сетка: 200 активов, 6 площадок")
    df_big, _ = synth(200, 6, 24)
    t0 = time.time()
    g_big = build_grid(df_big, settings=s, venue_kinds=["dex"],
                       max_assets=200, drop_suspicious=False)
    t_grid = time.time() - t0
    check("сетка построена", g_big.n_assets >= 150,
          f"{g_big.n_assets} активов, {len(g_big.venues)} площадок, "
          f"{t_grid:.1f} с")
    check("активов больше прежнего потолка в 40", g_big.n_assets > 40)
    check("площадок больше четырёх", len(g_big.venues) > 4)

    t0 = time.time()
    table, cycles = find_cycles(g_big, anchor="USDT", max_legs=4, top=20,
                                gas_per_dex_leg_usd=0.0, min_margin_pct=-100,
                                settings=s)
    t_search = time.time() - t0
    check("поиск уложился в 30 секунд", t_search < 30, f"{t_search:.1f} с")
    check("связки найдены", len(cycles) > 0, f"{len(cycles)}")
    check("заложенный перекос найден",
          any("T001" in c.label for c in cycles),
          "; ".join(c.label for c in cycles[:3]))
    check("полный перебор при таком n не запускался",
          g_big.n_assets > DENSE_LIMIT)

    print("\n4. Много площадок не мешает атрибуции")
    top = cycles[0]
    venues = top.dominant_venues()
    check("на каждую ногу назначена площадка",
          len(venues) == top.legs and all(v for v in venues),
          " → ".join(venues))

    print("\n" + ("ПРОВАЛЕНО: " + ", ".join(FAIL) if FAIL
                  else "Расчёт тянет сотни активов и произвольное число площадок"))
    return 1 if FAIL else 0


def _margin(grid, path):
    total = np.zeros(grid.n_times)
    for a, b in zip(path[:-1], path[1:]):
        total += grid.log_rate[:, a, b]
    with np.errstate(over="ignore"):
        return (np.exp(total) - 1.0) * 100.0


if __name__ == "__main__":
    sys.exit(main())
