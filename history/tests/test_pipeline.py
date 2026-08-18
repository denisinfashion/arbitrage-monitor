"""Сквозная проверка конвейера без сети: пишем синтетические котировки
в настоящую базу, читаем их обратно и прогоняем полный поиск связок.

Проверяется всё, кроме обращений к внешним API: схема SQLite, апсерт,
чтение окна, построение сетки, поиск, формирование таблицы и графика.

Запуск:  python -m history.tests.test_pipeline
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Отдельная база, чтобы не трогать рабочую
_TMP = tempfile.mkdtemp(prefix="arb_selftest_")
os.environ["ARB_DATA_DIR"] = _TMP

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    print("=" * 70)
    print("Сквозная проверка конвейера (без сети)")
    print("=" * 70)
    print(f"временная база: {_TMP}")

    from history import store
    from history.config import SETTINGS
    from history.paths import find_cycles
    from history.rates import build_grid

    # ---------------------------------------------------------------- схема
    print("\n1. Схема и запись")
    store.init()
    check("база создана", store.db_exists())

    now = int(time.time()) // 60 * 60
    n_steps = 240          # 4 часа минутных свечей
    candles = []

    # DEX-пулы на BNB Chain: USDT/BNB, BNB/CAKE, CAKE/USDT
    # с намеренным расхождением, дающим прибыльный цикл в части времени
    rng = np.random.default_rng(11)
    bnb = 600.0
    cake = 2.0
    for k in range(n_steps):
        ts = now - (n_steps - k) * 60
        bnb *= 1 + rng.normal(0, 0.0006)
        cake *= 1 + rng.normal(0, 0.0009)
        # в середине окна цена CAKE на второй площадке уходит вверх
        skew = 1.012 if 80 <= k < 140 else 1.0

        candles += [
            store.Candle(ts=ts, venue="pancakeswap_v3", venue_kind="dex", chain="bsc",
                         base="BNB", quote="USDT", close=bnb,
                         liquidity_usd=8_000_000, pool="0xpool1"),
            store.Candle(ts=ts, venue="pancakeswap_v3", venue_kind="dex", chain="bsc",
                         base="CAKE", quote="BNB", close=cake / bnb,
                         liquidity_usd=4_000_000, pool="0xpool2"),
            store.Candle(ts=ts, venue="biswap", venue_kind="dex", chain="bsc",
                         base="CAKE", quote="USDT", close=cake * skew,
                         liquidity_usd=2_000_000, pool="0xpool3"),
            store.Candle(ts=ts, venue="binance", venue_kind="cex", chain="",
                         base="BNB", quote="USDT", close=bnb * 1.0001, volume=5_000),
        ]

    written = store.write_candles(candles)
    check("свечи записаны", written == len(candles), f"{written} из {len(candles)}")

    # повторная запись тех же данных не должна плодить дубликаты
    again = store.write_candles(candles[:100])
    total = store.stats()["rows"]
    check("апсерт не плодит дубликаты", total == len(candles),
          f"строк в базе: {total}")

    store.write_pools([{"chain": "bsc", "pool": "0xpool1", "dex": "pancakeswap_v3",
                        "base": "BNB", "quote": "USDT", "reserve_usd": 8e6}])
    check("справочник пулов пишется", len(store.read_pools("bsc")) == 1)

    store.set_state("test", "key", last_ts=now, ok=True, rows=5)
    check("состояние сборщика пишется", store.get_last_ts("test", "key") == now)

    # ---------------------------------------------------------------- чтение
    print("\n2. Чтение окна")
    df = store.read_quotes(since_ts=now - 3600)
    check("окно читается", not df.empty, f"строк: {len(df)}")
    check("колонка времени есть", "dt" in df.columns)
    check("окно действительно ограничено", df["ts"].min() >= now - 3600)

    dex_only = store.read_quotes(since_ts=now - 7200, venue_kinds=["dex"])
    check("фильтр по типу площадки работает",
          set(dex_only["venue_kind"]) == {"dex"})

    # ------------------------------------------------------------- аналитика
    print("\n3. Сетка и поиск")
    quotes = store.read_quotes(since_ts=now - n_steps * 60 - 60)
    grid = build_grid(quotes, settings=SETTINGS, trade_size_usd=1000,
                      venue_kinds=["dex"], apply_slippage=True)
    check("сетка построена", grid.n_times > 200, f"точек: {grid.n_times}")
    check("активы распознаны", set(grid.assets) == {"USDT", "BNB", "CAKE"},
          str(grid.assets))
    check("площадки распознаны", "pancakeswap_v3" in grid.venues and "biswap" in grid.venues,
          str(grid.venues))
    print(f"      заполненность сетки: {grid.coverage() * 100:.1f}%")

    table, cycles = find_cycles(grid, anchor="USDT", max_legs=3, top=20,
                                gas_per_dex_leg_usd=0.10, min_margin_pct=-100,
                                settings=SETTINGS)
    check("поиск отработал", not table.empty, f"связок: {len(table)}")
    if table.empty:
        return 1

    print(f"\n      таблица связок:")
    for _, r in table.head(4).iterrows():
        print(f"        {r['Связка']:34s} медиана {r['Медиана %']:+.4f}%  "
              f"макс {r['Макс %']:+.4f}%  в плюсе {r['В плюсе %']}%")

    cols = set(table.columns)
    need = {"Связка", "Ног", "Макс %", "Медиана %",
            "В плюсе %", "Маршрут"}
    check("в таблице все нужные колонки", need <= cols, str(sorted(need - cols)))

    # Заложенное расхождение: цикл через оба DEX прибылен ровно в окне 80–140
    # из 240 точек, то есть четверть времени.
    target = table[table["Связка"] == "USDT → BNB → CAKE → USDT"]
    check("заложенная связка найдена", not target.empty)
    if target.empty:
        return 1
    row = target.iloc[0]
    check("связка проходит через оба DEX",
          "biswap" in row["Маршрут"] and "pancakeswap" in row["Маршрут"],
          row["Маршрут"])
    check("максимум маржи положителен", row["Макс %"] > 0,
          f"{row['Макс %']}%")
    check("доля прибыльного времени ≈ 25%",
          abs(row["В плюсе %"] - 25.0) < 3.0,
          f"{row['В плюсе %']}%")
    check("ранжирование по окнам ставит её первой",
          table.iloc[0]["Связка"] == "USDT → BNB → CAKE → USDT",
          table.iloc[0]["Связка"])

    # сортировка по медиане должна дать другой порядок
    t_med, _ = find_cycles(grid, anchor="USDT", max_legs=3, top=20,
                           gas_per_dex_leg_usd=0.10, min_margin_pct=-100,
                           sort_by="медиана", settings=SETTINGS)
    check("сортировка по медиане меняет порядок",
          not t_med.empty and t_med.iloc[0]["Связка"] != table.iloc[0]["Связка"],
          t_med.iloc[0]["Связка"] if not t_med.empty else "пусто")

    # ---------------------------------------------------------------- график
    print("\n4. Данные для графика")
    cyc = cycles[0]
    frame = cyc.to_frame()
    check("временной ряд построен", len(frame) == grid.n_times, f"{len(frame)} строк")
    check("есть колонка произведения курсов", "Произведение курсов" in frame.columns)
    check("есть колонка маржи", "Маржа, %" in frame.columns)
    check("есть колонки по ногам",
          sum(c.startswith("Нога ") for c in frame.columns) == cyc.legs)
    finite = np.isfinite(frame["Маржа, %"]).sum()
    check("ряд заполнен значениями", finite > grid.n_times * 0.5,
          f"{finite} из {grid.n_times}")

    # прибыльное окно должно попасть примерно в середину
    m = cyc.margin_pct()
    pos = np.where(np.nan_to_num(m, nan=-1) > 0)[0]
    if len(pos):
        print(f"      окно в плюсе: точки {pos.min()}–{pos.max()} "
              f"из {grid.n_times} (закладывали 80–140)")
        check("прибыльное окно там, где закладывали",
              60 <= pos.min() <= 100 and 120 <= pos.max() <= 160,
              f"{pos.min()}–{pos.max()}")

    # ---------------------------------------------------------------- прочее
    print("\n5. Обслуживание базы")
    before = store.stats()["rows"]
    deleted = store.prune(now - 3600)
    after = store.stats()["rows"]
    check("обрезка глубины работает", after < before and deleted > 0,
          f"{before} -> {after}, удалено {deleted}")

    s = store.stats()
    check("сводка считается", s["rows"] > 0 and s["venues"] > 0)
    print(f"      итог: {s['rows']} строк, {s['venues']} площадок, {s['db_mb']} МБ")

    print("\n" + "=" * 70)
    if FAILED:
        print(f"ПРОВАЛЕНО {len(FAILED)}:")
        for f in FAILED:
            print("  -", f)
        return 1
    print("Конвейер работает от записи до графика")
    return 0


if __name__ == "__main__":
    code = main()
    import shutil
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(code)
