"""Проверка переноса истории через снимок.

Главное, что здесь проверяется, — цикл, на котором держится облачный режим:
раннер GitHub одноразовый, поэтому каждый прогон разворачивает накопленное
из файла, дописывает свежее и выгружает обратно. Если этот цикл теряет данные
или ломается на повторном импорте, история в облаке никогда не накопится.

Запуск:  python -m history.tests.test_snapshot
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="arb_snap_")
os.environ["ARB_DATA_DIR"] = _TMP

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_candles(store, t_start: int, n: int, venue: str = "pancakeswap_v3"):
    out = []
    price = 600.0
    rng = np.random.default_rng(abs(hash(venue)) % 2**31)
    for k in range(n):
        price *= 1 + rng.normal(0, 0.0008)
        out.append(store.Candle(
            ts=t_start + k * 60, venue=venue, venue_kind="dex", chain="bsc",
            base="BNB", quote="USDT", close=price,
            volume=1000.0, liquidity_usd=5e6, pool="0xabc"))
    return out


def main() -> int:
    print("=" * 70)
    print("Проверка снимка истории")
    print("=" * 70)
    print(f"временная папка: {_TMP}")

    from history import snapshot, store
    from history.config import DATA_DIR

    now = int(time.time()) // 60 * 60

    # ---------------------------------------------------------------- цикл 1
    print("\n1. Первый прогон: пустой старт")
    store.init()
    n0 = snapshot.import_snapshot()
    check("импорт при отсутствии снимка не падает", n0 == 0)

    store.write_candles(make_candles(store, now - 300 * 60, 300))
    store.write_pools([{"chain": "bsc", "pool": "0xabc", "dex": "pancakeswap_v3",
                        "base": "BNB", "quote": "USDT", "reserve_usd": 5e6}])
    rows1 = store.stats()["rows"]
    path = snapshot.export_snapshot()
    check("снимок создан", path.exists(), f"{path.stat().st_size / 1024:.0f} КБ")
    check("файл пулов создан", path.with_name(snapshot.POOLS_NAME).exists())
    print(f"      записано {rows1} строк, снимок {path.stat().st_size / 1024:.0f} КБ")

    ratio = (rows1 * 60) / max(1, path.stat().st_size)
    print(f"      сжатие: ~{ratio:.1f}x против сырых данных")

    # ---------------------------------------------------------------- цикл 2
    print("\n2. Второй прогон: новый раннер, старая база стёрта")
    snapshot_bytes = path.read_bytes()
    pools_bytes = path.with_name(snapshot.POOLS_NAME).read_bytes()

    # эмулируем чистый раннер: удаляем базу, оставляем только файлы снимка
    for f in DATA_DIR.glob("*.sqlite*"):
        f.unlink()
    import history.store as st_mod
    for attr in ("conn_rw", "conn_ro"):
        if hasattr(st_mod._LOCAL, attr):
            delattr(st_mod._LOCAL, attr)

    path.write_bytes(snapshot_bytes)
    path.with_name(snapshot.POOLS_NAME).write_bytes(pools_bytes)

    store.init()
    check("база пуста после сброса", store.stats()["rows"] == 0)

    n = snapshot.import_snapshot()
    check("снимок развернулся", n == rows1, f"{n} против {rows1}")
    check("строки на месте", store.stats()["rows"] == rows1)
    check("пулы восстановлены", len(store.read_pools("bsc")) == 1)

    # дописываем свежее, как сделал бы сборщик
    store.write_candles(make_candles(store, now - 60 * 60, 60, venue="biswap"))
    rows2 = store.stats()["rows"]
    check("данные приросли", rows2 > rows1, f"{rows1} -> {rows2}")

    snapshot.export_snapshot()
    check("снимок перезаписан", path.exists())

    # ---------------------------------------------------------------- цикл 3
    print("\n3. Третий прогон: накопление не теряется")
    for f in DATA_DIR.glob("*.sqlite*"):
        f.unlink()
    for attr in ("conn_rw", "conn_ro"):
        if hasattr(st_mod._LOCAL, attr):
            delattr(st_mod._LOCAL, attr)
    store.init()
    snapshot.import_snapshot()
    rows3 = store.stats()["rows"]
    check("накопленное за два прогона сохранилось", rows3 == rows2,
          f"{rows3} против {rows2}")
    venues = {r["venue"] for _, r in snapshot.read_quotes().iterrows()}
    check("обе площадки в снимке", venues == {"pancakeswap_v3", "biswap"},
          str(sorted(venues)))

    # ------------------------------------------------------------ идемпотент
    print("\n4. Повторный импорт того же снимка")
    n_again = snapshot.import_snapshot()
    rows4 = store.stats()["rows"]
    check("апсерт не плодит дубликаты", rows4 == rows3, f"{rows3} -> {rows4}")
    check("импорт вернул число строк снимка", n_again == rows3)

    # -------------------------------------------------------------- фасад
    print("\n5. Фасад чтения")
    check("локальный режим определяется", not snapshot.cloud_mode())
    df = snapshot.read_quotes(since_ts=now - 100 * 60)
    check("окно фильтруется", not df.empty and df["ts"].min() >= now - 100 * 60,
          f"строк {len(df)}")
    s = snapshot.stats()
    check("сводка считается", s["rows"] == rows3 and s["source"] == "local")
    cov = snapshot.coverage()
    check("покрытие считается", len(cov) == 2, f"строк {len(cov)}")
    check("в покрытии нужные колонки",
          {"Площадка", "Тип", "Свечей", "Пар"} <= set(cov.columns),
          str(list(cov.columns)))
    check("подпись источника осмысленна", "локальная база" in snapshot.source_label(),
          snapshot.source_label())

    # ------------------------------------------------------- облачный режим
    print("\n6. Облачный режим определяется по настройке")
    os.environ[snapshot.ENV_SNAPSHOT_URL] = "https://example.invalid/history.parquet"
    check("режим переключился", snapshot.cloud_mode())
    check("подпись сменилась", "облак" in snapshot.source_label(),
          snapshot.source_label())
    # сети нет — фасад обязан вернуть пустоту, а не упасть
    empty = snapshot.read_quotes()
    check("недоступный снимок не роняет приложение", empty.empty)
    check("data_available отвечает честно", snapshot.data_available() is False)
    del os.environ[snapshot.ENV_SNAPSHOT_URL]

    print("\n" + "=" * 70)
    if FAILED:
        print(f"ПРОВАЛЕНО {len(FAILED)}:")
        for f in FAILED:
            print("  -", f)
        return 1
    print("Снимок переносит историю без потерь")
    return 0


if __name__ == "__main__":
    code = main()
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(code)
