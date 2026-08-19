"""Проверки отсева недостоверных данных.

Поводом послужил реальный случай: оповещения ушли со связкой
«USDT → BNB → MARSCOIN → USDT, +578%». Тесты воспроизводят обе причины,
по которым такое число может появиться, и проверяют, что теперь оно
не доходит ни до таблицы, ни до Telegram.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from history.quality import (CANONICAL, MIN_POOL_TURNOVER,
                             MIN_POOL_VOLUME_USD, implausible,
                             screen_pools)

REAL_USDT = CANONICAL["bsc"]["USDT"]
REAL_WBNB = CANONICAL["bsc"]["WBNB"]
FAKE = "0xdeadbeef00000000000000000000000000000001"
REAL_CAKE = CANONICAL["bsc"]["CAKE"]


def _pool(pool, base, quote, base_addr, quote_addr, volume=1e6, reserve=5e5):
    return {"chain": "bsc", "pool": pool, "dex": "pancakeswap_v3",
            "base": base, "quote": quote,
            "base_addr": base_addr, "quote_addr": quote_addr,
            "base_name": base, "quote_name": quote,
            "reserve_usd": reserve, "volume_24h": volume}


def test_1_clean_pools_pass():
    df = pd.DataFrame([
        _pool("0xaaa", "WBNB", "USDT", REAL_WBNB, REAL_USDT),
        _pool("0xbbb", "CAKE", "WBNB", REAL_CAKE, REAL_WBNB),
    ])
    s = screen_pools(df)
    assert not s.bad_pools, s.bad_pools
    assert not s.notes, s.notes
    assert s.address["CAKE"] == REAL_CAKE
    print("1 ok: чистый справочник проходит без потерь")


def test_2_quiet_pool_dropped():
    df = pd.DataFrame([
        _pool("0xaaa", "WBNB", "USDT", REAL_WBNB, REAL_USDT, volume=5e6),
        _pool("0xzzz", "GHOST", "USDT", FAKE, REAL_USDT, volume=10.0,
              reserve=9e5),
    ])
    s = screen_pools(df)
    assert "0xzzz" in s.bad_pools
    # Резерв у пустышки большой, но оборота нет — символ уходит целиком.
    assert "GHOST" in s.notes
    assert "оборот" in s.notes["GHOST"]
    print("2 ok: пул с накрученным резервом и без оборота отсеян")


def test_3_unknown_volume_is_not_punished():
    """Старый снимок мог не содержать оборота — это не повод всё выбросить."""
    df = pd.DataFrame([
        _pool("0xaaa", "WBNB", "USDT", REAL_WBNB, REAL_USDT, volume=float("nan")),
    ])
    s = screen_pools(df)
    assert not s.bad_pools
    print("3 ok: неизвестный оборот не считается нарушением")


def test_4_ticker_collision_resolved_by_canon():
    """Настоящий USDT и одноимённая подделка — разные узлы, не один."""
    df = pd.DataFrame([
        _pool("0xaaa", "WBNB", "USDT", REAL_WBNB, REAL_USDT, volume=9e6),
        # у подделки оборот выше — по весу победила бы она
        _pool("0xfff", "WBNB", "USDT", REAL_WBNB, FAKE, volume=9e9),
    ])
    s = screen_pools(df)
    assert s.address["USDT"] == REAL_USDT, s.address
    assert "0xfff" in s.bad_pools
    print("4 ok: спор тикеров решён в пользу канонического контракта")


def test_5_collision_without_canon_goes_by_volume():
    df = pd.DataFrame([
        _pool("0x111", "MARS", "USDT", FAKE, REAL_USDT, volume=1e5),
        _pool("0x222", "MARS", "USDT",
              "0xabc0000000000000000000000000000000000002", REAL_USDT,
              volume=4e6),
    ])
    s = screen_pools(df)
    assert s.address["MARS"].endswith("0002")
    assert "0x111" in s.bad_pools
    assert "0x222" not in s.bad_pools
    print("5 ok: без канона спор решается по обороту")


def test_6_anchor_never_dropped():
    """Даже если у USDT в снимке нет оборота, якорь остаётся."""
    df = pd.DataFrame([
        _pool("0xaaa", "WBNB", "USDT", REAL_WBNB, REAL_USDT, volume=1.0),
    ])
    s = screen_pools(df)
    assert "USDT" not in s.notes
    assert "WBNB" not in s.notes
    print("6 ok: опорные активы не отсеиваются")


def test_7_ceiling():
    assert implausible(578.6)
    assert implausible(5.1)
    assert not implausible(0.42)
    assert not implausible(float("nan"))
    print("7 ok: потолок правдоподобия срабатывает")


def test_8_grid_excludes_fake_pool():
    """Сквозная проверка: подделка не попадает в сетку курсов."""
    from history.config import SETTINGS
    from history.rates import build_grid

    ts = np.arange(1_700_000_000, 1_700_000_000 + 900 * 6, 900)
    rows = []
    for t in ts:
        rows.append(dict(ts=int(t), venue="pancakeswap_v3", venue_kind="dex",
                         chain="bsc", base="WBNB", quote="USDT", close=600.0,
                         volume=1e6, liquidity_usd=5e5, pool="0xaaa"))
        # тот же тикер USDT, но чужой контракт и абсурдный курс
        rows.append(dict(ts=int(t), venue="biswap", venue_kind="dex",
                         chain="bsc", base="WBNB", quote="USDT", close=4200.0,
                         volume=1e6, liquidity_usd=5e5, pool="0xfff"))
    quotes = pd.DataFrame(rows)

    pools = pd.DataFrame([
        _pool("0xaaa", "WBNB", "USDT", REAL_WBNB, REAL_USDT, volume=9e6),
        _pool("0xfff", "WBNB", "USDT", REAL_WBNB, FAKE, volume=9e9),
    ])

    import history.rates as rates
    orig = rates._pools_frame
    rates._pools_frame = lambda settings: pools
    try:
        grid = build_grid(quotes, settings=SETTINGS, venue_kinds=["dex"],
                          drop_suspicious=True)
        assert "biswap" not in grid.venues, grid.venues
        clean = build_grid(quotes, settings=SETTINGS, venue_kinds=["dex"],
                           drop_suspicious=False)
        assert "biswap" in clean.venues
    finally:
        rates._pools_frame = orig
    print("8 ok: пул с одноимённой подделкой не доходит до сетки")


def test_9_alert_dedup_and_ceiling():
    from history.alerts import AlertConfig, mute_key, pick

    class FakeGrid:
        trade_size_usd = 1000.0

    class FakeCycle:
        def __init__(self, assets, margin, liq=5e5):
            self.assets = tuple(assets)
            self._m = margin
            self._liq = liq
            self.grid = FakeGrid()

        @property
        def label(self):
            return " → ".join(self.assets)

        def margin_pct(self):
            return np.array([self._m])

        def bottleneck_liquidity(self):
            return self._liq

        def needs_transfer(self):
            return False

    cfg = AlertConfig()
    wild = FakeCycle(["USDT", "BNB", "MARSCOIN", "USDT"], 578.6)
    same_a = FakeCycle(["USDT", "BNB", "CAKE", "USDT"], 0.9)
    same_b = FakeCycle(["USDT", "CAKE", "BNB", "USDT"], 0.8)

    assert mute_key(same_a) == mute_key(same_b)
    chosen = pick([wild, same_a, same_b], cfg, {})
    assert len(chosen) == 1, [c.label for c in chosen]
    assert chosen[0] is same_a
    print("9 ok: абсурдная маржа не уходит, перестановки не дублируются")


def test_10_pandas_na_does_not_crash():
    """Регрессия: в облаке справочник приходит из Parquet с pd.NA.

    У pd.NA нет булева значения, поэтому проверка вида `if sym and addr`
    не возвращает False, а падает с TypeError. Локально этого не видно:
    из SQLite приходят обычные None. Приложение сломалось ровно так.
    """
    import history.rates as rates
    from history.config import SETTINGS
    from history.rates import _denullify, build_grid

    df = pd.DataFrame([
        _pool("0xaaa", "WBNB", "USDT", REAL_WBNB, REAL_USDT, volume=9e6),
        _pool("0xbbb", "CAKE", "WBNB", REAL_CAKE, REAL_WBNB, volume=3e6),
    ])
    # Имена и часть адресов отсутствуют — как у строк, записанных до
    # появления этих колонок. convert_dtypes даёт ровно те типы,
    # в которых pandas отдаёт Parquet.
    df.loc[0, "base_name"] = None
    df.loc[1, "quote_name"] = None
    df = df.convert_dtypes()
    assert str(df["base_name"].dtype) != "object", df.dtypes.to_dict()

    # Без приведения типов — TypeError, ради которого всё и написано.
    raised = False
    try:
        for r in df.to_dict("records"):
            if r["base_name"] and r["base_addr"]:
                pass
    except TypeError:
        raised = True
    print(f"   (проверка постановки задачи: сырой pd.NA даёт TypeError = {raised})")

    s = screen_pools(df)
    assert not s.bad_pools, s.bad_pools

    ts = np.arange(1_700_000_000, 1_700_000_000 + 900 * 5, 900)
    rows = []
    for t in ts:
        rows.append(dict(ts=int(t), venue="pancakeswap_v3", venue_kind="dex",
                         chain="bsc", base="WBNB", quote="USDT", close=600.0,
                         volume=1e6, liquidity_usd=5e5, pool="0xaaa"))
        rows.append(dict(ts=int(t), venue="pancakeswap_v3", venue_kind="dex",
                         chain="bsc", base="CAKE", quote="WBNB",
                         close=2.05 / 600, volume=1e6, liquidity_usd=5e5,
                         pool="0xbbb"))

    # Подменяем не _pools_frame, а источник под ним: приведение типов
    # должно происходить в самом производственном пути, а не в тесте.
    from history import snapshot
    orig = snapshot.pools
    snapshot.pools = lambda chain, min_reserve_usd=0.0: df
    try:
        grid = build_grid(pd.DataFrame(rows), settings=SETTINGS,
                          venue_kinds=["dex"], drop_suspicious=True)
        assert grid.token_address.get(("bsc", "CAKE")) == REAL_CAKE
        assert grid.token_name.get("CAKE") == "CAKE"
    finally:
        snapshot.pools = orig
    print("10 ok: пропуски из Parquet не роняют расчёт")


def test_11_broken_directory_degrades_quietly():
    """Сломанный справочник не должен уносить страницу целиком."""
    import history.rates as rates
    from history.config import SETTINGS
    from history.rates import build_grid

    class Exploding:
        empty = False
        columns = ["base"]

        def copy(self):
            raise RuntimeError("справочник повреждён")

    ts = np.arange(1_700_000_000, 1_700_000_000 + 900 * 4, 900)
    rows = [dict(ts=int(t), venue="pancakeswap_v3", venue_kind="dex",
                 chain="bsc", base="WBNB", quote="USDT", close=600.0,
                 volume=1e6, liquidity_usd=5e5, pool="0xaaa") for t in ts]

    orig = rates._pools_frame
    rates._pools_frame = lambda settings: Exploding()
    try:
        grid = build_grid(pd.DataFrame(rows), settings=SETTINGS,
                          venue_kinds=["dex"], drop_suspicious=True)
        assert "USDT" in grid.assets and "WBNB" in grid.assets
    finally:
        rates._pools_frame = orig
    print("11 ok: сбой справочника не ломает расчёт")


def test_12_turnover_beats_absolute_volume():
    """Живые данные по AAVE: абсолютный порог оборота отсекал здоровые пулы.

    Пул AAVE/USDT с резервом $12 159 и оборотом $4 696 за сутки
    оборачивается на 39% — работает. Прежний порог в $25 000 его
    выбрасывал, и у AAVE оставалась одна пара, из-за чего цикл через
    него не замыкался. Подделку же — резерв $900 000 при обороте $90 —
    порог по величине оборота пропустил бы.
    """
    A = "0xfb6115445bff7b52feb98650c87f44907e58f802"

    def pool(i, base, quote, reserve, volume, dex="uniswap-bsc"):
        addr = {"AAVE": A}.get
        return {"chain": "bsc", "pool": f"0x{i}", "dex": dex,
                "base": base, "quote": quote,
                "base_addr": addr(base) or f"0x{base.lower()}",
                "quote_addr": addr(quote) or f"0x{quote.lower()}",
                "base_name": "", "quote_name": "",
                "reserve_usd": reserve, "volume_24h": volume}

    df = pd.DataFrame([
        pool(0, "AAVE", "WBNB", 104_711, 58_057),                  # 55%
        pool(1, "AAVE", "WBNB", 49_119, 17_399, "pancakeswap-v3"), # 35%
        pool(2, "AAVE", "USDT", 12_159, 4_696),                    # 39%
        pool(3, "AAVE", "WBNB", 23_599, 492, "apeswap"),           # 2.1%
        pool(4, "BNB", "AAVE", 15_740, 18, "uniswap-v4"),          # 0.1%
        pool(5, "FAKE", "USDT", 900_000, 90),                      # 0.01%
    ])
    s = screen_pools(df)

    assert "0x0" not in s.bad_pools, "крупный живой пул выброшен"
    assert "0x1" not in s.bad_pools, "средний живой пул выброшен"
    assert "0x2" not in s.bad_pools, "маленький, но оборотистый пул выброшен"
    assert "0x4" in s.bad_pools, "мёртвый пул остался"
    assert "0x5" in s.bad_pools, "надутый резерв без оборота остался"

    # Главное следствие: у AAVE теперь два разных партнёра, а значит
    # цикл USDT -> AAVE -> WBNB -> USDT замыкается.
    alive = df[~df["pool"].isin(s.bad_pools)]
    partners = set(alive["base"]) | set(alive["quote"])
    partners -= {"AAVE"}
    assert {"WBNB", "USDT"} <= partners, partners
    assert "AAVE" not in s.notes, s.notes
    print(f"12 ok: оборот к резерву различает живой пул и надутый "
          f"(порог {MIN_POOL_TURNOVER * 100:.0f}% в сутки)")


def test_13_turnover_scales_with_pool_size():
    """Правило относительное: одинаковая оборачиваемость — одинаковый вердикт."""
    def pool(i, reserve, volume):
        return {"chain": "bsc", "pool": f"0x{i}", "dex": "d",
                "base": f"T{i}", "quote": "USDT",
                "base_addr": f"0xt{i}", "quote_addr": "0xu",
                "base_name": "", "quote_name": "",
                "reserve_usd": reserve, "volume_24h": volume}

    # 20% оборачиваемости при очень разных размерах
    df = pd.DataFrame([pool(0, 10_000, 2_000), pool(1, 10_000_000, 2_000_000)])
    s = screen_pools(df)
    assert not s.bad_pools, s.bad_pools

    # 0.5% оборачиваемости — тоже при разных размерах
    df2 = pd.DataFrame([pool(2, 100_000, 500), pool(3, 10_000_000, 50_000)])
    s2 = screen_pools(df2)
    assert {"0x2", "0x3"} <= s2.bad_pools, s2.bad_pools
    print("13 ok: вердикт зависит от оборачиваемости, а не от размера")


def run_all():
    import inspect
    mod = sys.modules[__name__]
    tests = [(n, f) for n, f in vars(mod).items()
             if n.startswith("test_") and inspect.isfunction(f)]
    tests.sort(key=lambda kv: kv[0])
    for name, fn in tests:
        fn()
    print(f"\nвсе проверки качества пройдены: {len(tests)}")


if __name__ == "__main__":
    run_all()
