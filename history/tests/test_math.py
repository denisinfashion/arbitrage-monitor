"""Проверка математики: подкладываем данные с ЗАРАНЕЕ ИЗВЕСТНОЙ связкой
и требуем, чтобы поиск нашёл именно её и посчитал маржу с точностью до
округления.

Запуск:  python -m history.tests.test_math
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from ..config import Settings
from ..paths import evaluate_path, find_cycles
from ..rates import RateGrid, build_grid, cex_slippage_factor, dex_slippage_factor

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# --------------------------------------------------------------------------


def test_slippage_models():
    print("\n1. Модели проскальзывания")

    # Пул постоянного произведения: резерв $200k -> R_in = $100k.
    # Своп на $1000: фактор = 100000 / 101000
    f = dex_slippage_factor(1_000, 200_000)
    check("DEX: формула x*y=k", close(f, 100_000 / 101_000),
          f"{f:.6f} против {100_000/101_000:.6f}")

    # Чем больше пул, тем ближе к единице
    check("DEX: большой пул -> нет удара",
          dex_slippage_factor(1_000, 1e9) > 0.999995)

    # Своп размером с половину резерва режет курс примерно вдвое
    check("DEX: своп в размер резерва", close(dex_slippage_factor(100_000, 200_000), 0.5))

    check("DEX: нет данных о резерве -> без поправки",
          dex_slippage_factor(1_000, None) == 1.0)

    # CEX: сделка в 1% минутного оборота -> удар 0.5%
    f = cex_slippage_factor(1_000, 100_000)
    check("CEX: доля оборота", close(f, 1 - 0.005), f"{f:.6f}")

    check("CEX: потолок удара", cex_slippage_factor(1e9, 1.0) == 1 - 200 / 1e4)


def make_quotes(with_arb: bool = True) -> pd.DataFrame:
    """Синтетический рынок с одной заложенной связкой.

    Три актива: USDT, BNB, CAKE. Курсы подобраны так, чтобы цикл
    USDT -> BNB -> CAKE -> USDT давал ровно +1% ДО комиссий.
    """
    t0 = 1_700_000_000
    rows = []
    n_steps = 60

    for k in range(n_steps):
        ts = t0 + k * 60
        # базовые курсы
        bnb_usdt = 600.0
        cake_bnb = 1 / 300.0          # 1 CAKE = 1/300 BNB  -> 2 USDT
        cake_usdt = 2.0
        if with_arb:
            # завышаем цену CAKE в USDT на 1% -> цикл USDT->BNB->CAKE->USDT в плюсе
            cake_usdt = 2.0 * 1.01

        for base, quote, price, venue, kind, liq, vol in [
            ("BNB", "USDT", bnb_usdt, "venueA", "cex", None, 1e9),
            ("CAKE", "BNB", cake_bnb, "venueA", "cex", None, 1e9),
            ("CAKE", "USDT", cake_usdt, "venueB", "cex", None, 1e9),
        ]:
            rows.append(dict(ts=ts, venue=venue, venue_kind=kind, chain="",
                             base=base, quote=quote, close=price,
                             volume=vol, liquidity_usd=liq, pool=None))
    return pd.DataFrame(rows)


def test_grid_and_cycle():
    print("\n2. Сетка курсов и расчёт связки")

    s = Settings()
    s.timeframe = "1m"
    s.staleness_sec = 180
    s.quote_asset = "USDT"

    df = make_quotes(with_arb=True)
    grid = build_grid(df, settings=s, trade_size_usd=1_000, apply_slippage=False)

    check("активы в сетке", set(grid.assets) == {"USDT", "BNB", "CAKE"},
          str(grid.assets))
    check("точек времени", grid.n_times == 60, str(grid.n_times))

    i_u, i_b, i_c = (grid.asset_index(a) for a in ("USDT", "BNB", "CAKE"))

    # курс USDT->BNB должен быть 1/600 с поправкой на комиссию 0.10%
    r = float(np.exp(grid.log_rate[0, i_u, i_b]))
    expected = (1 / 600.0) * (1 - 0.001)
    check("курс USDT->BNB с комиссией", close(r, expected, 1e-5),
          f"{r:.8f} против {expected:.8f}")

    # обратный курс BNB->USDT
    r2 = float(np.exp(grid.log_rate[0, i_b, i_u]))
    check("курс BNB->USDT с комиссией", close(r2, 600.0 * 0.999, 1e-5), f"{r2:.4f}")

    # Полный цикл. Три ноги по 0.10% -> множитель 0.999^3
    cyc = evaluate_path(grid, [i_u, i_b, i_c, i_u])
    got = float(cyc.margin_pct()[0])
    theory = (1.01 * 0.999 ** 3 - 1) * 100
    check("маржа цикла USDT->BNB->CAKE->USDT", close(got, theory, 1e-4),
          f"{got:.6f}% против {theory:.6f}%")
    check("цикл в плюсе", got > 0, f"{got:.4f}%")


def test_no_false_positive():
    print("\n3. Отсутствие ложных срабатываний на согласованном рынке")

    s = Settings()
    s.timeframe = "1m"
    df = make_quotes(with_arb=False)
    grid = build_grid(df, settings=s, trade_size_usd=1_000, apply_slippage=False)

    i_u, i_b, i_c = (grid.asset_index(a) for a in ("USDT", "BNB", "CAKE"))
    cyc = evaluate_path(grid, [i_u, i_b, i_c, i_u])
    got = float(cyc.margin_pct()[0])
    check("на согласованных курсах маржа отрицательна (съедена комиссией)",
          got < 0, f"{got:.6f}%")

    tbl, cycles = find_cycles(grid, anchor="USDT", max_legs=4, top=10,
                              gas_per_dex_leg_usd=0.0, settings=s)
    check("поиск не выдаёт прибыльных связок", tbl.empty or (tbl["Макс %"] <= 0).all(),
          f"строк: {len(tbl)}")


def test_search_finds_it():
    print("\n4. Поиск находит заложенную связку")

    s = Settings()
    s.timeframe = "1m"
    s.min_margin_pct = 0.0
    df = make_quotes(with_arb=True)
    grid = build_grid(df, settings=s, trade_size_usd=1_000, apply_slippage=False)

    tbl, cycles = find_cycles(grid, anchor="USDT", max_legs=4, top=20,
                              gas_per_dex_leg_usd=0.0, settings=s)
    check("что-то найдено", not tbl.empty, f"строк: {len(tbl)}")
    if tbl.empty:
        return

    best = tbl.iloc[0]["Связка"]
    print(f"      лучшая связка: {best}  ({tbl.iloc[0]['Медиана %']}%)")
    check("найден правильный цикл",
          best in ("USDT → BNB → CAKE → USDT",),
          best)
    check("все ноги начинаются и кончаются на USDT",
          all(c.assets[0] == "USDT" and c.assets[-1] == "USDT" for c in cycles))
    check("нет повторов активов внутри цикла",
          all(len(set(c.assets[1:-1])) == len(c.assets[1:-1]) for c in cycles))


def test_staleness():
    print("\n5. Ограничитель свежести")

    s = Settings()
    s.timeframe = "1m"
    s.staleness_sec = 120          # 2 шага

    t0 = 1_700_000_000
    rows = []
    # BNB/USDT есть всё время, CAKE/BNB обрывается на шаге 5
    for k in range(20):
        ts = t0 + k * 60
        rows.append(dict(ts=ts, venue="v", venue_kind="cex", chain="",
                         base="BNB", quote="USDT", close=600.0,
                         volume=1e9, liquidity_usd=None, pool=None))
        if k <= 5:
            rows.append(dict(ts=ts, venue="v", venue_kind="cex", chain="",
                             base="CAKE", quote="BNB", close=1 / 300.0,
                             volume=1e9, liquidity_usd=None, pool=None))
    grid = build_grid(pd.DataFrame(rows), settings=s, apply_slippage=False)

    i_c, i_b = grid.asset_index("CAKE"), grid.asset_index("BNB")
    series = grid.log_rate[:, i_c, i_b]
    check("до обрыва курс есть", np.isfinite(series[5]))
    check("протяжка работает в пределах допуска", np.isfinite(series[7]),
          "шаг 7 = обрыв + 2")
    check("за пределом допуска курса нет", not np.isfinite(series[9]),
          "шаг 9 = обрыв + 4")


def test_maxplus_correctness():
    print("\n6. Max-plus против прямого перебора")

    rng = np.random.default_rng(42)
    n, T = 8, 3
    W = rng.normal(0, 0.01, size=(T, n, n)).astype(np.float32)
    # немного дыр
    W[0, 2, 3] = -np.inf

    from ..paths import _maxplus
    M, arg = _maxplus(W, W)

    ok = True
    for t in range(T):
        for i in range(n):
            for k in range(n):
                brute = max(W[t, i, j] + W[t, j, k] for j in range(n))
                if not np.isclose(M[t, i, k], brute, atol=1e-5):
                    ok = False
    check("max-plus совпадает с перебором", ok)

    # argmax указывает на тот же j
    ok2 = all(
        np.isclose(W[t, i, arg[t, i, k]] + W[t, arg[t, i, k], k], M[t, i, k], atol=1e-5)
        for t in range(T) for i in range(n) for k in range(n)
        if arg[t, i, k] >= 0 and np.isfinite(M[t, i, k])
    )
    check("argmax указывает на верный промежуточный узел", ok2)


def make_four_leg_market() -> pd.DataFrame:
    """Рынок, где прибыльна РОВНО ОДНА связка и ровно из четырёх ног.

    Пары заданы цепочкой A/USDT, B/A, C/B, C/USDT, поэтому короткие циклы
    либо не замыкаются (нет прямой пары), либо уходят в минус на комиссии.
    Единственный плюс — USDT → A → B → C → USDT: 1·1·1·1.02.
    """
    t0 = 1_700_000_000
    rows = []
    for k in range(40):
        ts = t0 + k * 60
        for base, quote, price in [("A", "USDT", 1.0), ("B", "A", 1.0),
                                   ("C", "B", 1.0), ("C", "USDT", 1.02)]:
            rows.append(dict(ts=ts, venue="v", venue_kind="cex", chain="",
                             base=base, quote=quote, close=price,
                             volume=1e9, liquidity_usd=None, pool=None))
    return pd.DataFrame(rows)


def test_four_leg_reconstruction():
    print("\n7. Связка из четырёх ног и восстановление маршрута")

    s = Settings()
    s.timeframe = "1m"
    s.min_margin_pct = 0.0
    grid = build_grid(make_four_leg_market(), settings=s, apply_slippage=False)

    idx = [grid.asset_index(a) for a in ("USDT", "A", "B", "C", "USDT")]
    cyc = evaluate_path(grid, idx)
    got = float(cyc.margin_pct()[0])
    theory = (1.02 * 0.999 ** 4 - 1) * 100
    check("маржа четырёхногого цикла", close(got, theory, 1e-4),
          f"{got:.6f}% против {theory:.6f}%")

    # короткие циклы должны быть в минусе
    short = evaluate_path(grid, [grid.asset_index("USDT"), grid.asset_index("A"),
                                 grid.asset_index("USDT")])
    check("двухногий цикл в минусе", float(short.margin_pct()[0]) < 0,
          f"{float(short.margin_pct()[0]):.4f}%")

    tbl, cycles = find_cycles(grid, anchor="USDT", max_legs=4, top=10,
                              gas_per_dex_leg_usd=0.0, settings=s)
    check("поиск нашёл связку", not tbl.empty, f"строк: {len(tbl)}")
    if tbl.empty:
        return
    best = tbl.iloc[0]["Связка"]
    print(f"      лучшая связка: {best}  ({tbl.iloc[0]['Медиана %']}%)")
    check("восстановлен верный маршрут из четырёх ног",
          best == "USDT → A → B → C → USDT", best)
    check("длина маршрута = 4 ноги", int(tbl.iloc[0]["Ног"]) == 4)

    # при ограничении в 3 ноги эта связка найтись не должна
    tbl3, _ = find_cycles(grid, anchor="USDT", max_legs=3, top=10,
                          gas_per_dex_leg_usd=0.0, settings=s)
    check("при max_legs=3 прибыльных связок нет",
          tbl3.empty or (tbl3["Макс %"] <= 0).all(), f"строк: {len(tbl3)}")


def test_reconstruction_uses_correct_timestep():
    print("\n8. Восстановление маршрута берёт нужный момент времени")

    # Первая половина истории: прибылен USDT->A->B->C->USDT.
    # Вторая половина: прибылен USDT->A->D->C->USDT, а маршрут через B убыточен.
    # Если восстановление всегда смотрит на t=0, второй маршрут не найдётся.
    t0 = 1_700_000_000
    rows = []
    for k in range(40):
        ts = t0 + k * 60
        first_half = k < 20
        pairs = [("A", "USDT", 1.0), ("C", "USDT", 1.02),
                 ("B", "A", 1.0 if first_half else 1.30),
                 ("C", "B", 1.0 if first_half else 1.30),
                 ("D", "A", 1.30 if first_half else 1.0),
                 ("C", "D", 1.30 if first_half else 1.0)]
        for base, quote, price in pairs:
            rows.append(dict(ts=ts, venue="v", venue_kind="cex", chain="",
                             base=base, quote=quote, close=price,
                             volume=1e9, liquidity_usd=None, pool=None))

    s = Settings()
    s.timeframe = "1m"
    s.min_margin_pct = -100.0
    grid = build_grid(pd.DataFrame(rows), settings=s, apply_slippage=False)
    tbl, cycles = find_cycles(grid, anchor="USDT", max_legs=4, top=40,
                              gas_per_dex_leg_usd=0.0, settings=s)

    labels = set(tbl["Связка"]) if not tbl.empty else set()
    check("найден маршрут первой половины",
          "USDT → A → B → C → USDT" in labels)
    check("найден маршрут второй половины",
          "USDT → A → D → C → USDT" in labels,
          f"найдено: {sorted(labels)[:4]}")


def test_venue_attribution():
    print("\n9. Атрибуция площадок")

    # Две площадки на одну пару. Дешевле купить на 'cheap', дороже продать
    # на 'rich'. Порядок строк подобран так, чтобы «последняя выигрывает»
    # давала неверный ответ: в базе строки идут по алфавиту площадок.
    t0 = 1_700_000_000
    rows = []
    for k in range(30):
        ts = t0 + k * 60
        # BNB дешевле на 'aaa_cheap', дороже на 'zzz_rich'
        rows.append(dict(ts=ts, venue="aaa_cheap", venue_kind="cex", chain="",
                         base="BNB", quote="USDT", close=600.0,
                         volume=1e9, liquidity_usd=None, pool=None))
        rows.append(dict(ts=ts, venue="zzz_rich", venue_kind="cex", chain="",
                         base="BNB", quote="USDT", close=606.0,
                         volume=1e9, liquidity_usd=None, pool=None))

    s = Settings()
    s.timeframe = "1m"
    s.min_margin_pct = -100.0
    grid = build_grid(pd.DataFrame(rows), settings=s, apply_slippage=False)

    i_u, i_b = grid.asset_index("USDT"), grid.asset_index("BNB")

    # лучший курс USDT->BNB — там, где BNB дешевле
    v_buy = grid.venues[int(grid.venue_idx[0, i_u, i_b])]
    check("покупка отнесена к площадке с дешёвой ценой", v_buy == "aaa_cheap", v_buy)

    # лучший курс BNB->USDT — там, где BNB дороже
    v_sell = grid.venues[int(grid.venue_idx[0, i_b, i_u])]
    check("продажа отнесена к площадке с дорогой ценой", v_sell == "zzz_rich", v_sell)

    # и сам курс должен соответствовать лучшей цене, а не последней строке
    rate = float(np.exp(grid.log_rate[0, i_b, i_u]))
    check("курс взят по лучшей цене", close(rate, 606.0 * 0.999, 1e-5), f"{rate:.4f}")

    cyc = evaluate_path(grid, [i_u, i_b, i_u])
    got = float(cyc.margin_pct()[0])
    theory = ((606.0 / 600.0) * 0.999 ** 2 - 1) * 100
    check("маржа двухногого межбиржевого цикла", close(got, theory, 1e-4),
          f"{got:.4f}% против {theory:.4f}%")

    venues = cyc.dominant_venues()
    check("в цикле две РАЗНЫЕ площадки", venues == ["aaa_cheap", "zzz_rich"],
          " | ".join(venues))


def test_scale():
    print("\n10. Масштаб: 60 активов, 1440 точек")

    s = Settings()
    s.timeframe = "1m"
    rng = np.random.default_rng(7)
    n_assets, n_steps = 60, 1440
    assets = ["USDT"] + [f"T{i}" for i in range(n_assets - 1)]
    base_price = {a: float(rng.uniform(0.5, 500)) for a in assets}
    base_price["USDT"] = 1.0

    t0 = 1_700_000_000
    rows = []
    for k in range(n_steps):
        ts = t0 + k * 60
        drift = 1 + rng.normal(0, 0.0005)
        for a in assets[1:]:
            base_price[a] *= drift
            rows.append(dict(ts=ts, venue="cexA", venue_kind="cex", chain="",
                             base=a, quote="USDT", close=base_price[a],
                             volume=1e8, liquidity_usd=None, pool=None))
    df = pd.DataFrame(rows)
    print(f"      строк котировок: {len(df):,}")

    t1 = time.time()
    grid = build_grid(df, settings=s, trade_size_usd=1000, apply_slippage=False)
    t_grid = time.time() - t1
    mem = grid.log_rate.nbytes / 1e6
    print(f"      сетка построена за {t_grid:.1f} с, память {mem:.0f} МБ")
    check("сетка строится меньше чем за 60 с", t_grid < 60, f"{t_grid:.1f} с")

    for legs in (3, 4):
        t2 = time.time()
        tbl, cycles = find_cycles(grid, anchor="USDT", max_legs=legs, top=20,
                                  gas_per_dex_leg_usd=0.0, settings=s)
        t_search = time.time() - t2
        print(f"      поиск {legs} ног: {t_search:.1f} с, кандидатов {len(cycles)}")
        check(f"поиск {legs} ног меньше чем за 180 с", t_search < 180,
              f"{t_search:.1f} с")


def test_leveraged_filter():
    print("\n11. Отсев токенов с плечом")

    from ..config import filter_leveraged, is_leveraged_token

    known = {"BTC", "ETH", "BNB", "ADA", "LINK", "SOL", "DOT", "CAKE", "UNI"}

    # Реальные тикеры плечевых токенов с Gate, MEXC, KuCoin, Binance
    leveraged = ["BTC3L", "BTC3S", "ETH5L", "ETH5S", "ADA2L", "LINK3S",
                 "BTCUP", "BTCDOWN", "ETHUP", "ETHDOWN",
                 "BTCBULL", "ETHBEAR", "BTC3XLONG", "ETH3XSHORT"]
    for t in leveraged:
        check(f"{t} распознан как плечевой", is_leveraged_token(t, known))

    # Обычные активы, в том числе те, чьё имя формально похоже на шаблон
    normal = ["BTC", "ETH", "USDT", "CAKE", "UNI", "SUI", "APT", "TON",
              "JUP", "PEPE", "ONDO", "ENA", "WIF", "ATOM", "NEAR"]
    for t in normal:
        check(f"{t} не тронут", not is_leveraged_token(t, known))

    # Без справочника словесные маркеры не режем — слишком легко ошибиться
    check("без справочника BTCUP не режется", not is_leveraged_token("BTCUP"))
    check("без справочника BTC3L режется", is_leveraged_token("BTC3L"))

    got = filter_leveraged(["BTC", "BTC3L", "ETH", "ETHUP", "CAKE"], known)
    check("фильтр списка", got == ["BTC", "ETH", "CAKE"], str(got))


def test_leveraged_excluded_from_grid():
    print("\n12. Плечевые токены не попадают в сетку и в связки")

    # Рынок, где BTC3L даёт «идеальную» связку. Она должна исчезнуть
    # при spot_only=True и найтись при spot_only=False.
    t0 = 1_700_000_000
    rows = []
    for k in range(40):
        ts = t0 + k * 60
        for base, quote, price in [("BTC", "USDT", 60000.0),
                                   ("BTC3L", "USDT", 100.0),
                                   ("BTC3L", "BTC", 100.0 / 60000.0 * 1.05)]:
            rows.append(dict(ts=ts, venue="v", venue_kind="cex", chain="",
                             base=base, quote=quote, close=price,
                             volume=1e9, liquidity_usd=None, pool=None))

    s = Settings()
    s.timeframe = "1m"
    s.min_margin_pct = -100.0
    df = pd.DataFrame(rows)

    g_off = build_grid(df, settings=s, apply_slippage=False, spot_only=False)
    check("без фильтра BTC3L в сетке", "BTC3L" in g_off.assets, str(g_off.assets))

    g_on = build_grid(df, settings=s, apply_slippage=False, spot_only=True)
    check("с фильтром BTC3L отсутствует", "BTC3L" not in g_on.assets, str(g_on.assets))
    check("обычные активы остались", {"BTC", "USDT"} <= set(g_on.assets))

    t_off, _ = find_cycles(g_off, anchor="USDT", max_legs=3, top=20,
                           gas_per_dex_leg_usd=0.0, settings=s)
    t_on, _ = find_cycles(g_on, anchor="USDT", max_legs=3, top=20,
                          gas_per_dex_leg_usd=0.0, settings=s)
    off_has = (not t_off.empty) and t_off["Связка"].str.contains("BTC3L").any()
    on_has = (not t_on.empty) and t_on["Связка"].str.contains("BTC3L").any()
    check("без фильтра связка через BTC3L находится", off_has)
    check("с фильтром её нет", not on_has)

    # настройка по умолчанию — только спот
    check("по умолчанию фильтр включён", Settings().spot_only is True)


def test_route_order_and_links():
    print("\n13. Порядок площадок и ссылки на обмен")

    from ..links import chain_name, swap_url, token_name
    from ..store import init as store_init, write_pools

    # Имена площадок подобраны так, чтобы алфавитный порядок НЕ совпадал
    # с порядком исполнения: по алфавиту alpha, mike, zebra, а исполнять
    # надо zebra -> alpha -> mike. Если колонка когда-нибудь начнёт
    # сортировать площадки, тест это поймает.
    t0 = 1_700_000_000
    rows = []
    for k in range(20):
        ts = t0 + k * 60
        for venue, base, quote, price in [
            ("zebra", "A", "USDT", 1.00), ("alpha", "A", "USDT", 1.50),
            ("alpha", "B", "A", 1.00), ("mike", "B", "A", 1.50),
            ("mike", "B", "USDT", 1.30), ("zebra", "B", "USDT", 0.90),
        ]:
            rows.append(dict(ts=ts, venue=venue, venue_kind="dex", chain="bsc",
                             base=base, quote=quote, close=price,
                             volume=1e9, liquidity_usd=3e6, pool=f"0x{venue}"))

    s = Settings()
    s.timeframe = "1m"
    grid = build_grid(pd.DataFrame(rows), settings=s, apply_slippage=False)
    idx = [grid.asset_index(a) for a in ("USDT", "A", "B", "USDT")]
    cyc = evaluate_path(grid, idx)

    check("реестр площадок отсортирован по алфавиту",
          grid.venues == sorted(grid.venues), str(grid.venues))
    check("колонка идёт по порядку ИСПОЛНЕНИЯ, а не по алфавиту",
          cyc.dominant_venues() == ["zebra", "alpha", "mike"],
          " → ".join(cyc.dominant_venues()))

    st = cyc.stats()
    check("маршрут пронумерован по ногам",
          st["Маршрут"] == "1·zebra → 2·alpha → 3·mike", st["Маршрут"])

    # сеть и ликвидность
    check("сеть определена", st["Сеть"] == "BNB Chain", st["Сеть"])
    check("переводы не нужны: всё в одной сети на DEX", st["Переводы"] == "нет")
    check("узкое место по ликвидности найдено",
          cyc.bottleneck_liquidity() == 3e6, str(cyc.bottleneck_liquidity()))

    legs = cyc.leg_links()
    check("данные по каждой ноге", len(legs) == 3, str(len(legs)))
    check("ноги пронумерованы по порядку", [l["n"] for l in legs] == [1, 2, 3])
    check("направление обмена верное",
          [(l["from"], l["to"]) for l in legs] == [("USDT", "A"), ("A", "B"), ("B", "USDT")])

    # ссылки строятся, когда известны адреса контрактов
    url = swap_url("pancakeswap_v3", "bsc", "0xAAA", "0xBBB")
    check("ссылка на PancakeSwap собирается",
          url and "pancakeswap.finance/swap" in url and "0xAAA" in url and "0xBBB" in url,
          (url or "")[:80])
    check("сеть подставлена в ссылку", "chain=bsc" in (url or ""))

    url2 = swap_url("неизвестная_биржа", "bsc", "0xAAA", "0xBBB")
    check("для незнакомой площадки даётся агрегатор",
          url2 and "1inch" in url2 and "/56/" in url2, (url2 or "")[:70])

    check("без адресов ссылки нет", swap_url("pancakeswap", "bsc", "", "") is None)
    check("название сети человекочитаемо", chain_name("bsc") == "BNB Chain")


def test_token_names():
    print("\n14. Расшифровка тикеров")

    from ..links import describe_path, token_name

    check("встроенный справочник", token_name("CAKE") == "PancakeSwap")
    check("обёртка распознана", token_name("BTCB") == "Bitcoin BEP-20")
    check("имя из данных о пулах, если своего нет",
          token_name("ZZZ", {"ZZZ": "Zed Token"}) == "Zed Token")
    check("встроенное имя приоритетнее",
          token_name("CAKE", {"CAKE": "мусор"}) == "PancakeSwap")
    check("неизвестный тикер даёт пусто", token_name("QQQQ") == "")

    note = describe_path(("USDT", "CAKE", "BTCB", "USDT"))
    check("стартовый актив не повторяется в расшифровке", "USDT" not in note, note)
    check("промежуточные расшифрованы",
          "CAKE — PancakeSwap" in note and "BTCB — Bitcoin BEP-20" in note, note)


def main() -> int:
    print("=" * 70)
    print("Проверка математики исторического сканнера")
    print("=" * 70)
    test_slippage_models()
    test_grid_and_cycle()
    test_no_false_positive()
    test_search_finds_it()
    test_staleness()
    test_maxplus_correctness()
    test_four_leg_reconstruction()
    test_reconstruction_uses_correct_timestep()
    test_venue_attribution()
    test_scale()
    test_leveraged_filter()
    test_leveraged_excluded_from_grid()
    test_route_order_and_links()
    test_token_names()

    print("\n" + "=" * 70)
    if FAILED:
        print(f"ПРОВАЛЕНО {len(FAILED)}:")
        for f in FAILED:
            print("  -", f)
        return 1
    print("Все проверки пройдены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
