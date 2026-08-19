"""Проверка разбора связки по шагам.

Разбор нужен для самоконтроля перед сделкой, поэтому к нему требование
жёстче обычного: он обязан сходиться с таблицей до последнего знака.
Если разбор показывает одно, а таблица другое — доверять нельзя ни тому,
ни другому.

Запуск:  python -m history.tests.test_breakdown
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


def main() -> int:
    from history.breakdown import explain, mid_rate
    from history.config import Settings
    from history.paths import find_cycles
    from history.rates import build_grid

    # Рынок с заложенной связкой: CAKE на biswap стоит чуть дороже, чем
    # получается через BNB на pancake. Числа круглые, чтобы расхождение
    # было видно глазом.
    t0 = int(time.time()) // 900 * 900 - 30 * 900
    rows = []
    for k in range(30):
        ts = t0 + k * 900
        for v, b, q, price, liq in [
            ("pancakeswap_v3", "BNB", "USDT", 600.0, 8e6),
            ("pancakeswap_v3", "CAKE", "BNB", 2.10 / 600.0, 4e6),
            ("biswap", "CAKE", "USDT", 2.16, 6e6),
        ]:
            rows.append(dict(ts=ts, venue=v, venue_kind="dex", chain="bsc",
                             base=b, quote=q, close=price, volume=1e6,
                             liquidity_usd=liq, pool=f"0x{v[:4]}{b}"))
    df = pd.DataFrame(rows)

    s = Settings()
    s.analysis_timeframe = "15m"
    s.staleness_sec = 1800
    grid = build_grid(df, settings=s, venue_kinds=["dex"], trade_size_usd=1000.0,
                      drop_suspicious=False)
    table, cycles = find_cycles(grid, anchor="USDT", max_legs=3, top=10,
                                gas_per_dex_leg_usd=0.0, min_margin_pct=-100,
                                settings=s)
    check("связки найдены", len(cycles) > 0, f"{len(cycles)}")
    if not cycles:
        return 1

    target = None
    for c in cycles:
        if c.assets == ("USDT", "BNB", "CAKE", "USDT"):
            target = c
    check("заложенная связка найдена", target is not None,
          target.label if target else ", ".join(c.label for c in cycles[:3]))
    if target is None:
        return 1

    print("\n1. Разбор сходится с таблицей")
    br = explain(target)
    check("разбор построен", br is not None)
    if br is None:
        return 1

    margin_table = float(target.margin_pct()[-1])
    check("маржа совпадает с таблицей", abs(br.net_pct - margin_table) < 1e-6,
          f"разбор {br.net_pct:.6f}% против {margin_table:.6f}%")

    print("\n2. Цепочка сумм замкнута")
    check("шагов столько же, сколько ног", len(br.steps) == target.legs,
          f"{len(br.steps)}")
    chained = True
    for i in range(1, len(br.steps)):
        if abs(br.steps[i].amount_in - br.steps[i - 1].amount_out) > 1e-9:
            chained = False
    check("выход шага равен входу следующего", chained)
    check("итог равен выходу последнего шага",
          abs(br.amount_out - br.steps[-1].amount_out) < 1e-9)

    print("\n3. Вклады ног дают итог")
    prod = 1.0
    for st in br.steps:
        prod *= (1.0 + st.pnl_pct / 100.0)
    check("произведение вкладов равно итогу",
          abs((prod - 1.0) * 100.0 - br.gross_pct) < 1e-6,
          f"{(prod - 1) * 100:.6f}% против {br.gross_pct:.6f}%")
    best = br.best_leg()
    check("маржа приписана одной ноге", best is not None and best.pnl_pct > 0,
          f"{best.asset_in}→{best.asset_out} {best.pnl_pct:+.3f}%" if best else "")

    print("\n4. Издержки разложены")
    ok_costs = True
    for st in br.steps:
        # спот × (1 − комиссия) × проскальзывание должно давать курс сделки
        back = st.spot_rate * (1 - st.fee_pct / 100.0) * (1 - st.slippage_pct / 100.0)
        if abs(back - st.exec_rate) > abs(st.exec_rate) * 1e-9:
            ok_costs = False
    check("спот × издержки = курс сделки", ok_costs)
    check("комиссия положительна на всех ногах",
          all(st.fee_pct > 0 for st in br.steps))
    check("проскальзывание не отрицательно",
          all(st.slippage_pct >= 0 for st in br.steps))

    print("\n5. Объём меняет результат")
    small = explain(target, amount=100.0)
    big = explain(target, amount=100_000.0)
    check("на сотне разбор построен", small is not None)
    check("на маленьком объёме проскальзывание меньше",
          small.steps[0].slippage_pct < br.steps[0].slippage_pct,
          f"{small.steps[0].slippage_pct:.4f}% против {br.steps[0].slippage_pct:.4f}%")
    check("на большом объёме маржа хуже", big.gross_pct < br.gross_pct,
          f"{big.gross_pct:.3f}% против {br.gross_pct:.3f}%")
    check("пометка о пересчёте выставлена", not small.exact and br.exact)

    print("\n6. Газ учитывается")
    target.gas_usd = 0.45
    with_gas = explain(target, amount=100.0)
    without = explain(target, amount=10_000.0)
    check("на сотне газ заметен, на десяти тысячах — нет",
          (with_gas.gross_pct - with_gas.net_pct)
          > (without.gross_pct - without.net_pct),
          f"{with_gas.gross_pct - with_gas.net_pct:.3f}% против "
          f"{without.gross_pct - without.net_pct:.3f}%")
    target.gas_usd = 0.0

    print("\n7. Срединный курс")
    t_last = len(grid.times) - 1
    mid = mid_rate(grid, t_last, "BNB", "USDT")
    check("срединный курс BNB близок к заложенному", abs(mid - 600.0) < 1.0,
          f"{mid:.2f}")
    check("курс стартового актива к себе равен единице",
          mid_rate(grid, t_last, "USDT", "USDT") == 1.0)

    print("\n8. Текстовый вид")
    txt = br.as_text()
    print("─" * 66)
    print(txt)
    print("─" * 66)
    check("в тексте все ноги", all(f"{i + 1}. " in txt for i in range(len(br.steps))))
    check("в тексте есть итог", "Итог:" in txt)
    frame = br.to_frame()
    check("таблица разбора строится", len(frame) == len(br.steps))

    print("\n" + ("ПРОВАЛЕНО: " + ", ".join(FAIL) if FAIL
                  else "Разбор по шагам сходится с расчётом"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
