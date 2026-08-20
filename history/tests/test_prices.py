"""Откуда берётся курс пула — и почему это не всё равно.

Симптом был такой. В телеграм пошли связки целиком внутри PancakeSwap:
USDT → BNB → ETH → BTC → USDT, маржа 0.3–0.44%, «типичное окно около
пятнадцати минут». Активы — самые ликвидные в сети, площадка одна.
Настоящий треугольный арбитраж между такими пулами закрывают боты
за один блок, а не за пятнадцать минут; значит, маржа не настоящая.

Причина нашлась в одной строке. Курс пула мы получали делением двух
долларовых оценок: `base_token_price_usd / quote_token_price_usd`.
Но долларовая цена токена у источника — это оценка по самому ликвидному
пулу, и обновляется она своим темпом. У одного токена в двух разных
пулах она отличается на сотые доли процента просто потому, что снята
в разные моменты.

На одном плече это незаметно. Связка перемножает четыре курса, и четыре
независимые погрешности складываются. Отсюда треть процента из ниоткуда.

У источника есть собственное отношение пула — `base_token_price_quote_token`.
Оно не проходит через доллар вовсе, и замкнутый круг по согласованным
ценам даёт ровно единицу. Это здесь и проверяется.

Запуск:  python -m history.tests.test_prices
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("ARB_DATA_DIR", tempfile.mkdtemp())

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# Согласованный рынок: доллар, BNB по 600, ETH по 3000, BTC по 60000.
# Никакого арбитража в нём нет по построению.
TRUE_USD = {"USDT": 1.0, "BNB": 600.0, "ETH": 3000.0, "BTC": 60000.0}

# Долларовые оценки, снятые в разные моменты. Отклонения крошечные —
# по одной десятой процента, столько и бывает между пулами.
SKEWED = {
    ("USDT", "BNB"): {"USDT": 1.0, "BNB": 599.4},
    ("BNB", "ETH"): {"BNB": 600.0, "ETH": 2997.0},
    ("ETH", "BTC"): {"ETH": 3000.0, "BTC": 60000.0},
    ("BTC", "USDT"): {"BTC": 60000.0, "USDT": 1.0},
}

LOOP = [("USDT", "BNB"), ("BNB", "ETH"), ("ETH", "BTC"), ("BTC", "USDT")]


def pool_payload(base: str, quote: str, with_ratio: bool = True) -> dict:
    """Пул в том виде, в каком его отдаёт источник."""
    usd = SKEWED[(base, quote)]
    attrs = {
        "address": f"0x{base}{quote}",
        "name": f"{base} / {quote} 0.05%",
        "base_token_price_usd": str(usd[base]),
        "quote_token_price_usd": str(usd[quote]),
        "reserve_in_usd": "1000000",
        "volume_usd": {"h24": "500000"},
    }
    if with_ratio:
        # Собственное отношение пула — по настоящим ценам, без перекоса.
        attrs["base_token_price_quote_token"] = str(
            TRUE_USD[base] / TRUE_USD[quote])
    return {
        "id": f"bsc_{base}{quote}",
        "attributes": attrs,
        "relationships": {
            "base_token": {"data": {"id": f"bsc_{base}"}},
            "quote_token": {"data": {"id": f"bsc_{quote}"}},
            "dex": {"data": {"id": "pancakeswap-v3-bsc"}},
        },
    }


def included() -> list:
    out = []
    for sym in TRUE_USD:
        out.append({"id": f"bsc_{sym}",
                    "attributes": {"symbol": sym, "address": f"0xaddr{sym}",
                                   "name": sym}})
    return out


def main() -> int:
    from history.config import SETTINGS
    from history.sources.dex_gt import GeckoTerminalSource

    src = GeckoTerminalSource(SETTINGS)
    tokens = src._index_included(included())
    now = int(time.time())

    print("\n1. Курс берётся из собственного поля пула")

    parsed = src._parse_pool(pool_payload("USDT", "BNB"), tokens)
    check("отношение пула прочитано",
          abs(parsed["_ratio"] - 1 / 600.0) < 1e-15, str(parsed["_ratio"]))
    candle = src._live_candle(parsed, now)
    check("свеча берёт отношение, а не частное долларов",
          abs(candle.close - 1 / 600.0) < 1e-15, str(candle.close))
    check("частное долларов дало бы другое число",
          abs(1 / 599.4 - 1 / 600.0) > 1e-9)

    no_ratio = src._parse_pool(pool_payload("USDT", "BNB", with_ratio=False),
                               tokens)
    fallback = src._live_candle(no_ratio, now)
    check("без отношения работает запасной путь",
          abs(fallback.close - 1 / 599.4) < 1e-12, str(fallback.close))
    check("совсем без цен свечи нет",
          src._live_candle({"_ratio": 0, "_base_usd": 0, "_quote_usd": 0,
                            "dex": "d", "base": "A", "quote": "B",
                            "reserve_usd": 1, "pool": "0x"}, now) is None)

    print("\n2. Замкнутый круг по согласованным ценам даёт единицу")

    product_ratio = 1.0
    product_usd = 1.0
    for base, quote in LOOP:
        p = src._parse_pool(pool_payload(base, quote), tokens)
        product_ratio *= src._live_candle(p, now).close
        product_usd *= p["_base_usd"] / p["_quote_usd"]

    check("по отношениям пулов круг замыкается ровно",
          abs(product_ratio - 1.0) < 1e-12,
          f"{(product_ratio - 1) * 100:+.6f}%")

    fake_margin = (product_usd - 1.0) * 100
    check("по долларовым оценкам возникает маржа из ниоткуда",
          fake_margin > 0.15,
          f"{fake_margin:+.3f}% на четырёх плечах при отклонениях в 0.1%")
    check("и она того же порядка, что приходила в телеграм",
          0.15 < fake_margin < 1.0, f"{fake_margin:+.3f}% против 0.31–0.44%")

    print("\n3. То же самое в прямом запросе со страницы")

    from history import live

    payload = {"data": [pool_payload(b, q) for b, q in LOOP],
               "included": included()}
    rows = live._parse(payload, "bsc", now)
    check("разобраны все четыре пула", len(rows) == 4, str(len(rows)))

    by_pair = {(r["base"], r["quote"]): r["close"] for r in rows}
    loop_live = 1.0
    for base, quote in LOOP:
        loop_live *= by_pair[(base, quote)]
    check("прямой запрос тоже замыкает круг в единицу",
          abs(loop_live - 1.0) < 1e-12, f"{(loop_live - 1) * 100:+.6f}%")

    stripped = {"data": [pool_payload(b, q, with_ratio=False) for b, q in LOOP],
                "included": included()}
    rows2 = live._parse(stripped, "bsc", now)
    check("без отношения прямой запрос не падает", len(rows2) == 4)

    print("\n4. Связка из таких цен не проходит отбор")

    import numpy as np
    import pandas as pd
    from history.config import Settings
    from history.paths import find_cycles
    from history.rates import build_grid
    import history.rates as rates

    t0 = int(time.time()) // 300 * 300 - 10 * 300
    quotes = []
    for k in range(10):
        ts = t0 + k * 300
        for base, quote in LOOP:
            p = src._parse_pool(pool_payload(base, quote), tokens)
            quotes.append({
                "ts": ts, "venue": "pancakeswap-v3-bsc", "venue_kind": "dex",
                "chain": "bsc", "base": base, "quote": quote,
                "close": src._live_candle(p, ts).close, "volume": 1000.0,
                "liquidity_usd": 1_000_000.0, "pool": p["pool"],
            })
    df = pd.DataFrame(quotes)

    s = Settings()
    s.staleness_sec = 3600
    real = rates._pools_frame
    try:
        rates._pools_frame = lambda _s: None
        grid = build_grid(df, settings=s, trade_size_usd=1000.0,
                          drop_suspicious=False)
    finally:
        rates._pools_frame = real

    table, cycles = find_cycles(grid, anchor="USDT", max_legs=4, top=10,
                               min_margin_pct=0.0, settings=s)
    best = max((float(np.nanmax(c.margin_pct())) for c in cycles),
               default=float("-inf"))
    check("ни одна связка не в плюсе", best < 0,
          f"лучшая {best:+.3f}%" if best > float("-inf") else "связок нет")
    check("минус примерно равен сумме комиссий", -1.5 < best < -0.2,
          f"{best:+.3f}% при четырёх плечах")

    # Тот же рынок, но курсы из долларовых оценок — и связки появляются.
    skewed_rows = []
    for k in range(10):
        ts = t0 + k * 300
        for base, quote in LOOP:
            p2 = src._parse_pool(pool_payload(base, quote, with_ratio=False),
                                 tokens)
            skewed_rows.append({
                "ts": ts, "venue": "pancakeswap-v3-bsc", "venue_kind": "dex",
                "chain": "bsc", "base": base, "quote": quote,
                "close": src._live_candle(p2, ts).close, "volume": 1000.0,
                "liquidity_usd": 1_000_000.0, "pool": p2["pool"],
            })
    try:
        rates._pools_frame = lambda _s: None
        grid2 = build_grid(pd.DataFrame(skewed_rows), settings=s,
                           trade_size_usd=1000.0, drop_suspicious=False)
    finally:
        rates._pools_frame = real
    _, cycles2 = find_cycles(grid2, anchor="USDT", max_legs=4, top=10,
                             min_margin_pct=0.0, settings=s)
    best2 = max((float(np.nanmax(c.margin_pct())) for c in cycles2),
                default=float("-inf"))
    check("ложных связок не появилось и там", best2 < 0, f"{best2:+.3f}%")

    # Сравниваем ровно ту цепочку, что приходила в телеграм. Берём её
    # прямо из сетки, а не из отбора: на честных ценах она настолько
    # плоха, что в десятку лучших уже не попадает.
    def loop_margin(g):
        total = 0.0
        for a, b in LOOP:
            total += float(g.log_rate[-1, g.asset_index(a), g.asset_index(b)])
        return (np.exp(total) - 1.0) * 100

    m_ratio, m_usd = loop_margin(grid), loop_margin(grid2)
    check("на честных ценах цепочка стоит ровно комиссий",
          -1.5 < m_ratio < -0.8, f"{m_ratio:+.3f}%")
    check("на долларовых оценках она выглядит выгоднее, чем есть",
          m_usd - m_ratio > 0.15,
          f"{m_usd:+.3f}% против честных {m_ratio:+.3f}% — "
          f"разница {m_usd - m_ratio:+.3f}% взялась из ниоткуда")

    print("\n" + "=" * 70)
    if FAIL:
        print("НЕ ПРОЙДЕНО:", ", ".join(FAIL))
        return 1
    print("Курс пула больше не проходит через доллар — ложные треугольники "
          "внутри одной площадки исчезли")
    return 0


if __name__ == "__main__":
    sys.exit(main())
