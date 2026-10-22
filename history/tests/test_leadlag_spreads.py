"""Спреды фьючерсов между биржами: расписание funding, знаки ставок,
итог связки — на вручную посчитанных примерах.

Запуск:  python -m history.tests.test_leadlag_spreads
"""

from __future__ import annotations

import sys

import pandas as pd

from leadlag import funding, spreads
from leadlag.settings import ScreenSettings

FAILED = []
NOW = 1_800_000_000.0  # секунды


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def close(a, b, tol=1e-9) -> bool:
    return abs(a - b) <= tol


def test_settlements():
    print("\n1. Сколько расчётов funding попадёт в удержание")
    nxt = int((NOW + 1800) * 1000)  # через 30 минут
    check("8 ч, интервал 8 ч, ближайший через 30 мин → 1", funding.settlements(nxt, 8, NOW, 8) == 1)
    check("8 ч, интервал 4 ч → 2", funding.settlements(nxt, 4, NOW, 8) == 2)
    check("8 ч, интервал 1 ч → 8", funding.settlements(nxt, 1, NOW, 8) == 8)
    check("20 мин, ближайший через 30 мин → 0", funding.settlements(nxt, 8, NOW, 20 / 60) == 0)
    stale = int((NOW - 600) * 1000)  # время из ответа уже прошло
    check("устаревшее время сдвигается на интервал", funding.settlements(stale, 1, NOW, 2) == 2)
    check("время неизвестно — в среднем 24/8 = 3", funding.settlements(None, 8, NOW, 24) == 3)


def market(venue, bid, ask, rate, interval, next_in_s=1800, volume=5e6, taker=0.05, coin="AAA"):
    return {"venue": venue, "symbol": f"{coin}/USDT:USDT", "url": "", "coin": coin,
            "last": (bid + ask) / 2, "bid": bid, "ask": ask, "volume_usd": volume,
            "quote_ts": NOW * 1000, "f_rate": rate, "f_interval_h": interval,
            "f_next_ts": int((NOW + next_in_s) * 1000), "taker_pct": taker}


def test_pair_math():
    print("\n2. Итог связки")
    m = pd.DataFrame([
        market("a", 0.999, 1.000, -0.0002, 8),   # дешевле; отрицательная ставка — лонг получает
        market("b", 1.010, 1.011, 0.0005, 4),    # дороже; положительная — шорт получает
    ])
    p = spreads.pairs(m, 8, now=NOW)
    r = p[(p.venue_l == "a") & (p.venue_s == "b")].iloc[0]
    check("спред по bid шорта / ask лонга: 1,0%", close(r.spread_pct, 1.0, 1e-9), f"{r.spread_pct:.6f}")
    check("комиссии 2 × (0,05 + 0,05) = 0,2%", close(r.fees_pct, 0.2))
    # шорт b: 2 расчёта × 0,05% = +0,10; лонг a: 1 расчёт, ставка −0,02% → лонг получает +0,02
    check("funding за 8 ч = +0,12%", close(r.funding_hold_pct, 0.12, 1e-9), f"{r.funding_hold_pct:.6f}")
    check("funding в сутки = 0,05×6 + 0,02×3 = 0,36%", close(r.funding_day_pct, 0.36, 1e-9),
          f"{r.funding_day_pct:.6f}")
    check("итог при схождении 1,0 − 0,2 + 0,12 = 0,92%", close(r.pnl_converge_pct, 0.92, 1e-9))
    check("итог только funding 0,12 − 0,2 = −0,08%", close(r.pnl_carry_pct, -0.08, 1e-9))
    rev = p[(p.venue_l == "b") & (p.venue_s == "a")].iloc[0]
    check("обратное направление — отрицательный спред", rev.spread_pct < 0)
    check("подсказка: дорогой платит больший funding", "funding" in r.reasons, r.reasons)


def test_select():
    print("\n3. Отбор связок")
    m = pd.DataFrame([
        market("binance", 0.999, 1.000, 0.0, 8),
        market("gate", 1.010, 1.011, 0.0, 8),
        market("mexc", 1.010, 1.011, 0.0, 8, volume=1e4),          # мало объёма
        market("binance", 1.0, 1.0, 0.0, 8, coin="FAKE"),
        market("okx", 3.0, 3.0, 0.0, 8, coin="FAKE"),               # другой токен под тем же тикером
    ])
    q = spreads.select(spreads.pairs(m, 8, now=NOW), ScreenSettings())
    got = set(zip(q.coin, q.venue_l, q.venue_s))
    check("осталась одна связка AAA binance → gate", got == {("AAA", "binance", "gate")}, str(got))


def test_notify():
    print("\n4. Оповещения: только устойчивые, без повторов")
    import tempfile
    from pathlib import Path

    from history import alerts
    from leadlag import notify

    sent = []
    orig = (alerts.send, alerts.configured, notify.STATE_PATH)
    alerts.send = lambda text, *a, **k: sent.append(text) or True
    alerts.configured = lambda: True
    with tempfile.TemporaryDirectory() as d:
        notify.STATE_PATH = Path(d) / "sent.json"
        try:
            m = pd.DataFrame([market("binance", 0.999, 1.000, 0.0, 8),
                              market("gate", 1.010, 1.011, 0.0, 8),
                              market("binance", 0.999, 1.000, 0.0, 8, coin="BLINK"),
                              market("gate", 1.010, 1.011, 0.0, 8, coin="BLINK")])
            s = ScreenSettings()
            q = spreads.select(spreads.pairs(m, 8, now=NOW), s)
            hot = pd.DataFrame({"coin": ["AAA", "BLINK"], "venue_l": ["binance"] * 2,
                                "venue_s": ["gate"] * 2, "minutes_hot": [12, 3], "spread_avg": [1.0, 1.0]})
            check("отправлена одна (мелькнувшая BLINK — нет)", notify.notify_spreads(q, hot, s) == 1,
                  str(len(sent)))
            check("в сообщении монета и ссылки бирж", "AAA" in sent[0] and "Binance" in sent[0])
            check("повтор в течение 4 ч не шлётся", notify.notify_spreads(q, hot, s) == 0)
            s.alerts_enabled = False
            check("выключено — не шлётся", notify.notify_spreads(q, hot.assign(coin="AAA"), s) == 0)
        finally:
            alerts.send, alerts.configured, notify.STATE_PATH = orig


def main() -> int:
    test_settlements()
    test_pair_math()
    test_select()
    test_notify()
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
