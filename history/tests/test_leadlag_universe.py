"""Отбор кандидатов в связки «фьючерс ↔ DEX»: множители тикеров, выбор монеты
по цене при совпадении тикеров, применение галочек и порогов.

Запуск:  python -m history.tests.test_leadlag_universe
"""

from __future__ import annotations

import sys

import pandas as pd

from leadlag.settings import ScreenSettings
from leadlag.universe import apply_filters, pick_coin, split_multiplier

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def test_multiplier():
    print("\n1. Множители в тикерах фьючерсов")
    check("1000PEPE", split_multiplier("1000PEPE") == ("PEPE", 1000.0))
    check("1MBABYDOGE", split_multiplier("1MBABYDOGE") == ("BABYDOGE", 1_000_000.0))
    check("10000SATS", split_multiplier("10000SATS") == ("SATS", 10000.0))
    check("обычный тикер не трогается", split_multiplier("SOL") == ("SOL", 1.0))
    check("цифры без буквы следом — не множитель", split_multiplier("1000") == ("1000", 1.0))


def test_pick_coin():
    print("\n2. Монета по цене при совпадении тикеров")
    cands = [
        {"id": "real", "current_price": 0.0102, "market_cap": 5_000_000},
        {"id": "fake", "current_price": 3.5, "market_cap": 90_000_000},
        {"id": "dust", "current_price": 0.0101, "market_cap": 10_000},
    ]
    best = pick_coin(0.0100, cands)
    check("выбрана близкая по цене и крупнее", best and best["id"] == "real", str(best and best["id"]))
    check("нет совпадения по цене — нет монеты",
          pick_coin(1.0, [{"id": "x", "current_price": 2.0}]) is None)
    check("пустая цена не ломает выбор",
          pick_coin(1.0, [{"id": "x", "current_price": None}]) is None)


def test_filters():
    print("\n3. Галочки и пороги")
    base = dict(fut_venue="mexc", chain="bsc", spot_on="", fut_volume_usd=5e6,
                fut_range_pct=12.0, dex_liquidity_usd=4e5, dex_volume_usd=2e5,
                price_gap_pct=0.4, coin="AAA")
    rows = [
        base,
        {**base, "coin": "BIN", "spot_on": "binance,gate"},      # спот на tier 1
        {**base, "coin": "GATE", "spot_on": "gate"},             # спот только на tier 2
        {**base, "coin": "LOWV", "fut_volume_usd": 1e5},
        {**base, "coin": "CALM", "fut_range_pct": 2.0},
        {**base, "coin": "THIN", "dex_liquidity_usd": 1e4},
        {**base, "coin": "GAP", "price_gap_pct": 25.0},
        {**base, "coin": "ETH1", "chain": "ethereum"},
        {**base, "coin": "OTHER", "fut_venue": "unknownex"},
        {**base, "coin": "MU", "tradfi": True},                  # токенизированная акция
    ]
    rows = [{"tradfi": False, **r} for r in rows]
    s = ScreenSettings()
    got = set(apply_filters(pd.DataFrame(rows), s)["coin"])
    check("прошли только AAA и GATE", got == {"AAA", "GATE"}, str(sorted(got)))

    s.tier1_spot = s.tier1_spot + ["gate"]
    got = set(apply_filters(pd.DataFrame(rows), s)["coin"])
    check("галочка gate исключает GATE", got == {"AAA"}, str(sorted(got)))

    s.chains = s.chains + ["ethereum"]
    got = set(apply_filters(pd.DataFrame(rows), s)["coin"])
    check("галочка Ethereum добавляет ETH1", "ETH1" in got, str(sorted(got)))

    s.exclude_tradfi = False
    got = set(apply_filters(pd.DataFrame(rows), s)["coin"])
    check("без исключения акций MU проходит", "MU" in got, str(sorted(got)))


def test_tradfi_names():
    print("\n4. Токены традиционных активов по названию")
    from leadlag.universe import _TRADFI_NAME
    for name in ["Micron Technology (Backpack Securities)", "Tesla xStock", "Ondo Tokenized NVDA"]:
        check(f"акция: {name}", bool(_TRADFI_NAME.search(name)))
    for name in ["Rootstock", "Ondo", "Stockpile Finance", "Pepe"]:
        check(f"не акция: {name}", not _TRADFI_NAME.search(name))


def test_pick_pool():
    print("\n5. Выбор пула: к стейблкоину или WBNB, а не к редкому токену")
    from leadlag.universe import pick_pool
    usdt = "0x55d398326f99059ff775485246999027b3197955"
    pools = [{"pool": "rare", "liquidity_usd": 900_000, "pair_with_address": "0x" + "f" * 40},
             {"pool": "usdt", "liquidity_usd": 300_000, "pair_with_address": usdt}]
    check("BSC: пул к USDT, хоть и менее ликвидный", pick_pool("bsc", pools)["pool"] == "usdt")
    check("Solana: просто самый ликвидный", pick_pool("solana", pools)["pool"] == "rare")
    check("других нет — берётся редкий", pick_pool("bsc", pools[:1])["pool"] == "rare")


def main() -> int:
    test_multiplier()
    test_pick_coin()
    test_filters()
    test_tradfi_names()
    test_pick_pool()
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
