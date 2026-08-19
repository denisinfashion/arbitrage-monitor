"""Проверка отбора пулов DEX: площадки, квота, список наблюдения.

Повод конкретный. В результатах месяцами фигурировала одна площадка —
PancakeSwap, — и заказчик справедливо усомнился. Причина оказалась
не в математике: топ сети по обороту в BNB Chain почти целиком
пакейковский, а других источников пулов не было. Связка между
площадками не могла возникнуть, потому что вторая площадка
в наблюдение просто не попадала.

Сети тесты не требуют: ответы GeckoTerminal подменяются.

Запуск:  python -m history.tests.test_dex_discovery
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("ARB_DATA_DIR", tempfile.mkdtemp())

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


TOKENS = {
    "USDT": ("0x55d398326f99059ff775485246999027b3197955", "Tether USD"),
    "WBNB": ("0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c", "Wrapped BNB"),
    "CAKE": ("0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82", "PancakeSwap Token"),
    "AAVE": ("0xfb6115445bff7b52feb98650c87f44907e58f802", "Aave Token"),
    "DAI":  ("0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3", "Dai Token"),
}
# После norm_asset обёртка приводится к базовому тикеру, и в справочнике
# пулов лежит уже BNB, а не WBNB — срез спрашивает цены по этим записям.
TOKENS["BNB"] = TOKENS["WBNB"]


def _tok_obj(sym):
    addr, name = TOKENS[sym]
    return {"id": f"bsc_{addr}",
            "attributes": {"symbol": sym, "address": addr, "name": name}}


def _pool_item(addr, dex, base, quote, reserve=5e5, volume=1e6,
               base_usd=1.0, quote_usd=1.0):
    return {
        "attributes": {
            "address": addr,
            "name": f"{base} / {quote}",
            "reserve_in_usd": str(reserve),
            "volume_usd": {"h24": str(volume)},
            "base_token_price_usd": str(base_usd),
            "quote_token_price_usd": str(quote_usd),
        },
        "relationships": {
            "base_token": {"data": {"id": f"bsc_{TOKENS[base][0]}"}},
            "quote_token": {"data": {"id": f"bsc_{TOKENS[quote][0]}"}},
            "dex": {"data": {"id": dex}},
        },
    }


def _payload(items):
    syms = set()
    for it in items:
        for side in ("base_token", "quote_token"):
            ref = it["relationships"][side]["data"]["id"]
            for s, (a, _n) in TOKENS.items():
                if ref.endswith(a):
                    syms.add(s)
    return {"data": items, "included": [_tok_obj(s) for s in sorted(syms)]}


MULTI: dict = {}


def main() -> int:
    from history import store
    from history.config import Settings
    from history.sources import dex_gt

    store.init()

    calls = []

    def fake_get_json(url, params=None, headers=None, **kw):
        calls.append(url)
        params = params or {}

        # Топ сети: только PancakeSwap — ровно так выглядит BNB Chain.
        if url.endswith("/networks/bsc/pools"):
            page = int(params.get("page", 1))
            if page > 2:
                return {"data": [], "included": []}
            items = [
                _pool_item(f"0xtop{page}{i}", "pancakeswap-v3-bsc", "WBNB", "USDT",
                           reserve=9e6 - i, volume=5e6, base_usd=600.0)
                for i in range(20)
            ]
            return _payload(items)

        # Список площадок сети. Идентификаторы намеренно в том виде,
        # в каком их отдаёт источник: с дефисами и суффиксом сети.
        if url.endswith("/networks/bsc/dexes"):
            return {"data": [{"id": f"bsc_{d}"} for d in
                             ("pancakeswap-v3-bsc", "uniswap-bsc",
                              "biswap", "thena", "sushiswap-bsc")]}

        # Топ отдельной площадки.
        if "/dexes/" in url:
            dex = url.rsplit("/dexes/", 1)[1].split("/")[0]
            if dex in ("sushiswap-bsc",):
                # площадки нет в этой сети
                from history.http import HttpError
                raise HttpError(f"{dex}: 404")
            return _payload([
                _pool_item(f"0x{dex}{i}", dex, "CAKE", "USDT",
                           reserve=8e5 - i * 1000, volume=3e5, base_usd=2.05)
                for i in range(5)
            ])

        # Живой срез: цены известных пулов пачками по адресам.
        if "/pools/multi/" in url:
            addrs = url.rsplit("/pools/multi/", 1)[1].split(",")
            known = {a: d for a, d in MULTI.items()}
            items = [_pool_item(a, known[a][0], known[a][1], known[a][2],
                                reserve=known[a][3], volume=known[a][4],
                                base_usd=known[a][5])
                     for a in addrs if a in known]
            return _payload(items)

        # Список наблюдения.
        if url.endswith("/search/pools"):
            q = (params.get("query") or "").upper()
            if q not in TOKENS:
                return {"data": [], "included": []}
            return _payload([
                _pool_item(f"0xwatch{q}", "thena", q, "USDT",
                           reserve=4e5, volume=2e5, base_usd=90.0),
            ])

        return {"data": [], "included": []}

    dex_gt.get_json = fake_get_json

    s = Settings()
    s.dex_pool_limit = 30
    s.dex_venue_quota = 8
    s.dex_venues = []          # определяем автоматически
    s.watch_tokens = ["AAVE", "DAI"]
    s.min_pool_reserve_usd = 100_000

    src = dex_gt.GeckoTerminalSource(s)
    n = src.discover()

    pools = src._pools
    by_dex = {}
    for p in pools:
        by_dex[p["dex"]] = by_dex.get(p["dex"], 0) + 1

    print("\n1. Источники пулов")
    check("пулы получены", n > 0, f"{n} шт")
    check("площадок больше одной", len(by_dex) > 1, ", ".join(
        f"{d}:{c}" for d, c in sorted(by_dex.items())))
    check("недоступная площадка не роняет сбор",
          "sushiswap-bsc" not in by_dex and len(by_dex) >= 3)

    print("\n2. Квота на площадку")
    # Квота — это гарантированный минимум, а не потолок: остаток лимита
    # честно достаётся самым крупным пулам, где бы они ни были. Иначе
    # ради разнообразия выбрасывались бы как раз самые исполнимые пары.
    available = {"pancakeswap-v3-bsc": 40, "biswap": 5, "thena": 5 + 2}
    short = {d: (by_dex.get(d, 0), min(s.dex_venue_quota, k))
             for d, k in available.items()
             if by_dex.get(d, 0) < min(s.dex_venue_quota, k)}
    check("каждая площадка получила свою квоту", not short, str(short) if short else "")
    share = by_dex.get("pancakeswap-v3-bsc", 0) / max(1, len(pools))
    check("PancakeSwap не забрал весь список", share < 0.8,
          f"{by_dex.get('pancakeswap-v3-bsc', 0)} из {len(pools)} ({share:.0%})")

    print("\n2б. Идентификаторы площадок и комиссии")
    from history.config import dex_fee_pct
    check("идентификаторы взяты у источника, а не из настроек",
          any("-bsc" in d for d in by_dex), ", ".join(sorted(by_dex)))
    check("комиссия PancakeSwap распознана", dex_fee_pct("pancakeswap-v3-bsc") == 0.25)
    check("комиссия Uniswap распознана", dex_fee_pct("uniswap-bsc") == 0.30)
    check("комиссия Biswap распознана", dex_fee_pct("biswap") == 0.10)
    check("незнакомая площадка получает умолчание",
          dex_fee_pct("dinosaureggs") == 0.25)

    print("\n3. Список наблюдения")
    syms = {p["base"] for p in pools} | {p["quote"] for p in pools}
    check("AAVE добран принудительно", "AAVE" in syms)
    check("DAI добран принудительно", "DAI" in syms)
    # Список наблюдения складывается из трёх источников: файла
    # watchlist.txt, переменной окружения и настроек. Проверяем, что
    # запрос ушёл по каждому тикеру итогового списка, а не только
    # по тем, что заданы здесь.
    from history.config import load_watchlist
    expected = load_watchlist(s)
    check("поиск вызывался по каждому тикеру списка",
          sum(1 for u in calls if u.endswith("/search/pools")) == len(expected),
          f"{len(expected)} тикеров: {', '.join(expected)}")

    print("\n4. Живой срез дешевле полного поиска")
    # Полный поиск стоит десятки запросов: страницы топа, список площадок,
    # запрос на площадку и на каждый тикер наблюдения. На бесплатной квоте
    # в тридцать запросов в минуту этого хватало, чтобы упереться в 429
    # прямо посреди поиска. Срез спрашивает только цены известных пулов.
    for p in pools:
        MULTI[p["pool"]] = (p["dex"], p["base"], p["quote"],
                            p["reserve_usd"], p["volume_24h"], 1.0)
    calls.clear()
    written = src.pulse()
    multi_calls = sum(1 for u in calls if "/pools/multi/" in u)
    check("срез записал котировки", written > 0, f"{written}")
    check("запросов немного", len(calls) <= 6,
          f"{len(calls)} запросов на {len(pools)} пулов")
    check("использован эндпоинт multi", multi_calls == len(calls), f"{multi_calls}")
    check("поиск при срезе не выполнялся",
          not any("/search/pools" in u or "/dexes" in u for u in calls))

    print("\n5. Живые цены записаны")
    st = store.stats()
    check("свечи появились", st["rows"] > 0, f"{st['rows']} строк")
    check("площадок в базе больше одной", st["venues"] > 1, f"{st['venues']}")

    print("\n" + ("ПРОВАЛЕНО: " + ", ".join(FAIL) if FAIL
                  else "Отбор пулов охватывает несколько площадок"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
