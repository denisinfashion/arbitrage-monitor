"""Что на самом деле есть по токену в сети: пулы, размеры, обороты.

Отвечает на вопрос «почему связки через мой токен не находятся»,
обращаясь напрямую к источнику, а не к накопленной базе. Показывает
все пулы токена и по каждому — проходит ли он наши фильтры и какой
именно порог его не пустил.

Запуск:

    python probe_token.py AAVE
    python probe_token.py AAVE DAI CAKE
    python probe_token.py 0xfb6115445bff7b52feb98650c87f44907e58f802
    python probe_token.py AAVE --chain eth

Важное про маршрутизацию. Кошелёк меняет AAVE на DAI даже когда прямого
пула AAVE/DAI не существует: роутер сам разложит обмен на AAVE → WBNB →
DAI. В нашей сетке такого ребра нет — там только прямые пулы, — поэтому
«торгуется с» в интерфейсе означает «есть прямой пул», а не «можно
обменять в кошельке». Разница существенная: каждый скрытый переход стоит
своей комиссии пула, и связка из трёх ног в терминах кошелька может
оказаться пятью реальными обменами.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from history.config import SETTINGS, load_watchlist
from history.http import HttpError, get_json
from history.sources.dex_gt import endpoint


def money(x) -> str:
    if x is None:
        return "—"
    return f"${x:,.0f}".replace(",", " ")


def fetch_pools(api: str, headers: dict, chain: str, query: str) -> list:
    """Пулы токена. По адресу — точный запрос, по тикеру — поиск."""
    out, seen = [], set()

    def add(payload):
        tokens = {o.get("id", ""): o for o in payload.get("included", [])}
        for item in payload.get("data", []) or []:
            attrs = item.get("attributes", {}) or {}
            addr = attrs.get("address")
            if not addr or addr in seen:
                continue
            seen.add(addr)
            rel = item.get("relationships", {}) or {}

            def side(name):
                ref = ((rel.get(name) or {}).get("data") or {}).get("id", "")
                a = (tokens.get(ref) or {}).get("attributes", {}) or {}
                return (a.get("symbol") or "?").upper(), (a.get("address") or "")

            base, base_addr = side("base_token")
            quote, quote_addr = side("quote_token")
            out.append({
                "pool": addr,
                "name": attrs.get("name") or f"{base} / {quote}",
                "base": base, "quote": quote,
                "base_addr": base_addr, "quote_addr": quote_addr,
                "dex": ((rel.get("dex") or {}).get("data") or {}).get("id", "?"),
                "reserve": _f(attrs.get("reserve_in_usd")),
                "volume": _f((attrs.get("volume_usd") or {}).get("h24")),
            })

    params = {"include": "base_token,quote_token,dex"}
    if query.startswith("0x") and len(query) > 20:
        url = f"{api}/networks/{chain}/tokens/{query.lower()}/pools"
        add(get_json(url, params=params, headers=headers))
    else:
        add(get_json(f"{api}/search/pools",
                     params={**params, "query": query, "network": chain},
                     headers=headers))
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def probe(ticker: str, chain: str) -> int:
    api, headers = endpoint(SETTINGS)
    print(f"\n{'=' * 74}\n{ticker} в сети {chain}\n{'=' * 74}")
    try:
        pools = fetch_pools(api, headers, chain, ticker)
    except HttpError as exc:
        print(f"источник недоступен: {exc}")
        return 1

    want = ticker.upper()
    if not ticker.startswith("0x"):
        pools = [p for p in pools if want in (p["base"], p["quote"])]

    if not pools:
        print("пулов не найдено вовсе — проверьте написание тикера "
              "или укажите адрес контракта")
        return 0

    watch = {w.upper() for w in load_watchlist(SETTINGS)}
    in_watch = want in watch
    floor = (SETTINGS.watch_min_reserve_usd if in_watch
             else SETTINGS.min_pool_reserve_usd)

    try:
        from history.quality import MIN_POOL_VOLUME_USD as vol_floor
    except ImportError:
        vol_floor = 0.0

    print(f"в списке наблюдения: {'да' if in_watch else 'нет'} · "
          f"порог ликвидности {money(floor)} · порог оборота {money(vol_floor)}")
    print(f"\n{'пара':<22}{'площадка':<20}{'ликвидность':>14}"
          f"{'оборот 24ч':>14}  вердикт")
    print("-" * 88)

    pools.sort(key=lambda p: p["reserve"] or 0, reverse=True)
    partners, passed = set(), 0
    for p in pools:
        why = []
        if (p["reserve"] or 0) < floor:
            why.append("мал резерв")
        if vol_floor and p["volume"] is not None and p["volume"] < vol_floor:
            why.append("нет оборота")
        ok = not why
        if ok:
            passed += 1
            other = p["quote"] if p["base"] == want else p["base"]
            partners.add(other)
        pair = f"{p['base']} / {p['quote']}"
        print(f"{pair:<22}{p['dex']:<20}{money(p['reserve']):>14}"
              f"{money(p['volume']):>14}  "
              f"{'берём' if ok else 'мимо: ' + ', '.join(why)}")

    print(f"\nпулов всего {len(pools)}, проходит фильтры {passed}")
    if partners:
        print("прямые пары после фильтров: " + ", ".join(sorted(partners)))
        if len(partners) < 2:
            print("\nОдна пара — цикл USDT → … → USDT через этот токен "
                  "не замкнётся: нужен ещё один прямой пул.")
    else:
        print("\nНи один пул не прошёл. Понизить пороги можно в "
              "history/config.py: min_pool_reserve_usd и "
              "watch_min_reserve_usd, а порог оборота — "
              "MIN_POOL_VOLUME_USD в history/quality.py.")

    if not in_watch:
        print(f"\nЧтобы {want} собирался независимо от оборота, "
              "допишите тикер в watchlist.txt.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Показать пулы токена и почему он проходит или не проходит фильтры")
    p.add_argument("tickers", nargs="+", help="тикеры или адреса контрактов")
    p.add_argument("--chain", default=SETTINGS.chain, help="сеть, по умолчанию bsc")
    args = p.parse_args(argv)

    rc = 0
    for t in args.tickers:
        rc |= probe(t, args.chain)
    print("\nПрямой пул — не то же самое, что возможность обмена в кошельке:")
    print("роутер сам проложит путь через промежуточный токен, если прямого")
    print("пула нет. В расчёте такой обмен выглядит как две ноги, а не одна.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
