"""Диагностика: проверяет, что все источники данных доступны с этой машины.

Запуск из папки arb_calculator:

    python selftest.py

Скрипт по очереди дёргает каждый внешний источник, показывает реальный
ответ и объясняет, что делать при отказе. Ничего не пишет в базу.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Callable, List, Tuple

RESULTS: List[Tuple[str, bool, str]] = []


def section(title: str) -> None:
    print("\n" + "─" * 72)
    print(title)
    print("─" * 72)


def probe(name: str, fn: Callable[[], str], hint: str = "") -> bool:
    t0 = time.time()
    try:
        detail = fn()
        dt = time.time() - t0
        print(f"  [ОК]    {name}  ({dt:.1f} с)")
        if detail:
            for line in detail.splitlines():
                print(f"          {line}")
        RESULTS.append((name, True, detail))
        return True
    except Exception as exc:
        dt = time.time() - t0
        msg = f"{type(exc).__name__}: {exc}"
        print(f"  [СБОЙ]  {name}  ({dt:.1f} с)")
        print(f"          {msg[:300]}")
        if hint:
            print(f"          → {hint}")
        RESULTS.append((name, False, msg))
        return False


# --------------------------------------------------------------------------
# 1. Библиотеки
# --------------------------------------------------------------------------


def check_imports() -> None:
    section("1. Библиотеки")

    def _v(mod: str) -> Callable[[], str]:
        def inner() -> str:
            m = __import__(mod)
            return f"версия {getattr(m, '__version__', '?')}"
        return inner

    for mod, hint in [
        ("pandas", "pip install pandas"),
        ("numpy", "pip install numpy"),
        ("requests", "pip install requests"),
        ("ccxt", "pip install ccxt — без него не будет истории с бирж"),
        ("plotly", "pip install plotly — без него не построится график"),
        ("streamlit", "pip install streamlit"),
    ]:
        probe(mod, _v(mod), hint)


# --------------------------------------------------------------------------
# 2. Хранилище
# --------------------------------------------------------------------------


def check_storage() -> None:
    section("2. Хранилище")

    def create() -> str:
        from history import store
        from history.config import DB_PATH
        store.init()
        return f"база: {DB_PATH}"

    def write_read() -> str:
        from history.store import Candle, read_quotes, write_candles
        ts = int(time.time()) // 60 * 60
        n = write_candles([Candle(ts=ts, venue="__selftest__", venue_kind="cex",
                                  chain="", base="TEST", quote="USDT", close=1.23)])
        df = read_quotes(since_ts=ts - 60, venues=["__selftest__"])
        from history.store import transaction
        with transaction() as c:
            c.execute("DELETE FROM quotes WHERE venue='__selftest__'")
        assert len(df) >= 1, "запись прошла, а чтение вернуло пусто"
        return f"записано {n}, прочитано {len(df)} — запись и чтение работают"

    probe("создание базы", create, "проверьте права на запись в папку проекта")
    probe("запись и чтение", write_read)


# --------------------------------------------------------------------------
# 3. Биржи через ccxt
# --------------------------------------------------------------------------


def check_cex() -> None:
    section("3. Биржи (ccxt)")
    try:
        import ccxt
    except ImportError:
        print("  ccxt не установлен — раздел пропущен")
        return

    from history.config import SETTINGS

    for vid in SETTINGS.cex_venues:
        def make(vid=vid) -> Callable[[], str]:
            def inner() -> str:
                ex = getattr(ccxt, vid)({"enableRateLimit": True, "timeout": 20000})
                ex.load_markets()
                spot = [m for m in ex.markets.values() if m.get("spot")]
                ohlcv = ex.fetch_ohlcv("BTC/USDT", "1m", limit=5)
                if not ohlcv:
                    raise RuntimeError("fetch_ohlcv вернул пусто")
                newest = time.strftime("%H:%M", time.gmtime(ohlcv[-1][0] / 1000))
                age = (time.time() - ohlcv[-1][0] / 1000) / 60
                return (f"{len(spot)} спот-пар, свечи получены "
                        f"(последняя {newest} UTC, возраст {age:.0f} мин)")
            return inner

        probe(f"{vid}", make(),
              "биржа может блокировать регион — уберите её из cex_venues "
              "в history/config.py или используйте VPN")


# --------------------------------------------------------------------------
# 4. GeckoTerminal
# --------------------------------------------------------------------------


def check_geckoterminal() -> None:
    section("4. GeckoTerminal (история DEX)")

    from history.config import SETTINGS
    from history.http import get_json
    from history.sources.dex_gt import API, HEADERS

    state = {}

    def top_pools() -> str:
        d = get_json(f"{API}/networks/{SETTINGS.chain}/pools",
                     params={"page": 1, "include": "base_token,quote_token,dex"},
                     headers=HEADERS)
        items = d.get("data") or []
        if not items:
            raise RuntimeError("список пулов пуст")
        a = items[0].get("attributes", {})
        state["pool"] = a.get("address")
        state["name"] = a.get("name")
        return (f"{len(items)} пулов на странице; первый: {a.get('name')} "
                f"резерв ${float(a.get('reserve_in_usd') or 0):,.0f}".replace(",", " "))

    def structure() -> str:
        """Проверяем, что поля, на которые опирается парсер, действительно есть."""
        d = get_json(f"{API}/networks/{SETTINGS.chain}/pools",
                     params={"page": 1, "include": "base_token,quote_token,dex"},
                     headers=HEADERS)
        item = (d.get("data") or [{}])[0]
        attrs = item.get("attributes", {})
        rel = item.get("relationships", {})
        missing = [f for f in ("address", "name", "base_token_price_usd",
                               "quote_token_price_usd", "reserve_in_usd")
                   if f not in attrs]
        missing += [f"relationships.{f}" for f in ("base_token", "quote_token", "dex")
                    if f not in rel]
        included = d.get("included") or []
        if missing:
            raise RuntimeError(f"в ответе нет полей: {', '.join(missing)}")
        return f"все нужные поля на месте; included: {len(included)} объектов"

    def ohlcv() -> str:
        pool = state.get("pool")
        if not pool:
            raise RuntimeError("сначала должен пройти запрос списка пулов")
        d = get_json(f"{API}/networks/{SETTINGS.chain}/pools/{pool}/ohlcv/minute",
                     params={"aggregate": 1, "limit": 1000, "currency": "usd",
                             "token": "base"},
                     headers=HEADERS)
        rows = ((d.get("data") or {}).get("attributes") or {}).get("ohlcv_list") or []
        if not rows:
            raise RuntimeError("ohlcv_list пуст")
        span_h = (rows[0][0] - rows[-1][0]) / 3600
        return (f"{len(rows)} свечей за один запрос, глубина {span_h:.1f} ч; "
                f"формат строки: {rows[0]}")

    def rate_limit() -> str:
        """Проверяем реальный лимит: пять запросов подряд."""
        t0 = time.time()
        for i in range(5):
            get_json(f"{API}/networks/{SETTINGS.chain}/pools",
                     params={"page": i + 1}, headers=HEADERS)
        dt = time.time() - t0
        return (f"5 запросов за {dt:.1f} с (~{5 / dt * 60:.0f} запросов/мин "
                f"с учётом встроенного троттлинга)")

    ok = probe("список топ-пулов", top_pools,
               "проверьте доступ к api.geckoterminal.com из браузера")
    if ok:
        probe("структура ответа", structure,
              "API изменился — нужно поправить _parse_pool в history/sources/dex_gt.py")
        probe("исторические свечи OHLCV", ohlcv)
        probe("лимит запросов", rate_limit)


# --------------------------------------------------------------------------
# 5. Модули исходного проекта
# --------------------------------------------------------------------------


def check_legacy() -> None:
    section("5. Модули исходного проекта")

    def dexes_ok() -> str:
        import dexes
        chains = list(dexes.CHAINS)
        tokens = sum(len(c.get("tokens", {})) for c in dexes.CHAINS.values())
        return f"{len(chains)} сетей, {tokens} токенов"

    def dex_live() -> str:
        import dexes
        q = dexes.dex_quote("PancakeSwap", "BNB Chain", "USDT", "BNB", 1000.0)
        if not q.ok:
            raise RuntimeError(q.error or "котировка не получена")
        return f"1000 USDT -> {q.amount_out:.6f} BNB, газ ${q.gas_usd:.3f}"

    def exchanges_ok() -> str:
        import exchanges
        return f"{len(exchanges.EXCHANGES)} адаптеров CEX"

    probe("dexes.py импортируется", dexes_ok)
    probe("exchanges.py импортируется", exchanges_ok)
    probe("живая котировка PancakeSwap", dex_live,
          "публичный RPC может быть недоступен — проверьте bsc-rpc.publicnode.com")


# --------------------------------------------------------------------------
# 6. Математика
# --------------------------------------------------------------------------


def check_math() -> None:
    section("6. Математика (офлайн)")

    def run() -> str:
        from history.tests.test_math import main as run_tests
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run_tests()
        out = buf.getvalue()
        n_ok = out.count("[OK  ]")
        n_bad = out.count("[FAIL]")
        if code != 0:
            raise RuntimeError(f"провалено проверок: {n_bad}\n{out[-800:]}")
        return f"пройдено {n_ok} проверок"

    probe("проверки расчётов", run)


# --------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Диагностика исторического арбитражного сканнера")
    print("=" * 72)

    for fn in (check_imports, check_storage, check_math, check_cex,
               check_geckoterminal, check_legacy):
        try:
            fn()
        except Exception:
            print("  непредвиденный сбой раздела:")
            traceback.print_exc()

    section("Итог")
    ok = [r for r in RESULTS if r[1]]
    bad = [r for r in RESULTS if not r[1]]
    print(f"  Успешно: {len(ok)}   Отказов: {len(bad)}")
    if bad:
        print("\n  Не работает:")
        for name, _, msg in bad:
            print(f"    • {name}: {msg[:160]}")
        print("\n  Скопируйте этот вывод — по нему видно, что чинить.")
    else:
        print("\n  Все источники доступны. Запускайте сборщик:")
        print("      python -m history.collector")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
