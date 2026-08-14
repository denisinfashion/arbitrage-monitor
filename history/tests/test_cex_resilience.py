"""Устойчивость сбора с бирж к их реальным отказам.

Оба сценария взяты из живого прогона в GitHub Actions:

  * OKX отдал 50011 «Too Many Requests» по 75 парам из 200 — троттлинг ccxt
    рассчитан на последовательные вызовы и не держит несколько потоков
    на одном объекте биржи;

  * Gate отказал по ВСЕМ 200 парам: «Candlestick too long ago. Maximum 10000
    points ago are allowed». Семь суток минутных свечей — это 10080 точек,
    промах мимо лимита на восемьдесят штук.

Сеть здесь не нужна: объект биржи подменяется заглушкой, которая
воспроизводит эти ответы.

Запуск:  python -m history.tests.test_cex_resilience
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="arb_cex_")
os.environ["ARB_DATA_DIR"] = _TMP

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


class FakeExchange:
    """Заглушка биржи: считает вызовы и отдаёт заданные отказы."""

    def __init__(self, fail_times: int = 0, exc=None, depth_error: bool = False,
                 candles: int = 5):
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc
        self.depth_error = depth_error
        self.candles = candles
        self.seen_since = []
        self.markets = {}
        self.fees = {"trading": {"taker": 0.001}}
        self.has = {"fetchTickers": False}

    def load_markets(self):
        return self.markets

    def fetch_ohlcv(self, symbol, tf, since=None, limit=None):
        self.calls += 1
        self.seen_since.append(since)
        if self.depth_error:
            import ccxt
            raise ccxt.BadRequest(
                '{"label":"INVALID_PARAM_VALUE","message":"Candlestick too long '
                'ago. Maximum 10000 points ago are allowed"}')
        if self.calls <= self.fail_times and self.exc is not None:
            raise self.exc("okx {\"msg\":\"Too Many Requests\",\"code\":\"50011\"}")
        base = since or 0
        return [[base + i * 60_000, 1.0, 1.0, 1.0, 1.0 + i * 0.001, 10.0]
                for i in range(self.candles)]


def main() -> int:
    print("=" * 70)
    print("Устойчивость сбора с бирж")
    print("=" * 70)

    import ccxt
    from history import store
    from history.config import (CEX_MAX_HISTORY_CANDLES, CEX_MAX_WORKERS,
                                DEFAULT_CEX_WORKERS, SETTINGS)
    from history.sources.cex_ccxt import CexSource

    store.init()

    # ------------------------------------------------------------------ 1
    print("\n1. Отказ по лимиту запросов: повтор с паузой")

    src = CexSource("okx", SETTINGS)
    fake = FakeExchange(fail_times=2, exc=ccxt.RateLimitExceeded, candles=3)
    src.ex = fake
    src._markets_loaded = True

    t0 = time.time()
    got = src._fetch_symbol("BTC/USDT", int(time.time() * 1000) - 600_000,
                            int(time.time() * 1000))
    dt = time.time() - t0
    check("после двух отказов данные получены", len(got) > 0, f"{len(got)} свечей")
    check("сделано три попытки", fake.calls >= 3, f"вызовов: {fake.calls}")
    check("между попытками была пауза", dt >= 3.0, f"{dt:.1f} с")

    # не бесконечно
    src2 = CexSource("okx", SETTINGS)
    fake2 = FakeExchange(fail_times=99, exc=ccxt.RateLimitExceeded)
    src2.ex = fake2
    src2._markets_loaded = True
    try:
        src2._fetch_symbol("BTC/USDT", 0, int(time.time() * 1000))
        check("бесконечных повторов нет", False, "исключения не было")
    except RuntimeError as exc:
        check("бесконечных повторов нет", fake2.calls == 4, f"вызовов: {fake2.calls}")
        check("текст ошибки объясняет причину",
              "лимит запросов" in str(exc), str(exc)[:80])

    # ------------------------------------------------------------------ 2
    print("\n2. Биржа не отдаёт запрошенную глубину")

    src3 = CexSource("mexc", SETTINGS)   # у mexc известного лимита нет
    fake3 = FakeExchange(depth_error=True)
    src3.ex = fake3
    src3._markets_loaded = True

    check("до отказа ограничения нет", src3._max_history_candles() is None)
    got = src3._fetch_symbol("BTC/USDT", 0, int(time.time() * 1000))
    check("пара пропущена без исключения", got == [], f"{len(got)} свечей")
    check("ограничение запомнено", src3._depth_capped)
    check("дальше глубина урезается", src3._max_history_candles() == 9_800,
          str(src3._max_history_candles()))

    # ------------------------------------------------------------------ 3
    print("\n3. Обрезка глубины под лимит биржи")

    s = SETTINGS
    old_tf, old_days = s.timeframe, s.history_days
    s.timeframe, s.history_days = "1m", 7.0

    src4 = CexSource("gate", s)
    src4.ex = FakeExchange()
    src4._markets_loaded = True

    until = int(time.time() * 1000)
    want = until - int(7 * 86400 * 1000)          # 10080 минутных свечей
    clamped = src4._clamp_since(want, until)
    candles = (until - clamped) / 60_000
    check("для gate глубина урезана", clamped > want, f"{candles:.0f} свечей")
    check("уложились в лимит биржи", candles <= 10_000, f"{candles:.0f} против 10000")
    check("это ровно настроенный потолок", abs(candles - 9_800) < 1,
          f"{candles:.0f}")

    src5 = CexSource("mexc", s)
    src5.ex = FakeExchange()
    src5._markets_loaded = True
    check("для биржи без лимита глубина не трогается",
          src5._clamp_since(want, until) == want)

    # часовые свечи в лимит влезают целиком
    s.timeframe = "1h"
    check("на часовом таймфрейме обрезки нет",
          src4._clamp_since(want, until) == want,
          "7 дней это 168 свечей, лимит не задет")
    s.timeframe, s.history_days = old_tf, old_days

    # ------------------------------------------------------------------ 4
    print("\n4. Параллелизм под квоты площадок")

    check("okx ограничен", CexSource("okx", s)._workers(None) == CEX_MAX_WORKERS["okx"],
          str(CexSource("okx", s)._workers(None)))
    check("mexc по умолчанию",
          CexSource("mexc", s)._workers(None) == DEFAULT_CEX_WORKERS,
          str(CexSource("mexc", s)._workers(None)))
    check("явное значение имеет приоритет",
          CexSource("okx", s)._workers(7) == 7)

    # ------------------------------------------------------------------ 5
    print("\n5. Распознавание текста отказа")

    from history.sources.cex_ccxt import CexSource as C
    yes = ['Candlestick too long ago. Maximum 10000 points ago are allowed',
           'candlestick too old', 'Maximum 10000 points ago are allowed']
    no = ['Too Many Requests', 'symbol not found', 'invalid signature']
    for t in yes:
        check(f"распознан: {t[:40]}", C._is_depth_error(t))
    for t in no:
        check(f"не спутан: {t[:40]}", not C._is_depth_error(t))

    print("\n" + "=" * 70)
    if FAILED:
        print(f"ПРОВАЛЕНО {len(FAILED)}:")
        for f in FAILED:
            print("  -", f)
        return 1
    print("Сбор переживает отказы бирж")
    return 0


if __name__ == "__main__":
    code = main()
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(code)
