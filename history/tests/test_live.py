"""Быстрый канал свежих цен и модель глубины.

Две вещи, из-за которых монитор молчал при живых спредах на рынке.

Первая — задержка. Цены уезжали к странице только внутри снимка, то есть
раз в прогон, и старели в среднем на восемь минут при спреде, живущем
пять. Быстрый канал кладёт двадцать килобайт после каждого среза.

Вторая — проскальзывание. Пул V3 считался по формуле V2, глубина
занижалась на порядок, и связка на два процента не проходила расчёт
в принципе. Здесь это проверяется числом, а не рассуждением.

Запуск:  python -m history.tests.test_live
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

TMP = tempfile.mkdtemp()
os.environ.setdefault("ARB_DATA_DIR", TMP)

import pandas as pd

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    from history import live
    from history.config import SETTINGS
    from history.store import Candle

    print("\n1. Запись и чтение быстрого файла")

    candles = [
        Candle(ts=1000, venue="pancakeswap-v3-bsc", venue_kind="dex", chain="bsc",
               base="AAVE", quote="DAI", close=250.0, liquidity_usd=120_000.0,
               pool="0xaaa"),
        Candle(ts=1000, venue="uniswap-v3-bsc", venue_kind="dex", chain="bsc",
               base="DAI", quote="USDT", close=1.0002, liquidity_usd=900_000.0,
               pool="0xbbb"),
    ]
    rows = live.rows_from_candles(candles)
    check("свечи свернулись в строки", len(rows) == 2, str(rows[0]))
    check("ключи короткие", set(rows[0]) == {"t", "v", "c", "b", "q", "p", "l", "a"})

    path = live.write_live(rows, chain="bsc")
    check("файл записан", path is not None and path.exists(), str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    check("в файле есть отметка времени", int(payload["ts"]) > 0)
    check("сеть сохранена", payload["chain"] == "bsc")

    size = path.stat().st_size
    check("файл маленький", size < 5_000, f"{size} байт на 2 котировки")

    back = live.read_live()
    check("прочитано обратно", back is not None and len(back["rows"]) == 2)

    age = live.age_seconds(back)
    check("возраст считается", age is not None and age < 5, f"{age:.1f} с")
    check("пустой файл не пишется", live.write_live([]) is None)

    print("\n2. Превращение в котировки")

    df = live.as_frame(back)
    check("получился кадр", len(df) == 2, f"{len(df)} строк")
    check("колонки те же, что у истории",
          list(df.columns) == live.COLUMNS, str(list(df.columns)))
    check("тип площадки проставлен", set(df["venue_kind"]) == {"dex"})
    check("нулевые цены отброшены",
          live.as_frame({"ts": 1, "rows": [{"p": 0, "b": "A", "q": "B"}]}).empty)
    check("пустая нагрузка не роняет", live.as_frame(None).empty)

    print("\n3. Подмешивание к истории")

    hist = pd.DataFrame([{
        "ts": 900, "venue": "pancakeswap-v3-bsc", "venue_kind": "dex",
        "chain": "bsc", "base": "AAVE", "quote": "DAI", "close": 249.0,
        "volume": 10.0, "liquidity_usd": 120_000.0, "pool": "0xaaa",
    }])
    merged = live.merge(hist, df)
    check("история осталась", len(merged) == 3, f"{len(merged)} строк")
    check("свежая точка последняя", int(merged["ts"].iloc[-1]) == 1000)
    check("пустой срез ничего не меняет", len(live.merge(hist, df.iloc[0:0])) == 1)
    check("без истории отдаётся срез", len(live.merge(pd.DataFrame(), df)) == 2)

    print("\n4. Прямой запрос к источнику")

    import history.snapshot as snapshot
    import history.http as http

    fake_pools = pd.DataFrame([{"pool": "0xaaa", "reserve_usd": 500_000.0},
                               {"pool": "0xbbb", "reserve_usd": 100_000.0}])
    answer = {
        "data": [{
            "attributes": {"address": "0xaaa", "name": "AAVE / DAI",
                           "base_token_price_usd": "250.0",
                           "quote_token_price_usd": "1.0",
                           "reserve_in_usd": "500000"},
            "relationships": {
                "base_token": {"data": {"id": "bsc_1"}},
                "quote_token": {"data": {"id": "bsc_2"}},
                "dex": {"data": {"id": "pancakeswap-v3-bsc"}},
            },
        }],
        "included": [
            {"id": "bsc_1", "attributes": {"symbol": "AAVE"}},
            {"id": "bsc_2", "attributes": {"symbol": "DAI"}},
        ],
    }
    calls = []

    def fake_get_json(url, **kw):
        calls.append(url)
        return answer

    old_pools, old_get = snapshot.pools, http.get_json
    snapshot.pools = lambda chain, *a, **k: fake_pools
    http.get_json = fake_get_json
    try:
        got = live.fetch_now(SETTINGS, limit=50)
    finally:
        snapshot.pools, http.get_json = old_pools, old_get

    check("один запрос на пачку до тридцати адресов", len(calls) == 1, str(len(calls)))
    check("адреса ушли в запрос", "0xaaa" in calls[0] and "0xbbb" in calls[0])
    check("котировка разобрана", len(got) == 1, f"{len(got)} строк")
    if len(got):
        r = got.iloc[0]
        check("курс = цена базы / цену котировки", abs(r["close"] - 250.0) < 1e-9,
              str(r["close"]))
        check("площадка сохранена", r["venue"] == "pancakeswap-v3-bsc")
        check("ликвидность подхвачена", abs(r["liquidity_usd"] - 500_000) < 1)

    def boom(url, **kw):
        from history.http import HttpError
        raise HttpError("429 Too Many Requests", status=429)

    http.get_json = boom
    snapshot.pools = lambda chain, *a, **k: fake_pools
    try:
        empty = live.fetch_now(SETTINGS, limit=50)
    finally:
        snapshot.pools, http.get_json = old_pools, old_get
    check("отказ источника не роняет страницу", empty.empty)

    print("\n5. Глубина: V3 больше не считается по формуле V2")

    from history.rates import dex_depth_usd, dex_slippage_factor

    v2 = dex_slippage_factor(1_000, 100_000, venue="pancakeswap-v2-bsc",
                             base="AAVE", quote="BNB")
    v3 = dex_slippage_factor(1_000, 100_000, venue="pancakeswap-v3-bsc",
                             base="AAVE", quote="BNB")
    check("на V2 формула прежняя", abs(v2 - 50_000 / 51_000) < 1e-9,
          f"−{(1 - v2) * 100:.2f}%")
    check("на V3 проскальзывание меньше", v3 > v2,
          f"V3 −{(1 - v3) * 100:.3f}% против V2 −{(1 - v2) * 100:.2f}%")

    stable = dex_slippage_factor(1_000, 100_000, venue="pancakeswap-v3-bsc",
                                 base="DAI", quote="USDT")
    check("пара стейблов глубже обычной V3", stable > v3,
          f"−{(1 - stable) * 100:.4f}%")

    # Тот самый случай: три плеча по тысяче долларов на пулах в сто тысяч.
    # По старой модели это −5.8% и связка на два процента не проходила.
    legs = [
        dex_slippage_factor(1_000, 300_000, venue="pancakeswap-v3-bsc",
                            base="USDT", quote="AAVE"),
        dex_slippage_factor(1_000, 150_000, venue="uniswap-v3-bsc",
                            base="AAVE", quote="DAI"),
        dex_slippage_factor(1_000, 900_000, venue="pancakeswap-v3-bsc",
                            base="DAI", quote="USDT"),
    ]
    total = (1 - legs[0] * legs[1] * legs[2]) * 100
    check("связка на трёх плечах переживает проскальзывание", total < 1.0,
          f"суммарно −{total:.3f}% (раньше было около −6%)")

    check("без данных о резерве глубины нет",
          dex_depth_usd(None, "pancakeswap-v3-bsc") is None)
    check("нулевой резерв не даёт деления на ноль",
          dex_slippage_factor(1_000, 0) == 1.0)

    print("\n6. Разбор связки называет виновника")

    from history import diagnose as dg

    chain = dg.parse_chain("aave-dai")
    check("цепочка достроена до USDT", chain == ["USDT", "AAVE", "DAI", "USDT"],
          str(chain))
    check("разделители любые",
          dg.parse_chain("USDT → AAVE → DAI") == ["USDT", "AAVE", "DAI", "USDT"])

    now = int(time.time())
    q = pd.DataFrame([
        # Спот замкнут с прибылью 2%: 1/250 * 250 * 1.02 * ... — считаем
        # прямо, чтобы в тесте не гадать.
        {"ts": now, "venue": "pancakeswap-v3-bsc", "venue_kind": "dex",
         "chain": "bsc", "base": "USDT", "quote": "AAVE", "close": 1 / 250.0,
         "volume": None, "liquidity_usd": 300_000.0, "pool": "0x1"},
        {"ts": now, "venue": "uniswap-v3-bsc", "venue_kind": "dex",
         "chain": "bsc", "base": "AAVE", "quote": "DAI", "close": 255.0,
         "volume": None, "liquidity_usd": 150_000.0, "pool": "0x2"},
        {"ts": now, "venue": "pancakeswap-v3-bsc", "venue_kind": "dex",
         "chain": "bsc", "base": "DAI", "quote": "USDT", "close": 1.0,
         "volume": None, "liquidity_usd": 900_000.0, "pool": "0x3"},
    ])
    rep = dg.diagnose(chain, q, trade_usd=1_000)
    check("спот посчитан", abs(rep.spot_pct - 2.0) < 1e-6, f"{rep.spot_pct:+.4f}%")
    check("связка проходит после издержек", rep.ok, rep.verdict)
    check("плечей три", len(rep.legs) == 3)
    check("таблица плеч строится", len(dg.legs_frame(rep)) == 3)

    # То же самое, но на пулах V2 — там издержка настоящая.
    q2 = q.copy()
    q2["venue"] = ["pancakeswap-v2-bsc"] * 3
    rep2 = dg.diagnose(chain, q2, trade_usd=1_000)
    check("на V2 та же связка не проходит", not rep2.ok, rep2.verdict[:60])
    check("виноватым назван нужный шаг", rep2.stage in ("slippage", "fees"),
          rep2.stage)

    q3 = q.iloc[:2]
    rep3 = dg.diagnose(chain, q3, trade_usd=1_000)
    check("отсутствующее плечо названо", rep3.stage == "no_quotes"
          and rep3.missing == ["DAI/USDT"], str(rep3.missing))

    q4 = q.copy()
    q4.loc[1, "ts"] = now - 3600
    rep4 = dg.diagnose(chain, q4, trade_usd=1_000)
    check("разъехавшиеся по времени цены не считаются связкой",
          rep4.stage == "not_aligned", rep4.stage)

    q5 = q.copy()
    q5.loc[1, "close"] = 249.0
    rep5 = dg.diagnose(chain, q5, trade_usd=1_000)
    check("отрицательный спот назван прямо", rep5.stage == "no_spot", rep5.stage)

    print("\n" + "=" * 70)
    if FAIL:
        print("НЕ ПРОЙДЕНО:", ", ".join(FAIL))
        return 1
    print("Свежие цены доходят до страницы, глубина V3 больше не выдумывает "
          "издержку")
    return 0


if __name__ == "__main__":
    sys.exit(main())
