"""Налог на перевод: разбор ответа, хранение, влияние на расчёт.

Проверяется то, из-за чего модуль появился. В чат ушло оповещение
о связке USDT → SPCXB → MARSCOIN → USDT с маржой +0.363%, прошедшей
все фильтры, — а связка убыточна, потому что контракты этих токенов
удерживают процент при обмене. В цене пула этого нет, и увидеть можно
только симуляцией обмена.

Сеть в тестах не трогается: ответ источника подставляется вручную.
Проверяется наша сторона — разбор, хранение, арифметика.

Запуск:  python -m history.tests.test_taxes
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


HONEST = {
    "token": {"symbol": "CAKE", "address": "0x0e09"},
    "simulationSuccess": True,
    "simulationResult": {"buyTax": 0, "sellTax": 0, "transferTax": 0},
    "honeypotResult": {"isHoneypot": False},
    "summary": {"risk": "very_low", "riskLevel": 0},
    "contractCode": {"openSource": True},
}

TAXED = {
    "token": {"symbol": "SPCXB"},
    "simulationSuccess": True,
    "simulationResult": {"buyTax": 3.0, "sellTax": 5.0, "transferTax": 0},
    "honeypotResult": {"isHoneypot": False},
    "summary": {"riskLevel": 60},
    "contractCode": {"openSource": False},
}

TRAP = {
    "token": {"symbol": "MARSCOIN"},
    "simulationSuccess": True,
    "simulationResult": {"buyTax": 0, "sellTax": 100, "transferTax": 0},
    "honeypotResult": {"isHoneypot": True,
                       "honeypotReason": "Cannot sell token"},
    "summary": {"riskLevel": 100},
    "contractCode": {"openSource": False},
}


def main() -> int:
    from history import taxes
    from history.store import init, read_token_risk

    A_HONEST = "0x" + "1" * 40
    A_TAXED = "0x" + "2" * 40
    A_TRAP = "0x" + "3" * 40

    print("\n1. Разбор ответа симуляции")

    import history.http as http
    answers = {A_HONEST: HONEST, A_TAXED: TAXED, A_TRAP: TRAP}
    asked = []

    def fake_get_json(url, params=None, **kw):
        asked.append((url, (params or {}).get("address")))
        return answers[(params or {})["address"]]

    old = http.get_json
    http.get_json = fake_get_json
    try:
        honest = taxes.check(A_HONEST, "bsc")
        taxed = taxes.check(A_TAXED, "bsc")
        trap = taxes.check(A_TRAP, "bsc")
    finally:
        http.get_json = old

    check("сеть передана номером", asked and "IsHoneypot" in asked[0][0])
    check("честный токен: налога нет",
          honest.buy_pct == 0 and honest.sell_pct == 0 and honest.tradable)
    check("честный токен назван правильно", honest.label() == "налога нет")
    check("налог разобран", taxed.buy_pct == 3.0 and taxed.sell_pct == 5.0)
    check("суммарный налог на круг", taxed.round_trip_pct == 8.0)
    check("токен с налогом торговать можно", taxed.tradable)
    check("подпись человеческая", taxed.label() ==
          "налог 3.0% на вход, 5.0% на выход", taxed.label())
    check("honeypot опознан", trap.honeypot and not trap.tradable)
    check("причина сохранена", trap.reason == "Cannot sell token", trap.reason)

    def broken(url, params=None, **kw):
        from history.http import HttpError
        raise HttpError("503", status=503)

    http.get_json = broken
    try:
        none = taxes.check(A_HONEST, "bsc")
    finally:
        http.get_json = old
    check("недоступный источник даёт None, а не приговор", none is None)
    check("мусорный адрес не спрашивается", taxes.check("не адрес") is None)

    print("\n2. Хранение и срок годности")

    init()
    from history.store import write_token_risk
    write_token_risk([honest, taxed, trap])
    back = read_token_risk("bsc")
    check("три записи вернулись", len(back) == 3, str(len(back)))
    check("налог сохранился", back[A_TAXED].sell_pct == 5.0)
    check("запрет сохранился", not back[A_TRAP].tradable)
    check("свежая проверка не протухла", not back[A_HONEST].stale)

    old_risk = taxes.TokenRisk(address=A_HONEST, chain="bsc",
                               checked_at=int(time.time()) - taxes.TTL_SEC - 1)
    check("вчерашняя проверка протухла", old_risk.stale)

    print("\n3. Только непроверенное спрашивается заново")

    asked.clear()
    http.get_json = fake_get_json
    try:
        # Все три уже в базе и свежие — запросов быть не должно.
        taxes.refresh([A_HONEST, A_TAXED, A_TRAP], "bsc", budget=10)
        check("свежее не перепроверяется", not asked, str(len(asked)))

        A_NEW = "0x" + "4" * 40
        answers[A_NEW] = HONEST
        taxes.refresh([A_NEW], "bsc", budget=10)
        check("новое спрашивается", len(asked) == 1, str(len(asked)))

        asked.clear()
        answers["0x" + "5" * 40] = HONEST
        answers["0x" + "6" * 40] = HONEST
        taxes.refresh(["0x" + "5" * 40, "0x" + "6" * 40], "bsc", budget=1)
        check("бюджет соблюдается", len(asked) == 1, str(len(asked)))
    finally:
        http.get_json = old

    print("\n4. Множитель плеча")

    check("продажа облагается",
          abs(taxes.leg_factor(taxed, None) - 0.95) < 1e-12)
    check("покупка облагается",
          abs(taxes.leg_factor(None, taxed) - 0.97) < 1e-12)
    check("обе стороны сразу",
          abs(taxes.leg_factor(taxed, taxed) - 0.95 * 0.97) < 1e-12)
    check("без проверки множитель единичный",
          taxes.leg_factor(None, None) == 1.0)

    print("\n5. Налог входит в сетку курсов")

    from history.rates import asset_tax_map, build_grid, leg_tax_factor
    from history.config import Settings

    pools = pd.DataFrame([{
        "chain": "bsc", "pool": "0xp1", "dex": "pancakeswap-v3-bsc",
        "base": "SPCXB", "quote": "USDT",
        "base_addr": A_TAXED, "quote_addr": "0xusdt",
        "reserve_usd": 500_000.0, "volume_24h": 100_000.0, "fee_pct": 0.25,
        "base_tax_buy": 3.0, "base_tax_sell": 5.0, "base_tradable": True,
        "base_risk_note": "",
        "quote_tax_buy": 0.0, "quote_tax_sell": 0.0, "quote_tradable": True,
        "quote_risk_note": "",
    }])
    tmap = asset_tax_map(pools)
    check("карта налогов собрана", tmap.get("SPCXB") == (3.0, 5.0, True),
          str(tmap.get("SPCXB")))
    check("чистый токен тоже в карте", tmap.get("USDT") == (0.0, 0.0, True))

    check("покупка токена с налогом дороже на налог покупки",
          abs(leg_tax_factor(tmap, "USDT", "SPCXB", "dex") - 0.97) < 1e-12)
    check("продажа — на налог продажи",
          abs(leg_tax_factor(tmap, "SPCXB", "USDT", "dex") - 0.95) < 1e-12)
    check("на бирже налога нет",
          leg_tax_factor(tmap, "SPCXB", "USDT", "cex") == 1.0)

    # Сквозь build_grid: тот же рынок с налогом и без него.
    t0 = int(time.time()) // 300 * 300 - 20 * 300
    rows = []
    for k in range(20):
        ts = t0 + k * 300
        rows.append({"ts": ts, "venue": "pancakeswap-v3-bsc", "venue_kind": "dex",
                     "chain": "bsc", "base": "SPCXB", "quote": "USDT",
                     "close": 2.0, "volume": 1000.0,
                     "liquidity_usd": 500_000.0, "pool": "0xp1"})
    quotes = pd.DataFrame(rows)

    s = Settings()
    s.staleness_sec = 3600

    import history.rates as rates
    real_pools = rates._pools_frame

    def with_tax(_settings):
        return pools

    def without_tax(_settings):
        return pools.drop(columns=[c for c in pools.columns if "tax" in c
                                   or "tradable" in c or "risk_note" in c])

    try:
        rates._pools_frame = without_tax
        clean = build_grid(quotes, settings=s, trade_size_usd=1000.0,
                           apply_slippage=False, drop_suspicious=False)
        rates._pools_frame = with_tax
        dirty = build_grid(quotes, settings=s, trade_size_usd=1000.0,
                           apply_slippage=False, drop_suspicious=False)
    finally:
        rates._pools_frame = real_pools

    i_u, i_s = clean.assets.index("USDT"), clean.assets.index("SPCXB")
    buy_clean = float(np.exp(clean.log_rate[-1, i_u, i_s]))
    buy_dirty = float(np.exp(dirty.log_rate[-1, i_u, i_s]))
    check("покупка стала хуже ровно на налог покупки",
          abs(buy_dirty / buy_clean - 0.97) < 1e-6,
          f"{buy_dirty / buy_clean:.6f}")

    sell_clean = float(np.exp(clean.log_rate[-1, i_s, i_u]))
    sell_dirty = float(np.exp(dirty.log_rate[-1, i_s, i_u]))
    check("продажа — ровно на налог продажи",
          abs(sell_dirty / sell_clean - 0.95) < 1e-6,
          f"{sell_dirty / sell_clean:.6f}")
    check("направления не перепутаны", abs(buy_dirty / buy_clean
                                           - sell_dirty / sell_clean) > 1e-3)
    check("сетка знает налоги по тикерам",
          dirty.asset_tax.get("SPCXB") == (3.0, 5.0, True))

    print("\n6. Непродаваемый токен до расчёта не доходит")

    trap_pools = pools.copy()
    trap_pools["base_tradable"] = False
    trap_pools["base_risk_note"] = "Cannot sell token"

    def with_trap(_settings):
        return trap_pools

    try:
        rates._pools_frame = with_trap
        try:
            build_grid(quotes, settings=s, trade_size_usd=1000.0,
                       apply_slippage=False, drop_suspicious=True)
            gone = False
        except ValueError:
            gone = True
    finally:
        rates._pools_frame = real_pools
    check("рынок из одного honeypot схлопывается в пустоту", gone)

    print("\n7. Оповещение по такому токену не уходит")

    from history.alerts import AlertConfig, pick
    from history.paths import Cycle
    from history.rates import RateGrid

    def cycle_with(tax_map):
        v = "pancakeswap-v3-bsc"
        g = RateGrid(times=np.array([int(time.time())]),
                     assets=["USDT", "SPCXB", "MARSCOIN"], venues=[v],
                     log_rate=np.zeros((1, 1, 1), dtype=np.float32),
                     venue_idx=np.zeros((1, 1, 1), dtype=np.int16),
                     trade_size_usd=1000.0, venue_kind={v: "dex"},
                     venue_chain={v: "bsc"}, asset_tax=tax_map)
        for x, y in (("USDT", "SPCXB"), ("SPCXB", "MARSCOIN"),
                     ("MARSCOIN", "USDT")):
            g.pair_liquidity[(v, x, y)] = 287_402.0
            g.pair_liquidity[(v, y, x)] = 287_402.0
        return Cycle(assets=("USDT", "SPCXB", "MARSCOIN", "USDT"),
                     log_margin=np.array([np.log(1.00363)]),
                     venues=[np.array([0])] * 3, grid=g)

    cfg = AlertConfig(min_margin_pct=0.3, min_liquidity_usd=30_000,
                      require_known_tokens=False)

    ok_cycle = cycle_with({"SPCXB": (0.0, 0.0, True),
                           "MARSCOIN": (0.0, 0.0, True)})
    check("чистая связка уходит", len(pick([ok_cycle], cfg, {})) == 1)

    bad_cycle = cycle_with({"SPCXB": (3.0, 5.0, True),
                            "MARSCOIN": (0.0, 100.0, False)})
    check("связка с непродаваемым токеном не уходит",
          pick([bad_cycle], cfg, {}) == [])

    from history.quality import taxed_in
    only_taxed = cycle_with({"SPCXB": (3.0, 5.0, True),
                             "MARSCOIN": (1.0, 1.0, True)})
    check("связка с налогом остаётся — он уже вычтен из маржи",
          len(pick([only_taxed], cfg, {})) == 1)
    check("налоги перечислены для сообщения",
          taxed_in(only_taxed) == {"SPCXB": 8.0, "MARSCOIN": 2.0},
          str(taxed_in(only_taxed)))

    from history.alerts import format_message
    text = format_message(only_taxed, 0.363)
    check("налог виден в тексте оповещения",
          "Налог контракта" in text and "SPCXB 8.0%" in text)

    print("\n8. Разбор связки показывает налог отдельной строкой")

    from history import diagnose as dg

    now = int(time.time())
    q = pd.DataFrame([
        {"ts": now, "venue": "pancakeswap-v3-bsc", "venue_kind": "dex",
         "chain": "bsc", "base": "USDT", "quote": "SPCXB", "close": 0.5,
         "volume": None, "liquidity_usd": 500_000.0, "pool": "0xp1"},
        {"ts": now, "venue": "pancakeswap-v3-bsc", "venue_kind": "dex",
         "chain": "bsc", "base": "SPCXB", "quote": "USDT", "close": 2.05,
         "volume": None, "liquidity_usd": 500_000.0, "pool": "0xp2"},
    ])
    chain = ["USDT", "SPCXB", "USDT"]

    clean_rep = dg.diagnose(chain, q, trade_usd=1000.0, fees={}, taxes={})
    check("без налога связка проходит", clean_rep.ok, clean_rep.verdict)

    taxed_rep = dg.diagnose(chain, q, trade_usd=1000.0, fees={},
                            taxes={"SPCXB": (3.0, 5.0, True)})
    check("налог посчитан на обоих плечах",
          abs(taxed_rep.tax_pct - 8.0) < 1e-9, f"{taxed_rep.tax_pct:.3f}%")
    check("итог упал", taxed_rep.net_pct < clean_rep.net_pct,
          f"{taxed_rep.net_pct:+.2f}% против {clean_rep.net_pct:+.2f}%")
    check("виноватым назван налог", taxed_rep.stage == "tax", taxed_rep.stage)
    check("в вердикте сказано про контракт",
          "налог контрактов" in taxed_rep.verdict.lower(), taxed_rep.verdict)
    check("колонка налога есть в таблице",
          "Налог, %" in dg.legs_frame(taxed_rep).columns)

    print("\n" + "=" * 70)
    if FAIL:
        print("НЕ ПРОЙДЕНО:", ", ".join(FAIL))
        return 1
    print("Налог на перевод виден расчёту, а непродаваемые токены "
          "до оповещений не доходят")
    return 0


if __name__ == "__main__":
    sys.exit(main())
