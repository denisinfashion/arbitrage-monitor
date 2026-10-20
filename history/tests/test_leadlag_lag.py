"""Сбор сделок и поиск лага: подкладываем данные с ЗАРАНЕЕ ИЗВЕСТНЫМ лагом
и требуем, чтобы анализ нашёл именно его; разбор свопов из логов Transfer
и ленты GeckoTerminal — на вручную собранных примерах.

Запуск:  python -m history.tests.test_leadlag_lag
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag import evm, lag, ticks

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def synthetic(n: int, lag_s: int, seed: int, sparse_every: int = 0):
    """Ведущий — случайное блуждание со скачками; ведомый повторяет его через lag_s
    секунд с шумом. sparse_every > 0 — ведомый торгуется редко (как пул DEX)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1e-4, n) + (rng.random(n) < 0.02) * rng.normal(0, 3e-3, n)
    lead = 1.0 * np.exp(np.cumsum(steps))
    follow = np.empty(n)
    follow[lag_s:] = lead[:n - lag_s]
    follow[:lag_s] = lead[0]
    follow *= np.exp(rng.normal(0, 2e-4, n))
    if sparse_every:
        mask = rng.random(n) < 1 / sparse_every
        follow = pd.Series(np.where(mask, follow, np.nan)).ffill().bfill().to_numpy()
    return lead, follow


def test_known_lag():
    print("\n1. Известный лаг находится")
    for lag_s in (3, 7, 20):
        a, b = synthetic(3600, lag_s, seed=lag_s)
        res = lag.analyse(lag.returns(a), lag.returns(b), 60)
        check(f"лаг {lag_s} с: A ведёт B", res.lag == lag_s, f"найдено {res.lag}, corr {res.corr:.2f}")
        res = lag.analyse(lag.returns(b), lag.returns(a), 60)
        check(f"лаг {lag_s} с: в обратном порядке знак минус", res.lag == -lag_s, f"найдено {res.lag}")
    a, b = synthetic(3600, 5, seed=11, sparse_every=8)
    res = lag.analyse(lag.returns(a), lag.returns(b), 60)
    check("редкие сделки у ведомого (раз в ~8 с): лаг ≥ 5 с",
          5 <= res.lag <= 13, f"найдено {res.lag}, corr {res.corr:.2f}")


def test_no_relation():
    print("\n2. Нет связи — нет значимого пика")
    a, _ = synthetic(3600, 1, seed=1)
    b, _ = synthetic(3600, 1, seed=2)
    res = lag.analyse(lag.returns(a), lag.returns(b), 60)
    check("независимые ряды: корреляция меньше порога", res.corr < 0.10, f"corr {res.corr:.3f}")


def test_summarize():
    print("\n3. Итог по часам")
    hours = pd.DataFrame({
        "lag": [5, 5, 6, 4, 5, -30, 5], "corr": [0.4, 0.5, 0.3, 0.35, 0.45, 0.12, 0.05],
        "corr0": [0.05] * 7, "llr": [8.0] * 7, "n_a": [500] * 7, "n_b": [100] * 7,
    })
    r = lag.summarize(hours, 0.10)
    check("направление A → B", r["direction"] == 1)
    check("лаг 5 с", r["lag_s"] == 5, str(r["lag_s"]))
    check("устойчивых часов 5", r["hours_stable"] == 5, str(r["hours_stable"]))
    check("значимых часов 6 (последний слабый)", r["hours_sig"] == 6, str(r["hours_sig"]))
    thin = hours.assign(n_b=10)
    check("мало сделок — часы не в счёт", lag.summarize(thin, 0.10)["hours_data"] == 0)


def test_db_roundtrip():
    print("\n4. База: секунды, повторы свопов, расчёт по часам")
    with tempfile.TemporaryDirectory() as d:
        con = ticks.connect(Path(d) / "t.sqlite")
        hour = 1_800_000_000 // 3600 * 3600
        a, b = synthetic(3600, 4, seed=5)
        ticks.write_seconds(con, [("fut:x:A", hour + i, float(a[i]), 1.0, 1) for i in range(3600)])
        rows = [("dex:bsc:P", f"tx{i}", hour + i, float(b[i]), 2.0) for i in range(0, 3600, 3)]
        check("свопы записаны", ticks.store_swaps(con, rows) == 1200)
        check("повторный опрос не дублирует", ticks.store_swaps(con, rows) == 0)
        vol = con.execute("SELECT sum(vol) FROM prices WHERE inst='dex:bsc:P'").fetchone()[0]
        check("объём не удвоился", abs(vol - 2400.0) < 1e-6, str(vol))
        ra, na = lag.price_grid(con, "fut:x:A", hour, hour + 3600)
        rb, nb = lag.price_grid(con, "dex:bsc:P", hour, hour + 3600)
        res = lag.analyse(lag.returns(ra), lag.returns(rb), 60)
        check("из базы находится лаг 4–6 с", 4 <= res.lag <= 6, f"найдено {res.lag}")
        con.close()


def test_evm_swaps():
    print("\n5. Свопы из переводов ERC-20")
    pool, tok, usdt = "0x" + "a" * 40, "0x" + "b" * 40, "0x" + "c" * 40
    other = "0x" + "d" * 40
    meta = {pool: evm.PoolMeta(pool, "dex:bsc:P", tok, usdt, 18, 18)}

    def tr(tx, idx, token, frm, to, amount, ts=100):
        return {"transactionHash": tx, "logIndex": hex(idx), "address": token,
                "topics": [evm.TRANSFER, evm._topic(frm), evm._topic(to)],
                "data": hex(amount), "blockTimestamp": hex(ts)}

    e18 = 10 ** 18
    logs = [
        tr("t1", 1, usdt, other, pool, 50 * e18),     # покупка: 50 USDT в пул
        tr("t1", 2, tok, pool, other, 1000 * e18),    # 1000 токенов из пула
        tr("t1", 2, tok, pool, other, 1000 * e18),    # тот же лог из второго запроса
        tr("t2", 1, usdt, other, pool, 10 * e18),     # добавление ликвидности —
        tr("t2", 2, tok, other, pool, 200 * e18),     # оба токена в пул
    ]
    swaps = evm.swaps_from_net(evm.net_amounts(logs, meta), meta)
    check("одна сделка (ликвидность и повтор лога не считаются)", len(swaps) == 1, str(len(swaps)))
    check("цена 0,05 USDT", abs(swaps[0][3] - 0.05) < 1e-12, str(swaps[0][3]))
    check("объём 50 USDT", abs(swaps[0][4] - 50) < 1e-9)

    pools = [evm.PoolMeta(f"0x{i:040x}", "", f"0x{i + 100:040x}", usdt, 18, 18) for i in range(20)]
    groups = evm.group_pools(pools)
    sizes = [len({p.token for p in g} | {p.quote for p in g}) for g in groups]
    check("в группе не больше 10 адресов", max(sizes) <= 10, str(sizes))
    check("все пулы распределены", sum(len(g) for g in groups) == 20)


def test_gt_parse():
    print("\n6. Лента GeckoTerminal")
    tok = "0x" + "b" * 40
    items = [
        {"attributes": {"block_timestamp": "2026-10-20T15:41:44Z", "tx_hash": "0x1", "block_number": 7,
                        "from_token_address": tok, "to_token_address": "0xq", "from_token_amount": "200",
                        "to_token_amount": "10", "volume_in_usd": "10", "is_anomalous": False}},
        {"attributes": {"block_timestamp": "2026-10-20T15:41:45Z", "tx_hash": "0x2", "block_number": 8,
                        "from_token_address": "0xq", "to_token_address": tok.upper().replace("0X", "0x"),
                        "from_token_amount": "5", "to_token_amount": "100", "volume_in_usd": "5",
                        "is_anomalous": False}},
        {"attributes": {"block_timestamp": "2026-10-20T15:41:46Z", "tx_hash": "0x3", "block_number": 9,
                        "from_token_address": tok, "to_token_address": "0xq", "from_token_amount": "1",
                        "to_token_amount": "99", "volume_in_usd": "99", "is_anomalous": True}},
    ]
    out = ticks.parse_gt_trades(items, tok, "bsc")
    check("аномальная сделка отброшена", len(out) == 2)
    check("цена продажи 10/200", abs(out[0][2] - 0.05) < 1e-12)
    check("цена покупки 5/100, адрес без учёта регистра", abs(out[1][2] - 0.05) < 1e-12)


def main() -> int:
    test_known_lag()
    test_no_relation()
    test_summarize()
    test_db_roundtrip()
    test_evm_swaps()
    test_gt_parse()
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
