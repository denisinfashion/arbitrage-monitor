"""Проверка инвентаря токенов.

Смысл инвентаря в том, чтобы различать «связки нет» и «токен вообще
не проверялся». Тесты проверяют именно это различие, а не оформление.

Запуск:  python -m history.tests.test_inventory
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("ARB_DATA_DIR", tempfile.mkdtemp())

import pandas as pd

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    from history import inventory

    now = 1_700_000_000
    rows = []
    for k in range(6):
        ts = now - (5 - k) * 300
        rows += [
            dict(ts=ts, venue="pancakeswap_v3", venue_kind="dex", base="WBNB",
                 quote="USDT"),
            dict(ts=ts, venue="biswap", venue_kind="dex", base="CAKE",
                 quote="USDT"),
            dict(ts=ts, venue="pancakeswap_v3", venue_kind="dex", base="CAKE",
                 quote="WBNB"),
            # токен с единственной парой: цикл через него не замкнётся
            dict(ts=ts, venue="apeswap", venue_kind="dex", base="LONELY",
                 quote="USDT"),
        ]
    # протухший токен: последняя котировка два часа назад
    rows.append(dict(ts=now - 7200, venue="thena", venue_kind="dex",
                     base="OLD", quote="USDT"))
    rows.append(dict(ts=now - 7200, venue="thena", venue_kind="dex",
                     base="OLD", quote="WBNB"))
    quotes = pd.DataFrame(rows)

    pools = pd.DataFrame([
        dict(chain="bsc", pool="0x1", dex="pancakeswap_v3", base="WBNB",
             quote="USDT", base_name="Wrapped BNB", quote_name="Tether USD",
             reserve_usd=8e6, volume_24h=9e6),
        dict(chain="bsc", pool="0x2", dex="biswap", base="CAKE", quote="USDT",
             base_name="PancakeSwap Token", quote_name="Tether USD",
             reserve_usd=6e5, volume_24h=4e5),
    ])

    notes = {"FAKE": "тикер занят чужим контрактом (пулов 2)"}
    watch = ["AAVE", "CAKE"]

    inv = inventory.build("bsc", quotes, pools, notes=notes,
                          watchlist=watch, window_h=6.0, now=now)

    print("\n1. Таблица собрана")
    check("токены найдены", not inv.tokens.empty, f"{len(inv.tokens)} строк")
    check("имя подтянуто из справочника",
          inv.tokens.loc[inv.tokens["Тикер"] == "CAKE", "Имя"].iloc[0]
          == "PancakeSwap Token")
    check("число пар посчитано",
          int(inv.tokens.loc[inv.tokens["Тикер"] == "CAKE", "Пар"].iloc[0]) == 2)
    check("свежесть посчитана",
          float(inv.tokens.loc[inv.tokens["Тикер"] == "WBNB",
                               "Свежесть, мин"].iloc[0]) == 0.0)

    print("\n2. Токен из списка, которого нет в данных")
    check("AAVE в таблице есть", inv.known("AAVE"))
    check("AAVE помечен как ненайденный",
          int(inv.tokens.loc[inv.tokens["Тикер"] == "AAVE", "Пар"].iloc[0]) == 0)
    v = inv.verdict("AAVE")
    check("вердикт: нет в данных", v["status"] == "нет в данных", v["text"][:70])
    check("сказано, что он в списке наблюдения",
          "списк" in v["text"].lower())

    print("\n3. Токен, которого нет ни в данных, ни в списке")
    v = inv.verdict("ZZZZ")
    check("вердикт: нет в данных", v["status"] == "нет в данных")
    check("совет — добавить в watchlist", "watchlist" in v["advice"])

    print("\n4. Отсеянный токен")
    v = inv.verdict("FAKE")
    check("вердикт: отсеян", v["status"] == "отсеян", v["text"][:70])
    check("причина названа", "чужим контрактом" in v["text"])

    print("\n5. Токен с одной парой")
    v = inv.verdict("LONELY")
    check("вердикт: мало пар", v["status"] == "мало пар", v["text"][:70])
    # Формулировка важна не меньше вердикта: «торгуется только с USDT»
    # читалось как «обменять больше не на что», хотя речь про прямые пулы.
    check("сказано про прямой пул", "прямой пул" in v["text"])
    check("объяснено, что кошелёк проложит путь сам",
          "промежуточный" in v["advice"])
    check("подсказана проверка источника", "probe_token.py" in v["advice"])

    print("\n6. Протухший токен")
    v = inv.verdict("OLD")
    check("вердикт: устарел", v["status"] == "устарел", v["text"][:70])

    print("\n7. Нормальный токен")
    v = inv.verdict("CAKE")
    check("вердикт: проверяется", v["status"] == "проверяется", v["text"][:70])
    check("сказано, что связки нет не из-за пропуска",
          "не сходится" in v["advice"])

    print("\n8. Сводка")
    s = inventory.summary(inv)
    check("посчитано отсеянных", s["отсеяно"] == 1)
    check("из списка найден один", s["из списка наблюдения"] == 1, str(s))
    check("из списка не найден один", s["не найдено из списка"] == 1)

    print("\n9. Пустые данные не роняют")
    empty = inventory.build("bsc", pd.DataFrame(), None, watchlist=["AAVE"])
    check("пустой инвентарь строится", empty.tokens.empty)
    check("сводка на пустых данных считается",
          inventory.summary(empty)["токенов"] == 0)

    print("\n10. Список наблюдения из файла")
    from history.config import load_watchlist
    wl = load_watchlist()
    check("файл watchlist.txt прочитан", len(wl) > 0, ", ".join(wl[:5]))
    check("комментарии отброшены", all("#" not in w for w in wl))
    check("регистр приведён", all(w == w.upper() for w in wl))

    print("\n" + ("ПРОВАЛЕНО: " + ", ".join(FAIL) if FAIL
                  else "Инвентарь различает «нет связки» и «нет данных»"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
