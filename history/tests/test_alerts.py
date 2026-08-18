"""Проверка оповещений: отбор, защита от повторов, формат сообщения.

Сети не требует: отправка проверяется на том, что при недоступном
Telegram функция возвращает False, а не роняет прогон сбора.

Запуск:  python -m history.tests.test_alerts
"""

import os, sys, tempfile, time
os.environ["ARB_DATA_DIR"] = tempfile.mkdtemp()
import numpy as np, pandas as pd
from history.config import Settings
from history.rates import build_grid
from history.paths import find_cycles
from history import alerts, store

FAIL=[]
def check(n,c,d=""):
    print(f"  [{'OK  ' if c else 'FAIL'}] {n}" + (f"  — {d}" if d else ""))
    if not c: FAIL.append(n)

store.init()
ADDR={"USDT":"0x55d3","CAKE":"0x0E09","BNB":"0xbb4C"}
store.write_pools([
  dict(chain="bsc",pool="0xp1",dex="pancakeswap_v3",base="BNB",quote="USDT",
       base_addr=ADDR["BNB"],quote_addr=ADDR["USDT"],base_name="BNB",quote_name="Tether USD",reserve_usd=8e6),
  dict(chain="bsc",pool="0xp2",dex="biswap",base="CAKE",quote="USDT",
       base_addr=ADDR["CAKE"],quote_addr=ADDR["USDT"],base_name="PancakeSwap Token",quote_name="Tether USD",reserve_usd=5e5),
])
t0=int(time.time())//900*900 - 20*900
rows=[]
for k in range(20):
    ts=t0+k*900
    for v,b,q,p_,liq in [("pancakeswap_v3","BNB","USDT",600.0,8e6),
                         ("pancakeswap_v3","CAKE","BNB",1/300*1.02,4e6),
                         ("biswap","CAKE","USDT",2.06,5e5)]:
        rows.append(dict(ts=ts,venue=v,venue_kind="dex",chain="bsc",base=b,quote=q,
                         close=p_,volume=1e6,liquidity_usd=liq,pool="0xp1"))
s=Settings(); s.analysis_timeframe="15m"; s.staleness_sec=1800
g=build_grid(pd.DataFrame(rows),settings=s,apply_slippage=False)
tbl,cycles=find_cycles(g,anchor="USDT",max_legs=3,top=20,gas_per_dex_leg_usd=0.0,min_margin_pct=-100,settings=s)
print(f"  связок: {len(cycles)}")

print("\n1. Отбор")
cfg = alerts.AlertConfig(min_margin_pct=0.3, min_liquidity_usd=1e5, max_per_run=5)
picked = alerts.pick(cycles, cfg, {})
check("что-то отобрано", len(picked)>0, f"{len(picked)} шт")
if picked:
    check("все прибыльны сейчас", all(float(c.margin_pct()[-1])>=0.3 for c in picked))
    check("все без переводов", all(not c.needs_transfer() for c in picked))

print("\n2. Порог по ликвидности отсекает мелкие пулы")
strict = alerts.AlertConfig(min_margin_pct=0.3, min_liquidity_usd=1e7)
check("с высоким порогом ничего не проходит", len(alerts.pick(cycles, strict, {}))==0)

print("\n3. Дедупликация")
if picked:
    # Ключ молчания — набор токенов, а не подпись маршрута: одна и та же
    # возможность приходит в нескольких перестановках, и по подписи они
    # выглядели бы разными связками.
    key = alerts.mute_key(picked[0])
    sent = {key: time.time()}
    again = alerts.pick(cycles, cfg, sent)
    check("уже отправленное не повторяется", key not in [alerts.mute_key(c) for c in again])
    check("перестановки не считаются разными", 
          len({alerts.mute_key(c) for c in picked}) == len(picked))
    old = {key: time.time() - 3*3600}
    pruned = alerts._prune_sent(old, 90)
    check("через срок молчания забывается", key not in pruned)

print("\n4. Настройка")
check("без токена не настроено", not alerts.configured())
os.environ["TELEGRAM_BOT_TOKEN"]="123:ABC"; os.environ["TELEGRAM_CHAT_ID"]="42"
check("с токеном настроено", alerts.configured())
os.environ["ALERT_MIN_MARGIN"]="0.75"
check("порог читается из окружения", alerts.AlertConfig.from_env().min_margin_pct==0.75)

print("\n5. Текст сообщения")
if picked:
    txt = alerts.format_message(picked[0], float(picked[0].margin_pct()[-1]))
    print("─"*66); print(txt); print("─"*66)
    check("есть маршрут", picked[0].label in txt)
    check("есть маржа", "Маржа сейчас" in txt)
    check("есть сеть", "BNB Chain" in txt)
    check("есть ссылки на обмен", "pancakeswap.finance" in txt or "1inch" in txt)
    check("есть оговорка про историю", "Окно могло уже закрыться" in txt)
    check("ноги пронумерованы", "1. " in txt and "2. " in txt)
    check("разметка HTML закрыта корректно",
          txt.count("<b>")==txt.count("</b>") and txt.count("<i>")==txt.count("</i>"))

print("\n6. Отправка без сети не роняет процесс")
ok = alerts.send("проба")
check("вернула False, исключения нет", ok is False)

for k in ("TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID","ALERT_MIN_MARGIN"): os.environ.pop(k,None)
print("\n" + ("ПРОВАЛЕНО: "+", ".join(FAIL) if FAIL else "Оповещения работают"))
sys.exit(1 if FAIL else 0)
