"""Что именно собирается: токены, пулы, площадки, свежесть.

Страница отвечает на один вопрос — «проверяется ли мой токен». Без неё
отсутствие связки выглядит одинаково и когда возможности нет, и когда
токена вообще нет в данных, а это разные ситуации с разными действиями.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history import inventory, snapshot
from history.config import SETTINGS, WATCHLIST_FILE, load_watchlist
from history.ui import FULL

try:
    from history.config import CODE_VERSION
except ImportError:
    CODE_VERSION = "неизвестна"

st.set_page_config(page_title="Токены под наблюдением", layout="wide")
st.title("Токены под наблюдением")
st.caption(f"Источник данных: {snapshot.source_label()} · версия {CODE_VERSION}")

if not snapshot.data_available():
    st.warning("Нет данных. Как их получить — на странице «Сбор данных».")
    st.stop()

with st.sidebar:
    st.header("Параметры")
    window_h = st.slider("Окно, часов", 1.0, 48.0, 6.0, step=1.0,
                         help="За какой период смотреть наблюдения. "
                              "Токен считается собираемым, если в этом окне "
                              "по нему есть хотя бы одна котировка.")
    kinds = st.multiselect("Типы площадок", ["dex", "cex"], default=["dex"],
                           format_func=lambda k: "DEX" if k == "dex" else "Биржи (CEX)")
    only_watch = st.toggle("Только из списка наблюдения", value=False)
    refresh = st.button("Обновить", type="primary", **FULL)


@st.cache_data(ttl=120, show_spinner=False)
def load(window_h: float, kinds: tuple, _bust: int):
    since = int(time.time() - window_h * 3600)
    quotes = snapshot.read_quotes(since_ts=since, venue_kinds=list(kinds))
    try:
        pools = snapshot.pools(SETTINGS.chain)
    except Exception:
        pools = None

    notes = {}
    try:
        from history.quality import screen_pools
        from history.rates import _denullify
        notes = dict(screen_pools(_denullify(pools)).notes)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Проверка качества не отработала: {exc}")

    inv = inventory.build(SETTINGS.chain, quotes, pools, notes=notes,
                          watchlist=load_watchlist(SETTINGS),
                          window_h=window_h)
    return inv


if "bust_tokens" not in st.session_state:
    st.session_state.bust_tokens = 0
if refresh:
    st.session_state.bust_tokens += 1

with st.spinner("Собираю инвентарь…"):
    inv = load(window_h, tuple(kinds), st.session_state.bust_tokens)

if inv.tokens.empty:
    st.info("В выбранном окне нет наблюдений. Расширьте окно или проверьте сбор.")
    st.stop()

s = inventory.summary(inv)
c = st.columns(5)
c[0].metric("Токенов в данных", s["токенов"])
c[1].metric("В расчёте", s["в расчёте"])
c[2].metric("Отсеяно", s["отсеяно"])
c[3].metric("Из списка наблюдения", s["из списка наблюдения"])
c[4].metric("Списка не найдено", s["не найдено из списка"],
            help="Тикеры из watchlist.txt, по которым не пришло ни одного пула")

# --------------------------------------------------------------------------
# Проверка конкретного тикера
# --------------------------------------------------------------------------

st.subheader("Проверить тикер")
q = st.text_input("Тикер", placeholder="например AAVE",
                  label_visibility="collapsed").strip().upper()
if q:
    v = inv.verdict(q)
    icon = {"проверяется": st.success, "отсеян": st.warning,
            "нет в данных": st.error, "мало пар": st.warning,
            "устарел": st.warning}.get(v["status"], st.info)
    icon(f"**{v['status'].capitalize()}.** {v['text']}")
    if v["advice"]:
        st.caption(v["advice"])
    row = inv.row(q)
    if row:
        st.dataframe(pd.DataFrame([row]), hide_index=True, **FULL)

# --------------------------------------------------------------------------
# Полная таблица
# --------------------------------------------------------------------------

st.subheader("Все токены")
view = inv.tokens
if only_watch:
    view = view[view["В списке"]]

st.dataframe(
    view, hide_index=True, **FULL,
    column_config={
        "Тикер": st.column_config.TextColumn("Тикер", width="small"),
        "Имя": st.column_config.TextColumn("Имя", width="medium"),
        "В расчёте": st.column_config.CheckboxColumn(
            "В расчёте", width="small",
            help="Токен участвует в поиске связок. Снят — значит отсеян "
                 "как недостоверный, причина ниже."),
        "В списке": st.column_config.CheckboxColumn(
            "Список", width="small",
            help="Токен указан в watchlist.txt и собирается принудительно"),
        "Пар": st.column_config.NumberColumn(
            "Пар", width="small", format="%d",
            help="С каким числом других активов торгуется. "
                 "Меньше двух — цикл через токен не замкнётся."),
        "Площадок": st.column_config.NumberColumn("Площ.", width="small", format="%d"),
        "Пулов": st.column_config.NumberColumn("Пулов", width="small", format="%d"),
        "Ликвидность": st.column_config.NumberColumn(
            "Ликвидн. $", width="small", format="%.0f",
            help="Сумма резервов всех пулов токена"),
        "Оборот": st.column_config.NumberColumn(
            "Оборот $", width="small", format="%.0f",
            help="Суммарный оборот пулов за сутки"),
        "Свежесть, мин": st.column_config.NumberColumn(
            "Свежесть", width="small", format="%.0f",
            help="Сколько минут назад получена последняя котировка"),
        "Торгуется с": st.column_config.TextColumn("Торгуется с", width="medium"),
        "Площадки": st.column_config.TextColumn("Где", width="medium"),
        "Наблюдений": st.column_config.NumberColumn("Точек", width="small", format="%d"),
    },
)
st.download_button("Скачать список CSV",
                   inv.tokens.to_csv(index=False).encode("utf-8-sig"),
                   file_name="токены.csv", mime="text/csv")

if inv.notes:
    with st.expander(f"Отсеяно как недостоверное — тикеров: {len(inv.notes)}"):
        st.dataframe(
            pd.DataFrame({"Тикер": list(inv.notes),
                          "Причина": list(inv.notes.values())}),
            hide_index=True, **FULL)

# --------------------------------------------------------------------------
# Список наблюдения
# --------------------------------------------------------------------------

st.subheader("Список принудительного наблюдения")

found = set(inv.tokens.loc[inv.tokens["Пар"] > 0, "Тикер"])
if inv.watchlist:
    st.dataframe(
        pd.DataFrame([{
            "Тикер": w,
            "Собирается": w in found,
            "Пулов": int(inv.tokens.loc[inv.tokens["Тикер"] == w, "Пулов"].sum()),
        } for w in inv.watchlist]),
        hide_index=True, **FULL,
        column_config={
            "Тикер": st.column_config.TextColumn("Тикер", width="small"),
            "Собирается": st.column_config.CheckboxColumn(
                "Собирается", width="small",
                help="По тикеру пришёл хотя бы один пул"),
            "Пулов": st.column_config.NumberColumn("Пулов", width="small", format="%d"),
        })
else:
    st.info("Список наблюдения пуст.")

st.markdown(
    f"""
**Куда вписывать токены.** Файл `watchlist.txt` в корне проекта —
один тикер в строке, после `#` комментарий. Адрес контракта не нужен,
поиск идёт по имени в сети `{SETTINGS.chain}`.

```
AAVE
DAI
XVS     # Venus
```

После правки — `git add -A`, `git commit`, `git push`. Новый список
подхватывается следующим прогоном сбора, то есть в течение 15 минут;
здесь тикер появится с галочкой «Собирается».

Второй способ, если менять файл неудобно, — переменная
`ARB_WATCH_TOKENS` в `.github/workflows/collect.yml`, тикеры через
запятую. Оба источника складываются, а не заменяют друг друга.
"""
)
st.caption(f"Файл читается по пути: {WATCHLIST_FILE}")
