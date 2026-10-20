"""Зависимости: какой инструмент ведёт, какой следует и с каким лагом.

Пары внутри одного токена: фьючерс ↔ DEX, фьючерс ↔ фьючерс на разных
биржах. Оценка по часам за сутки — связка считается рабочей, если
направление и лаг держатся много часов подряд, а не мелькнули один раз.
"""

from __future__ import annotations

import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history.ui import FULL
from leadlag import lag, ticks
from leadlag.catalog import CHAINS
from leadlag.settings import ScreenSettings

st.set_page_config(page_title="Зависимости", layout="wide")
st.title("Зависимости: кто ведёт, кто следует")
st.caption("Корреляция посекундных изменений цены со сдвигом ±60 с, по каждому часу. "
           "Время — биржевое у фьючерсов и время блока у DEX.")

s = ScreenSettings.load()

if not ticks.TICKS_DB.exists():
    st.info("Сбор сделок ещё не запускался: `python -m leadlag.ticks`.")
    st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def instruments() -> pd.DataFrame:
    with closing(ticks.connect()) as con:
        return pd.read_sql_query("SELECT * FROM instruments", con)


@st.cache_data(ttl=60, show_spinner=False)
def coverage() -> pd.DataFrame:
    since = int(time.time()) - 3600
    with closing(ticks.connect()) as con:
        return pd.read_sql_query(
            "SELECT i.kind, i.venue, count(DISTINCT p.inst) AS active, sum(p.n) AS trades, "
            "max(p.ts) AS last_ts FROM prices p JOIN instruments i ON i.inst = p.inst "
            "WHERE p.ts >= ? GROUP BY i.kind, i.venue", con, params=(since,))


inst = instruments().set_index("inst")


def label(i: str) -> str:
    if i not in inst.index:
        return i
    r = inst.loc[i]
    if r["kind"] == "dex":
        chain = CHAINS[r["venue"]].label if r["venue"] in CHAINS else r["venue"]
        return f"{r['coin']} · DEX {chain}"
    return f"{r['coin']} · фьючерс {str(r['venue']).capitalize()}"


with st.expander("Сбор данных за последний час"):
    cov = coverage()
    if cov.empty:
        st.write("За последний час данных нет.")
    else:
        total = inst.groupby(["kind", "venue"]).size().rename("всего")
        cov = cov.set_index(["kind", "venue"]).join(total).reset_index()
        cov["отставание, с"] = (time.time() - cov["last_ts"]).round(0)
        st.dataframe(cov.rename(columns={"kind": "тип", "venue": "площадка", "active": "со сделками",
                                         "trades": "сделок"}).drop(columns="last_ts"),
                     hide_index=True, **FULL)

with st.sidebar:
    st.header("Фильтры")
    kinds = st.multiselect("Пары", ["dex-fut", "fut-fut", "dex-dex"], default=["dex-fut", "fut-fut"],
                           format_func={"dex-fut": "фьючерс ↔ DEX", "fut-fut": "фьючерс ↔ фьючерс",
                                        "dex-dex": "DEX ↔ DEX"}.get)
    min_stable = st.slider("Устойчивых часов, не меньше", 0, 24, 3,
                           help="Часов, где направление то же и лаг отличается от медианного "
                                "не больше чем на 2 с (или 30%).")
    min_lag = st.slider("Лаг, секунд, от", 0, 30, 1)
    hide_sync = st.toggle("Скрыть синхронные (лаг 0)", value=True)
    if st.button("Пересчитать сейчас", **FULL):
        with st.spinner("Считаю закрытые часы…"):
            lag.run_once(s)
        st.cache_data.clear()

res = lag.load_results()
if res is None or res.empty:
    st.info("Итогов пока нет: нужен хотя бы один закрытый час данных. "
            "Анализ запускается раз в 30 минут или кнопкой «Пересчитать сейчас».")
    st.stop()

view = res[res["pair_kind"].isin(kinds)]
view = view[view["hours_stable"].fillna(0) >= min_stable]
view = view[view["lag_s"].fillna(0) >= min_lag]
if hide_sync:
    view = view[~view["synchronous"]]
view = view.sort_values(["hours_stable", "corr"], ascending=False)

c = st.columns(4)
c[0].metric("Пар с оценкой", len(res))
c[1].metric("Прошли фильтры", len(view))
c[2].metric("Монет", view["coin"].nunique())
c[3].metric("Окно, часов", s.lag_window_hours)

table = pd.DataFrame({
    "Монета": view["coin"],
    "Ведёт": view["leader"].map(label),
    "Ссылка ведущего": view["leader_url"],
    "Следует": view["follower"].map(label),
    "Ссылка ведомого": view["follower_url"],
    "Лаг, с": view["lag_s"],
    "Корреляция на лаге": view["corr"].round(2),
    "Корреляция без сдвига": view["corr0"].round(2),
    "Устойчивых ч": view["hours_stable"],
    "Значимых ч": view["hours_sig"],
    "Часов с данными": view["hours_data"],
    "Последний час: лаг": view["last_lag"],
    "Последний час: корр.": view["last_corr"].round(2),
})
st.dataframe(table, hide_index=True, height=520, **FULL, column_config={
    "Ссылка ведущего": st.column_config.LinkColumn(display_text="открыть"),
    "Ссылка ведомого": st.column_config.LinkColumn(display_text="открыть"),
})
st.caption("Лаг у редко торгуемого ведомого (пул DEX) включает ожидание следующей сделки — "
           "это время, через которое изменение реально появляется в его цене.")

if view.empty:
    st.stop()

st.subheader("Разбор пары")
options = list(zip(view["leader"], view["follower"]))
pick = st.selectbox("Пара", options, format_func=lambda p: f"{label(p[0])}  →  {label(p[1])}")
a, b = sorted(pick)
sign = 1 if pick[0] == a else -1

with closing(ticks.connect()) as con:
    con.executescript(lag.SCHEMA)
    hours = pd.read_sql_query("SELECT * FROM lag_hours WHERE a=? AND b=? ORDER BY hour", con, params=(a, b))
    last_hour = int(time.time()) // 3600 * 3600 - 3600
    ga, _ = lag.price_grid(con, pick[0], last_hour, last_hour + 3600)
    gb, _ = lag.price_grid(con, pick[1], last_hour, last_hour + 3600)

left, right = st.columns(2)
with left:
    curve = lag.xcorr(lag.returns(ga), lag.returns(gb), s.max_lag_s)
    taus = np.arange(-s.max_lag_s, s.max_lag_s + 1)
    fig = go.Figure(go.Bar(x=taus, y=curve))
    fig.update_layout(title="Корреляция по сдвигу за последний закрытый час",
                      xaxis_title="сдвиг, с (плюс — ведущий раньше)", yaxis_title="корреляция",
                      height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, **FULL)
with right:
    h = hours.dropna(subset=["corr"]).copy()
    h["time"] = pd.to_datetime(h["hour"], unit="s", utc=True).dt.tz_convert("Europe/Moscow")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["time"], y=h["lag"] * sign, mode="markers+lines", name="лаг, с"))
    fig.add_trace(go.Scatter(x=h["time"], y=h["corr"], mode="lines", name="корреляция", yaxis="y2"))
    fig.update_layout(title="Лаг и сила связи по часам", height=360,
                      yaxis=dict(title="лаг, с"), yaxis2=dict(title="корр.", overlaying="y", side="right"),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, **FULL)
