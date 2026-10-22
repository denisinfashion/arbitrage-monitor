"""Бумажная торговля на лаге: что заработал бы бот на сигналах ведущего.

Без денег: сигнал — сдвиг стакана ведущего, вход и выход по стакану
ведомого на заданный объём, комиссии тейкера. Итоги — по всем сигналам
и только по подтверждённым реальной сделкой ведомого.
"""

from __future__ import annotations

import sys
import time
from contextlib import closing
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history.ui import FULL
from leadlag import paper

st.set_page_config(page_title="Бумажная торговля", layout="wide")
st.title("Бумажная торговля на лаге")
st.caption(f"Сигнал: середина стакана ведущего сдвинулась на ≥ {paper.MIN_MOVE_PCT:g}% за "
           f"{paper.WINDOW_MS} мс. Вход через {paper.REACT_MS} мс по стакану ведомого на "
           f"${paper.NOTIONAL_USD:,.0f}, выход через 1/2/5 с, минус 2 комиссии тейкера. "
           "Время — момент получения котировок сервером.")

if not paper.PAPER_DB.exists():
    st.info("Сервис ещё не запускался: `python -m leadlag.paper`.")
    st.stop()

with st.sidebar:
    days = st.slider("Период, дней", 1, 14, 3)
    min_move = st.number_input("Сдвиг ведущего, % от", 0.15, 5.0, 0.15, 0.05,
                               help="Отбор сигналов по силе движения ведущего.")
    hold = st.radio("Выход через", [1000, 2000, 5000], index=1, format_func=lambda x: f"{x // 1000} с")
    only_conf = st.toggle("Только подтверждённые сделкой", value=False,
                          help="У ведомого в течение 0,5 с после входа была реальная сделка по цене "
                               "входа или лучше. Защищает от «старого» стакана, но у редко торгуемых "
                               "монет отсекает большинство сигналов.")


@st.cache_data(ttl=60, show_spinner=False)
def load(since_ms: int) -> pd.DataFrame:
    with closing(paper.connect()) as con:
        return pd.read_sql_query("SELECT * FROM paper_trades WHERE ts >= ?", con, params=(since_ms,))


d = load(int((time.time() - days * 86400) * 1000))
if d.empty:
    st.info("Сигналов за период нет.")
    st.stop()
d = d[(d["move_pct"].abs() >= min_move) & (d["hold_ms"] == hold)]
if only_conf:
    d = d[d["confirmed"] == 1]
if d.empty:
    st.info("Под фильтры сигналов нет.")
    st.stop()

c = st.columns(5)
c[0].metric("Сигналов", len(d))
c[1].metric("Средний итог, %", f"{d['pnl_pct'].mean():+.3f}")
c[2].metric("Медиана, %", f"{d['pnl_pct'].median():+.3f}")
c[3].metric("В плюс", f"{(d['pnl_pct'] > 0).mean():.0%}")
c[4].metric(f"Сумма на ${paper.NOTIONAL_USD:,.0f}", f"${d['pnl_pct'].sum() / 100 * paper.NOTIONAL_USD:+,.0f}")

d = d.assign(time=pd.to_datetime(d["ts"], unit="ms", utc=True).dt.tz_convert("Europe/Moscow"))
cum = d.sort_values("time")
fig = go.Figure(go.Scatter(x=cum["time"], y=(cum["pnl_pct"] / 100 * paper.NOTIONAL_USD).cumsum(),
                           mode="lines", name="накопленный итог, $"))
fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="$")
st.plotly_chart(fig, **FULL)

st.subheader("По парам")
g = d.groupby(["leader", "follower"]).agg(
    signals=("pnl_pct", "size"), avg=("pnl_pct", "mean"), median=("pnl_pct", "median"),
    win=("pnl_pct", lambda x: (x > 0).mean()), confirmed=("confirmed", "mean"),
    book_age_ms=("book_age_ms", "median"), already=("follower_pre_pct", "mean"),
).reset_index().sort_values("avg", ascending=False)
st.dataframe(g.rename(columns={
    "leader": "Ведущий", "follower": "Ведомый", "signals": "Сигналов", "avg": "Средний, %",
    "median": "Медиана, %", "win": "В плюс", "confirmed": "Подтверждено",
    "book_age_ms": "Возраст стакана, мс", "already": "Ведомый уже сдвинулся, %",
}).round(3), hide_index=True, **FULL)
st.caption("«Возраст стакана» — сколько миллисекунд назад пришло последнее обновление стакана ведомого "
           "к моменту входа. Большой возраст — стакан мог устареть, итог завышен. "
           "«Ведомый уже сдвинулся» — его движение за то же окно до сигнала, в сторону сигнала.")
