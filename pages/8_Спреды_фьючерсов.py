"""Спреды одного бессрочного фьючерса между биржами — с учётом funding.

Лонг там, где дешевле, шорт там, где дороже. Итог считается за выбранное
время удержания: спред − комиссии + funding по фактическому расписанию
расчётов каждой биржи.
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
from leadlag import spreads
from leadlag.catalog import FUTURES, FUTURES_TAKER_PCT
from leadlag.settings import ScreenSettings

st.set_page_config(page_title="Спреды фьючерсов", layout="wide")
st.title("Спреды фьючерсов между биржами")
st.caption("Лонг по ask там, где дешевле, шорт по bid там, где дороже. "
           "Итог = спред − комиссии (4 сделки тейкером) + funding за время удержания. "
           "Ставка funding берётся текущая — она меняется на каждом расчёте.")

s = ScreenSettings.load()
defaults = ScreenSettings()

m = spreads.load_snapshot()
if m is None or m.empty:
    st.info("Снимков ещё нет: запустите `python -m leadlag.spreads`.")
    st.stop()

with st.sidebar:
    st.header("Связка")
    s.spread_hold_hours = st.slider("Время удержания, ч", 1.0, 72.0, float(s.spread_hold_hours), 1.0,
                                    help="Сколько расчётов funding попадёт в позицию — по расписанию "
                                         f"каждой биржи. По умолчанию {defaults.spread_hold_hours:g} ч.")
    s.spread_min_pnl_pct = st.number_input(
        "Итог за удержание, % от", -5.0, 50.0, float(s.spread_min_pnl_pct), 0.05,
        help="Лучший из двух итогов: «спред сошёлся» или «только funding». "
             f"По умолчанию {defaults.spread_min_pnl_pct:g}%.")
    s.spread_min_volume_usd = st.number_input(
        "Объём каждой ноги за 24 ч, $ от", 0.0, value=float(s.spread_min_volume_usd), step=250_000.0,
        help=f"По умолчанию ${defaults.spread_min_volume_usd:,.0f}.")
    s.spread_max_price_ratio = st.number_input(
        "Отношение цен до", 1.0, 10.0, float(s.spread_max_price_ratio), 0.05,
        help="Больше — почти наверняка разные токены под одним тикером. "
             f"По умолчанию {defaults.spread_max_price_ratio:g}.")
    s.futures_exchanges = st.multiselect("Биржи", list(FUTURES), default=s.futures_exchanges,
                                         format_func=str.capitalize)
    mode = st.radio("Что ищем", ["любой доход", "спред (доход при схождении)", "funding (доход без схождения)"])
    st.caption("Комиссии тейкера, %: " + ", ".join(f"{k} {v:g}" for k, v in FUTURES_TAKER_PCT.items()))

if s.__dict__ != ScreenSettings.load().__dict__:
    s.save()

snap_ts = int(m["snap_ts"].iloc[0]) if "snap_ts" in m else None
if snap_ts:
    st.write(f"Снимок цен {time.strftime('%H:%M:%S', time.localtime(snap_ts))} "
             f"({time.time() - snap_ts:.0f} с назад): {len(m)} рынков, "
             f"{int((m.groupby('coin')['venue'].nunique() > 1).sum())} монет на двух и более биржах.")

p = spreads.pairs(m, s.spread_hold_hours)
q = spreads.select(p, s)
if mode.startswith("спред"):
    q = q[q["pnl_converge_pct"] >= s.spread_min_pnl_pct].sort_values("pnl_converge_pct", ascending=False)
elif mode.startswith("funding"):
    q = q[q["pnl_carry_pct"] >= s.spread_min_pnl_pct].sort_values("pnl_carry_pct", ascending=False)

with closing(spreads.hist_connect()) as con:
    hot = spreads.persistence(con, 60)
q = q.merge(hot, on=["coin", "venue_l", "venue_s"], how="left")

c = st.columns(4)
c[0].metric("Связок", len(q))
c[1].metric("Монет", q["coin"].nunique())
c[2].metric("Лучший итог при схождении, %", f"{q['pnl_converge_pct'].max():.2f}" if len(q) else "—")
c[3].metric("Лучший итог только funding, %", f"{q['pnl_carry_pct'].max():.2f}" if len(q) else "—")


def next_in(ts_ms) -> str:
    if ts_ms is None or pd.isna(ts_ms):
        return "—"
    mins = (ts_ms / 1000 - time.time()) / 60
    return f"{mins:.0f} мин" if mins < 120 else f"{mins / 60:.1f} ч"


def rate(r, side: str) -> str:
    v, iv = r[f"f_rate_{side}"], r[f"f_interval_h_{side}"]
    if pd.isna(v):
        return "—"
    return f"{v * 100:+.4f}% / {iv:g} ч"


table = pd.DataFrame({
    "Монета": q["coin"],
    "Лонг": q["venue_l"].str.capitalize(),
    "Ссылка лонга": q["url_l"],
    "Шорт": q["venue_s"].str.capitalize(),
    "Ссылка шорта": q["url_s"],
    "Спред, %": q["spread_pct"].round(3),
    "Комиссии, %": q["fees_pct"].round(3),
    f"Funding за {s.spread_hold_hours:g} ч, %": q["funding_hold_pct"].round(3),
    "Funding в сутки, %": q["funding_day_pct"].round(3),
    "Итог: спред сошёлся, %": q["pnl_converge_pct"].round(3),
    "Итог: только funding, %": q["pnl_carry_pct"].round(3),
    "Ставка лонга": [rate(r, "l") for _, r in q.iterrows()],
    "Расчёт лонга через": q["f_next_ts_l"].map(next_in),
    "Ставка шорта": [rate(r, "s") for _, r in q.iterrows()],
    "Расчёт шорта через": q["f_next_ts_s"].map(next_in),
    "Объём меньшей ноги, $": q["min_volume_usd"].round(0),
    "Минут в спреде из 60": q["minutes_hot"].fillna(0).astype(int),
    "Возможные причины": q["reasons"],
})
st.dataframe(table, hide_index=True, height=560, **FULL, column_config={
    "Ссылка лонга": st.column_config.LinkColumn(display_text="открыть"),
    "Ссылка шорта": st.column_config.LinkColumn(display_text="открыть"),
    "Объём меньшей ноги, $": st.column_config.NumberColumn(format="%,.0f"),
})
st.caption("«Минут в спреде из 60» — сколько минутных снимков за последний час спред этой пары "
           f"был не меньше {spreads.HIST_MIN_SPREAD_PCT:g}%. Спред, который держится, — "
           "кандидат в работу; мелькнувший — скорее шум котировок.")

if q.empty:
    st.stop()

st.subheader("История спреда пары")
opts = list(zip(q["coin"], q["venue_l"], q["venue_s"]))
coin, vl, vs = st.selectbox("Пара", opts, format_func=lambda t: f"{t[0]}: лонг {t[1]} / шорт {t[2]}")
with closing(spreads.hist_connect()) as con:
    h = pd.read_sql_query("SELECT ts, spread_pct, funding_day_pct FROM spread_hist "
                          "WHERE coin=? AND long=? AND short=? AND ts >= ? ORDER BY ts", con,
                          params=(coin, vl, vs, int(time.time()) - 86400))
if h.empty:
    st.write(f"За сутки спред этой пары не поднимался до {spreads.HIST_MIN_SPREAD_PCT:g}%.")
else:
    h["time"] = pd.to_datetime(h["ts"], unit="s", utc=True).dt.tz_convert("Europe/Moscow")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["time"], y=h["spread_pct"], mode="markers+lines", name="спред, %"))
    fig.add_trace(go.Scatter(x=h["time"], y=h["funding_day_pct"], mode="lines", name="funding в сутки, %"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"),
                      yaxis_title="%")
    st.plotly_chart(fig, **FULL)
    st.caption(f"Точки — минуты, когда спред был не меньше {spreads.HIST_MIN_SPREAD_PCT:g}%; "
               "разрывы — спред был ниже.")
