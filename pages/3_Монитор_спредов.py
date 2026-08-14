"""Монитор спредов: постоянно смотрит котировки и подсказывает,
где разница между площадками превышает издержки.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history import monitor
from history.config import SETTINGS
from history.ui import FULL

st.set_page_config(page_title="Монитор спредов", layout="wide")
st.title("Монитор спредов")

st.caption(
    "Опрашивает площадки и показывает, где разница цен больше суммарных комиссий. "
    "Это ориентир по цене последней сделки, а не проверка исполнимости: "
    "объём под связку смотрите в основном калькуляторе по реальному стакану."
)

# --------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Что отслеживать")

    use_cex = st.toggle("Спред между крипто-биржами", value=True)
    use_stable = st.toggle("Депег стейблкоинов", value=True)
    use_basis = st.toggle("Базис спот ↔ перп", value=True)
    use_dexcex = st.toggle("DEX против CEX", value=True,
                           help=f"Сеть {SETTINGS.chain.upper()}, котировка прямо из пула")
    use_other = st.toggle("Акции и металлы", value=False,
                          help="Бесплатных биржевых данных по ним нет — "
                               "адаптеры вернут пояснение вместо сигналов")

    st.divider()
    venues = st.multiselect("Биржи", SETTINGS.cex_venues + ["kraken", "bitget", "htx"],
                            default=SETTINGS.cex_venues[:4])
    quote = st.selectbox("Котируемая валюта", ["USDT", "USDC", "BTC"], index=0)
    spot_only = st.toggle("Только спотовые активы", value=SETTINGS.spot_only,
                          help="Отсекает токены с плечом (BTC3L, ETHUP и подобные): "
                               "их спред неисполним из-за перебалансировки NAV")
    min_net = st.slider("Порог чистого спреда, б.п.", -50, 300, 10, step=5,
                        help="10 б.п. = 0.1%")

    st.divider()
    auto = st.toggle("Автообновление", value=False)
    period = st.slider("Период обновления, сек", 15, 300, 60, step=15,
                       disabled=not auto)

    scan = st.button("Обновить сейчас", type="primary", **FULL)


# --------------------------------------------------------------------------
# Опрос
# --------------------------------------------------------------------------


def build_adapters():
    out = []
    if use_cex:
        try:
            out.append(monitor.CexSpread(venues=venues, quote=quote,
                                         spot_only=spot_only))
        except Exception as exc:
            st.warning(f"Спред между биржами недоступен: {exc}")
    if use_stable:
        try:
            out.append(monitor.StablecoinDepeg(venues=venues))
        except Exception as exc:
            st.warning(f"Депег стейблов недоступен: {exc}")
    if use_basis:
        try:
            out.append(monitor.PerpBasis(venue=venues[0] if venues else "binance"))
        except Exception as exc:
            st.warning(f"Базис недоступен: {exc}")
    if use_dexcex:
        try:
            out.append(monitor.DexCexSpread(cex=venues[0] if venues else "binance"))
        except Exception as exc:
            st.warning(f"DEX-адаптер недоступен: {exc}")
    if use_other:
        out += [monitor.equities_adapter(), monitor.metals_adapter()]
    return out


@st.cache_data(ttl=15, show_spinner=False)
def run_scan(_adapters, min_net: float, _bust: int) -> pd.DataFrame:
    return monitor.scan_all(_adapters, min_net_bps=min_net)


if "bust_mon" not in st.session_state:
    st.session_state.bust_mon = 0
if scan or auto:
    st.session_state.bust_mon += 1

adapters = build_adapters()
if not adapters:
    st.info("Включите хотя бы один источник в боковой панели.")
    st.stop()

with st.spinner("Опрашиваю площадки…"):
    df = run_scan(adapters, float(min_net), st.session_state.bust_mon)

st.caption(f"Обновлено: {time.strftime('%H:%M:%S')} · источников: {len(adapters)}")

if df.empty:
    st.info(
        f"Спредов выше {min_net} б.п. не найдено. Это обычная ситуация на "
        "ликвидных парах — рынок эффективен большую часть времени. "
        "Снизьте порог, чтобы увидеть картину целиком."
    )
    st.stop()

# --------------------------------------------------------------------------
# Результат
# --------------------------------------------------------------------------

real = df[df["Исполнимо"] == "да"]
info = df[df["Исполнимо"] != "да"]

m = st.columns(4)
m[0].metric("Сигналов", len(real))
if not real.empty:
    m[1].metric("Лучший спред", f"{real['Чистый спред, %'].max():.3f} %")
    m[2].metric("Медиана", f"{real['Чистый спред, %'].median():.3f} %")
    m[3].metric("Классов", real["Класс"].nunique())

if not real.empty:
    st.subheader("Найденные спреды")
    st.dataframe(
        real.style.background_gradient(subset=["Чистый спред, б.п."], cmap="RdYlGn")
                  .format({"Цена покупки": "{:.8g}", "Цена продажи": "{:.8g}"}),
        **FULL, hide_index=True, height=440,
    )
    st.download_button("Скачать CSV", real.to_csv(index=False).encode("utf-8-sig"),
                       file_name="монитор_спредов.csv", mime="text/csv")

    st.subheader("По классам инструментов")
    agg = real.groupby("Класс").agg(
        Сигналов=("Инструмент", "count"),
        Лучший=("Чистый спред, %", "max"),
        Медиана=("Чистый спред, %", "median"),
    ).round(3).reset_index()
    st.dataframe(agg, **FULL, hide_index=True)

if not info.empty:
    st.subheader("Индикативные источники")
    for _, row in info.iterrows():
        st.info(f"**{row['Класс'].capitalize()}** — {row['Примечание']}")

st.caption(
    "Комиссии учтены тейкерские, обе ноги. Не учтены: комиссия вывода в сеть, "
    "время перевода между площадками и движение цены за это время, лимиты и "
    "заморозки вывода. Для межбиржевых связок это существенно — прогоняйте "
    "найденное через основной калькулятор."
)

if auto:
    time.sleep(period)
    st.rerun()
