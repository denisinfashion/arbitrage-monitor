"""Поиск арбитражных связок по истории: таблица и график.

Таблица — связки USDT → … → USDT, отсортированные по марже.
График — произведение курсов вдоль выбранной связки во времени,
уже за вычетом комиссий, проскальзывания и газа.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history import snapshot, store
from history.config import SETTINGS
from history.ui import FULL
from history.paths import find_cycles
from history.rates import build_grid

st.set_page_config(page_title="Связки по истории", layout="wide")
st.title("Арбитражные связки по истории")
st.caption(f"Источник данных: {snapshot.source_label()}")

if not snapshot.data_available():
    st.warning("Нет данных. Как их получить — на странице «Сбор данных».")
    st.stop()

stats = snapshot.stats()

# Бесплатный тариф Streamlit Cloud даёт около 1 ГБ памяти. Сетка курсов
# занимает T x N x N ячеек по 4 байта: 60 активов на 10 тысяч точек — это
# уже 145 МБ только под неё, плюс исходный DataFrame и накладные расходы
# pandas. Поэтому в облаке потолки ниже, а по умолчанию берётся окно поуже.
CLOUD = snapshot.cloud_mode()
LIM_ASSETS = 40 if CLOUD else 120
DEF_ASSETS = 30 if CLOUD else 60
DEF_WINDOW = 12.0 if CLOUD else 24.0

# --------------------------------------------------------------------------
# Параметры
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Параметры поиска")

    max_hours = max(1.0, (stats["t1"] - stats["t0"]) / 3600) if stats["t0"] else 24.0
    window_h = st.slider("Окно анализа, часов", 1.0, float(round(max_hours, 1)),
                         float(min(DEF_WINDOW, round(max_hours, 1))), step=1.0,
                         help="Сколько истории от текущего момента назад анализировать")

    anchor = st.text_input("Стартовый и конечный актив", SETTINGS.quote_asset).upper()

    max_legs = st.slider("Максимум ног в связке", 2, 5, SETTINGS.max_legs,
                         help="Каждая нога добавляет комиссию и газ. "
                              "Длинные циклы почти всегда убыточны, но проверить можно.")

    trade_size = st.number_input("Размер сделки, $", 100.0, 1_000_000.0,
                                 SETTINGS.trade_size_usd, step=100.0,
                                 help="Под этот объём считается проскальзывание")

    granularity = st.selectbox(
        "Гранулярность анализа", ["1m", "5m", "15m", "1h"], index=2,
        help="Шаг сетки времени. Биржи отдают минутные данные, а пулы DEX "
             "на бесплатной инфраструктуре обновляются раз в несколько минут — "
             "на мелком шаге строки DEX будут считаться протухшими и связки "
             "через них не найдутся.")

    st.divider()
    kinds = st.multiselect("Типы площадок", ["dex", "cex"],
                           default=["dex"],
                           format_func=lambda k: "DEX" if k == "dex" else "Биржи (CEX)",
                           help="По умолчанию DEX — обмен в одной сети без переводов")

    spot_only = st.toggle("Только спотовые активы", value=SETTINGS.spot_only,
                          help="Отсекает токены с плечом: BTC3L, ETHUP, BTCBULL "
                               "и подобные. Биржи размещают их на спотовом рынке, "
                               "но внутри у них ежедневная перебалансировка — "
                               "цена следует за NAV, а не за базовым активом, "
                               "и разница между площадками неисполнима.")

    one_chain = st.toggle("Только без переводов", value=True,
                          help="Оставить связки, где все обмены идут в одной "
                               "сети на DEX — это последовательность свопов "
                               "из кошелька. Как только появляется биржа или "
                               "вторая сеть, между ногами нужен перевод: "
                               "комиссия сети, время, иногда заморозка вывода. "
                               "Расчёт этого не учитывает.")

    apply_slip = st.toggle("Учитывать проскальзывание", value=True,
                           help="Для DEX — точная формула по резерву пула. "
                                "Для CEX — оценка по обороту свечи.")

    staleness = st.slider("Допустимый возраст котировки, сек", 60, 3600,
                          1200, step=60,
                          help="Главный предохранитель от ложных связок: "
                               "цена старше этого порога в расчёт не идёт")

    gas_leg = st.number_input("Газ за своп на DEX, $", 0.0, 50.0, 0.15, step=0.05)

    st.divider()
    max_assets = st.slider("Активов в анализе", 10, LIM_ASSETS, DEF_ASSETS, step=10,
                           help="Больше активов — больше связок, но и дольше расчёт. "
                                + ("Потолок снижен: в облаке около 1 ГБ памяти."
                                   if CLOUD else ""))
    top_n = st.slider("Строк в таблице", 10, 200, 50, step=10)
    min_margin = st.number_input("Порог маржи, %", -5.0, 10.0, 0.0, step=0.05,
                                 help="Показывать связки, у которых максимум за период выше порога")

    sort_by = st.selectbox(
        "Сортировать по",
        ["окна", "максимум", "медиана", "сейчас", "ликвидность"],
        format_func={
            "окна": "доле прибыльного времени",
            "максимум": "лучшей марже за период",
            "медиана": "типичной марже",
            "сейчас": "марже в последней точке",
            "ликвидность": "ликвидности пулов",
        }.get,
        help="«Доля прибыльного времени» — практичный выбор: связка с редкими, "
             "но реальными окнами полезнее той, что стабильно чуть ниже нуля.",
    )

    st.divider()
    detailed = st.toggle("Показать все колонки", value=False,
                         help="По умолчанию в таблице только то, по чему "
                              "принимается решение. Остальное — сеть, окна, "
                              "расшифровка тикеров — прячется, чтобы таблица "
                              "влезала в экран без прокрутки вбок.")

    run = st.button("Найти связки", type="primary", **FULL)


# --------------------------------------------------------------------------
# Расчёт
# --------------------------------------------------------------------------


@st.cache_data(ttl=120, show_spinner=False)
def compute(window_h: float, anchor: str, max_legs: int, trade_size: float,
            kinds: tuple, apply_slip: bool, staleness: int, gas_leg: float,
            max_assets: int, top_n: int, min_margin: float, sort_by: str,
            spot_only: bool, granularity: str, one_chain: bool, _bust: int):
    since = int(time.time() - window_h * 3600)
    quotes = snapshot.read_quotes(since_ts=since, venue_kinds=list(kinds))
    if quotes.empty:
        return None, None, None, "В выбранном окне нет котировок."

    s = SETTINGS
    old_stale, old_tf = s.staleness_sec, s.analysis_timeframe
    s.staleness_sec = staleness
    s.analysis_timeframe = granularity
    try:
        grid = build_grid(quotes, settings=s, trade_size_usd=trade_size,
                          venue_kinds=list(kinds), max_assets=max_assets,
                          apply_slippage=apply_slip, spot_only=spot_only)
        table, cycles = find_cycles(grid, anchor=anchor, max_legs=max_legs,
                                    top=top_n, gas_per_dex_leg_usd=gas_leg,
                                    min_margin_pct=min_margin,
                                    sort_by=sort_by, settings=s)
        if one_chain and cycles:
            keep = [i for i, c in enumerate(cycles) if not c.needs_transfer()]
            if keep:
                names = {cycles[i].label for i in keep}
                table = table[table["Связка"].isin(names)].reset_index(drop=True)
                cycles = [cycles[i] for i in keep]
    except MemoryError as exc:
        return None, None, None, (
            f"{exc} Уменьшите окно анализа или число активов в боковой панели."
        )
    except (ValueError, KeyError) as exc:
        return None, None, None, str(exc)
    finally:
        s.staleness_sec, s.analysis_timeframe = old_stale, old_tf
    return grid, table, cycles, None


if "bust_paths" not in st.session_state:
    st.session_state.bust_paths = 0
if run:
    st.session_state.bust_paths += 1

with st.spinner("Строю сетку курсов и ищу связки…"):
    grid, table, cycles, err = compute(
        window_h, anchor, max_legs, trade_size, tuple(kinds), apply_slip,
        staleness, gas_leg, max_assets, top_n, min_margin, sort_by, spot_only,
        granularity, one_chain, st.session_state.bust_paths,
    )

if err:
    st.error(err)
    st.stop()

# --------------------------------------------------------------------------
# Сводка по сетке
# --------------------------------------------------------------------------

info = grid.summary()
c = st.columns(5)
c[0].metric("Точек времени", info["точек времени"])
c[1].metric("Активов", info["активов"])
c[2].metric("Площадок", info["площадок"])
c[3].metric("Заполненность сетки", info["заполненность"])
c[4].metric("Связок найдено", len(table) if table is not None else 0)

cov = grid.coverage()
if cov < 0.02:
    st.warning(
        f"Заполненность сетки всего {cov * 100:.1f}%. Пар, торгуемых напрямую "
        "друг к другу, мало — большинство связок не замыкается. Соберите больше "
        "данных или добавьте биржи к DEX."
    )

if table is None or table.empty:
    st.info(
        "Прибыльных связок выше порога не найдено. Это нормальный результат: "
        "на ликвидных парах спред обычно меньше суммарных комиссий. "
        "Попробуйте снизить порог маржи, расширить окно или добавить площадки."
    )
    st.stop()

# --------------------------------------------------------------------------
# Таблица
# --------------------------------------------------------------------------

SORT_LABEL = {
    "окна": "доле прибыльного времени",
    "максимум": "лучшей марже за период",
    "медиана": "типичной марже",
    "сейчас": "марже в последней точке",
    "ликвидность": "ликвидности пулов",
}
st.subheader(f"Связки, отсортированные по {SORT_LABEL.get(sort_by, sort_by)}")
st.caption(
    f"Окно {window_h:.0f} ч · шаг {granularity} · объём ${trade_size:,.0f} · "
    f"газ ${gas_leg:.2f} за своп · возраст котировки до {staleness} с"
    .replace(",", " ")
    + (" · только спот" if spot_only else " · с плечевыми токенами")
    + (" · без переводов" if one_chain else "")
)

display = table.drop(columns=["Точек"], errors="ignore")

# Короткий набор: только то, по чему принимается решение. Прочее — по кнопке.
BRIEF = ["Связка", "Ног", "Сейчас %", "Макс %", "В плюсе %",
         "Окно средн, мин", "Ликвидность $", "Маршрут"]
if not detailed:
    display = display[[c for c in BRIEF if c in display.columns]]

# Заголовки намеренно короткие. Ширина колонки в таблице тянется по самому
# длинному содержимому, а заголовок обычно длиннее значения: «Доля
# прибыльного времени, %» против «9.7». На узком экране из-за одних только
# подписей приходилось мотать таблицу вбок. Расшифровки убраны в подсказки.
COLS = {
    "Связка":         st.column_config.TextColumn("Связка", width="medium",
                        help="Цепочка обменов от стартового актива и обратно"),
    "Ног":            st.column_config.NumberColumn("Ног", width="small", format="%d"),
    "Сейчас %":       st.column_config.NumberColumn("Сейчас", width="small",
                        format="%+.3f", help="Маржа в последней точке истории"),
    "Макс %":         st.column_config.NumberColumn("Макс", width="small",
                        format="%+.3f", help="Лучшая маржа за выбранное окно"),
    "Медиана %":      st.column_config.NumberColumn("Медиана", width="small",
                        format="%+.3f", help="Типичная маржа за окно"),
    "В плюсе %":      st.column_config.NumberColumn("В плюсе", width="small",
                        format="%.1f%%", help="Какую долю времени связка была прибыльна"),
    "Окон":           st.column_config.NumberColumn("Окон", width="small", format="%d",
                        help="Сколько раз возникала возможность"),
    "Окно макс, мин": st.column_config.NumberColumn("Окно макс", width="small",
                        format="%d м",
                        help="Самое длинное окно возможности, минут"),
    "Окно средн, мин": st.column_config.NumberColumn("Окно средн", width="small",
                        format="%d м",
                        help="Типичная длительность окна — столько есть на то, "
                             "чтобы зайти и провернуть обмен"),
    "Ликвидность $":  st.column_config.NumberColumn("Ликвидн.", width="small",
                        format="compact",
                        help="Самый мелкий пул в цепочке — он и ограничивает объём"),
    "Сеть":           st.column_config.TextColumn("Сеть", width="small"),
    "Переводы":       st.column_config.TextColumn("Перев.", width="small",
                        help="Нужен ли вывод между площадками. «нет» — все обмены "
                             "в одной сети из кошелька"),
    "Маршрут":        st.column_config.TextColumn("Маршрут", width="medium",
                        help="Площадки по порядку исполнения обмена"),
    "Данные %":       st.column_config.NumberColumn("Данн.", width="small",
                        format="%.0f%%", help="Покрытие окна данными"),
    "Токены":         st.column_config.TextColumn("Токены", width="medium",
                        help="Расшифровка тикеров маршрута"),
}
# Градиент вешаем на максимум маржи, а не на медиану: медиана почти всегда
# отрицательна, и зелёный на отрицательных числах читается как «хорошо».
# Шкала центрирована на нуле, чтобы цвет означал знак.
_mx = float(display["Макс %"].abs().max() or 1.0)
st.dataframe(
    display.style.background_gradient(subset=["Макс %"], cmap="RdYlGn",
                                      vmin=-_mx, vmax=_mx),
    **FULL, hide_index=True, height=420, column_config=COLS,
)

if (display["Сейчас %"].fillna(-1) <= 0).all():
    st.caption(
        "Ни одна связка не прибыльна прямо сейчас — обычная ситуация. "
        "«Макс» показывает лучший момент окна, «В плюсе» — насколько такие "
        "моменты часты, «Окно средн» — сколько времени обычно есть на сделку."
    )

st.download_button("Скачать таблицу CSV",
                   table.to_csv(index=False).encode("utf-8-sig"),
                   file_name="связки.csv", mime="text/csv")

# --------------------------------------------------------------------------
# График
# --------------------------------------------------------------------------

st.subheader("График связки во времени")

labels = [c.label for c in cycles]
pick = st.selectbox("Связка", range(len(labels)), format_func=lambda i: labels[i])
cyc = cycles[pick]

df = cyc.to_frame()
leg_cols = [c for c in df.columns if c.startswith("Нога ")]

g1, g2 = st.columns([3, 1])
with g2:
    st.markdown("**Обмены по порядку**")
    legs = cyc.leg_links()
    for leg in legs:
        head = f"{leg['n']}. `{leg['from']} → {leg['to']}`"
        st.markdown(head)
        bits = [f"**{leg['venue']}**"]
        if leg["chain_name"] != "—":
            bits.append(leg["chain_name"])
        if leg["liquidity"]:
            bits.append(f"пул ${leg['liquidity']:,.0f}".replace(",", " "))
        st.caption(" · ".join(bits))
        # Ссылка ведёт на страницу обмена с уже подставленными адресами
        # контрактов — её можно открыть во встроенном браузере кошелька.
        links = []
        if leg["swap"]:
            links.append(f"[открыть обмен]({leg['swap']})")
        if leg["pool_page"]:
            links.append(f"[пул]({leg['pool_page']})")
        if links:
            st.markdown(" · ".join(links))
        # Показываем расшифровку, только если она что-то добавляет:
        # строка «BNB — BNB» занимает место и не сообщает ничего.
        if leg["name_to"] and leg["name_to"].upper() != leg["to"].upper():
            st.caption(f"{leg['to']} — {leg['name_to']}")

    if cyc.needs_transfer():
        st.warning(
            "Связка требует переводов между площадками или сетями. "
            "Комиссия вывода, время перевода и движение цены за это время "
            "в расчёт не входят.", icon="⚠️")

    st.markdown("---")
    m = cyc.margin_pct()
    ok = np.isfinite(m)
    if ok.any():
        st.metric("Маржа медиана", f"{np.median(m[ok]):.3f} %")
        st.metric("Маржа максимум", f"{m[ok].max():.3f} %")
        st.metric("Время в плюсе", f"{(m[ok] > 0).mean() * 100:.1f} %")
    bl = cyc.bottleneck_liquidity()
    if bl:
        st.metric("Узкое место по ликвидности", f"${bl:,.0f}".replace(",", " "))
    st.caption(f"Газ учтён: ${cyc.gas_usd:.2f}")

with g1:
    show_what = st.radio("По вертикали", ["Произведение курсов", "Маржа, %"],
                         horizontal=True, label_visibility="collapsed")
    y = df[show_what]
    baseline = 1.0 if show_what == "Произведение курсов" else 0.0

    fig = go.Figure()
    # Порядок трасс важен: fill="tonexty" заливает область между текущей
    # трассой и предыдущей. Сначала кладём линию безубыточности, затем
    # верхнюю огибающую max(y, baseline) — заливка ложится ТОЛЬКО выше
    # безубыточности. Если поменять их местами, закрасится убыточная зона.
    fig.add_trace(go.Scatter(
        x=df["Время"], y=np.full(len(df), baseline, dtype=float),
        mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df["Время"], y=np.where(np.isfinite(y), np.maximum(y, baseline), baseline),
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(46,204,113,0.35)", name="в плюсе", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["Время"], y=y, mode="lines", name=cyc.label,
        line=dict(width=1.4, color="#2E86DE"),
        hovertemplate="%{x|%d.%m %H:%M}<br>%{y:.6f}<extra></extra>",
    ))
    fig.add_hline(y=baseline, line_dash="dash", line_color="#888",
                  annotation_text="безубыточность", annotation_position="right")
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Время (UTC)",
        yaxis_title=("Итог / старт, с учётом комиссий"
                     if show_what == "Произведение курсов" else "Чистая маржа, %"),
        hovermode="x unified", showlegend=False,
    )
    st.plotly_chart(fig, **FULL)

with st.expander("Данные графика"):
    st.dataframe(df, **FULL, hide_index=True)
    st.download_button("Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"связка_{'_'.join(cyc.assets)}.csv", mime="text/csv")

st.caption(
    "Как читать: значение выше линии безубыточности означает, что в этот момент "
    "цепочка обменов возвращала больше USDT, чем было вложено, уже за вычетом "
    "комиссий площадок, проскальзывания под заданный объём и газа. "
    "Короткие одиночные всплески чаще всего неисполнимы: на BNB Chain такие окна "
    "забирают MEV-боты в том же блоке. Практический интерес представляют связки "
    "с высокой долей прибыльного времени и длинными окнами."
)
