"""Интерактивный калькулятор межбиржевого арбитража.

Запуск:  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from arb_core import (
    Leg,
    Transfer,
    breakeven_volumes,
    evaluate_trade,
    parse_manual_book,
    scan_volumes,
    synthetic_book,
)
from dexes import AGGREGATORS, CHAINS, aggregators_for, cross_venue_triangles, dex_spot, fetch_dex_books
from exchanges import EXCHANGES, best_pairs, best_triangles, fetch_book, fetch_many, fetch_triangle_books, spread_matrix

st.set_page_config(page_title="Арбитражный калькулятор", page_icon="⚖", layout="wide")

# Ориентировочные комиссии вывода: (для стейблкоина, для «дорогой» монеты типа BTC).
# Значения — только подсказка по умолчанию, всегда редактируются вручную.
NETWORKS = {
    "BEP-20 (BNB Chain)": (0.29, 0.000010),
    "TRC-20 (Tron)": (1.0, 0.000015),
    "ERC-20 (Ethereum)": (4.5, 0.00025),
    "Arbitrum One": (0.15, 0.000008),
    "Base": (0.10, 0.000005),
    "Optimism": (0.15, 0.000008),
    "Polygon": (0.10, 0.000006),
    "Avalanche C-Chain": (0.50, 0.000010),
    "Solana": (0.02, 0.000004),
    "Bitcoin": (0.0002, 0.0002),
    "Lightning": (0.0, 0.0),
    "Внутренний перевод": (0.0, 0.0),
    "Своё значение": (0.0, 0.0),
}

EX_NAMES = list(EXCHANGES)


# ======================================================================================
# Кэшированные обёртки над API бирж
# ======================================================================================
@st.cache_data(ttl=15, show_spinner=False)
def cached_book(ex_name: str, base: str, quote: str, limit: int, _bust: int):
    snap = fetch_book(ex_name, base, quote, limit)
    return snap.symbol, snap.ask, snap.bid, snap.error


@st.cache_data(ttl=20, show_spinner=False)
def cached_dex_books(agg: str, chain: str, base: str, quote: str, depth: float, steps: int, _bust: int):
    r = fetch_dex_books(agg, chain, base, quote, depth, steps)
    return r.ask, r.bid, r.gas_usd, r.error


@st.cache_data(ttl=20, show_spinner=False)
def cached_scan(names: tuple, base: str, quote: str, limit: int, _bust: int):
    snaps = fetch_many(list(names), base, quote, limit)
    return (
        best_pairs(snaps, top=15),
        spread_matrix(snaps),
        {n: (s.ok, s.symbol, s.best_ask, s.best_bid, s.error) for n, s in snaps.items()},
    )


@st.cache_data(ttl=20, show_spinner=False)
def cached_triangle_scan(names: tuple, assets: tuple, limit: int, start_asset: str, start_amount: float, _bust: int):
    books, status = fetch_triangle_books(list(names), list(assets), limit)
    return best_triangles(books, start_asset, start_amount), status


@st.cache_data(ttl=30, show_spinner=False)
def cached_cross_venue_triangles(chain: str, venues: tuple, assets: tuple, start_asset: str, start_amount: float, _bust: int):
    return cross_venue_triangles(chain, list(venues), list(assets), start_asset, start_amount)


WRAPPED = {"WETH": "ETH", "WBTC": "BTC", "BTCB": "BTC", "WBNB": "BNB", "WMATIC": "MATIC",
           "WAVAX": "AVAX", "WSOL": "SOL", "USDC.E": "USDC", "XBT": "BTC",
           "WETH.E": "ETH", "WBTC.E": "BTC", "BTC.B": "BTC", "CBBTC": "BTC",
           "SOLVBTC": "BTC", "WCRO": "CRO", "WFTM": "FTM", "WMNT": "MNT",
           "WXDAI": "DAI", "XDAI": "DAI", "USDBC": "USDC", "WBETH": "ETH"}


def norm_asset(sym: str) -> str:
    s = (sym or "").upper()
    return WRAPPED.get(s, s)


def add_data_log(message: str) -> None:
    logs = st.session_state.setdefault("data_log", [])
    if not logs or logs[-1]["message"] != message:
        logs.append({"time": datetime.now().strftime("%H:%M:%S"), "message": message})
    st.session_state.data_log = logs[-200:]


def render_data_log() -> None:
    st.subheader("Лог получения данных")
    clear_column, info_column = st.columns([1, 6])
    if clear_column.button("Очистить лог", key="clear_data_log"):
        st.session_state.data_log = []
        st.rerun()
    info_column.caption("Запросы и ответы API отображаются в хронологическом порядке. Хранятся последние 200 событий.")
    with st.container(height=260, border=True):
        logs = st.session_state.get("data_log", [])
        if not logs:
            st.caption("Запросов пока не было.")
        else:
            st.code("\n".join(f"[{entry['time']}] {entry['message']}" for entry in logs), language=None)


def render_cross_venue_page() -> None:
    st.header("Межплощадочный треугольный арбитраж")
    st.caption(
        "Три свопа выполняются на выбранных DEX в одной сети. Сканнер перебирает все упорядоченные "
        "сочетания промежуточных активов и площадок, исключая циклы на одной DEX."
    )
    cross_chain = st.selectbox(
        "Рабочая сеть для всех трёх ног", list(CHAINS), index=list(CHAINS).index("BNB Chain"), key="page_cross_chain",
        help="По умолчанию BNB Chain. Мосты между сетями в расчёт не входят.",
    )
    available_venues = aggregators_for(cross_chain)
    default_venues = [venue for venue in ("SushiSwap", "PancakeSwap") if venue in available_venues]
    cross_venues = st.multiselect(
        "DEX-площадки для ног A, B и C", available_venues, default=default_venues, key="page_cross_venues",
        help="Нужно минимум две площадки. Для каждой ноги сканнер сам выбирает площадку из этого списка.",
    )
    chain_tokens = list(CHAINS[cross_chain]["tokens"])
    default_assets = [asset for asset in ("USDT", "AAVE", "DAI") if asset in chain_tokens]
    cross_assets = st.multiselect(
        "Активы для треугольников", chain_tokens, default=default_assets, max_selections=4, key="page_cross_assets",
        help="Выберите от трёх до четырёх токенов одной сети. Проверяются все порядки промежуточных активов.",
    )
    cross_start = st.selectbox("Стартовый и конечный актив", cross_assets, key="page_cross_start") if cross_assets else None
    cross_amount = st.number_input(
        "Стартовая сумма", min_value=0.01, value=100.0, step=10.0, key="page_cross_amount",
        help="Каждая следующая нога запрашивается на фактический выход предыдущей. Для стейблкоинов оценка газа вычитается из результата.",
    )
    if st.button("Сканировать межплощадочные треугольники", type="primary", disabled=len(cross_venues) < 2 or len(cross_assets) < 3):
        add_data_log(
            f"Запрос DEX-треугольников: сеть {cross_chain}; площадки {', '.join(cross_venues)}; "
            f"активы {', '.join(cross_assets)}; старт {cross_amount:g} {cross_start}."
        )
        with st.spinner("Получаю последовательные котировки DEX для всех сочетаний…"):
            table, status = cached_cross_venue_triangles(
                cross_chain, tuple(cross_venues), tuple(cross_assets), cross_start, float(cross_amount), st.session_state.bust
            )
        st.caption(
            f"Маршрутов активов: {status['routes']} · сочетаний DEX: {status['venue_sets']} · "
            f"готовых циклов: {status['completed']} · недоступных: {status['failed']}"
        )
        add_data_log(
            f"DEX-ответ: маршрутов активов {status['routes']}, сочетаний площадок {status['venue_sets']}, "
            f"получено циклов {status['completed']}, ошибок {status['failed']}."
        )
        if table.empty:
            st.warning("Готовых циклов нет: проверьте токены в выбранной сети, доступность DEX и ликвидность пар.")
        else:
            start_column = f"Старт, {cross_start}"
            before_gas_column = f"Финиш до газа, {cross_start}"
            finish_column = f"Финиш после газа, {cross_start}"
            profit_column = f"Прибыль, {cross_start}"
            st.dataframe(
                table.style.format(
                    {
                        start_column: "{:,.6f}", before_gas_column: "{:,.6f}", "Газ, USD": "{:,.4f}",
                        finish_column: "{:,.6f}", profit_column: "{:,.6f}", "Чистый спред, б.п.": "{:,.2f}",
                    }
                ).background_gradient(cmap="RdYlGn", subset=["Чистый спред, б.п."]),
                use_container_width=True, hide_index=True, height=560,
            )
            st.download_button("Скачать результаты CSV", table.to_csv(index=False).encode("utf-8-sig"),
                               file_name="cross_venue_triangular_arbitrage.csv", mime="text/csv")
    render_data_log()


def render_cex_triangle_page() -> None:
    st.header("Треугольный арбитраж")
    st.caption(
        "Внутрибиржевой цикл из трёх сделок: стартовая валюта → монета 1 → монета 2 → стартовая валюта. "
        "Расчёт использует лучшие цены стакана и тейкерские комиссии выбранной CEX."
    )
    cex_assets = st.multiselect(
        "Активы для треугольников", ["USDT", "USDC", "BTC", "ETH", "BNB", "SOL", "XRP", "AAVE", "DAI"],
        default=["USDT", "BTC", "ETH"], max_selections=5, key="page_cex_assets",
    )
    cex_start = st.selectbox("Стартовая и конечная валюта", cex_assets, key="page_cex_start") if cex_assets else None
    cex_amount = st.number_input("Сумма для проверки", min_value=0.01, value=1_000.0, step=100.0, key="page_cex_amount")
    cex_exchanges = st.multiselect(
        "CEX-площадки", EX_NAMES, default=["Binance", "OKX", "Gate.io"], key="page_cex_exchanges",
        help="Каждый маршрут исполняется на одной CEX; для межплощадочных DEX-циклов используйте соседний лист.",
    )
    if st.button("Сканировать CEX-треугольники", type="primary", disabled=len(cex_assets) < 3 or not cex_exchanges):
        add_data_log(
            f"Запрос CEX-треугольников: биржи {', '.join(cex_exchanges)}; активы {', '.join(cex_assets)}; "
            f"старт {cex_amount:g} {cex_start}."
        )
        with st.spinner("Загружаю пары и перебираю тройки…"):
            table, status = cached_triangle_scan(
                tuple(cex_exchanges), tuple(cex_assets), 20, cex_start, float(cex_amount), st.session_state.bust
            )
        available_pairs = sum(value[0] for value in status.values())
        requested_pairs = sum(value[1] for value in status.values())
        st.caption(f"Доступно пар: {available_pairs} из {requested_pairs} · бирж: {len(cex_exchanges)}")
        for exchange, (received, requested) in status.items():
            add_data_log(f"CEX-ответ {exchange}: получено пар {received} из {requested}.")
        if table.empty:
            st.warning("Для выбранных активов не найдено полных треугольников.")
        else:
            start_column = f"Старт, {cex_start}"
            finish_column = f"Финиш, {cex_start}"
            profit_column = f"Прибыль, {cex_start}"
            st.dataframe(
                table.style.format(
                    {
                        start_column: "{:,.6f}", finish_column: "{:,.6f}", profit_column: "{:,.6f}",
                        "Грязный спред, б.п.": "{:,.2f}", "Комиссии тейкера, б.п.": "{:,.2f}", "Чистый спред, б.п.": "{:,.2f}",
                    }
                ).background_gradient(cmap="RdYlGn", subset=["Чистый спред, б.п."]),
                use_container_width=True, hide_index=True, height=560,
            )
            st.download_button("Скачать результаты CSV", table.to_csv(index=False).encode("utf-8-sig"),
                               file_name="cex_triangular_arbitrage.csv", mime="text/csv")
    render_data_log()


def leg_controls(key: str, title: str, live: bool, default_cex: str, default_chain: str) -> dict:
    """Блок параметров одной ноги сделки: ЦЕФИ или DEX-агрегатор."""
    st.header(title)
    cfg: dict = {"kind": "cex"}
    if live:
        cfg["kind"] = (
            "dex"
            if st.radio(
                "Тип площадки", ["CEX (биржа)", "DEX (агрегатор)"], horizontal=True, key=f"kind_{key}"
            ).startswith("DEX")
            else "cex"
        )

    if cfg["kind"] == "dex":
        chain = st.selectbox("Сеть", list(CHAINS), index=list(CHAINS).index(default_chain), key=f"chain_{key}")
        aggs = aggregators_for(chain)
        agg = st.selectbox("Агрегатор", aggs, key=f"agg_{key}_{chain}")
        tokens = list(CHAINS[chain]["tokens"])
        cfg["chain"], cfg["agg"] = chain, agg
        cfg["base"] = st.selectbox("Монета", tokens, key=f"db_{key}_{chain}")
        cfg["quote"] = st.selectbox(
            "Котировка", tokens,
            index=tokens.index("USDT") if "USDT" in tokens else 1, key=f"dq_{key}_{chain}",
        )
        cfg["probe"] = st.number_input(
            "Глубина зондирования, монеты", value=20.0, step=1.0, min_value=0.001,
            help="Максимальный размер свопа, на котором строится кривая price impact.",
            key=f"probe_{key}",
        )
        cfg["name"] = f"{agg} · {chain}"
        cfg["maker"] = cfg["taker"] = 0.0
        st.caption("Комиссия пула уже зашита в котировку агрегатора, отдельный тейкерский сбор не взимается.")
    elif live:
        name = st.selectbox("Биржа", EX_NAMES, index=EX_NAMES.index(default_cex), key=f"ex_{key}")
        cfg["name"] = name
        cfg["maker"] = st.number_input(
            "Комиссия мейкера, %", value=float(EXCHANGES[name].maker_pct),
            step=0.01, format="%.4f", key=f"m_{key}_{name}",
        )
        cfg["taker"] = st.number_input(
            "Комиссия тейкера, %", value=float(EXCHANGES[name].taker_pct),
            step=0.01, format="%.4f", key=f"t_{key}_{name}",
        )
    else:
        cfg["name"] = st.text_input("Название", default_cex, key=f"nm_{key}")
        cfg["maker"] = st.number_input("Комиссия мейкера, %", value=0.02, step=0.01, format="%.4f", key=f"m_{key}")
        cfg["taker"] = st.number_input("Комиссия тейкера, %", value=0.10, step=0.01, format="%.4f", key=f"t_{key}")

    cfg["type"] = st.radio("Тип ордера", ["taker", "maker"], horizontal=True, key=f"ord_{key}")
    return cfg


st.title("⚖ Арбитражный калькулятор с учётом стакана")
st.caption(
    "Модель: покупка по ask на бирже A → комиссия биржи → вывод в сети → продажа по bid "
    "на бирже B → комиссия биржи → возврат котируемой валюты. Проскальзывание считается "
    "проходом по уровням стакана (VWAP)."
)

if "bust" not in st.session_state:
    st.session_state.bust = 0

active_page = st.radio(
    "Лист калькулятора",
    ["Арбитражный калькулятор", "Межплощадочный треугольный арбитраж", "Треугольный арбитраж"],
    horizontal=True,
    label_visibility="collapsed",
)

if active_page == "Межплощадочный треугольный арбитраж":
    render_cross_venue_page()
    st.stop()
if active_page == "Треугольный арбитраж":
    render_cex_triangle_page()
    st.stop()

# ======================================================================================
# Боковая панель — параметры
# ======================================================================================
with st.sidebar:
    st.header("Источник стакана")
    source = st.radio(
        "Откуда брать ликвидность",
        ["Реальные стаканы (API бирж)", "Синтетический", "Вставить вручную"],
        label_visibility="collapsed",
    )
    live = source.startswith("Реальные")

    st.divider()
    st.header("Инструмент")
    base = st.text_input("Базовая монета", "BTC").strip().upper()
    quote = st.text_input("Котируемая валюта", "USDT").strip().upper()
    depth_limit = st.slider("Уровней стакана запрашивать", 20, 500, 100, 10) if live else 100

    st.divider()
    leg_a = leg_controls("a", "Нога A — покупка", live, "Binance", "BNB Chain")
    st.divider()
    leg_b = leg_controls("b", "Нога B — продажа", live, "OKX", "BNB Chain")
    st.divider()
    leg_c = leg_controls("c", "Нога C — третья сделка", live, "Gate.io", "BNB Chain")

    ex_a, fee_a_maker, fee_a_taker, type_a = leg_a["name"], leg_a["maker"], leg_a["taker"], leg_a["type"]
    ex_b, fee_b_maker, fee_b_taker, type_b = leg_b["name"], leg_b["maker"], leg_b["taker"], leg_b["type"]
    ex_c, fee_c_maker, fee_c_taker, type_c = leg_c["name"], leg_c["maker"], leg_c["taker"], leg_c["type"]

    st.divider()
    st.header("Перевод между биржами")
    net_name = st.selectbox("Сеть вывода", list(NETWORKS.keys()), index=0)
    asset_kind = st.radio(
        "Тип базового актива",
        ["крипто (BTC/ETH)", "стейблкоин"],
        horizontal=True,
        help="Влияет только на подсказанное значение комиссии сети.",
    )
    default_fee = NETWORKS[net_name][1 if asset_kind.startswith("крипто") else 0]
    net_fee_coin = st.number_input(
        f"Комиссия сети, {base}",
        value=float(default_fee),
        step=0.0001,
        format="%.6f",
        key=f"netfee_{net_name}_{asset_kind}",
    )
    quote_back_fee = st.number_input(
        f"Комиссия возврата {quote} на A", value=1.0, step=0.5, format="%.4f"
    )
    minutes = st.number_input("Время перевода, мин", value=10.0, step=1.0)

    st.divider()
    fee_in_coin = st.checkbox("Комиссия биржи A списывается монетой", value=True)

# ======================================================================================
# Получение стаканов
# ======================================================================================
ask_book = bid_book = None
auto = False

if live:
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("🔄 Обновить стаканы", use_container_width=True):
        st.session_state.bust += 1
    auto = c2.toggle("Автообновление 15 c", value=False)
    if auto:
        st.session_state.bust += 1

    def load_leg(cfg: dict, label: str):
        """Возвращает (symbol, ask_book, bid_book, error, gas_usd) для любой площадки."""
        if cfg["kind"] == "dex":
            add_data_log(
                f"Запрос {label}: {cfg['agg']} · {cfg['chain']} · {cfg['base']}/{cfg['quote']} · глубина {cfg['probe']:g}."
            )
            a, b_, gas, err = cached_dex_books(
                cfg["agg"], cfg["chain"], cfg["base"], cfg["quote"],
                float(cfg["probe"]), 6, st.session_state.bust,
            )
            return f"{cfg['base']}/{cfg['quote']}", a, b_, err, gas
        add_data_log(f"Запрос {label}: {cfg['name']} · {base}/{quote} · уровней {depth_limit}.")
        s, a, b_, err = cached_book(cfg["name"], base, quote, depth_limit, st.session_state.bust)
        return s, a, b_, err, 0.0

    with st.spinner("Загружаю стаканы с площадок…"):
        sym_a, ask_a, bid_a, err_a, gas_a = load_leg(leg_a, "ноги A")
        sym_b, ask_b, bid_b, err_b, gas_b = load_leg(leg_b, "ноги B")

    add_data_log(
        f"Ответ ноги A: {ex_a} · {sym_a} · "
        f"{'ошибка: ' + err_a if err_a else f'ask {len(ask_a)}, bid {len(bid_a)}'}"
    )
    add_data_log(
        f"Ответ ноги B: {ex_b} · {sym_b} · "
        f"{'ошибка: ' + err_b if err_b else f'ask {len(ask_b)}, bid {len(bid_b)}'}"
    )

    if err_a:
        st.error(f"{ex_a} ({sym_a}): {err_a}")
    if err_b:
        st.error(f"{ex_b} ({sym_b}): {err_b}")
    if err_a or err_b:
        st.info(
            "Выберите другую биржу или переключитесь на синтетический стакан в боковой панели. "
            "Часть бирж блокирует доступ по региону сервера."
        )
        render_data_log()
        st.stop()

    asset_a = norm_asset(leg_a["base"] if leg_a["kind"] == "dex" else base)
    asset_b = norm_asset(leg_b["base"] if leg_b["kind"] == "dex" else base)
    if asset_a != asset_b:
        st.warning(
            f"Ноги торгуют разные активы: покупка {asset_a}, продажа {asset_b}. "
            "Арбитраж возможен только по одной и той же монете — цифры ниже не имеют экономического смысла."
        )
    ask_book, bid_book = ask_a, bid_b
    if gas_a or gas_b:
        st.caption(f"Газ свопов по оценке агрегаторов: ≈ {gas_a + gas_b:,.2f} USD (учтите в поле «Комиссия возврата»).")
    c3.caption(
        f"Данные обновлены {datetime.now():%H:%M:%S} · "
        f"покупка {ex_a} `{sym_a}` ({len(ask_book)} уровней) · "
        f"продажа {ex_b} `{sym_b}` ({len(bid_book)} уровней)"
    )
    ask = float(ask_book["price"].iloc[0])
    bid = float(bid_book["price"].iloc[0])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Ask на {ex_a}", f"{ask:,.4f} {quote}")
    k2.metric(f"Bid на {ex_b}", f"{bid:,.4f} {quote}")
    k3.metric(f"Глубина ask, {base}", f"{ask_book['size'].sum():,.4f}")
    k4.metric(f"Глубина bid, {base}", f"{bid_book['size'].sum():,.4f}")

elif source == "Синтетический":
    st.subheader("Ликвидность в стакане")
    c1, c2, c3, c4 = st.columns(4)
    ask = c1.number_input(f"Лучший Ask, {quote}", value=64_000.0, step=1.0, format="%.6f")
    bid = c2.number_input(f"Лучший Bid, {quote}", value=64_300.0, step=1.0, format="%.6f")
    depth_ask = c3.number_input(f"Глубина ask, {base}", value=40.0, step=0.5, format="%.6f")
    depth_bid = c4.number_input(f"Глубина bid, {base}", value=30.0, step=0.5, format="%.6f")
    c5, c6, c7 = st.columns(3)
    step_bps = c5.number_input("Шаг между уровнями, б.п.", value=2.0, step=0.5)
    shape = c6.slider("Профиль ликвидности (<1 — плотный топ)", 0.2, 3.0, 1.0, 0.1)
    levels = c7.slider("Число уровней", 5, 60, 25)
    ask_book = synthetic_book(ask, "ask", depth_ask, levels, step_bps, shape)
    bid_book = synthetic_book(bid, "bid", depth_bid, levels, step_bps, shape)
    add_data_log(f"Синтетические стаканы обновлены: {base}/{quote}; ask {len(ask_book)}, bid {len(bid_book)}.")

else:
    st.subheader("Стакан вручную")
    c1, c2 = st.columns(2)
    txt_a = c1.text_area(
        f"Ask-стакан {ex_a} (цена, объём)",
        "64000, 0.8\n64010, 1.2\n64025, 2.0\n64050, 3.0",
        height=180,
    )
    txt_b = c2.text_area(
        f"Bid-стакан {ex_b} (цена, объём)",
        "64300, 0.6\n64280, 1.0\n64250, 1.8\n64210, 2.5",
        height=180,
    )
    ask_book = parse_manual_book(txt_a)
    bid_book = parse_manual_book(txt_b)
    if ask_book is None or bid_book is None:
        st.error("Не удалось разобрать стакан. Формат строки: цена, объём")
        add_data_log("Ошибка ручного стакана: не удалось разобрать введённые уровни.")
        render_data_log()
        st.stop()
    ask = float(ask_book["price"].iloc[0])
    bid = float(bid_book["price"].iloc[0])
    add_data_log(f"Ручные стаканы обновлены: {base}/{quote}; ask {len(ask_book)}, bid {len(bid_book)}.")

max_qty = float(min(ask_book["size"].sum(), bid_book["size"].sum()))

# ======================================================================================
# Расчёт
# ======================================================================================
buy_leg = Leg(ex_a, fee_a_maker, fee_a_taker, type_a)
sell_leg = Leg(ex_b, fee_b_maker, fee_b_taker, type_b)
transfer = Transfer(net_name, net_fee_coin, quote_back_fee, minutes)

st.subheader("Объём сделки")
qty = st.slider(
    f"Объём, {base}",
    min_value=float(max_qty / 200),
    max_value=float(max_qty),
    value=float(max_qty / 10),
    step=float(max_qty / 200),
)

res = evaluate_trade(qty, ask_book, bid_book, buy_leg, sell_leg, transfer, fee_in_coin)
be = breakeven_volumes(max_qty, ask_book, bid_book, buy_leg, sell_leg, transfer, fee_in_coin)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Спред по лучшим ценам", f"{res.spread_bps:,.1f} б.п.")
m2.metric("Чистая прибыль", f"{res.net_quote:,.2f} {quote}")
m3.metric("Рентабельность", f"{res.net_margin_pct:,.3f} %")
m4.metric("Проскальзывание покупки", f"{res.buy.slippage_bps:,.1f} б.п.")
m5.metric("Проскальзывание продажи", f"{res.sell.slippage_bps:,.1f} б.п.")

if not res.liquidity_ok:
    st.warning("Ликвидности в стакане не хватает на заданный объём — сделка исполнена частично.")

if res.net_quote > 0:
    st.success(
        f"Сделка прибыльна: {res.net_quote:,.2f} {quote} "
        f"({res.net_margin_pct:.3f}% от оборота {res.cost_quote:,.2f} {quote})."
    )
else:
    st.error(
        f"Сделка убыточна: {res.net_quote:,.2f} {quote}. "
        "Спреда не хватает на комиссии и проскальзывание."
    )

# --- Точка безубыточности ------------------------------------------------------------
st.subheader("Точка безубыточности по объёму")
b1, b2, b3 = st.columns(3)
if be["lower"] is not None:
    lower_txt = f"{be['lower']:.6f} {base}"
elif be["profitable_anywhere"]:
    lower_txt = "любой объём"
else:
    lower_txt = "нет прибыльной зоны"

b1.metric(
    "Минимальный объём в плюс",
    lower_txt,
    help="Ниже этого объёма фиксированная комиссия сети съедает всю прибыль.",
)
b2.metric(
    "Максимальный объём в плюс",
    f"{be['upper']:.6f} {base}" if be["upper"] else ("выше глубины стакана" if be["profitable_anywhere"] else "—"),
    help="Выше этого объёма проскальзывание в стакане делает сделку убыточной.",
)
b3.metric(
    "Оптимальный объём",
    f"{be['q_opt']:.6f} {base}",
    delta=f"{be['net_opt']:,.2f} {quote}",
    help="Объём, при котором чистая прибыль максимальна.",
)

curve = be["curve"].rename(columns={"qty": f"Объём, {base}", "net": f"Чистая прибыль, {quote}"})
st.line_chart(curve, x=f"Объём, {base}", y=f"Чистая прибыль, {quote}", height=280)

# --- Разложение P&L -------------------------------------------------------------------
st.subheader("Разложение результата")
pnl = pd.DataFrame(
    [
        ("Покупка по VWAP на " + ex_a, -res.buy.quote),
        (f"Комиссия биржи {ex_a} ({buy_leg.order_type}, {buy_leg.fee_pct}%)", -res.buy_fee_quote),
        (f"Комиссия сети {net_name}", -res.withdraw_fee_quote),
        ("Продажа по VWAP на " + ex_b, res.sell.quote),
        (f"Комиссия биржи {ex_b} ({sell_leg.order_type}, {sell_leg.fee_pct}%)", -res.sell_fee_quote),
        (f"Возврат {quote} на {ex_a}", -res.quote_fee_quote),
        ("ИТОГО чистая прибыль", res.net_quote),
    ],
    columns=["Статья", f"Сумма, {quote}"],
)
st.dataframe(
    pnl.style.format({f"Сумма, {quote}": "{:,.2f}"}),
    use_container_width=True,
    hide_index=True,
)

d1, d2 = st.columns(2)
d1.markdown(
    f"""
**Исполнение покупки ({ex_a})**
- Лучший ask: `{res.buy.best:,.4f}`
- VWAP: `{res.buy.vwap:,.4f}`
- Уровней задето: `{res.buy.levels_used}`
- Куплено: `{res.buy.qty:.6f} {base}`
"""
)
d2.markdown(
    f"""
**Исполнение продажи ({ex_b})**
- Лучший bid: `{res.sell.best:,.4f}`
- VWAP: `{res.sell.vwap:,.4f}`
- Уровней задето: `{res.sell.levels_used}`
- Продано: `{res.sell.qty:.6f} {base}` (после вывода {net_fee_coin:g} {base})
"""
)

# --- Таблица рентабельности ------------------------------------------------------------
st.subheader("Таблица рентабельности по объёмам")
points = st.slider("Число строк", 5, 60, 20)
table = scan_volumes(max_qty, points, ask_book, bid_book, buy_leg, sell_leg, transfer, fee_in_coin)


def _highlight(row: pd.Series):
    color = (
        "background-color: rgba(0,160,80,0.15)"
        if row["Чистая прибыль"] > 0
        else "background-color: rgba(200,40,40,0.12)"
    )
    return [color] * len(row)


st.dataframe(
    table.style.apply(_highlight, axis=1).format(
        {
            "Объём (монета)": "{:.6f}",
            "Оборот (котируемая)": "{:,.2f}",
            "VWAP покупки": "{:,.4f}",
            "VWAP продажи": "{:,.4f}",
            "Проскальзывание покупки, б.п.": "{:,.1f}",
            "Проскальзывание продажи, б.п.": "{:,.1f}",
            "Комиссии бирж": "{:,.2f}",
            "Комиссия сети": "{:,.2f}",
            "Чистая прибыль": "{:,.2f}",
            "Рентабельность, %": "{:,.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
    height=460,
)

st.download_button(
    "Скачать таблицу CSV",
    table.to_csv(index=False).encode("utf-8-sig"),
    file_name="arbitrage_profitability.csv",
    mime="text/csv",
)

# --- Сканер спредов по биржам ------------------------------------------------------------
if live:
    st.subheader("Сканер связок по биржам")
    st.caption(
        f"Опрашивает публичные API выбранных бирж по паре {base}/{quote} и сортирует связки "
        "по спреду за вычетом тейкерских комиссий. Комиссии сети здесь не учитываются."
    )
    sel = st.multiselect("Биржи для сканирования", EX_NAMES, default=EX_NAMES)
    if st.button("Сканировать спреды", type="primary") and sel:
        add_data_log(f"Запрос сканера спредов: {base}/{quote}; биржи {', '.join(sel)}; уровней 20.")
        with st.spinner("Опрашиваю биржи…"):
            pairs, matrix, status = cached_scan(tuple(sel), base, quote, 20, st.session_state.bust)

        okc = sum(1 for v in status.values() if v[0])
        st.caption(f"Успешно опрошено бирж: {okc} из {len(sel)} · {datetime.now():%H:%M:%S}")

        failed = {n: v[4] for n, v in status.items() if not v[0]}
        for exchange, value in status.items():
            if value[0]:
                add_data_log(f"Сканер спредов: {exchange} · {value[1]} · ask {value[2]:.8g}, bid {value[3]:.8g}.")
            else:
                add_data_log(f"Сканер спредов: {exchange} · ошибка: {value[4]}.")
        if failed:
            with st.expander(f"Недоступные биржи ({len(failed)})"):
                for n, e in failed.items():
                    st.write(f"**{n}** — {e}")

        if not pairs.empty:
            st.dataframe(
                pairs.style.format(
                    {
                        "Ask": "{:,.4f}",
                        "Bid": "{:,.4f}",
                        "Спред, б.п.": "{:,.2f}",
                        "Комиссии тейкера, б.п.": "{:,.2f}",
                        "Спред за вычетом комиссий, б.п.": "{:,.2f}",
                    }
                ).background_gradient(
                    cmap="RdYlGn", subset=["Спред за вычетом комиссий, б.п."]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Матрица спредов, б.п.: строка — где покупаем, столбец — где продаём.")
            st.dataframe(
                matrix.style.background_gradient(cmap="RdYlGn", axis=None).format("{:,.1f}", na_rep="—"),
                use_container_width=True,
            )
        else:
            st.warning("Ни одна биржа не ответила по этой паре.")

# --- Межплощадочный DEX-сканер ----------------------------------------------------------
if live and active_page == "Межплощадочный треугольный арбитраж":
    st.subheader("Межплощадочный треугольный арбитраж: A → B → C")
    st.caption(
        "Три свопа выполняются на выбранных DEX в одной сети. Сканнер перебирает все упорядоченные "
        "сочетания двух промежуточных активов и площадок, но исключает циклы, где все три ноги на одной DEX."
    )
    cross_chain = st.selectbox(
        "Рабочая сеть для всех трёх ног", list(CHAINS), index=list(CHAINS).index("BNB Chain"), key="cross_chain",
        help="По умолчанию BNB Chain. Межсетевые мосты в расчёт не входят: все активы должны существовать в одной сети.",
    )
    available_venues = aggregators_for(cross_chain)
    default_venues = [venue for venue in ("SushiSwap", "PancakeSwap") if venue in available_venues]
    cross_venues = st.multiselect(
        "DEX-площадки для ног A, B и C", available_venues, default=default_venues, key="cross_venues",
        help="Нужно минимум две площадки. Для каждой ноги сканнер сам выбирает площадку из этого списка.",
    )
    chain_tokens = list(CHAINS[cross_chain]["tokens"])
    default_cross_assets = [asset for asset in ("USDT", "AAVE", "DAI") if asset in chain_tokens]
    cross_assets = st.multiselect(
        "Активы для межплощадочных треугольников", chain_tokens, default=default_cross_assets,
        max_selections=4, key="cross_assets",
        help="Выберите от трёх до четырёх токенов одной сети. Будут проверены все порядки промежуточных активов.",
    )
    cross_start = st.selectbox("Стартовый и конечный актив", cross_assets, key="cross_start") if cross_assets else None
    cross_amount = st.number_input(
        "Стартовая сумма для DEX-цикла", min_value=0.01, value=100.0, step=10.0, key="cross_amount",
        help="Каждая следующая нога запрашивается на фактический выход предыдущей. Для USDT/USDC/DAI оценка газа вычитается из результата.",
    )
    if st.button(
        "Сканировать межплощадочные треугольники", type="primary",
        disabled=len(cross_venues) < 2 or len(cross_assets) < 3,
    ):
        with st.spinner("Получаю последовательные котировки DEX для всех сочетаний…"):
            cross_triangles, cross_status = cached_cross_venue_triangles(
                cross_chain, tuple(cross_venues), tuple(cross_assets), cross_start, float(cross_amount), st.session_state.bust
            )
        st.caption(
            f"Маршрутов активов: {cross_status['routes']} · сочетаний DEX: {cross_status['venue_sets']} · "
            f"готовых циклов: {cross_status['completed']} · недоступных: {cross_status['failed']}"
        )
        if cross_triangles.empty:
            st.warning("Готовых циклов нет: проверьте токены в выбранной сети, доступность DEX и ликвидность пар.")
        else:
            start_column = f"Старт, {cross_start}"
            before_gas_column = f"Финиш до газа, {cross_start}"
            finish_column = f"Финиш после газа, {cross_start}"
            profit_column = f"Прибыль, {cross_start}"
            st.dataframe(
                cross_triangles.style.format(
                    {
                        start_column: "{:,.6f}",
                        before_gas_column: "{:,.6f}",
                        "Газ, USD": "{:,.4f}",
                        finish_column: "{:,.6f}",
                        profit_column: "{:,.6f}",
                        "Чистый спред, б.п.": "{:,.2f}",
                    }
                ).background_gradient(cmap="RdYlGn", subset=["Чистый спред, б.п."]),
                use_container_width=True,
                hide_index=True,
                height=560,
            )
            st.download_button(
                "Скачать межплощадочные треугольники CSV",
                cross_triangles.to_csv(index=False).encode("utf-8-sig"),
                file_name="cross_venue_triangular_arbitrage.csv",
                mime="text/csv",
            )

# --- Треугольный сканер -----------------------------------------------------------------
if live and active_page == "Треугольный арбитраж":
    st.subheader("Треугольный арбитраж: 3 торговые ноги")
    st.caption(
        "Перебирает циклы на одной бирже: стартовая валюта → монета 1 → монета 2 → стартовая валюта. "
        "В расчёте учитываются тейкерские комиссии, но не глубина стакана и не переводы между биржами."
    )
    default_triangle_assets = list(dict.fromkeys([quote, base, "ETH", "BNB"]))
    triangle_assets = st.multiselect(
        "Активы для треугольников", default_triangle_assets, default=default_triangle_assets,
        help="Нужно не менее трёх активов. Для N активов сканер проверяет все доступные циклы из трёх сделок.",
    )
    triangle_start = st.selectbox("Стартовая и конечная валюта", triangle_assets, key="triangle_start") if triangle_assets else None
    triangle_amount = st.number_input(
        "Сумма для проверки треугольника", min_value=0.01, value=1_000.0, step=100.0,
        help="Сумма используется для показа результата в стартовой валюте; спред не зависит от неё при расчёте по лучшей цене.",
    )
    triangle_defaults = [name for name in dict.fromkeys((ex_a, ex_b, ex_c)) if name in EX_NAMES]
    triangle_exchanges = st.multiselect(
        "Биржи для треугольников", EX_NAMES, default=triangle_defaults, key="triangle_exchanges",
        help="По умолчанию используются CEX-площадки, выбранные в ногах A, B и C.",
    )

    if st.button("Сканировать треугольники", type="primary", disabled=len(triangle_assets) < 3 or not triangle_exchanges):
        with st.spinner("Загружаю пары и перебираю тройки…"):
            triangles, triangle_status = cached_triangle_scan(
                tuple(triangle_exchanges), tuple(triangle_assets), 20, triangle_start, float(triangle_amount), st.session_state.bust
            )

        available_pairs = sum(value[0] for value in triangle_status.values())
        requested_pairs = sum(value[1] for value in triangle_status.values())
        st.caption(
            f"Доступно пар: {available_pairs} из {requested_pairs} · бирж: {len(triangle_exchanges)} · {datetime.now():%H:%M:%S}"
        )
        if triangles.empty:
            st.warning("Для выбранных активов не найдено полных треугольников. Добавьте ликвидные активы или выберите другие биржи.")
        else:
            start_column = f"Старт, {triangle_start}"
            finish_column = f"Финиш, {triangle_start}"
            profit_column = f"Прибыль, {triangle_start}"
            st.dataframe(
                triangles.style.format(
                    {
                        start_column: "{:,.4f}",
                        finish_column: "{:,.4f}",
                        profit_column: "{:,.4f}",
                        "Грязный спред, б.п.": "{:,.2f}",
                        "Комиссии тейкера, б.п.": "{:,.2f}",
                        "Чистый спред, б.п.": "{:,.2f}",
                    }
                ).background_gradient(cmap="RdYlGn", subset=["Чистый спред, б.п."]),
                use_container_width=True,
                hide_index=True,
                height=520,
            )
            st.download_button(
                "Скачать треугольные связки CSV",
                triangles.to_csv(index=False).encode("utf-8-sig"),
                file_name="triangular_arbitrage.csv",
                mime="text/csv",
            )

render_data_log()

# --- Стаканы ---------------------------------------------------------------------------
with st.expander("Показать стаканы"):
    c1, c2 = st.columns(2)
    c1.caption(f"Ask — {ex_a}")
    c1.dataframe(ask_book.head(50), hide_index=True, use_container_width=True)
    c2.caption(f"Bid — {ex_b}")
    c2.dataframe(bid_book.head(50), hide_index=True, use_container_width=True)

# --- Чувствительность --------------------------------------------------------------------
with st.expander("Матрица чувствительности: спред × комиссия сети"):
    spreads = np.linspace(0, max(res.spread_bps * 2, 40), 9)
    fees = np.unique(np.round(np.linspace(0, max(net_fee_coin * 2, 0.0001), 6), 8))
    grid = []
    for f in fees:
        row = {}
        for s in spreads:
            bid_alt = ask * (1 + s / 10_000.0)
            bb = synthetic_book(bid_alt, "bid", float(bid_book["size"].sum()), len(bid_book))
            t = Transfer(net_name, float(f), quote_back_fee, minutes)
            r = evaluate_trade(qty, ask_book, bb, buy_leg, sell_leg, t, fee_in_coin)
            row[f"{s:.1f} б.п."] = r.net_quote
        grid.append(row)
    sens = pd.DataFrame(grid, index=[f"{f:.6g} {base}" for f in fees])
    sens = sens.loc[~sens.index.duplicated(), ~sens.columns.duplicated()]
    st.dataframe(
        sens.style.background_gradient(cmap="RdYlGn", axis=None).format("{:,.1f}"),
        use_container_width=True,
    )
    st.caption("Строки — комиссия сети, столбцы — спред между биржами. Значения — чистая прибыль.")

st.caption(
    "Данные стаканов — публичные REST API бирж, без ключей и авторизации; часть площадок может "
    "блокировать запросы по региону сервера. Модель не учитывает ценовой риск за время перевода "
    f"(~{minutes:.0f} мин), лимиты вывода, задержки сети и частичные отмены заявок."
)

if live and auto:
    import time

    time.sleep(15)
    st.rerun()
