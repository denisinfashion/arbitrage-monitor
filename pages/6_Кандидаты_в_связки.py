"""Кандидаты в связки «фьючерс ↔ DEX» — шаг 1 поиска зависимостей.

Токены, у которых есть бессрочный фьючерс, но нет спота на крупных биржах:
цена у них формируется на DEX, и фьючерс может следовать за ней с запаздыванием.
Фьючерс и пул сопоставлены по адресу контракта, а не по тикеру.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from history.ui import FULL
from leadlag import universe
from leadlag.catalog import CHAINS, FUTURES, SPOT
from leadlag.settings import LEADLAG_DIR, ScreenSettings

RUN_LOG = LEADLAG_DIR / "universe_run.log"
RUN_PID = LEADLAG_DIR / "universe_run.pid"

st.set_page_config(page_title="Кандидаты в связки", layout="wide")
st.title("Кандидаты в связки: фьючерс ↔ DEX")
st.caption("Фьючерс и пул сопоставлены по адресу контракта. Токены со спотом на "
           "отмеченных крупных биржах исключены — у них цену формирует не DEX.")


def checkbox_grid(title: str, options: dict, selected: list, key: str, cols: int = 4) -> list:
    st.markdown(f"**{title}**")
    grid = st.columns(cols)
    out = []
    for i, (value, label) in enumerate(options.items()):
        if grid[i % cols].checkbox(label, value=value in selected, key=f"{key}:{value}"):
            out.append(value)
    return out


def running() -> bool:
    try:
        pid = int(RUN_PID.read_text())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


s = ScreenSettings.load()
defaults = ScreenSettings()

with st.sidebar:
    st.header("Пороги")
    s.min_fut_volume_usd = st.number_input(
        "Объём фьючерса за 24 ч, $ от", 0.0, value=float(s.min_fut_volume_usd), step=250_000.0,
        help=f"По умолчанию ${defaults.min_fut_volume_usd:,.0f}: на меньшем объёме "
             "вход и выход двигают цену.")
    s.min_fut_range_pct = st.number_input(
        "Диапазон цены фьючерса за 24 ч, % от", 0.0, value=float(s.min_fut_range_pct), step=1.0,
        help=f"(максимум − минимум) / цена. По умолчанию {defaults.min_fut_range_pct:g}%: "
             "без движений зависимость нечем монетизировать.")
    s.min_dex_liquidity_usd = st.number_input(
        "Ликвидность пулов DEX, $ от", 0.0, value=float(s.min_dex_liquidity_usd), step=25_000.0,
        help=f"Сумма по всем пулам токена в сети. По умолчанию ${defaults.min_dex_liquidity_usd:,.0f}.")
    s.min_dex_volume_usd = st.number_input(
        "Объём DEX за 24 ч, $ от", 0.0, value=float(s.min_dex_volume_usd), step=25_000.0,
        help=f"По умолчанию ${defaults.min_dex_volume_usd:,.0f}: пул без торгов цену не формирует.")
    s.max_price_gap_pct = st.number_input(
        "Расхождение цены фьючерса и пула, % до", 0.0, value=float(s.max_price_gap_pct), step=0.5,
        help=f"Контроль сопоставления. По умолчанию {defaults.max_price_gap_pct:g}%: "
             "большое расхождение — скорее другой токен или мёртвый пул.")
    if st.button("Вернуть значения по умолчанию", **FULL):
        defaults.save()
        for k in list(st.session_state):
            if ":" in str(k):
                del st.session_state[k]
        st.rerun()

with st.expander("Площадки и сети", expanded=False):
    s.futures_exchanges = checkbox_grid(
        "Биржи фьючерсов, на которых торгую", {k: k.capitalize() for k in FUTURES},
        s.futures_exchanges, "fut")
    s.tier1_spot = checkbox_grid(
        "Исключать токены со спотом на биржах (tier 1)", {k: k.capitalize() for k in SPOT},
        s.tier1_spot, "spot")
    s.chains = checkbox_grid(
        "Сети DEX", {k: f"{c.label} · блок {c.block_s:g} с" for k, c in CHAINS.items()},
        s.chains, "chain", cols=3)
    s.exclude_tradfi = st.checkbox(
        "Исключать токенизированные акции, фонды и сырьё", value=s.exclude_tradfi, key="opt:tradfi",
        help="Их фьючерс идёт за фондовым рынком, а не за DEX.")
    st.caption("Время блока — точность метки времени свопа. При блоке 12 с "
               "(Ethereum) лаг в несколько секунд не измерить.")

if asdict(s) != asdict(ScreenSettings.load()):
    s.save()

df, meta = universe.load()

top = st.columns([3, 1])
with top[0]:
    if meta:
        age_h = (time.time() - meta["built_at"]) / 3600
        st.write(f"Данные собраны {time.strftime('%d.%m %H:%M', time.localtime(meta['built_at']))} "
                 f"({age_h:.1f} ч назад), сбор шёл {meta['duration_s']} с.")
    else:
        st.info("Данных ещё нет — запустите сбор.")
with top[1]:
    if running():
        st.button("Сбор идёт…", disabled=True, **FULL)
    elif st.button("Собрать заново", type="primary", **FULL):
        LEADLAG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(RUN_LOG, "w", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-m", "leadlag.universe"], cwd=ROOT,
                             stdout=log, stderr=subprocess.STDOUT)
        RUN_PID.write_text(str(p.pid))
        st.toast("Сбор запущен, займёт несколько минут")
        time.sleep(1)
        st.rerun()

if df is None or df.empty:
    st.stop()

shown = universe.apply_filters(df, s)

f = meta.get("funnel", {}) if meta else {}
c = st.columns(6)
c[0].metric("Монет с фьючерсами", f.get("futures_coins", "—"))
c[1].metric("Опознаны в CoinGecko", f.get("coins_matched", "—"))
c[2].metric("С контрактом в сетях", f.get("coins_with_contract", "—"))
c[3].metric("С пулами на DEX", f.get("coins_with_pools", "—"))
c[4].metric("Прошли фильтры, монет", shown["coin"].nunique())
c[5].metric("Пар фьючерс × сеть", len(shown))

view = pd.DataFrame({
    "Монета": shown["coin"],
    "Название": shown["coin_name"],
    "Биржа": shown["fut_venue"],
    "Фьючерс": shown["fut_url"],
    "Объём фьюч 24ч, $": shown["fut_volume_usd"].round(0),
    "Диапазон 24ч, %": shown["fut_range_pct"].round(1),
    "Изм 24ч, %": shown["fut_change_pct"].round(1),
    "Сеть": shown["chain"].map(lambda k: CHAINS[k].label if k in CHAINS else k),
    "Пул": shown["dex_url"],
    "DEX": shown["dex"],
    "Пара в пуле": shown["dex_pair_with"],
    "Ликвидность DEX, $": shown["dex_liquidity_usd"].round(0),
    "Объём DEX 24ч, $": shown["dex_volume_usd"].round(0),
    "Пулов": shown["dex_pools"],
    "Расхождение, %": shown["price_gap_pct"].round(2),
    "Спот на биржах": shown["spot_on"],
    "Контракт": shown["contract"],
})

st.dataframe(
    view, hide_index=True, height=640, **FULL,
    column_config={
        "Фьючерс": st.column_config.LinkColumn(display_text=r".*/([^/]+)/?$"),
        "Пул": st.column_config.LinkColumn(display_text="DexScreener"),
        "Объём фьюч 24ч, $": st.column_config.NumberColumn(format="%,.0f"),
        "Ликвидность DEX, $": st.column_config.NumberColumn(format="%,.0f"),
        "Объём DEX 24ч, $": st.column_config.NumberColumn(format="%,.0f"),
    },
)
st.download_button("Скачать CSV", view.to_csv(index=False).encode("utf-8-sig"),
                   "candidates.csv", "text/csv")

if meta and meta.get("errors"):
    with st.expander(f"Ошибки источников при сборе: {len(meta['errors'])}"):
        st.json(meta["errors"])
