"""Состояние сбора: что накоплено, насколько свежо, где ошибки.

Страница работает в двух режимах. Локально данные читаются из SQLite,
которую наполняет фоновый сборщик. В облаке локальной базы нет — данные
приходят готовым снимком, который собирает GitHub Actions. Какой режим
активен, определяет наличие ссылки на снимок в настройках.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history import snapshot, store
from history.config import DB_PATH, LOG_PATH, SETTINGS
from history.ui import FULL

st.set_page_config(page_title="Сбор данных", layout="wide")
st.title("Сбор котировок")

CLOUD = snapshot.cloud_mode()

st.caption(
    "Данные собирает GitHub Actions по расписанию, приложение их только читает."
    if CLOUD else
    "Сборщик работает отдельным процессом и пишет в общую базу. "
    "Интерфейс только читает — можно закрыть браузер, сбор продолжится."
)

# --------------------------------------------------------------------------
# Как запустить
# --------------------------------------------------------------------------

if not CLOUD:
    with st.expander("Как запустить сборщик", expanded=not store.db_exists()):
        st.markdown(
            f"""
Откройте отдельное окно терминала в папке `arb_calculator`:

```bash
pip install -r requirements.txt
python -m history.collector
```

Полезные ключи:

```bash
python -m history.collector --days 7 --timeframe 1m --pools 100
python -m history.collector --no-dex          # только биржи
python -m history.collector --once -v         # один проход, подробный вывод
```

История бирж выкачивается за минуты — ccxt отдаёт до 1000 свечей за запрос.
Глубина по DEX набирается постепенно: бесплатный лимит GeckoTerminal —
30 запросов в минуту, поэтому на {SETTINGS.dex_pool_limit} пулов уходит
порядка 30–40 минут. Текущие цены пулов при этом обновляются сразу.

База: `{DB_PATH}`
Журнал: `{LOG_PATH}`
"""
        )

c1, c2, _ = st.columns([1, 1, 4])
if c1.button("Обновить экран", **FULL):
    st.cache_data.clear()
    st.rerun()
auto = c2.toggle("Автообновление", value=False, help="перечитывать раз в 30 секунд")

# --------------------------------------------------------------------------
# Сводка
# --------------------------------------------------------------------------

stats = snapshot.stats()
st.caption(f"Источник: {snapshot.source_label()}")

if stats["rows"] == 0:
    if CLOUD:
        st.warning(
            "Снимок пока пуст или недоступен. Если публикация только что "
            "настроена — подождите первого прогона сборщика: вкладка Actions "
            "в репозитории, задача «Сбор котировок»."
        )
    else:
        st.warning("Данных нет. Запустите сборщик — команда выше.")
    st.stop()

m = st.columns(5)
m[0].metric("Строк котировок", f"{stats['rows']:,}".replace(",", " "))
m[1].metric("Площадок", stats["venues"])
m[2].metric("Пар", stats["pairs"])
m[3].metric("Размер данных", f"{stats['db_mb']} МБ")

if stats["t0"] and stats["t1"]:
    depth_h = (stats["t1"] - stats["t0"]) / 3600
    age_min = (time.time() - stats["t1"]) / 60
    m[4].metric("Глубина", f"{depth_h:.1f} ч",
                delta=f"свежесть {age_min:.0f} мин", delta_color="off")

    target_h = SETTINGS.history_days * 24
    st.progress(min(1.0, depth_h / target_h),
                text=f"Набрано {depth_h:.1f} ч из целевых {target_h:.0f} ч")

    # В облаке сбор идёт раз в 15 минут, плюс задержки очереди GitHub —
    # порог отставания должен это учитывать, иначе будет ложная тревога.
    limit = 45 if CLOUD else 15
    if age_min > limit:
        st.warning(
            f"Свежих данных нет уже {age_min:.0f} минут. "
            + ("Проверьте вкладку Actions — возможно, прогон упал "
               "или GitHub отключил расписание за неактивностью репозитория."
               if CLOUD else "Похоже, сборщик остановлен.")
        )

k = stats.get("by_kind", {})
if k:
    st.markdown(
        " · ".join(
            f"**{'Биржи' if kind == 'cex' else 'DEX'}**: "
            f"{v['rows']:,} строк, {v['venues']} площадок".replace(",", " ")
            for kind, v in k.items()
        )
    )

# --------------------------------------------------------------------------
# Покрытие
# --------------------------------------------------------------------------

st.subheader("Покрытие по площадкам")

cov = snapshot.coverage()
if not cov.empty:
    now = time.time()
    cov = cov.copy()
    cov["Глубина, ч"] = ((cov["t1"] - cov["t0"]) / 3600).round(1)
    cov["Отставание, мин"] = ((now - cov["t1"]) / 60).round(0)
    cov["Тип"] = cov["Тип"].map({"cex": "биржа", "dex": "DEX"}).fillna(cov["Тип"])
    st.dataframe(
        cov.drop(columns=["t0", "t1"]).style.background_gradient(
            subset=["Свечей"], cmap="Greens"),
        **FULL, hide_index=True,
    )

# --------------------------------------------------------------------------
# Пулы
# --------------------------------------------------------------------------

pools = snapshot.pools(SETTINGS.chain)
if not pools.empty:
    st.subheader(f"Пулы {SETTINGS.chain.upper()} под наблюдением ({len(pools)})")
    cols = [c for c in ["dex", "base", "quote", "reserve_usd", "volume_24h",
                        "fee_pct", "pool"] if c in pools.columns]
    show = pools[cols].copy()
    show.columns = [{"dex": "DEX", "base": "База", "quote": "Котировка",
                     "reserve_usd": "Резерв, $", "volume_24h": "Оборот 24ч, $",
                     "fee_pct": "Комиссия, %", "pool": "Адрес"}[c] for c in cols]
    st.dataframe(show, **FULL, hide_index=True, column_config={
        "Резерв, $": st.column_config.NumberColumn(format="%.0f"),
        "Оборот 24ч, $": st.column_config.NumberColumn(format="%.0f"),
    })

# --------------------------------------------------------------------------
# Ошибки и журнал — только для локального режима
# --------------------------------------------------------------------------

if not CLOUD:
    state = store.read_state()
    if not state.empty:
        bad = state[state["ok"] == 0]
        st.subheader("Состояние источников")
        if bad.empty:
            st.success("Ошибок нет.")
        else:
            st.warning(f"Источников с ошибкой: {len(bad)}")
            show = bad[["source", "key", "error", "last_run_dt"]].copy()
            show.columns = ["Источник", "Ключ", "Ошибка", "Последняя попытка"]
            st.dataframe(show.head(50), **FULL, hide_index=True)

        with st.expander("Все источники"):
            show = state[["source", "key", "rows_total", "last_ts_dt",
                          "last_run_dt", "ok"]].copy()
            show.columns = ["Источник", "Ключ", "Строк", "Последняя свеча",
                            "Последний запуск", "ОК"]
            st.dataframe(show, **FULL, hide_index=True)

    if LOG_PATH.exists():
        with st.expander("Журнал сборщика (последние 80 строк)"):
            try:
                lines = LOG_PATH.read_text(encoding="utf-8",
                                           errors="replace").splitlines()
                st.code("\n".join(lines[-80:]), language="log")
            except OSError as exc:
                st.caption(f"журнал недоступен: {exc}")
else:
    st.info(
        "Журнал сбора и ошибки источников смотрите на вкладке **Actions** "
        "в репозитории проекта — там лог каждого прогона."
    )

if auto:
    time.sleep(30)
    st.rerun()
