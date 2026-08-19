"""Проверка конкретной связки: почему её нет в таблице.

Таблица результатов отвечает на вопрос «что нашлось». Эта страница
отвечает на обратный — «почему не нашлось вот это». Вводится цепочка
целиком, и по каждому плечу показано, что у нас есть: цена, её возраст,
площадка, ликвидность, комиссия и оценка проскальзывания. Внизу —
шаг, на котором связка выпала.

Отдельно здесь же живёт прямой запрос к источнику. Он нужен ровно
затем, зачем и вся страница: сравнить расчёт с тем, что происходит
на рынке прямо сейчас, а не десять минут назад.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history import diagnose as dg
from history import snapshot
from history.config import SETTINGS
from history.ui import FULL

try:
    from history.config import CODE_VERSION
except ImportError:
    CODE_VERSION = "неизвестна"

try:
    from history import live
except ImportError:      # модули под страницей старее самой страницы
    live = None

st.set_page_config(page_title="Разбор связки", layout="wide")
st.title("Разбор связки")
st.caption(f"Источник данных: {snapshot.source_label()} · версия {CODE_VERSION}")

st.markdown(
    "Введите цепочку так, как её исполняете: `USDT-AAVE-DAI-USDT`. "
    "Замыкание в USDT дописывается само."
)

col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
text = col_a.text_input("Цепочка", value="USDT-AAVE-DAI-USDT",
                        label_visibility="collapsed")
amount = col_b.number_input("Сумма, USDT", min_value=10.0, max_value=1_000_000.0,
                            value=float(SETTINGS.trade_size_usd), step=100.0)
window_h = col_c.number_input("Окно, ч", min_value=0.25, max_value=48.0,
                              value=6.0, step=0.25)
max_age_min = col_d.number_input(
    "Не старше, мин", min_value=1.0, max_value=240.0, value=15.0, step=5.0,
    help="Цены старше этого возраста в расчёт не идут. Если по плечу "
         "нет ничего свежее, оно будет показано с пометкой «цена "
         "устарела» — связка по таким ценам не существует.")

with st.sidebar:
    st.header("Модель глубины")
    st.caption(
        "Проскальзывание считается по глубине у текущей цены. Для V2 это "
        "половина TVL. Для V3 ликвидность стоит в узких диапазонах, и "
        "глубина в разы больше — из TVL её не вывести, поэтому множитель "
        "здесь можно подобрать под то, что реально получается на свопе."
    )
    v3 = st.number_input("Глубина V3, × от формулы V2", min_value=1.0,
                         max_value=100.0,
                         value=float(getattr(SETTINGS, "dex_v3_depth_multiple", 10.0)),
                         step=1.0)
    stable = st.number_input("Запас для пары стейблов, ×", min_value=1.0,
                             max_value=100.0,
                             value=float(getattr(SETTINGS, "dex_stable_depth_multiple", 5.0)),
                             step=1.0)
    SETTINGS.dex_v3_depth_multiple = v3
    SETTINGS.dex_stable_depth_multiple = stable

    st.divider()
    st.caption(
        "Если реальный своп прошёл лучше расчёта — множитель занижен. "
        "Если хуже — завышен. Это единственный честный способ его настроить: "
        "глубину V3 источник не отдаёт."
    )

col_s, col_k = st.columns([3, 2])
source = col_s.radio(
    "Откуда брать цены",
    ["История", "Свежий срез", "Спросить источник сейчас"],
    horizontal=True,
    help="История — накопленный снимок. Свежий срез — файл, который сборщик "
         "кладёт после каждого прохода. Прямой запрос идёт к источнику "
         "мимо снимка и сборщика и занимает секунд десять.",
)
where = col_k.radio(
    "Где исполняем",
    ["Только на DEX", "DEX и биржи"],
    horizontal=True,
    help="Связка в одном блоке на DEX и связка с переводом на биржу — "
         "разные вещи. Если смешать их в одном разборе, получится "
         "маршрут, которого не существует.",
)

go = st.button("Разобрать", type="primary", **FULL)

if not go and "diag_last" not in st.session_state:
    st.info("Введите цепочку и нажмите «Разобрать».")
    st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def _history(window_h: float, _bust: int) -> pd.DataFrame:
    since = int(time.time() - window_h * 3600)
    return snapshot.read_quotes(since_ts=since)


def _load(kind: str) -> tuple:
    """Котировки и подпись к ним."""
    if kind == "Спросить источник сейчас":
        if live is None:
            return pd.DataFrame(), "прямой запрос недоступен: обновите модули"
        with st.spinner("Спрашиваем цены у источника…"):
            df = live.fetch_now(SETTINGS)
        return df, f"прямой запрос, {len(df)} котировок, возраст — секунды"
    if kind == "Свежий срез":
        if live is None:
            return pd.DataFrame(), "быстрый канал недоступен: обновите модули"
        payload = live.read_live()
        df = live.as_frame(payload)
        age = live.age_seconds(payload)
        label = ("свежего среза ещё нет" if df.empty else
                 f"срез от сборщика, возраст {age / 60:.1f} мин, "
                 f"{len(df)} котировок")
        return df, label
    df = _history(window_h, int(time.time() // 60))
    return df, f"история за {window_h:g} ч, {len(df)} котировок"


if go:
    st.session_state["diag_last"] = (text, source, amount, window_h, where,
                                     max_age_min)

(text, source, amount, window_h, where,
 max_age_min) = st.session_state["diag_last"]
chain = dg.parse_chain(text)
if len(chain) < 3:
    st.error("Не разобрал цепочку. Пример: USDT-AAVE-DAI-USDT")
    st.stop()

quotes, label = _load(source)
st.caption(" → ".join(chain) + " · " + label)

if quotes.empty:
    st.warning("Котировок нет — разбирать нечего. "
               "Проверьте, что сбор данных проходит.")
    st.stop()

kinds = ["dex"] if where == "Только на DEX" else ["dex", "cex"]
rep = dg.diagnose(chain, quotes, trade_usd=amount, window_h=window_h,
                  venue_kinds=kinds, max_age_sec=max_age_min * 60)

stale = [l for l in rep.legs if "устарел" in (l.note or "")]
if stale:
    st.warning(
        "Свежих цен нет по " + str(len(stale)) + " из " + str(len(rep.legs))
        + " плеч, взяты последние доступные. Цифры ниже показывают, "
        "что было, а не что есть: спред живёт минуты. Нажмите "
        "«Спросить источник сейчас», чтобы посчитать по текущим ценам.",
        icon="⏳",
    )

# Раскрытый маршрут показываем до вердикта: иначе непонятно, откуда
# в таблице взялось плечо, которого человек не вводил.
if rep.routed:
    st.info(
        "Прямого рынка по некоторым плечам нет, обмен идёт через "
        "промежуточный актив — так же, как это делает маршрутизатор "
        "кошелька: **" + "**, **".join(rep.routed) + "**. Для кошелька это "
        "одна операция, для расчёта — две ноги с двумя комиссиями. "
        "Итоговая цепочка: " + " → ".join(rep.tickers) + ".",
        icon="🔀",
    )

# Вердикт первым делом: ради него страница и открывается.
if rep.ok:
    st.success(rep.verdict)
elif rep.stage in ("no_quotes", "not_aligned"):
    st.warning(rep.verdict)
else:
    st.error(rep.verdict)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Спот", "—" if rep.spot_pct != rep.spot_pct else f"{rep.spot_pct:+.2f}%",
          help="Разница по чистым ценам, до комиссий и проскальзывания.")
c2.metric("Комиссии", f"−{rep.fees_pct:.2f}%")
c3.metric("Проскальзывание", f"−{rep.slip_pct:.2f}%",
          help="Оценка по модели глубины. Настраивается в боковой панели.")
c4.metric("Итого", "—" if rep.net_pct != rep.net_pct else f"{rep.net_pct:+.2f}%")

st.subheader("Плечи")
st.dataframe(
    dg.legs_frame(rep),
    hide_index=True,
    column_config={
        "Курс": st.column_config.NumberColumn(format="%.8g"),
        "Ликвидность, $": st.column_config.NumberColumn(format="%.0f"),
        "Глубина, $": st.column_config.NumberColumn(
            format="%.0f",
            help="Сколько долларов стоит у текущей цены по нашей модели. "
                 "Именно от неё считается проскальзывание, а не от TVL."),
    },
    **FULL,
)

# Если плеча не хватает — вопрос уже не в расчёте, а в сборе.
if rep.missing:
    st.subheader("Чего не хватает")
    st.markdown(
        "По этим парам котировок в окне нет: **" + ", ".join(rep.missing) + "**.\n\n"
        "Прямой рынок по ним не найден, и обходного пути через "
        + ", ".join(dg.INTERMEDIATES[:3]) + " и другие крупные активы — тоже. "
        "Значит, пул либо не попал в справочник, либо отсеян отбором. "
        "Что именно — видно на странице «Токены под наблюдением»: там по "
        "каждому тикеру написано, собирается он, отсеян или отсутствует. "
        "Добавить токен принудительно можно в файле `watchlist.txt`."
    )

with st.expander("Как это считается"):
    st.markdown(
        f"""
Курс плеча берётся не из самой свежей котировки, а из самой выгодной:
среди всех площадок с этой парой выбирается та, где после комиссии
и проскальзывания получится больше. Направление не важно: если записано
`DAI/USDT`, для плеча `USDT → DAI` курс переворачивается.

Если прямого рынка по плечу нет, оно раскрывается через промежуточный
актив — обёрнутый BNB, стейбл, крупную монету. Так же поступает
маршрутизатор кошелька: пула AAVE/DAI может не существовать вовсе,
и обмен идёт AAVE → WBNB → DAI. Разница в том, что комиссию мы считаем
за оба перехода, а не за один.

Издержки на плечо складываются из двух частей.

**Комиссия пула** — своя у каждого пула, если источник сообщил уровень:
GeckoTerminal нередко пишет его прямо в имени, «WBNB / USDT 0.05%».
У V3 уровни 0.01 / 0.05 / 0.25 / 1 процента, то есть между крайними
разница в сто раз, и брать среднее по площадке нельзя: паре стейблов
на 0.01% мы приписывали 0.25%, а на трёх ногах это три четверти
процента выдуманных издержек. Когда уровень неизвестен, подставляется
типичный для площадки — это видно по круглому значению 0.25 или 0.3.

**Проскальзывание** — оценка, а не факт. Своп размера S через глубину D
даёт курс хуже спота в D/(D+S) раз. Весь вопрос в D. Для пула V2
это половина TVL и формула точна. Для V3 ликвидность стоит в диапазонах,
выбранных поставщиками, и у текущей цены её кратно больше; сколько
именно — источник не сообщает. Сейчас взято ×{getattr(SETTINGS, 'dex_v3_depth_multiple', 10):g}
для V3 и ещё ×{getattr(SETTINGS, 'dex_stable_depth_multiple', 5):g} сверху
для пары из двух стейблов.

Раньше множителей не было вовсе, и V3 считался по формуле V2. Своп на
$1000 через пул с TVL $100 000 давал −1.96% на плечо, три плеча — около
−6%. Связка на 1–2% не могла пройти в принципе, а всё, что могло бы,
отсекалось потолком правдоподобия в 5% как артефакт. Таблица оставалась
пустой при живых спредах — ровно та картина, которая и наблюдалась.
"""
    )
