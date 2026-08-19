"""Отсев недостоверных данных DEX.

Появился после конкретного случая: оповещения ушли со связкой
«USDT → BNB → MARSCOIN → USDT, +578%». Маржа такого размера — это не
находка, а признак того, что одна из цен неверна. Реальный арбитраж на
ликвидных парах живёт в долях процента; всё, что измеряется сотнями
процентов, объясняется данными, а не рынком.

Три механизма, из-за которых цена оказывается неверной.

**Один тикер — разные токены.** Сетка курсов адресуется по символу:
узел «CAKE» один. В сети BNB создать токен с любым символом стоит
копейки, и на верхних строчках по обороту регулярно оказываются
одноимённые подделки. Если пул A торгует настоящий CAKE, а пул B —
одноимённую пустышку, алгоритм считает их одним активом и видит между
ними «разницу цен», которой не существует: купить в одном и продать
в другом невозможно, это разные контракты.

**Накрученный резерв.** Отбор по reserve_in_usd не защищает: долларовая
оценка пула считается по цене самого токена, а цену задаёт тот же пул.
Достаточно завести в пул много собственного токена по выдуманному курсу,
и пул выглядит на миллион. Оборот подделать сложнее — он требует
встречных сделок, — поэтому вторым фильтром идёт объём за сутки.

**Потолок правдоподобия.** Даже когда токен настоящий и пул живой,
трёхзначная маржа означает, что войти в позицию не выйдет: либо в токене
есть налог на перевод, либо продажа заблокирована, либо котировка
пришла из пула, где никто не торгует. Значение потолка — не истина,
а граница между «стоит посмотреть» и «почти наверняка артефакт».

Модуль ничего не удаляет из базы: он только сообщает, какие пулы и
символы не стоит пускать в расчёт. Сырые данные остаются на месте,
и порог всегда можно опустить.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

log = logging.getLogger(__name__)


MAX_PLAUSIBLE_MARGIN_PCT = 5.0
"""Выше этого связка считается артефактом данных, а не возможностью.

Ориентир: на паре с реальной глубиной конкуренция ботов съедает спред
за секунды, и остаток измеряется десятыми долями процента. Пять
процентов оставлены с запасом на редкие всплески волатильности.
"""

MIN_POOL_VOLUME_USD = 500.0
"""Абсолютный минимум оборота за сутки. Ниже — пул мёртв, а не мал."""

MIN_POOL_TURNOVER = 0.02
"""Минимальный оборот за сутки в долях от резерва.

Абсолютный порог здесь не работает, и это выяснилось на живых данных.
Пул AAVE/USDT с резервом $12 000 и оборотом $4 700 отсеивался порогом
в $25 000, хотя оборачивается за сутки на 39% — это здоровый рабочий
пул, просто небольшой. А подделка с накрученным резервом в $900 000
и оборотом в $90 порог по величине оборота прошла бы легко.

Различает их не размер оборота, а его отношение к резерву. Настоящий
пул оборачивается за сутки на проценты и десятки процентов; пул,
надутый собственным токеном по выдуманной цене, — на сотые доли.
"""

NEVER_DROP = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "BNB", "WBNB",
              "ETH", "WETH", "BTC", "BTCB", "WBTC"}
"""Опорные активы. Отсев не должен оставить расчёт без якоря."""


# Канонические адреса основных токенов. Список нужен только для разрешения
# спора: когда один символ встречается с несколькими адресами, настоящим
# считается этот. Если символа здесь нет — спор решается по обороту, и
# отсутствие записи ничего не ломает.
CANONICAL: Dict[str, Dict[str, str]] = {
    "bsc": {
        "USDT": "0x55d398326f99059ff775485246999027b3197955",
        "USDC": "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",
        "BUSD": "0xe9e7cea3dedca5984780bafc599bd69add087d56",
        "FDUSD": "0xc5f0f7b66764f6ec8c8dff7ba683102295e16409",
        "DAI": "0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3",
        # Binance-Peg Aave Token. Добавлен после разбора связки
        # USDT - AAVE - DAI - USDT: тикер AAVE в BNB Chain носит
        # не только он, и подделка с тем же тикером — самый дешёвый
        # способ показать спред, которого нет.
        "AAVE": "0xfb6115445bff7b52feb98650c87f44907e58f802",
        "WBNB": "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c",
        "BNB": "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c",
        "BTCB": "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c",
        "ETH": "0x2170ed0880ac9a755fd29b2688956bd959f933f8",
        "CAKE": "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82",
        "XRP": "0x1d2f0da169ceb9fc7b3144628db156f3f6c60dbe",
        "ADA": "0x3ee2200efb3400fabb9aacf31297cbdd1d435d47",
        "DOGE": "0xba2ae424d960c26247dd6c32edc70b295c744c43",
        "LINK": "0xf8a0bf9cf54bb92f17374d9e9a321e6a111a51bd",
        "TWT": "0x4b0f1812e5df2a09796481ff14017e6005508003",
    },
}


@dataclass
class Screen:
    """Результат проверки справочника пулов."""

    bad_pools: Set[str] = field(default_factory=set)
    """Адреса пулов, которые не следует пускать в расчёт."""

    notes: Dict[str, str] = field(default_factory=dict)
    """Символ -> человеческая причина, почему он отсеян полностью."""

    address: Dict[str, str] = field(default_factory=dict)
    """Символ -> адрес токена, признанный настоящим."""

    name: Dict[str, str] = field(default_factory=dict)
    """Символ -> полное имя токена."""

    def dropped_symbols(self) -> Set[str]:
        return set(self.notes)

    def summary(self) -> str:
        if not self.notes and not self.bad_pools:
            return "подозрительного не найдено"
        parts = []
        if self.notes:
            parts.append(f"символов отсеяно: {len(self.notes)}")
        if self.bad_pools:
            parts.append(f"пулов отсеяно: {len(self.bad_pools)}")
        return ", ".join(parts)


def _missing(value) -> bool:
    """Пропуск любого вида: None, NaN или pd.NA.

    Проверять истинность напрямую нельзя: из Parquet приходит pd.NA,
    у которого нет булева значения, и `if value` падает с TypeError.
    """
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _norm_addr(value) -> str:
    if _missing(value):
        return ""
    s = str(value).strip().lower()
    return "" if s in ("", "none", "nan", "<na>") else s


def screen_pools(pools: Optional[pd.DataFrame],
                 min_volume_usd: float = MIN_POOL_VOLUME_USD,
                 turnover: float = MIN_POOL_TURNOVER) -> Screen:
    """Проверяет справочник пулов и решает, чему можно верить.

    Ожидаемые колонки: chain, pool, base, quote, base_addr, quote_addr,
    base_name, quote_name, reserve_usd, volume_24h. Недостающие колонки
    не считаются ошибкой — соответствующая проверка просто не работает.
    """
    out = Screen()
    if pools is None or getattr(pools, "empty", True):
        return out

    df = pools.copy()
    for col in ("chain", "pool", "base", "quote", "base_addr", "quote_addr",
                "base_name", "quote_name"):
        if col not in df.columns:
            df[col] = ""
    for col in ("reserve_usd", "volume_24h"):
        if col not in df.columns:
            df[col] = float("nan")

    df["base_addr"] = df["base_addr"].map(_norm_addr)
    df["quote_addr"] = df["quote_addr"].map(_norm_addr)

    # --- 1. Тихие пулы ----------------------------------------------------
    # Оборот подделать дороже, чем резерв: он требует встречных сделок.
    # Пул без оборота цену не задаёт, каким бы большим он ни выглядел.
    #
    # Сравнивается не величина оборота, а его отношение к резерву:
    # маленький пул с высокой оборачиваемостью живой, огромный с нулевой —
    # нет. Абсолютный порог остаётся только как отсечка совсем мёртвых.
    if min_volume_usd > 0 or turnover > 0:
        vol = pd.to_numeric(df["volume_24h"], errors="coerce")
        res = pd.to_numeric(df["reserve_usd"], errors="coerce")
        # NaN означает «оборот неизвестен» — такие пулы не наказываем,
        # иначе на старом снимке отсеется всё сразу.
        known = vol.notna()
        too_small = known & (vol < min_volume_usd)
        too_slow = known & res.notna() & (res > 0) & (vol < res * turnover)
        quiet = too_small | too_slow
        if quiet.any():
            out.bad_pools |= set(df.loc[quiet, "pool"].astype(str))
            log.info("тихих пулов: %d (оборот меньше $%s или меньше %.0f%% "
                     "от резерва за сутки)", int(quiet.sum()),
                     f"{min_volume_usd:,.0f}".replace(",", " "), turnover * 100)
        df = df[~quiet]

    if df.empty:
        return out

    # --- 2. Один символ — несколько контрактов ----------------------------
    # Разворачиваем пулы в строки «символ, адрес, вес», где вес — то,
    # по чему решается спор. Оборот важнее резерва ровно по причине из
    # пункта 1, поэтому он идёт первым.
    sides = []
    for sym_col, addr_col, name_col in (("base", "base_addr", "base_name"),
                                        ("quote", "quote_addr", "quote_name")):
        part = df[[sym_col, addr_col, name_col, "chain", "pool",
                   "reserve_usd", "volume_24h"]].copy()
        part.columns = ["symbol", "addr", "name", "chain", "pool",
                        "reserve_usd", "volume_24h"]
        sides.append(part)
    tokens = pd.concat(sides, ignore_index=True)
    tokens = tokens[(tokens["symbol"].astype(str) != "") & (tokens["addr"] != "")]
    if tokens.empty:
        return out

    tokens["volume_24h"] = pd.to_numeric(tokens["volume_24h"], errors="coerce").fillna(0.0)
    tokens["reserve_usd"] = pd.to_numeric(tokens["reserve_usd"], errors="coerce").fillna(0.0)

    for symbol, grp in tokens.groupby("symbol", sort=False):
        addrs = grp["addr"].unique()
        chain = str(grp["chain"].iloc[0] or "")
        canon = CANONICAL.get(chain, {}).get(str(symbol).upper(), "")

        if len(addrs) == 1:
            out.address[str(symbol)] = str(addrs[0])
            _remember_name(out, symbol, grp)
            # Известный символ по неизвестному адресу — повод не для отсева,
            # а для записи в заметки: список канонических адресов неполон
            # и устаревает, ошибаться в эту сторону дороже, чем промолчать.
            continue

        # Спор. Настоящим считаем канонический адрес, а если его нет
        # в списке — тот, за которым больше оборота.
        weight = (grp.groupby("addr")[["volume_24h", "reserve_usd"]]
                  .sum().sort_values(["volume_24h", "reserve_usd"], ascending=False))
        winner = canon if canon in set(addrs) else str(weight.index[0])
        losers = [a for a in addrs if a != winner]

        # Пулы проигравших адресов выкидываем: их цена относится к другому
        # контракту, и «разница» с настоящим токеном неисполнима.
        drop = set(grp.loc[grp["addr"].isin(losers), "pool"].astype(str))
        out.bad_pools |= drop
        out.address[str(symbol)] = winner
        _remember_name(out, symbol, grp[grp["addr"] == winner])

        log.warning("символ %s встречается с %d разными контрактами — "
                    "оставлен %s…%s, отсеяно пулов: %d",
                    symbol, len(addrs), winner[:6], winner[-4:], len(drop))

    # --- 3. Символы, от которых ничего не осталось ------------------------
    # Считаем по исходному справочнику, а не по отфильтрованному: символ,
    # все пулы которого оказались тихими, до второго шага просто не дошёл.
    _note_orphans(out, pools, min_volume_usd, turnover)
    return out


def _note_orphans(out: Screen, pools: pd.DataFrame, min_volume_usd: float,
                  turnover: float = MIN_POOL_TURNOVER) -> None:
    """Заполняет причины для символов, у которых не осталось ни одного пула."""
    have = {"base", "quote", "pool"} <= set(pools.columns)
    if not have:
        return
    alive: Dict[str, int] = {}
    total: Dict[str, int] = {}
    vol_col = "volume_24h" if "volume_24h" in pools.columns else None
    quiet_only: Dict[str, bool] = {}

    for r in pools[["base", "quote", "pool"] +
                   ([vol_col] if vol_col else [])].to_dict("records"):
        pool = str(r.get("pool") or "")
        dead = pool in out.bad_pools
        for sym in (r.get("base"), r.get("quote")):
            sym = str(sym or "")
            if not sym:
                continue
            total[sym] = total.get(sym, 0) + 1
            if not dead:
                alive[sym] = alive.get(sym, 0) + 1
            elif vol_col:
                v = r.get(vol_col)
                is_quiet = False
                if not _missing(v):
                    try:
                        is_quiet = float(v) < min_volume_usd
                    except (TypeError, ValueError):
                        is_quiet = False
                quiet_only[sym] = quiet_only.get(sym, True) and is_quiet

    # Опорные активы не отсеиваются никогда. Если убрать USDT из-за того,
    # что в снимке не оказалось оборота, расчёт останется без якоря и
    # выдаст пустоту вместо понятной ошибки.
    protected = set(NEVER_DROP)
    for table in CANONICAL.values():
        protected |= set(table)

    for sym, n in total.items():
        if alive.get(sym, 0) or sym.upper() in protected:
            continue
        if quiet_only.get(sym, False):
            out.notes[sym] = f"нет пулов с оборотом (проверено {n})"
        else:
            out.notes[sym] = f"тикер занят чужим контрактом (пулов {n})"


def _remember_name(out: Screen, symbol, grp) -> None:
    names = [str(x).strip() for x in grp["name"].tolist() if not _missing(x)]
    names = [n for n in names if n and n.lower() not in ("nan", "none", "<na>")]
    if names:
        out.name.setdefault(str(symbol), names[0])


def implausible(margin_pct: float,
                ceiling: float = MAX_PLAUSIBLE_MARGIN_PCT) -> bool:
    """Маржа выше потолка — почти наверняка артефакт данных."""
    try:
        return float(margin_pct) > float(ceiling)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Доверие к токену
# --------------------------------------------------------------------------


def credible_assets(grid) -> Set[str]:
    """Тикеры, которым можно верить настолько, чтобы будить человека ночью.

    Повод: в чат ушло оповещение о связке USDT → SPCXB → MARSCOIN → USDT
    с маржой +0.363%, а на деле она отрицательная. Ни один из наших
    фильтров тут не помог, и не мог: пулы крупные, оборот есть, маржа
    скромная и правдоподобная, цепочка в одной сети. Всё сходится.

    Не сходится другое — сами токены. У монет такого рода почти всегда
    есть налог на перевод: контракт удерживает три-десять процентов при
    покупке или продаже. В цене пула этого не видно вовсе, и никакой
    источник котировок про это не расскажет. Расчёт по ценам получается
    честным, а сделка — убыточной.

    Проверить наличие налога, не читая контракт, нельзя. Зато есть
    хороший косвенный признак: токен, прошедший листинг на бирже.
    Биржа проверяет контракт до листинга, и токен с налогом на перевод
    туда не попадает — он ломает биржевые кошельки. Поэтому «торгуется
    хотя бы на одной бирже» здесь означает не ликвидность, а то, что
    контракт кто-то читал.

    Список наблюдения тоже считается достаточным основанием: раз токен
    вписали руками, человек берёт риск на себя сознательно.
    """
    known: Set[str] = set(NEVER_DROP)
    for table in CANONICAL.values():
        known.update(table.keys())

    for (venue, a, b) in getattr(grid, "pair_liquidity", {}):
        if grid.venue_kind.get(venue) == "cex":
            known.add(a)
            known.add(b)
    for (venue, a, b) in getattr(grid, "pair_volume_usd", {}):
        if grid.venue_kind.get(venue) == "cex":
            known.add(a)
            known.add(b)

    try:
        from .config import SETTINGS, load_watchlist
        known.update(w.upper() for w in load_watchlist(SETTINGS))
    except Exception:  # noqa: BLE001 — список наблюдения необязателен
        pass
    return known


def exotic_in(cycle, known: Optional[Set[str]] = None) -> List[str]:
    """Активы связки, которых нет ни на одной бирже и ни в одном списке."""
    if known is None:
        known = credible_assets(cycle.grid)
    seen, out = set(), []
    for a in getattr(cycle, "assets", ()):  # noqa: B007
        a = str(a).upper()
        if a in known or a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out
