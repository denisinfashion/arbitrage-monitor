"""Поиск арбитражных связок USDT -> ... -> USDT по истории.

Задача комбинаторно тяжёлая: при 80 активах и 4 ногах это 80^3 ≈ 512 тысяч
маршрутов, и каждый нужно посчитать в каждой точке истории (7 дней минутных
свечей — 10080 точек). Прямой перебор даёт 5 миллиардов операций.

Решение из двух шагов.

  Шаг 1 — поиск кандидатов. Маржа связки это произведение курсов, а в
  логарифмах — сумма. Значит лучшую цепочку можно искать алгеброй max-plus:
  «умножение матриц», где вместо суммы произведений берётся максимум сумм.
  Одно такое умножение сразу даёт лучший двухшаговый переход между всеми
  парами активов. Делается это на прореженной по времени выборке.

  Шаг 2 — точный расчёт. Кандидаты (обычно несколько сотен маршрутов)
  считаются уже по всей истории поточечно. Это дёшево: сотни маршрутов
  на десять тысяч точек — миллионы операций вместо миллиардов.

Такая схема даёт и ранжированную таблицу, и полный временной ряд по любой
выбранной связке для графика.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import SETTINGS
from .rates import NEG_INF, RateGrid

log = logging.getLogger(__name__)


@dataclass
class Cycle:
    """Одна связка и её поведение во времени."""

    assets: Tuple[str, ...]        # ('USDT', 'BNB', 'CAKE', 'USDT')
    log_margin: np.ndarray         # (T,) ln(итог/старт), NaN где данных нет
    venues: List[np.ndarray]       # по ноге: (T,) индексы площадок
    grid: RateGrid = field(repr=False)
    gas_usd: float = 0.0

    @property
    def legs(self) -> int:
        return len(self.assets) - 1

    @property
    def label(self) -> str:
        return " → ".join(self.assets)

    def margin_pct(self) -> np.ndarray:
        """Чистая маржа в процентах с учётом газа."""
        gross = np.exp(self.log_margin) - 1.0
        if self.gas_usd:
            gross = gross - self.gas_usd / self.grid.trade_size_usd
        return gross * 100.0

    def venue_names(self, t: int) -> List[str]:
        out = []
        for v in self.venues:
            i = int(v[t])
            out.append(self.grid.venues[i] if 0 <= i < len(self.grid.venues) else "—")
        return out

    def profitable_mask(self) -> np.ndarray:
        m = self.margin_pct()
        return np.isfinite(m) & (m > 0)

    def dominant_venues(self, only_profitable: bool = True) -> List[str]:
        """Площадка на каждой ноге, наиболее частая в прибыльные моменты.

        Считать по всему периоду нельзя: связка бывает в плюсе малую долю
        времени, и «самая частая площадка» тогда описывает те моменты,
        когда связка убыточна. Для двухногого цикла это давало откровенную
        бессмыслицу — одну и ту же биржу на покупку и на продажу.
        Поэтому по умолчанию берём только прибыльные точки, а если их нет —
        падаем на весь период.
        """
        mask = self.profitable_mask() if only_profitable else None
        if mask is not None and not mask.any():
            mask = None

        out = []
        for v in self.venues:
            vals = v[mask] if mask is not None else v
            vals = vals[vals >= 0]
            if len(vals) == 0:
                out.append("—")
                continue
            counts = np.bincount(vals.astype(np.int64), minlength=len(self.grid.venues))
            out.append(self.grid.venues[int(counts.argmax())])
        return out

    def venues_at_best(self) -> List[str]:
        """Площадки в точке максимальной маржи."""
        m = self.margin_pct()
        if not np.isfinite(m).any():
            return ["—"] * self.legs
        return self.venue_names(int(np.nanargmax(m)))

    # ---------------------------------------------------------------- сеть
    def chains(self) -> List[str]:
        """Сети, задействованные в связке, в порядке исполнения."""
        return [self.grid.venue_chain.get(v, "") for v in self.dominant_venues()]

    def single_chain(self) -> Optional[str]:
        """Единственная сеть связки или None, если их несколько."""
        uniq = {c for c in self.chains() if c}
        kinds = {self.grid.venue_kind.get(v, "") for v in self.dominant_venues()}
        if kinds == {"dex"} and len(uniq) == 1:
            return uniq.pop()
        return None

    def needs_transfer(self) -> bool:
        """Требует ли связка переводов между площадками или сетями.

        Обмен внутри одной сети на DEX — это последовательность свопов
        из одного кошелька: быстро и без вывода. Как только в цепочке
        появляется биржа или вторая сеть, между ногами возникает перевод:
        комиссия сети, время подтверждения, а иногда и заморозка вывода.
        Расчёт этого не учитывает, поэтому такие связки надо помечать.
        """
        return self.single_chain() is None

    # --------------------------------------------------------- ликвидность
    def leg_liquidity(self) -> List[Optional[float]]:
        """Ликвидность пула на каждой ноге, в долларах."""
        out = []
        for v, a, b in zip(self.dominant_venues(),
                           self.assets[:-1], self.assets[1:]):
            out.append(self.grid.pair_liquidity.get((v, a, b)))
        return out

    def bottleneck_liquidity(self) -> Optional[float]:
        """Самая мелкая ликвидность в цепочке — она и ограничивает объём."""
        vals = [x for x in self.leg_liquidity() if x]
        return min(vals) if vals else None

    # -------------------------------------------------------------- ссылки
    def leg_links(self) -> List[dict]:
        """Данные по каждой ноге для интерфейса: площадка, сеть, ссылки."""
        from .links import chain_name, pool_url, swap_url, token_name

        out = []
        venues = self.dominant_venues()
        liq = self.leg_liquidity()
        for i, (v, a, b) in enumerate(zip(venues, self.assets[:-1], self.assets[1:])):
            chain = self.grid.venue_chain.get(v, "")
            addr_a = self.grid.token_address.get((chain, a), "")
            addr_b = self.grid.token_address.get((chain, b), "")
            pool = self.grid.pair_pool.get((v, a, b), "")
            out.append({
                "n": i + 1,
                "from": a,
                "to": b,
                "venue": v,
                "kind": self.grid.venue_kind.get(v, ""),
                "chain": chain,
                "chain_name": chain_name(chain) if chain else "—",
                "liquidity": liq[i],
                "swap": swap_url(v, chain, addr_a, addr_b) if chain else None,
                "pool_page": pool_url(chain, pool) if chain and pool else None,
                "name_from": token_name(a, self.grid.token_name),
                "name_to": token_name(b, self.grid.token_name),
            })
        return out

    def first_swap_url(self) -> Optional[str]:
        """Ссылка на первый обмен — с него начинается исполнение."""
        legs = self.leg_links()
        return legs[0]["swap"] if legs else None

    def token_note(self) -> str:
        """Расшифровка тикеров маршрута."""
        from .links import describe_path
        return describe_path(self.assets, self.grid.token_name)

    def stats(self) -> dict:
        m = self.margin_pct()
        ok = np.isfinite(m)
        if not ok.any():
            return {"Связка": self.label, "Ног": self.legs, "Точек": 0}
        mv = m[ok]
        positive = mv > 0
        last = mv[-1] if ok[-1] else np.nan
        venues = self.dominant_venues()
        step = self.grid_step_min()
        liq = self.bottleneck_liquidity()
        chain = self.single_chain()

        from .links import chain_name

        # Площадки нумеруются по порядку исполнения обмена. Без номеров
        # список читается как перечисление, хотя это последовательность.
        route = " → ".join(f"{i + 1}·{v}" for i, v in enumerate(venues))

        return {
            "Связка": self.label,
            "Ног": self.legs,
            "Сейчас %": round(float(last), 3) if last == last else None,
            "Макс %": round(float(mv.max()), 3),
            "Медиана %": round(float(np.median(mv)), 3),
            "В плюсе %": round(float(positive.mean() * 100), 1),
            "Окон": int(_count_runs(positive)),
            "Окно макс, мин": int(_longest_run(positive) * step),
            "Окно средн, мин": int(round(_mean_run(positive) * step)),
            "Ликвидность $": round(liq) if liq else None,
            "Сеть": chain_name(chain) if chain else "несколько",
            "Переводы": "нет" if chain else "да",
            "Маршрут": route,
            "Данные %": round(float(ok.mean() * 100), 0),
            "Токены": self.token_note(),
            "Точек": int(ok.sum()),
        }

    def grid_step_min(self) -> float:
        if len(self.grid.times) < 2:
            return 1.0
        return float(self.grid.times[1] - self.grid.times[0]) / 60.0

    def to_frame(self) -> pd.DataFrame:
        """Временной ряд для графика."""
        df = pd.DataFrame({
            "Время": pd.to_datetime(self.grid.times, unit="s", utc=True),
            "Произведение курсов": np.exp(self.log_margin),
            "Маржа, %": self.margin_pct(),
        })
        for i, v in enumerate(self.venues):
            names = [self.grid.venues[j] if 0 <= j < len(self.grid.venues) else None
                     for j in v]
            df[f"Нога {i + 1}: {self.assets[i]}→{self.assets[i + 1]}"] = names
        return df


def _count_runs(mask: np.ndarray) -> int:
    """Число непрерывных участков True."""
    if mask.size == 0:
        return 0
    return int(np.sum(mask[1:] & ~mask[:-1]) + (1 if mask[0] else 0))


def _mean_run(mask: np.ndarray) -> float:
    """Средняя длина непрерывного участка True — типичное окно возможности."""
    runs, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# --------------------------------------------------------------------------
# Max-plus: поиск кандидатов
# --------------------------------------------------------------------------


def _maxplus(a: np.ndarray, b: np.ndarray, chunk: Optional[int] = None
             ) -> Tuple[np.ndarray, np.ndarray]:
    """«Умножение» матриц в полукольце max-plus, по оси времени пачками.

    c[t,i,k] = max_j (a[t,i,j] + b[t,j,k]),  arg[t,i,k] = argmax_j

    Смысл: если a — лучший курс за m шагов, а b — за один, то c —
    лучший курс за m+1 шаг. Так наращивается длина связки.

    Промежуточный тензор имеет размер chunk*n^3, поэтому размер пачки
    подбирается под число активов: держим примерно 64 МБ на шаг.
    """
    T, n, _ = a.shape
    if chunk is None:
        chunk = max(1, min(T, int(64e6 / max(1, n ** 3 * 4))))
    out = np.full((T, n, n), NEG_INF, dtype=np.float32)
    arg = np.full((T, n, n), -1, dtype=np.int16)

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        # (t, i, j, 1) + (t, 1, j, k) -> (t, i, j, k)
        cand = a[s:e, :, :, None] + b[s:e, None, :, :]
        np.nan_to_num(cand, copy=False, nan=NEG_INF, neginf=NEG_INF)
        out[s:e] = cand.max(axis=2)
        arg[s:e] = cand.argmax(axis=2).astype(np.int16)
        del cand
    return out, arg


def _candidate_paths(
    grid: RateGrid,
    anchor_idx: int,
    max_legs: int,
    subsample: int,
    per_step: int,
) -> List[Tuple[int, ...]]:
    """Собирает набор перспективных маршрутов на прореженной выборке."""
    W = grid.log_rate
    T = grid.n_times
    sel = np.arange(0, T, max(1, subsample))
    Ws = np.ascontiguousarray(W[sel])
    n = grid.n_assets

    found: set = set()

    # --- 2 ноги: USDT -> X -> USDT ---------------------------------------
    two = Ws[:, anchor_idx, :] + Ws[:, :, anchor_idx]
    _harvest_1d(two, per_step, lambda i, t: (anchor_idx, i, anchor_idx), found)

    if max_legs < 3:
        return sorted(found)

    # --- 3 ноги: USDT -> i -> j -> USDT ----------------------------------
    three = (Ws[:, anchor_idx, :, None] + Ws + Ws[:, None, :, anchor_idx])
    _harvest_2d(three, per_step,
                lambda i, j, t: (anchor_idx, i, j, anchor_idx), found)
    del three

    if max_legs < 4:
        return sorted(found)

    # --- 4+ ног: наращиваем через max-plus -------------------------------
    # M — лучший переход за (L-2) промежуточных шага между i и k.
    # chain_args хранит таблицы argmax каждого наращивания: по ним
    # восстанавливаются промежуточные узлы маршрута.
    M = Ws
    chain_args: List[np.ndarray] = []
    for _ in range(2, max_legs - 1):
        M, argM = _maxplus(M, Ws)
        chain_args.append(argM)
        total = (Ws[:, anchor_idx, :, None] + M + Ws[:, None, :, anchor_idx])
        _harvest_2d(
            total, per_step,
            lambda i, k, t, _args=list(chain_args): _reconstruct(anchor_idx, i, k, _args, t),
            found,
        )
        del total

    return sorted(found)


def _reconstruct(anchor: int, i: int, k: int, chain_args: Sequence[np.ndarray],
                 t: int) -> Tuple[int, ...]:
    """Восстанавливает промежуточные узлы маршрута из таблиц argmax.

    Последняя таблица даёт узел, разбивающий переход i..k на более
    короткий (i..mid) и одиночный шаг (mid..k); предыдущие таблицы
    раскрывают левую часть рекурсивно.

    t — момент времени, на котором маршрут был найден. Раньше здесь
    жёстко стоял ноль, из-за чего для кандидатов, найденных в середине
    истории, восстанавливались узлы от первой точки — маршрут получался
    не тот, что дал высокий балл.
    """
    if not chain_args:
        return (anchor, i, k, anchor)
    mid = int(chain_args[-1][t, i, k])
    if mid < 0:
        return (anchor, i, k, anchor)
    inner = _reconstruct(anchor, i, mid, chain_args[:-1], t)
    body = list(inner[1:-1]) + [k]
    return tuple([anchor] + body + [anchor])


def _harvest_1d(scores: np.ndarray, per_step: int, make, found: set) -> None:
    for t in range(scores.shape[0]):
        row = scores[t]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        k = min(per_step, int(finite.sum()))
        idx = np.argpartition(np.where(finite, row, NEG_INF), -k)[-k:]
        for i in idx:
            if np.isfinite(row[i]):
                p = make(int(i), t)
                if _valid(p):
                    found.add(p)


def _harvest_2d(scores: np.ndarray, per_step: int, make, found: set) -> None:
    T, n, _ = scores.shape
    for t in range(T):
        mat = scores[t]
        finite = np.isfinite(mat)
        if not finite.any():
            continue
        flat = np.where(finite, mat, NEG_INF).ravel()
        k = min(per_step, int(finite.sum()))
        idx = np.argpartition(flat, -k)[-k:]
        for f in idx:
            if not np.isfinite(flat[f]):
                continue
            i, j = divmod(int(f), n)
            p = make(i, j, t)
            if _valid(p):
                found.add(p)


def _valid(path: Tuple[int, ...]) -> bool:
    """Отбрасывает вырожденные маршруты: повтор актива внутри цикла."""
    body = path[1:-1]
    return len(body) == len(set(body)) and path[0] not in body


# --------------------------------------------------------------------------
# Точный расчёт по всей истории
# --------------------------------------------------------------------------


def evaluate_path(grid: RateGrid, path: Sequence[int], gas_usd: float = 0.0) -> Cycle:
    """Считает временной ряд маржи для одного маршрута по всей истории."""
    total = np.zeros(grid.n_times, dtype=np.float64)
    venues: List[np.ndarray] = []
    for a, b in zip(path[:-1], path[1:]):
        leg = grid.log_rate[:, a, b].astype(np.float64)
        total += leg
        venues.append(grid.venue_idx[:, a, b].copy())
    total[~np.isfinite(total)] = np.nan
    return Cycle(
        assets=tuple(grid.assets[i] for i in path),
        log_margin=total,
        venues=venues,
        grid=grid,
        gas_usd=gas_usd,
    )


def find_cycles(
    grid: RateGrid,
    *,
    anchor: Optional[str] = None,
    max_legs: Optional[int] = None,
    top: int = 50,
    subsample: Optional[int] = None,
    per_step: int = 12,
    gas_per_dex_leg_usd: float = 0.15,
    min_margin_pct: Optional[float] = None,
    max_margin_pct: Optional[float] = None,
    sort_by: str = "окна",
    settings=SETTINGS,
) -> Tuple[pd.DataFrame, List[Cycle]]:
    """Главная функция: находит и ранжирует связки.

    sort_by задаёт смысл слова «лучшая»:
      'окна'    — сначала те, что дольше всего были в плюсе, затем по максимуму.
                  Практичный выбор: связка с редким всплеском полезнее той,
                  что стабильно держится чуть ниже нуля;
      'максимум'— по лучшей марже за период;
      'медиана' — по типичной марже. Строгий критерий: почти всегда пусто,
                  потому что рынок эффективен большую часть времени;
      'сейчас'  — по марже в последней точке истории.

    Возвращает таблицу для отображения и список объектов Cycle,
    из которых строится график.
    """
    anchor = (anchor or settings.quote_asset).upper()
    max_legs = max_legs or settings.max_legs
    min_margin = settings.min_margin_pct if min_margin_pct is None else min_margin_pct

    try:
        a_idx = grid.asset_index(anchor)
    except KeyError:
        raise ValueError(
            f"актив {anchor} не найден. Доступны: {', '.join(grid.assets[:20])}…"
        )

    if subsample is None:
        # целимся примерно в 400 точек для этапа поиска кандидатов
        subsample = max(1, grid.n_times // 400)

    log.info("поиск кандидатов: до %d ног, прореживание 1/%d", max_legs, subsample)
    raw = _candidate_paths(grid, a_idx, max_legs, subsample, per_step)
    log.info("кандидатов: %d", len(raw))

    if not raw:
        return pd.DataFrame(), []

    # газ считаем только за ноги, проходящие через DEX
    dex_venues = {i for i, v in enumerate(grid.venues) if not _looks_like_cex(v)}

    cycles: List[Cycle] = []
    for path in raw:
        c = evaluate_path(grid, path, gas_usd=0.0)
        n_dex = sum(
            1 for v in c.venues
            if len(v) and int(np.bincount(v[v >= 0].astype(np.int64),
                                          minlength=len(grid.venues)).argmax()) in dex_venues
        ) if any(len(v) for v in c.venues) else 0
        c.gas_usd = gas_per_dex_leg_usd * n_dex
        cycles.append(c)

    rows = []
    for i, c in enumerate(cycles):
        s = c.stats()
        if s.get("Точек", 0) == 0:
            continue
        s["_i"] = i
        rows.append(s)

    if not rows:
        return pd.DataFrame(), cycles

    df = pd.DataFrame(rows)
    df = df[df["Макс %"] > min_margin]

    # Потолок правдоподобия. Трёхзначная маржа — это не находка, а неверная
    # цена: одноимённая подделка, налог на перевод, пул без сделок. Такие
    # строки не просто бесполезны, они вытесняют настоящие из верха таблицы.
    if max_margin_pct is not None and not df.empty:
        wild = df["Макс %"] > float(max_margin_pct)
        if wild.any():
            log.warning("отсеяно как недостоверное (маржа выше %.1f%%): %d — %s",
                        float(max_margin_pct), int(wild.sum()),
                        "; ".join(df.loc[wild, "Связка"].head(5)))
            df = df[~wild]

    if df.empty:
        return df, cycles

    sort_keys = {
        "окна": ["В плюсе %", "Макс %"],
        "максимум": ["Макс %", "В плюсе %"],
        "медиана": ["Медиана %", "В плюсе %"],
        "сейчас": ["Сейчас %", "Макс %"],
        "ликвидность": ["Ликвидность $", "Макс %"],
    }.get(sort_by, ["В плюсе %", "Макс %"])

    df = df.sort_values(sort_keys, ascending=False, na_position="last")
    df = df.head(top).reset_index(drop=True)
    order = df["_i"].tolist()
    return df.drop(columns=["_i"]), [cycles[i] for i in order]


def _looks_like_cex(venue: str) -> bool:
    from .config import CEX_TAKER_PCT
    return venue in CEX_TAKER_PCT
