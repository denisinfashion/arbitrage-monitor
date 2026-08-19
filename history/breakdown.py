"""Разбор связки по шагам: сколько чего получится на каждом обмене.

Таблица связок отвечает на вопрос «где есть маржа». Этот модуль отвечает
на другой, не менее важный: **откуда она берётся и что я увижу в кошельке
на каждом шаге**. Без него связка остаётся числом, которое нечем
перепроверить перед сделкой.

Что считается.

**Цепочка сумм.** Берём стартовую сумму и ведём её через все обмены:
1000 USDT → 1.6612 BNB → 486.31 CAKE → 1004.24 USDT. Эти числа точные —
они получены из тех же курсов, по которым строилась таблица, а не
пересчитаны заново по другой формуле. Их и надо сверять с тем, что
показывает кошелёк перед подтверждением обмена.

**Разложение издержек.** Исполнимый курс = спот × (1 − комиссия) ×
проскальзывание. Комиссия площадки известна точно, проскальзывание для
пула считается по резерву, для биржи оценивается от оборота. Отсюда
восстанавливается спот — курс без издержек — и видно, сколько именно
съедает каждая нога.

**Где создаётся маржа.** Стоимость позиции на каждом шаге пересчитывается
в стартовый актив по срединному курсу. Срединный курс берётся как
корень из отношения прямого и обратного исполнимых курсов: у прямого
издержки в знаменателе, у обратного в числителе, и при делении они
сокращаются. Прибавка и убыль по каждой ноге в сумме дают ровно итоговую
маржу — это не оценка, а тождество: произведение отношений телескопически
сворачивается в отношение конца к началу.

Отсюда сразу видно то, ради чего разбор и нужен: маржа почти никогда
не размазана по цепочке. Обычно одна нога даёт плюс, а остальные —
минус на комиссиях, и вопрос только в том, перевешивает ли первое второе.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .rates import cex_slippage_factor, dex_slippage_factor, venue_fee_pct


@dataclass
class Step:
    """Одна нога обмена со всеми числами, которые можно проверить."""

    n: int
    asset_in: str
    asset_out: str
    venue: str
    kind: str
    chain: str

    amount_in: float
    amount_out: float

    spot_rate: float          # курс без издержек
    exec_rate: float          # курс, по которому реально исполнится
    fee_pct: float            # комиссия площадки, %
    slippage_pct: float       # проскальзывание, %

    liquidity_usd: Optional[float]
    volume_usd: Optional[float]

    value_in: float           # стоимость позиции до обмена, в стартовом активе
    value_out: float          # стоимость позиции после обмена
    pnl_pct: float            # вклад ноги в итог, %

    swap_url: Optional[str] = None
    pool_url: Optional[str] = None
    note_in: str = ""
    note_out: str = ""

    @property
    def cost_pct(self) -> float:
        """Суммарные издержки ноги в процентах от суммы."""
        return (1.0 - (1.0 - self.fee_pct / 100.0)
                * (1.0 - self.slippage_pct / 100.0)) * 100.0


@dataclass
class Breakdown:
    """Полный разбор одной связки в один момент времени."""

    cycle: object = field(repr=False)
    ts: int = 0
    amount_in: float = 0.0
    amount_out: float = 0.0
    gas_usd: float = 0.0
    steps: List[Step] = field(default_factory=list)
    exact: bool = True
    """Совпадает ли объём с тем, под который строилась сетка.

    При другом объёме проскальзывание пересчитывается, и числа перестают
    в точности совпадать с таблицей — это ожидаемо и в интерфейсе оговорено.
    """

    @property
    def anchor(self) -> str:
        return self.cycle.assets[0]

    @property
    def gross_pct(self) -> float:
        """Маржа до вычета газа."""
        if not self.amount_in:
            return float("nan")
        return (self.amount_out / self.amount_in - 1.0) * 100.0

    @property
    def net_pct(self) -> float:
        """Маржа после газа."""
        if not self.amount_in:
            return float("nan")
        return ((self.amount_out - self.gas_usd) / self.amount_in - 1.0) * 100.0

    @property
    def profit(self) -> float:
        return self.amount_out - self.gas_usd - self.amount_in

    def total_fees(self) -> float:
        """Сколько всего съели комиссии, в стартовом активе."""
        return sum(s.value_in * s.fee_pct / 100.0 for s in self.steps)

    def total_slippage(self) -> float:
        return sum(s.value_in * s.slippage_pct / 100.0 for s in self.steps)

    def best_leg(self) -> Optional[Step]:
        """Нога, на которой создаётся маржа."""
        return max(self.steps, key=lambda s: s.pnl_pct) if self.steps else None

    def to_frame(self):
        import pandas as pd
        rows = []
        for s in self.steps:
            rows.append({
                "№": s.n,
                "Обмен": f"{s.asset_in} → {s.asset_out}",
                "Площадка": s.venue,
                "Отдаём": round(s.amount_in, _digits(s.amount_in)),
                "Курс": _round_rate(s.exec_rate),
                "Получаем": round(s.amount_out, _digits(s.amount_out)),
                "Комиссия %": round(s.fee_pct, 3),
                "Проскальз. %": round(s.slippage_pct, 3),
                "Вклад %": round(s.pnl_pct, 3),
                f"Стоимость, {self.anchor}": round(s.value_out, 2),
            })
        return pd.DataFrame(rows)

    def as_text(self) -> str:
        """Разбор в виде текста — чтобы скопировать и держать перед глазами."""
        a = self.anchor
        out = [f"{self.cycle.label}   объём {_fmt(self.amount_in)} {a}"]
        for s in self.steps:
            out.append(
                f"{s.n}. {_fmt(s.amount_in)} {s.asset_in} → "
                f"{_fmt(s.amount_out)} {s.asset_out}   "
                f"курс {_round_rate(s.exec_rate)} · {s.venue} · "
                f"издержки {s.cost_pct:.2f}% · вклад {s.pnl_pct:+.3f}%"
            )
        out.append(
            f"Итог: {_fmt(self.amount_out)} {a}"
            + (f" − газ {self.gas_usd:.2f}" if self.gas_usd else "")
            + f" = {self.net_pct:+.3f}% ({self.profit:+.2f} {a})"
        )
        return "\n".join(out)


def _digits(x: float) -> int:
    """Сколько знаков после запятой показывать, чтобы не терять смысл."""
    if not x or not math.isfinite(x):
        return 2
    ax = abs(x)
    if ax >= 1000:
        return 2
    if ax >= 1:
        return 4
    if ax >= 0.001:
        return 6
    return 10


def _round_rate(r: float) -> float:
    return round(r, _digits(r)) if math.isfinite(r) else float("nan")


def _fmt(x: float) -> str:
    return f"{x:,.{_digits(x)}f}".replace(",", " ")


def mid_rate(grid, t: int, asset: str, anchor: str) -> float:
    """Срединный курс актива к стартовому, без издержек.

    Прямой и обратный исполнимые курсы несут издержки в противоположных
    направлениях: f = m·c₁, b = (1/m)·c₂. Отношение f/b равно m² с
    точностью до различия площадок, поэтому корень из него — оценка
    чистого курса, не требующая знать издержки.
    """
    if asset == anchor:
        return 1.0
    try:
        i, j = grid.asset_index(asset), grid.asset_index(anchor)
    except KeyError:
        return float("nan")
    f = float(np.exp(grid.log_rate[t, i, j]))
    b = float(np.exp(grid.log_rate[t, j, i]))
    if f > 0 and b > 0 and math.isfinite(f) and math.isfinite(b):
        return math.sqrt(f / b)
    if f > 0 and math.isfinite(f):
        return f          # обратного курса нет — берём прямой как есть
    return float("nan")


def pick_moment(cycle, prefer: str = "сейчас") -> Optional[int]:
    """Индекс момента, по которому строить разбор."""
    m = cycle.margin_pct()
    ok = np.isfinite(m)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    if prefer == "лучший":
        return int(idx[int(np.argmax(m[idx]))])
    return int(idx[-1])


def explain(cycle, t: Optional[int] = None, amount: Optional[float] = None,
            prefer: str = "сейчас") -> Optional[Breakdown]:
    """Считает разбор связки по шагам.

    t — индекс момента в сетке; если не задан, берётся по prefer.
    amount — стартовая сумма; по умолчанию та, под которую строилась сетка.
    """
    grid = cycle.grid
    if t is None:
        t = pick_moment(cycle, prefer)
    if t is None:
        return None

    base_size = float(grid.trade_size_usd)
    amount = float(amount) if amount else base_size
    anchor = cycle.assets[0]

    n_legs = len(cycle.assets) - 1
    dex_legs = 0
    steps: List[Step] = []
    cur = amount

    from .links import chain_name, pool_url, swap_url, token_name  # noqa: F401

    for i in range(n_legs):
        a, b = cycle.assets[i], cycle.assets[i + 1]
        ia, ib = grid.asset_index(a), grid.asset_index(b)
        lr = float(grid.log_rate[t, ia, ib])
        if not math.isfinite(lr):
            return None
        exec_rate = math.exp(lr)

        vi = int(cycle.venues[i][t]) if len(cycle.venues[i]) > t else -1
        venue = grid.venues[vi] if 0 <= vi < len(grid.venues) else "?"
        kind = grid.venue_kind.get(venue, "")
        chain = grid.venue_chain.get(venue, "")
        liq = grid.pair_liquidity.get((venue, a, b))
        vol = grid.pair_volume_usd.get((venue, a, b))

        # Восстанавливаем спот: сетка хранит уже исполнимый курс, а разложить
        # его на составляющие нужно по тем же формулам, по которым он и
        # получен, иначе разбор будет описывать не то, что посчитано.
        # Комиссия своя у пула, если источник её сообщил. Берётся через
        # сетку, чтобы разбор и расчёт не разошлись: сетка считала
        # исполнимый курс ровно по этому же числу.
        fee_pct = (grid.fee_for(venue, kind, a, b)
                   if hasattr(grid, "fee_for") else venue_fee_pct(venue, kind))
        fee_mult = 1.0 - fee_pct / 100.0
        slip_grid = (dex_slippage_factor(base_size, liq, venue=venue,
                                         base=a, quote=b)
                     if kind == "dex" else cex_slippage_factor(base_size, vol))
        denom = fee_mult * slip_grid
        spot = exec_rate / denom if denom > 0 else exec_rate

        # Под другой объём проскальзывание другое: в этом весь смысл вопроса
        # «а что будет на ста долларах вместо тысячи».
        if abs(amount - base_size) > 1e-9:
            value_usd = cur * mid_rate(grid, t, a, anchor)
            if not math.isfinite(value_usd) or value_usd <= 0:
                value_usd = amount
            slip = (dex_slippage_factor(value_usd, liq, venue=venue,
                                        base=a, quote=b)
                    if kind == "dex" else cex_slippage_factor(value_usd, vol))
            exec_rate = spot * fee_mult * slip
        else:
            slip = slip_grid

        amount_out = cur * exec_rate
        mid_a = mid_rate(grid, t, a, anchor)
        mid_b = mid_rate(grid, t, b, anchor)
        value_in = cur * mid_a
        value_out = amount_out * mid_b
        pnl = (value_out / value_in - 1.0) * 100.0 if value_in else float("nan")

        if kind == "dex":
            dex_legs += 1

        addr_a = grid.token_address.get((chain, a), "")
        addr_b = grid.token_address.get((chain, b), "")
        pool = grid.pair_pool.get((venue, a, b), "")

        steps.append(Step(
            n=i + 1, asset_in=a, asset_out=b, venue=venue, kind=kind,
            chain=chain, amount_in=cur, amount_out=amount_out,
            spot_rate=spot, exec_rate=exec_rate, fee_pct=fee_pct,
            slippage_pct=(1.0 - slip) * 100.0,
            liquidity_usd=liq, volume_usd=vol,
            value_in=value_in, value_out=value_out, pnl_pct=pnl,
            swap_url=swap_url(venue, chain, addr_a, addr_b) if chain else None,
            pool_url=pool_url(chain, pool) if chain and pool else None,
            note_in=token_name(a, grid.token_name),
            note_out=token_name(b, grid.token_name),
        ))
        cur = amount_out

    # Газ пропорционален числу ног через DEX и не зависит от суммы —
    # именно поэтому на маленьком объёме он и съедает всю маржу.
    gas = float(cycle.gas_usd) if cycle.gas_usd else 0.0
    if n_legs and dex_legs and cycle.gas_usd:
        gas = float(cycle.gas_usd)

    return Breakdown(cycle=cycle, ts=int(grid.times[t]), amount_in=amount,
                     amount_out=cur, gas_usd=gas, steps=steps,
                     exact=abs(amount - base_size) <= 1e-9)
