"""Построение сетки исполнимых курсов из сырых котировок.

Отвечает на вопрос: «сколько Y я реально получу за 1 X на площадке V
в момент t, если торгую на сумму S» — с учётом комиссии и проскальзывания.

Три вещи, которые здесь делаются намеренно строго:

1. **Ограничитель свежести.** Котировка старше staleness_sec не участвует
   вообще. Наивный forward-fill протухшей цены — источник номер один
   ложных связок: цена «застыла» на одной площадке, а на другой ушла,
   и разница выглядит как арбитраж, которого не было.

2. **Проскальзывание, а не спот.** Для DEX резерв пула даёт точную формулу
   постоянного произведения. Для CEX глубины стакана в истории нет,
   поэтому используется модель от объёма свечи — консервативная и явно
   помеченная как оценка.

3. **Логарифмы.** Курсы хранятся как ln(rate), потому что маржа связки —
   это произведение курсов, а в логарифмах произведение становится суммой.
   Это позволяет искать лучшие цепочки алгеброй max-plus вместо перебора.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (CEX_TAKER_PCT, DEX_POOL_FEE_PCT, SETTINGS, USD_LIKE,
                     is_leveraged_token)

log = logging.getLogger(__name__)

NEG_INF = -np.inf


@dataclass
class RateGrid:
    """Трёхмерная сетка курсов в логарифмах.

    log_rate[t, i, j] — натуральный логарифм курса «сколько актива j
    за единицу актива i» в момент времени t, уже за вычетом комиссии
    и проскальзывания. -inf означает «обмена нет или данные протухли».

    venue_idx[t, i, j] — индекс площадки, давшей этот курс.
    """

    times: np.ndarray          # (T,) unix-секунды
    assets: List[str]          # (N,) тикеры
    venues: List[str]          # (V,) площадки
    log_rate: np.ndarray       # (T, N, N) float32
    venue_idx: np.ndarray      # (T, N, N) int16, -1 если курса нет
    trade_size_usd: float

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_times(self) -> int:
        return len(self.times)

    def asset_index(self, symbol: str) -> int:
        try:
            return self.assets.index(symbol.upper())
        except ValueError as exc:
            raise KeyError(f"актив {symbol!r} отсутствует в сетке") from exc

    def coverage(self) -> float:
        """Доля заполненных ячеек — индикатор качества данных."""
        total = self.log_rate.size
        return float(np.isfinite(self.log_rate).sum() / total) if total else 0.0

    def summary(self) -> dict:
        return {
            "точек времени": self.n_times,
            "активов": self.n_assets,
            "площадок": len(self.venues),
            "заполненность": f"{self.coverage() * 100:.1f}%",
            "объём сделки": self.trade_size_usd,
        }


# --------------------------------------------------------------------------
# Модели издержек
# --------------------------------------------------------------------------


def dex_slippage_factor(trade_usd: float, reserve_usd: Optional[float]) -> float:
    """Множитель к курсу для пула постоянного произведения.

    Для пары резервов (R_in, R_out) своп размера S даёт
        amount_out = R_out * S / (R_in + S)
    то есть эффективный курс хуже спота ровно в R_in / (R_in + S) раз.
    Резерв пула в долларах делится пополам между сторонами, поэтому
    R_in ≈ reserve_usd / 2.

    Для V3 ликвидность сконцентрирована около текущей цены и фактическая
    глубина выше, чем следует из общего TVL. Оценка получается заниженной —
    ошибка в безопасную сторону.
    """
    if not reserve_usd or reserve_usd <= 0:
        return 1.0
    r_in = reserve_usd / 2.0
    return r_in / (r_in + trade_usd)


def cex_slippage_factor(trade_usd: float, candle_volume_usd: Optional[float],
                        cap_bps: float = 200.0) -> float:
    """Грубая оценка проскальзывания на CEX по объёму свечи.

    Стакана в исторических данных нет, поэтому точную VWAP-цену
    восстановить невозможно. Эмпирическое допущение: сделка размером
    в долю v от минутного оборота двигает цену примерно на v/2 в долях,
    с потолком cap_bps.

    Это ОЦЕНКА, а не расчёт. Для ликвидных пар (оборот в минуту сильно
    больше сделки) поправка близка к нулю и не искажает результат.
    Для неликвидных — режет заведомо неисполнимые связки, что и требуется.
    """
    if not candle_volume_usd or candle_volume_usd <= 0:
        return 1.0 - cap_bps / 1e4
    ratio = trade_usd / candle_volume_usd
    impact = min(ratio / 2.0, cap_bps / 1e4)
    return 1.0 - impact


def venue_fee_pct(venue: str, venue_kind: str, pool_fee: Optional[float] = None) -> float:
    """Комиссия площадки в процентах за одну ногу."""
    if venue_kind == "cex":
        return CEX_TAKER_PCT.get(venue, 0.10)
    if pool_fee is not None and pool_fee > 0:
        return float(pool_fee)
    return DEX_POOL_FEE_PCT.get(venue, DEX_POOL_FEE_PCT["default"])


# --------------------------------------------------------------------------
# Сборка сетки
# --------------------------------------------------------------------------


def build_grid(
    quotes: pd.DataFrame,
    *,
    settings=SETTINGS,
    trade_size_usd: Optional[float] = None,
    assets: Optional[Sequence[str]] = None,
    venue_kinds: Optional[Sequence[str]] = None,
    max_assets: int = 80,
    apply_slippage: bool = True,
    spot_only: Optional[bool] = None,
    bucket_seconds: Optional[int] = None,
) -> RateGrid:
    """Превращает таблицу котировок в сетку исполнимых курсов.

    quotes — то, что вернул store.read_quotes: колонки ts, venue,
    venue_kind, chain, base, quote, close, volume, liquidity_usd.
    """
    if quotes.empty:
        raise ValueError("нет котировок: сначала запустите сборщик "
                         "(python -m history.collector)")

    df = quotes.copy()
    if venue_kinds:
        df = df[df["venue_kind"].isin(venue_kinds)]
    if df.empty:
        raise ValueError("после фильтра по типу площадок не осталось данных")

    # Токены с плечом отсекаются здесь, а не только при сборе: в уже
    # накопленной истории они могли остаться, а пересобирать её долго.
    if spot_only if spot_only is not None else settings.spot_only:
        known = set(df["base"]) | set(df["quote"])
        lev = {a for a in known if is_leveraged_token(a, known)}
        if lev:
            df = df[~df["base"].isin(lev) & ~df["quote"].isin(lev)]
            log.info("отсеяно токенов с плечом: %d (%s)", len(lev),
                     ", ".join(sorted(lev)[:8]))
        if df.empty:
            raise ValueError("после отсева токенов с плечом не осталось данных")

    trade = float(trade_size_usd if trade_size_usd is not None else settings.trade_size_usd)
    # Гранулярность анализа отдельна от гранулярности сбора: биржи отдают
    # минутные свечи, а DEX на бесплатной инфраструктуре обновляется раз
    # в несколько минут. Сводим всё к общему интервалу, иначе строки DEX
    # окажутся протухшими почти всегда и связки не найдутся.
    tf = int(bucket_seconds or settings.analysis_seconds())

    # --- 1. Сетка времени -------------------------------------------------
    df["bucket"] = (df["ts"] // tf) * tf
    t_min, t_max = int(df["bucket"].min()), int(df["bucket"].max())
    times = np.arange(t_min, t_max + tf, tf, dtype=np.int64)
    t_pos = {int(t): i for i, t in enumerate(times)}

    # --- 2. Набор активов -------------------------------------------------
    if assets:
        keep = {a.upper() for a in assets}
    else:
        keep = _select_assets(df, settings.quote_asset, max_assets)
    df = df[df["base"].isin(keep) & df["quote"].isin(keep)]
    if df.empty:
        raise ValueError("после отбора активов не осталось пар")

    asset_list = sorted(keep & (set(df["base"]) | set(df["quote"])))
    a_pos = {a: i for i, a in enumerate(asset_list)}
    n = len(asset_list)
    T = len(times)

    venue_list = sorted(df["venue"].unique())
    v_pos = {v: i for i, v in enumerate(venue_list)}

    log.info("сетка: %d точек x %d активов x %d площадок", T, n, len(venue_list))
    # Потолок по памяти. Сетка занимает T*n*n ячеек float32 плюс столько же
    # int16 под индексы площадок — примерно 6 байт на ячейку. В облаке
    # с гигабайтом памяти позволить себе можно заметно меньше.
    import os
    cap = 60_000_000 if os.environ.get("ARB_SNAPSHOT_URL") else 400_000_000
    cells = T * n * n
    if cells > cap:
        raise MemoryError(
            f"Сетка {T}x{n}x{n} — это {cells * 6 / 1e6:.0f} МБ, "
            f"больше допустимых {cap * 6 / 1e6:.0f} МБ."
        )

    # --- 3. Свежесть ------------------------------------------------------
    # Котировка относится к своему интервалу; если наблюдение старше
    # staleness_sec от границы интервала — не используем.
    max_age_buckets = max(0, int(settings.staleness_sec // tf))

    # --- 4. Заполнение ----------------------------------------------------
    log_rate = np.full((T, n, n), NEG_INF, dtype=np.float32)
    venue_idx = np.full((T, n, n), -1, dtype=np.int16)

    df = df.sort_values("ts")
    ti = df["bucket"].map(t_pos).to_numpy()
    bi = df["base"].map(a_pos).to_numpy()
    qi = df["quote"].map(a_pos).to_numpy()
    vi = df["venue"].map(v_pos).to_numpy()
    price = df["close"].to_numpy(dtype=np.float64)
    kinds = df["venue_kind"].to_numpy()
    liq = df["liquidity_usd"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)

    # комиссия на строку
    fee = np.array([
        venue_fee_pct(v, k) for v, k in zip(df["venue"].to_numpy(), kinds)
    ], dtype=np.float64)
    fee_mult = 1.0 - fee / 100.0

    # проскальзывание на строку
    if apply_slippage:
        slip = np.ones(len(df), dtype=np.float64)
        is_dex = kinds == "dex"
        for idx in np.flatnonzero(is_dex):
            slip[idx] = dex_slippage_factor(trade, liq[idx] if liq[idx] == liq[idx] else None)
        for idx in np.flatnonzero(~is_dex):
            v_usd = vol[idx] * price[idx] if vol[idx] == vol[idx] else None
            slip[idx] = cex_slippage_factor(trade, v_usd)
    else:
        slip = np.ones(len(df), dtype=np.float64)

    eff = fee_mult * slip
    valid = (price > 0) & (eff > 0)

    fwd = np.log(np.where(valid, price * eff, 1.0))        # base -> quote
    bwd = np.log(np.where(valid, (1.0 / np.where(price > 0, price, 1.0)) * eff, 1.0))

    # Разворачиваем в сетку, оставляя лучший курс среди площадок.
    _scatter_max(log_rate, venue_idx, ti[valid], bi[valid], qi[valid], fwd[valid], vi[valid])
    _scatter_max(log_rate, venue_idx, ti[valid], qi[valid], bi[valid], bwd[valid], vi[valid])

    # --- 5. Протяжка в пределах допустимого возраста ----------------------
    if max_age_buckets > 0:
        _forward_fill(log_rate, venue_idx, max_age_buckets)

    # --- 6. Собственный курс актива в себя = 1 (log 0) --------------------
    diag = np.arange(n)
    log_rate[:, diag, diag] = 0.0
    venue_idx[:, diag, diag] = -1

    return RateGrid(times=times, assets=asset_list, venues=venue_list,
                    log_rate=log_rate, venue_idx=venue_idx, trade_size_usd=trade)


def _scatter_max(log_rate, venue_idx, ti, bi, qi, values, vi) -> None:
    """Записывает курсы в сетку, сохраняя максимум по площадкам,
    и запоминает, какая площадка этот максимум дала.

    Тонкость, из-за которой здесь два прохода. На одну ячейку (момент,
    пара) приходится несколько строк — по одной на площадку. Сравнить
    каждую строку с текущим значением ДО записи нельзя: все они увидят
    одно и то же исходное значение и все окажутся «лучше». А обычное
    присваивание по повторяющимся индексам оставляет не максимум,
    а последнюю строку по порядку.

    Поэтому сначала np.maximum.at честно считает максимум, а потом
    вторым проходом отмечаются те строки, которые этого максимума
    достигли: их площадка и записывается. Совпадения по значению
    равнозначны, так что выбор любой из них корректен.
    """
    if len(ti) == 0:
        return
    vals32 = values.astype(np.float32)
    np.maximum.at(log_rate, (ti, bi, qi), vals32)

    achieved = log_rate[ti, bi, qi]
    # допуск на округление float32
    tol = np.maximum(np.abs(achieved), 1.0) * 1e-6
    is_max = vals32 >= achieved - tol
    idx = np.flatnonzero(is_max)
    if len(idx):
        venue_idx[ti[idx], bi[idx], qi[idx]] = vi[idx].astype(np.int16)


def _forward_fill(log_rate: np.ndarray, venue_idx: np.ndarray, max_age: int) -> None:
    """Протягивает последнее известное значение вперёд, но не дольше max_age шагов.

    Это компромисс: без всякой протяжки данные с разных площадок почти
    никогда не совпадут по времени и связок не найдётся вообще.
    С неограниченной протяжкой появятся ложные связки на протухших ценах.
    Ограничение по возрасту — то, что делает результат осмысленным.
    """
    T = log_rate.shape[0]
    age = np.zeros(log_rate.shape[1:], dtype=np.int32)
    last_val = np.full(log_rate.shape[1:], NEG_INF, dtype=np.float32)
    last_ven = np.full(log_rate.shape[1:], -1, dtype=np.int16)

    for t in range(T):
        cur = log_rate[t]
        fresh = np.isfinite(cur)
        # обновляем «последнее известное» там, где пришло новое значение
        last_val = np.where(fresh, cur, last_val)
        last_ven = np.where(fresh, venue_idx[t], last_ven)
        age = np.where(fresh, 0, age + 1)
        # протягиваем туда, где значения нет, но возраст в пределах допустимого
        usable = (~fresh) & (age <= max_age) & np.isfinite(last_val)
        if usable.any():
            log_rate[t] = np.where(usable, last_val, cur)
            venue_idx[t] = np.where(usable, last_ven, venue_idx[t])


def _select_assets(df: pd.DataFrame, anchor: str, max_assets: int) -> set:
    """Отбирает активы: якорь плюс самые представленные в данных.

    Критерий — число наблюдений: актив, который встречается часто,
    даёт плотный ряд и меньше дыр в сетке.
    """
    anchor = anchor.upper()
    counts: Dict[str, int] = {}
    for col in ("base", "quote"):
        for a, c in df[col].value_counts().items():
            counts[a] = counts.get(a, 0) + int(c)
    counts.pop(anchor, None)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[: max_assets - 1]
    return {anchor} | {a for a, _ in top}
