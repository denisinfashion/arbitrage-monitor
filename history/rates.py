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
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (CEX_TAKER_PCT, DEX_POOL_FEE_PCT, SETTINGS, USD_LIKE,
                     dex_fee_pct, is_concentrated, is_leveraged_token,
                     is_stable)

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

    # Справочники о площадках и парах. Хранятся отдельными словарями,
    # а не ещё одним трёхмерным массивом: значения почти не меняются
    # во времени, а память на бесплатном тарифе на счету.
    venue_chain: Dict[str, str] = field(default_factory=dict)
    """площадка -> сеть ('bsc'), пусто для бирж"""

    venue_kind: Dict[str, str] = field(default_factory=dict)
    """площадка -> 'cex' или 'dex'"""

    pair_liquidity: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    """(площадка, актив, актив) -> типичная ликвидность пула в долларах"""

    pair_pool: Dict[Tuple[str, str, str], str] = field(default_factory=dict)
    """(площадка, актив, актив) -> адрес пула"""

    pool_fee_pct: Dict[str, float] = field(default_factory=dict)
    """адрес пула -> его собственная комиссия в процентах.

    Не средняя по площадке. У V3 уровни различаются в сто раз: пара
    стейблов стоит 0.01%, а мы приписывали ей общие для PancakeSwap
    0.25%. На трёхногой связке это три четверти процента выдуманных
    издержек — больше, чем сама маржа, которую мы ищем.
    """

    pair_volume_usd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    """(площадка, актив, актив) -> типичный оборот свечи в долларах.

    Нужен разбору по шагам: на биржах проскальзывание оценивается от
    оборота, и без этого числа издержки ноги не разложить на комиссию
    и проскальзывание.
    """

    token_address: Dict[Tuple[str, str], str] = field(default_factory=dict)
    """(сеть, тикер) -> адрес контракта"""

    token_name: Dict[str, str] = field(default_factory=dict)
    """тикер -> полное имя токена"""

    quality_notes: Dict[str, str] = field(default_factory=dict)
    """тикер -> почему он не участвует в расчёте (для интерфейса)"""

    asset_tax: Dict[str, tuple] = field(default_factory=dict)
    """тикер -> (налог на покупку %, налог на продажу %, можно ли торговать).

    Налог удерживает сам контракт токена при переводе, и в цене пула его
    не видно. Связка USDT → SPCXB → MARSCOIN → USDT показывала +0.363%
    и была убыточной ровно поэтому.
    """

    def fee_for(self, venue: str, kind: str, a: str, b: str) -> float:
        """Комиссия плеча: своя у пула, иначе общая для площадки.

        Один вход для сетки, разбора и диагностики — иначе они начнут
        расходиться в третьем знаке, и доверять нельзя будет ни одному.
        """
        pool = self.pair_pool.get((venue, a, b))
        own = self.pool_fee_pct.get(pool) if pool else None
        return venue_fee_pct(venue, kind, own)

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


def dex_depth_usd(reserve_usd: Optional[float], venue: str = "",
                  base: str = "", quote: str = "", settings=SETTINGS) -> Optional[float]:
    """Сколько долларов реально стоит на пути свопа у текущей цены.

    Раньше здесь молча стояло `reserve_usd / 2` — половина TVL, как
    в пуле постоянного произведения. Для V2 это верно. Для V3 — нет,
    и ошибка выходила не на проценты, а на порядок.

    В V3 поставщик выбирает диапазон цены, и вся его ликвидность стоит
    внутри него. У текущей цены собирается кратно больше, чем дала бы
    та же сумма, размазанная по всей кривой от нуля до бесконечности.
    Насколько больше — зависит от того, какие диапазоны выбрали
    поставщики, и из TVL это не выводится. Но порядок известен:
    для обычной пары это единицы, для пары стейблов — десятки, потому
    что там диапазоны в доли процента.

    Чем это обошлось. Своп на $1000 через пул с TVL $100 000 давал
    по старой формуле −1.96% на плечо, три плеча — почти −6%. Ни одна
    настоящая связка на 1–2% не проходила: расчёт съедал её целиком
    ещё до отбора. При этом потолок правдоподобия стоит на 5%, то есть
    всё, что могло бы пробиться сквозь такое проскальзывание, мы
    отбрасывали уже как артефакт. Между двумя порогами не оставалось
    ничего — и таблица была пустой при живых спредах на рынке.

    Множители намеренно вынесены в настройки: их правильное значение
    проверяется только реальным свопом. Если расчёт расходится с тем,
    что получилось на самом деле, крутить надо здесь.
    """
    if not reserve_usd or reserve_usd <= 0:
        return None
    depth = float(reserve_usd) / 2.0
    if is_concentrated(venue):
        depth *= max(1.0, float(getattr(settings, "dex_v3_depth_multiple", 10.0)))
    if base and quote and is_stable(base) and is_stable(quote):
        depth *= max(1.0, float(getattr(settings, "dex_stable_depth_multiple", 5.0)))
    return depth


def dex_slippage_factor(trade_usd: float, reserve_usd: Optional[float],
                        venue: str = "", base: str = "", quote: str = "",
                        settings=SETTINGS) -> float:
    """Множитель к курсу: во сколько раз исполнение хуже спота.

    Для глубины D своп размера S даёт amount_out = R_out·S/(D+S),
    то есть курс хуже ровно в D/(D+S) раз. Вся содержательная часть —
    в оценке D, она в dex_depth_usd выше.
    """
    depth = dex_depth_usd(reserve_usd, venue, base, quote, settings)
    if not depth:
        return 1.0
    return depth / (depth + trade_usd)


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
    return dex_fee_pct(venue)


# --------------------------------------------------------------------------
# Сколько активов помещается
# --------------------------------------------------------------------------

BYTES_PER_CELL = 6
"""float32 на курс плюс int16 на индекс площадки."""


def budget_bytes() -> int:
    """Сколько памяти отводится под сетку курсов.

    Раньше потолок задавался числом ячеек и был один на все случаи.
    Это неудобно: при часовом шаге точек мало и активов помещается
    в разы больше, чем при минутном, а пользователь видел один и тот же
    жёсткий предел в сорок штук и справедливо спрашивал, почему нельзя
    взять все. Теперь ограничение считается от памяти и раскрывается
    в число активов под конкретное окно.

    Переопределяется переменной `ARB_GRID_BUDGET_MB`.
    """
    default = 250 if os.environ.get("ARB_SNAPSHOT_URL") else 1200
    try:
        mb = float(os.environ.get("ARB_GRID_BUDGET_MB", default))
    except (TypeError, ValueError):
        mb = default
    return int(max(16.0, mb) * 1e6)


def grid_bytes(n_times: int, n_assets: int) -> int:
    return int(n_times) * int(n_assets) ** 2 * BYTES_PER_CELL


def max_assets_for(n_times: int, budget: Optional[int] = None) -> int:
    """Сколько активов помещается в бюджет при данном числе точек."""
    budget = budget or budget_bytes()
    if n_times <= 0:
        return 1000
    n = int((budget / (BYTES_PER_CELL * n_times)) ** 0.5)
    return max(10, min(1000, n))


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
    drop_suspicious: bool = True,
    min_pool_volume_usd: Optional[float] = None,
    min_liquidity_usd: float = 0.0,
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

    # Недостоверные данные DEX отсеиваются до отбора активов, иначе
    # подделка с накрученным оборотом займёт место настоящего токена
    # в списке из max_assets штук.
    pools_df = _pools_frame(settings)
    quality_notes: Dict[str, str] = {}
    from .quality import MIN_POOL_VOLUME_USD, Screen, screen_pools
    vol_floor = (MIN_POOL_VOLUME_USD if min_pool_volume_usd is None
                 else float(min_pool_volume_usd))
    try:
        screen = screen_pools(pools_df, vol_floor if drop_suspicious else 0.0)
    except Exception as exc:  # noqa: BLE001
        # Отсев — предохранитель, а не обязательный этап. Если он сам
        # споткнулся о данные, правильное поведение — посчитать без него
        # и сказать об этом, а не оставить пользователя без страницы.
        log.warning("проверка качества пулов не отработала: %s", exc)
        screen = Screen()

    if drop_suspicious:
        if screen.bad_pools and "pool" in df.columns:
            bad = df["pool"].astype("string").isin(screen.bad_pools)
            if bad.any():
                log.info("отсеяно строк по качеству пулов: %d (%s)",
                         int(bad.sum()), screen.summary())
                df = df[~bad]
        quality_notes = dict(screen.notes)
        # Токены, которые нельзя продать, отсекаются здесь же. Это не
        # «подозрительно», а «симуляция обмена не прошла»: honeypot,
        # заблокированная продажа, налог в десятки процентов. Считать
        # по ним связку бессмысленно — она не исполнится ни при каком
        # спреде, и место в списке активов займёт зря.
        for sym, why in _untradable_symbols(pools_df).items():
            quality_notes.setdefault(sym, why)

        if quality_notes:
            drop = set(quality_notes)
            df = df[~df["base"].isin(drop) & ~df["quote"].isin(drop)]
        if df.empty:
            raise ValueError("после отсева недостоверных пулов не осталось данных")

    trade = float(trade_size_usd if trade_size_usd is not None else settings.trade_size_usd)

    # Порог ликвидности имеет смысл только относительно объёма сделки.
    # Проскальзывание в пуле постоянного произведения зависит не от
    # размера пула, а от отношения сделки к нему: тысяча долларов против
    # пяти тысяч и сто против пятисот дают одни и те же 29% потерь.
    # Поэтому здесь отсекается не «мелкий пул», а «мелкий для этой суммы».
    if min_liquidity_usd and "liquidity_usd" in df.columns:
        liq_col = pd.to_numeric(df["liquidity_usd"], errors="coerce")
        is_dex = df["venue_kind"].to_numpy() == "dex"
        thin = is_dex & liq_col.notna().to_numpy() & (liq_col.to_numpy() < min_liquidity_usd)
        if thin.any():
            log.info("отсеяно строк по ликвидности (пул меньше $%s под объём $%s): %d",
                     f"{min_liquidity_usd:,.0f}".replace(",", " "),
                     f"{trade:,.0f}".replace(",", " "), int(thin.sum()))
            df = df[~thin]
        if df.empty:
            raise ValueError(
                "После отсева по ликвидности не осталось данных. "
                "Уменьшите требование к размеру пула или объём сделки.")
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
    limit = max_assets_for(T)
    if n > limit:
        raise MemoryError(
            f"Сетка {T}×{n}×{n} заняла бы {grid_bytes(T, n) / 1e6:.0f} МБ "
            f"при бюджете {budget_bytes() / 1e6:.0f} МБ. "
            f"При шаге сетки в {T} точек помещается до {limit} активов."
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

    # Комиссия на строку. Своя у пула, если источник её сообщил, —
    # иначе общая для площадки. Разница не косметическая: уровни V3
    # отличаются в сто раз, и пара стейблов на 0.01% считалась у нас
    # по 0.25%.
    # Справочник комиссий пулов: адрес -> процент. Пустой словарь
    # означает лишь «источник уровня не сообщил» — тогда работает
    # прежнее поведение, комиссия по площадке.
    pool_fees = pool_fee_map(pools_df)

    pool_col = (df["pool"].to_numpy() if "pool" in df.columns
                else np.array([None] * len(df), dtype=object))
    fee = np.array([
        venue_fee_pct(v, k, pool_fees.get(p) if p is not None else None)
        for v, k, p in zip(df["venue"].to_numpy(), kinds, pool_col)
    ], dtype=np.float64)
    fee_mult = 1.0 - fee / 100.0

    # проскальзывание на строку
    if apply_slippage:
        slip = np.ones(len(df), dtype=np.float64)
        is_dex = kinds == "dex"
        venues = df["venue"].to_numpy()
        bases, quotes = df["base"].to_numpy(), df["quote"].to_numpy()
        for idx in np.flatnonzero(is_dex):
            slip[idx] = dex_slippage_factor(
                trade, liq[idx] if liq[idx] == liq[idx] else None,
                venue=str(venues[idx]), base=str(bases[idx]),
                quote=str(quotes[idx]), settings=settings)
        for idx in np.flatnonzero(~is_dex):
            v_usd = vol[idx] * price[idx] if vol[idx] == vol[idx] else None
            slip[idx] = cex_slippage_factor(trade, v_usd)
    else:
        slip = np.ones(len(df), dtype=np.float64)

    # Налог на перевод. Считается отдельно по направлениям, и это не
    # придирка: в обмене a -> b токен `a` продаётся, токен `b` покупается,
    # а ставки у них разные и принадлежат разным контрактам. До сих пор
    # издержка была одна на обе стороны строки, потому что комиссия
    # и проскальзывание симметричны. Налог — нет.
    taxes = asset_tax_map(pools_df)
    if taxes:
        tax_fwd = np.array([leg_tax_factor(taxes, str(a), str(b), k)
                            for a, b, k in zip(df["base"].to_numpy(),
                                               df["quote"].to_numpy(), kinds)],
                           dtype=np.float64)
        tax_bwd = np.array([leg_tax_factor(taxes, str(b), str(a), k)
                            for a, b, k in zip(df["base"].to_numpy(),
                                               df["quote"].to_numpy(), kinds)],
                           dtype=np.float64)
    else:
        tax_fwd = tax_bwd = np.ones(len(df), dtype=np.float64)

    eff = fee_mult * slip
    valid = (price > 0) & (eff > 0)

    fwd = np.log(np.where(valid, price * eff * tax_fwd, 1.0))       # base -> quote
    bwd = np.log(np.where(valid, (1.0 / np.where(price > 0, price, 1.0))
                          * eff * tax_bwd, 1.0))

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

    meta = _collect_meta(df, settings, pools_df)
    # При споре тикеров ссылка должна вести на признанный настоящим
    # контракт, а не на тот, что первым попался в справочнике.
    for sym, addr in screen.address.items():
        meta["token_address"][(settings.chain, sym)] = addr

    return RateGrid(times=times, assets=asset_list, venues=venue_list,
                    log_rate=log_rate, venue_idx=venue_idx, trade_size_usd=trade,
                    quality_notes=quality_notes, **meta)


def _untradable_symbols(pools_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Тикер -> почему по нему нельзя торговать, по проверке контракта."""
    out: Dict[str, str] = {}
    if pools_df is None or pools_df.empty:
        return out
    for side in ("base", "quote"):
        cols = (side, f"{side}_tradable", f"{side}_risk_note")
        if not all(c in pools_df.columns for c in cols):
            continue
        for sym, ok, note in zip(*(pools_df[c] for c in cols)):
            sym = str(sym or "").upper()
            if not sym or ok is None or ok != ok or bool(ok):
                continue
            out[sym] = str(note or "продать нельзя: проверка контракта")
    return out


def asset_tax_map(pools_df: Optional[pd.DataFrame]) -> Dict[str, tuple]:
    """Тикер -> (налог на покупку, налог на продажу, можно ли торговать).

    Берётся из справочника пулов: проверка контракта едет вместе с ним,
    продублированная на каждую строку. Тикер, а не адрес, потому что
    сетка курсов оперирует тикерами; подделку с чужим адресом отсеивает
    отбор качества раньше, до этого места.
    """
    out: Dict[str, tuple] = {}
    if pools_df is None or pools_df.empty:
        return out
    for side in ("base", "quote"):
        cols = (side, f"{side}_tax_buy", f"{side}_tax_sell", f"{side}_tradable")
        if not all(c in pools_df.columns for c in cols):
            continue
        for sym, buy, sell, ok in zip(*(pools_df[c] for c in cols)):
            sym = str(sym or "").upper()
            if not sym or buy is None or buy != buy:
                continue
            tradable = True if ok is None or ok != ok else bool(ok)
            out[sym] = (float(buy or 0.0), float(sell or 0.0), tradable)
    return out


def leg_tax_factor(taxes: Dict[str, tuple], sell_asset: str, buy_asset: str,
                   kind: str) -> float:
    """Множитель к курсу плеча из-за налогов контрактов.

    На бирже налога нет: там торгуется запись в базе биржи, а не токен
    в сети, и контракт при этом не вызывается.
    """
    if kind != "dex":
        return 1.0
    factor = 1.0
    out = taxes.get(str(sell_asset).upper())
    if out and out[1] > 0:
        factor *= max(0.0, 1.0 - out[1] / 100.0)
    inn = taxes.get(str(buy_asset).upper())
    if inn and inn[0] > 0:
        factor *= max(0.0, 1.0 - inn[0] / 100.0)
    return factor


def pool_fee_map(pools_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Адрес пула -> его собственная комиссия в процентах.

    Отдельной функцией, потому что этим словарём пользуются трое: сетка
    курсов, разбор по шагам и диагностика связки. Считать его в каждом
    месте заново — верный способ развести их в третьем знаке.
    """
    out: Dict[str, float] = {}
    if pools_df is None or pools_df.empty or "fee_pct" not in pools_df.columns:
        return out
    if "pool" not in pools_df.columns:
        return out
    for addr, fee_val in zip(pools_df["pool"], pools_df["fee_pct"]):
        if addr is None or fee_val is None or fee_val != fee_val:
            continue
        try:
            fee_val = float(fee_val)
        except (TypeError, ValueError):
            continue
        if 0 < fee_val <= 100:
            out[str(addr)] = fee_val
    return out


def _pools_frame(settings) -> Optional[pd.DataFrame]:
    """Справочник пулов — из локальной базы или из облачного снимка.

    Раньше он читался напрямую из SQLite. В облаке базы нет вовсе:
    приложение живёт на одном скачанном parquet-файле, и справочник
    молча оказывался пустым — вместе с ним пропадали адреса токенов,
    а значит ссылки на обмен и расшифровка тикеров.
    """
    try:
        from . import snapshot
        return _denullify(snapshot.pools(settings.chain))
    except Exception as exc:  # noqa: BLE001 — справочник необязателен
        log.debug("справочник пулов недоступен: %s", exc)
        return None


def _denullify(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Приводит пропуски к обычным None и NaN.

    Снимок читается из Parquet, и pandas отдаёт колонки в «расширенных»
    типах, где пропуск — это pd.NA. У этого значения нет истинности:
    выражение `if sym and addr` на нём не возвращает False, а падает с
    TypeError. Ровно так приложение и сломалось в облаке, хотя локально
    на той же логике работало — там справочник приходил из SQLite
    с обычными None.

    Чинить каждую проверку по отдельности бессмысленно: пропуск может
    появиться в любой колонке. Приводим типы один раз на входе.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_extension_array_dtype(out[col].dtype):
            if pd.api.types.is_numeric_dtype(out[col].dtype):
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
            else:
                out[col] = out[col].astype(object).where(out[col].notna(), None)
    return out


def _collect_meta(df: pd.DataFrame, settings,
                  pools_df: Optional[pd.DataFrame] = None) -> dict:
    """Собирает справочники: сети площадок, ликвидность пар, адреса токенов.

    Всё это нужно интерфейсу, чтобы показать, в какой сети идёт обмен,
    сколько в пуле денег и куда вести ссылку на своп. В самой математике
    не участвует.
    """
    venue_chain, venue_kind = {}, {}
    for v, grp in df.groupby("venue", sort=False):
        venue_kind[v] = str(grp["venue_kind"].iloc[0])
        chains = [c for c in grp["chain"].unique() if c]
        venue_chain[v] = str(chains[0]) if chains else ""

    pair_liquidity, pair_pool = {}, {}
    if "liquidity_usd" in df.columns:
        liq = df.dropna(subset=["liquidity_usd"])
        if not liq.empty:
            for (v, b, q), grp in liq.groupby(["venue", "base", "quote"], sort=False):
                val = float(grp["liquidity_usd"].median())
                # пара симметрична: ликвидность одна и та же в обе стороны
                pair_liquidity[(v, b, q)] = val
                pair_liquidity[(v, q, b)] = val
                pools = grp["pool"].dropna()
                if not pools.empty:
                    pair_pool[(v, b, q)] = str(pools.iloc[0])
                    pair_pool[(v, q, b)] = str(pools.iloc[0])

    # Оборот в долларах: для DEX он уже в долларах, для CEX объём свечи
    # выражен в базовом активе и переводится по цене той же свечи.
    pair_volume_usd = {}
    if {"volume", "close"} <= set(df.columns):
        vol = df.dropna(subset=["volume", "close"])
        if not vol.empty:
            v = vol.copy()
            is_cex = v["venue_kind"].to_numpy() == "cex"
            v["_usd"] = np.where(is_cex, v["volume"] * v["close"], v["volume"])
            for (ven, b, q), grp in v.groupby(["venue", "base", "quote"], sort=False):
                val = float(grp["_usd"].median())
                if val > 0:
                    pair_volume_usd[(ven, b, q)] = val
                    pair_volume_usd[(ven, q, b)] = val

    # Адреса и имена токенов лежат в справочнике пулов, а не в котировках
    token_address, token_name = {}, {}
    if pools_df is None:
        pools_df = _pools_frame(settings)
    if pools_df is not None and not pools_df.empty:
        cols = set(pools_df.columns)

        def text(value) -> str:
            """Пусто для любого вида пропуска, иначе строка.

            Проверять истинность значения напрямую нельзя: из Parquet
            приходит pd.NA, у которого нет булева значения.
            """
            if value is None:
                return ""
            try:
                if pd.isna(value):
                    return ""
            except (TypeError, ValueError):
                pass
            s = str(value).strip()
            return "" if s.lower() in ("nan", "none", "<na>") else s

        try:
            for r in pools_df.to_dict("records"):
                chain = text(r.get("chain")) or settings.chain
                for sym_c, addr_c, name_c in (("base", "base_addr", "base_name"),
                                              ("quote", "quote_addr", "quote_name")):
                    if sym_c not in cols:
                        continue
                    sym = text(r.get(sym_c))
                    addr = text(r.get(addr_c)) if addr_c in cols else ""
                    name = text(r.get(name_c)) if name_c in cols else ""
                    if sym and addr:
                        token_address.setdefault((chain, sym), addr)
                    if sym and name:
                        token_name.setdefault(sym, name)
        except Exception as exc:  # noqa: BLE001
            # Справочник — украшение: имена токенов и ссылки на обмен.
            # Ронять из-за него расчёт нельзя.
            log.warning("справочник токенов прочитан не полностью: %s", exc)

    return {"venue_chain": venue_chain, "venue_kind": venue_kind,
            "pair_liquidity": pair_liquidity, "pair_pool": pair_pool,
            "pool_fee_pct": pool_fee_map(pools_df),
            "asset_tax": asset_tax_map(pools_df),
            "pair_volume_usd": pair_volume_usd,
            "token_address": token_address, "token_name": token_name}


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
