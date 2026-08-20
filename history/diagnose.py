"""Разбор конкретной связки: где именно она у нас теряется.

Вопрос, ради которого написан модуль, звучал так: «я только что сделал
USDT → AAVE → DAI → USDT и получил два процента, почему монитор этого
не показал». Ответить на него по таблице результатов нельзя — там
показано то, что нашлось, а нужно обратное: что не нашлось и почему.

Причин может быть шесть, и они выясняются по очереди:

1. **Токена нет в справочнике пулов.** Мы его просто не собираем.
2. **Пул есть, но отсеян отбором качества** — по обороту или по порогу
   ликвидности.
3. **Котировок нет в окне анализа.** Пул известен, но за выбранные часы
   цены по нему не записалось.
4. **Плечи не совпали по времени.** Каждая цена есть, но в разные
   моменты, а связка считается по одному срезу.
5. **Спота не хватает.** Цены сошлись, но разница между ними меньше
   комиссий — связки действительно нет.
6. **Съело проскальзывание.** Спот есть, комиссии прошли, но модель
   глубины решила, что объём не исполнится.

Шестой пункт — самый коварный, потому что снаружи выглядит как «связок
нет на рынке», хотя на рынке они есть. Именно он и оказался виноват:
для V3 глубина считалась по формуле V2 и занижалась на порядок.

Разбор ничего не чинит. Он показывает по каждому плечу, что у нас есть,
и называет шаг, на котором связка выпала.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .config import (SETTINGS, dex_fee_pct, is_concentrated, is_stable,
                     norm_asset)
from .rates import (asset_tax_map, cex_slippage_factor, dex_depth_usd,
                    dex_slippage_factor, leg_tax_factor, pool_fee_map,
                    venue_fee_pct)


@dataclass
class LegReport:
    """Одно плечо: что мы о нём знаем."""

    frm: str
    to: str
    found: bool = False
    venue: str = ""
    venue_kind: str = ""
    price: Optional[float] = None
    ts: Optional[int] = None
    liquidity_usd: Optional[float] = None
    volume_usd: Optional[float] = None
    pool: str = ""
    fee_pct: float = 0.0
    depth_usd: Optional[float] = None
    slip_pct: float = 0.0
    concentrated: bool = False
    tax_pct: float = 0.0
    """Налог контракта на этом плече: удержание при продаже `frm` плюс
    удержание при покупке `to`. В цене пула его не видно."""
    alternatives: int = 0
    note: str = ""

    @property
    def age_sec(self) -> Optional[float]:
        return None if self.ts is None else max(0.0, time.time() - self.ts)

    @property
    def cost_pct(self) -> float:
        """Полная издержка плеча: комиссия, проскальзывание и налог."""
        return self.fee_pct + self.slip_pct + self.tax_pct


@dataclass
class ChainReport:
    """Связка целиком: плечи, итог и вердикт."""

    tickers: List[str]
    legs: List[LegReport] = field(default_factory=list)
    trade_usd: float = 1000.0
    window_h: float = 6.0
    spot_pct: float = float("nan")
    """Маржа по чистым ценам, без единой издержки."""
    net_pct: float = float("nan")
    """Маржа после комиссий и проскальзывания."""
    fees_pct: float = 0.0
    slip_pct: float = 0.0
    tax_pct: float = 0.0
    verdict: str = ""
    stage: str = ""
    """Короткий код шага, на котором связка выпала."""
    missing: List[str] = field(default_factory=list)
    routed: List[str] = field(default_factory=list)
    """Плечи, раскрытые через промежуточный актив: «AAVE → DAI через BNB»."""
    quotes_seen: int = 0

    @property
    def ok(self) -> bool:
        return self.stage == "ok"


def parse_chain(text: str, anchor: str = "USDT") -> List[str]:
    """«USDT-AAVE-DAI-USDT», «usdt aave dai» -> список тикеров.

    Замыкание дописывается само: человек редко печатает якорь дважды,
    а связка обязана возвращаться в него.
    """
    raw = (text or "").replace("→", "-").replace(">", "-").replace(",", "-")
    parts = [norm_asset(p.strip().upper()) for p in raw.replace(" ", "-").split("-")]
    parts = [p for p in parts if p]
    if not parts:
        return []
    if parts[0] != anchor.upper() and anchor:
        parts.insert(0, norm_asset(anchor.upper()))
    if parts[-1] != parts[0]:
        parts.append(parts[0])
    return parts


# Через что маршрутизатор обычно ведёт обмен, когда прямого пула нет.
# Порядок не случаен: в BNB Chain почти всё котируется против обёрнутого
# BNB, дальше идут стейблы, дальше крупные монеты.
INTERMEDIATES = ["BNB", "USDT", "USDC", "ETH", "BTC", "CAKE"]


def _rows_for(quotes: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Все котировки пары, приведённые к направлению a -> b.

    Направление хранения не важно: цена пары обратима. Важно не
    перепутать её при развороте — отсюда отдельная колонка `_rate`.
    """
    fwd = quotes[(quotes["base"] == a) & (quotes["quote"] == b)].copy()
    bwd = quotes[(quotes["base"] == b) & (quotes["quote"] == a)].copy()
    if not fwd.empty:
        fwd["_rate"] = pd.to_numeric(fwd["close"], errors="coerce")
    if not bwd.empty:
        price = pd.to_numeric(bwd["close"], errors="coerce")
        bwd["_rate"] = 1.0 / price.where(price > 0)
    parts = [f for f in (fwd, bwd) if not f.empty]
    if not parts:
        return fwd
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        return out
    return out[out["_rate"].notna() & (out["_rate"] > 0)]


def _pools(settings=SETTINGS):
    try:
        from . import snapshot
        return snapshot.pools(settings.chain)
    except Exception:  # noqa: BLE001 — справочник необязателен
        return None


def pool_fees(settings=SETTINGS) -> dict:
    """Комиссии пулов из справочника. Пусто, если справочник недоступен."""
    return pool_fee_map(_pools(settings))


def asset_taxes(settings=SETTINGS) -> dict:
    """Налоги контрактов по тикерам. Пусто, если проверок ещё не было."""
    return asset_tax_map(_pools(settings))


def _leg_cost(row, a: str, b: str, trade_usd: float, settings,
              fees: Optional[dict] = None) -> tuple:
    """Комиссия и проскальзывание одной котировки. Возвращает (fee_pct, slip)."""
    kind = str(row.get("venue_kind", ""))
    venue = str(row.get("venue", ""))
    if kind == "dex":
        liq = row.get("liquidity_usd")
        liq = float(liq) if liq == liq and liq is not None else None
        pool = row.get("pool")
        own = (fees or {}).get(str(pool)) if pool else None
        return (venue_fee_pct(venue, kind, own),
                dex_slippage_factor(trade_usd, liq, venue=venue, base=a,
                                    quote=b, settings=settings))
    from .config import CEX_TAKER_PCT
    vol, price = row.get("volume"), row.get("close")
    vol_usd = (float(vol) * float(price)
               if vol == vol and vol is not None and price == price else None)
    return CEX_TAKER_PCT.get(venue, 0.10), cex_slippage_factor(trade_usd, vol_usd)


def _best_row(quotes: pd.DataFrame, a: str, b: str, trade_usd: float = 1000.0,
              settings=SETTINGS, max_age: Optional[float] = None,
              fees: Optional[dict] = None):
    """Лучшая исполнимая котировка пары, а не просто самая свежая.

    Сперва здесь бралась последняя по времени строка. Это оказалось
    неверно вдвойне. Во-первых, «последняя» могла прийти с биржи, хотя
    рядом лежал пул той же пары — и разбор показывал плечо на KuCoin
    в цепочке, которую человек исполняет в одном блоке на DEX.
    Во-вторых, среди нескольких площадок выгодна не свежайшая,
    а та, где после комиссии и проскальзывания получишь больше.

    Поэтому: отбрасываем протухшее, остальное сравниваем по чистому
    курсу — ровно так же, как это делает основной расчёт.
    """
    rows = _rows_for(quotes, a, b)
    if rows.empty:
        return None

    now = time.time()
    stale = False
    if max_age:
        fresh = rows[rows["ts"] >= now - max_age]
        if fresh.empty:
            # Протухшую цену не прячем, но и молча за свежую не выдаём:
            # цена трёхчасовой давности в связке — это не связка.
            stale = True
        else:
            rows = fresh

    best, best_net = None, -1.0
    for _, r in rows.iterrows():
        fee_pct, slip = _leg_cost(r, a, b, trade_usd, settings, fees)
        net = float(r["_rate"]) * (1.0 - fee_pct / 100.0) * slip
        if net > best_net:
            best, best_net = r.copy(), net
    if best is None:
        return None
    best["_alternatives"] = len(rows)
    best["_stale"] = stale
    return best


def expand_route(chain: List[str], quotes: pd.DataFrame,
                 trade_usd: float = 1000.0, settings=SETTINGS,
                 max_age: Optional[float] = None,
                 fees: Optional[dict] = None) -> tuple:
    """Достраивает промежуточные активы там, где прямого рынка нет.

    Человек говорит «USDT → AAVE → DAI → USDT», потому что именно так
    он это и вводит в интерфейсе биржи. Но маршрутизатор внутри почти
    никогда не делает один обмен: прямого пула AAVE/DAI может не
    существовать вовсе, и своп идёт через обёрнутый BNB — AAVE → WBNB,
    WBNB → DAI. Для кошелька это одна операция, для нас — две ноги
    с двумя комиссиями.

    Раньше разбор писал по такому плечу «нет котировок в окне» и на этом
    останавливался. Формально верно, по существу — нет: рынок есть,
    просто составной. Здесь плечо раскрывается в два, и обе комиссии
    честно попадают в расчёт.
    """
    if not chain:
        return chain, []
    out, routed = [chain[0]], []
    for a, b in zip(chain, chain[1:]):
        if _best_row(quotes, a, b, trade_usd, settings, max_age, fees) is not None:
            out.append(b)
            continue
        best_mid, best_net = None, -1.0
        for m in INTERMEDIATES:
            if m in (a, b):
                continue
            first = _best_row(quotes, a, m, trade_usd, settings, max_age, fees)
            second = _best_row(quotes, m, b, trade_usd, settings, max_age, fees)
            if first is None or second is None:
                continue
            net = 1.0
            for row, x, y in ((first, a, m), (second, m, b)):
                fee_pct, slip = _leg_cost(row, x, y, trade_usd, settings, fees)
                net *= float(row["_rate"]) * (1.0 - fee_pct / 100.0) * slip
            if net > best_net:
                best_mid, best_net = m, net
        if best_mid:
            out.extend([best_mid, b])
            routed.append(f"{a} → {b} через {best_mid}")
        else:
            out.append(b)
    return out, routed


def diagnose(chain: List[str], quotes: pd.DataFrame, trade_usd: float = 1000.0,
             window_h: float = 6.0, settings=SETTINGS,
             venue_kinds: Optional[List[str]] = None,
             route: bool = True, max_age_sec: Optional[float] = 900.0,
             fees: Optional[dict] = None,
             taxes: Optional[dict] = None) -> ChainReport:
    """Проходит связку плечо за плечом и объясняет результат.

    `venue_kinds` ограничивает площадки. Это не украшение: связка,
    исполняемая в одном блоке на DEX, и связка с переводом на биржу —
    разные вещи с разным риском, и смешивать их в одном разборе значит
    показывать несуществующий маршрут.

    `route` разрешает раскрывать плечо через промежуточный актив, когда
    прямого рынка нет.
    """
    if venue_kinds and not quotes.empty and "venue_kind" in quotes.columns:
        quotes = quotes[quotes["venue_kind"].isin(list(venue_kinds))]

    rep = ChainReport(tickers=list(chain), trade_usd=trade_usd,
                      window_h=window_h, quotes_seen=len(quotes))
    if len(chain) < 3:
        rep.stage, rep.verdict = "input", "Нужно хотя бы два перехода."
        return rep

    if fees is None:
        fees = pool_fees(settings)
    if taxes is None:
        taxes = asset_taxes(settings)

    if route and not quotes.empty:
        chain, rep.routed = expand_route(chain, quotes, trade_usd, settings,
                                         max_age_sec, fees)
        rep.tickers = list(chain)

    product = 1.0
    net_product = 1.0
    for a, b in zip(chain, chain[1:]):
        leg = LegReport(frm=a, to=b)
        row = (_best_row(quotes, a, b, trade_usd, settings, max_age_sec, fees)
               if not quotes.empty else None)
        if row is None:
            leg.note = "нет котировок в окне"
            rep.legs.append(leg)
            rep.missing.append(f"{a}/{b}")
            continue

        leg.found = True
        leg.venue = str(row.get("venue", ""))
        leg.venue_kind = str(row.get("venue_kind", ""))
        leg.price = float(row["_rate"])
        leg.ts = int(row["ts"])
        leg.pool = str(row.get("pool") or "")
        leg.alternatives = int(row.get("_alternatives", 1))
        if any(f"через {a}" in r or f"через {b}" in r for r in rep.routed):
            leg.note = "плечо раскрыто: прямого рынка нет"

        liq = row.get("liquidity_usd")
        leg.liquidity_usd = float(liq) if liq == liq and liq is not None else None
        vol = row.get("volume")
        price = float(row["close"]) if row["close"] == row["close"] else 0.0
        leg.volume_usd = (float(vol) * price
                          if vol == vol and vol is not None else None)

        if bool(row.get("_stale")):
            leg.note = (leg.note + "; " if leg.note else "") + "цена устарела"

        if leg.venue_kind == "dex":
            own = fees.get(str(leg.pool)) if leg.pool else None
            leg.fee_pct = venue_fee_pct(leg.venue, "dex", own)
            leg.concentrated = is_concentrated(leg.venue)
            leg.depth_usd = dex_depth_usd(leg.liquidity_usd, leg.venue, a, b,
                                          settings)
            slip = dex_slippage_factor(trade_usd, leg.liquidity_usd,
                                       venue=leg.venue, base=a, quote=b,
                                       settings=settings)
        else:
            from .config import CEX_TAKER_PCT
            leg.fee_pct = CEX_TAKER_PCT.get(leg.venue, 0.10)
            slip = cex_slippage_factor(trade_usd, leg.volume_usd)
        leg.slip_pct = (1.0 - slip) * 100.0

        tax_factor = leg_tax_factor(taxes, a, b, leg.venue_kind)
        leg.tax_pct = (1.0 - tax_factor) * 100.0

        product *= leg.price
        net_product *= (leg.price * (1.0 - leg.fee_pct / 100.0) * slip
                        * tax_factor)
        rep.legs.append(leg)

    rep.fees_pct = sum(l.fee_pct for l in rep.legs if l.found)
    rep.slip_pct = sum(l.slip_pct for l in rep.legs if l.found)
    rep.tax_pct = sum(l.tax_pct for l in rep.legs if l.found)

    if rep.missing:
        rep.stage = "no_quotes"
        rep.verdict = ("Не по всем плечам есть котировки: "
                       + ", ".join(rep.missing)
                       + ". Пока цепочка не собрана целиком, считать нечего.")
        return rep

    rep.spot_pct = (product - 1.0) * 100.0
    rep.net_pct = (net_product - 1.0) * 100.0

    stamps = [l.ts for l in rep.legs if l.ts]
    spread_sec = (max(stamps) - min(stamps)) if stamps else 0
    if spread_sec > settings.staleness_sec:
        rep.stage = "not_aligned"
        rep.verdict = (
            f"Цены по плечам разъехались во времени на "
            f"{spread_sec // 60} мин при допуске "
            f"{settings.staleness_sec // 60} мин. Связка считается по одному "
            f"срезу, поэтому такая цепочка в расчёт не попадает — и это "
            f"правильно: разница между ценами из разных моментов не арбитраж, "
            f"а движение рынка.")
        return rep

    if rep.spot_pct <= 0:
        rep.stage = "no_spot"
        rep.verdict = (
            f"По чистым ценам связка даёт {rep.spot_pct:+.2f}% — до всяких "
            f"издержек. В этот момент её действительно не было.")
        return rep

    if rep.net_pct <= 0 and rep.spot_pct - rep.fees_pct <= 0:
        rep.stage = "fees"
        rep.verdict = (
            f"Спот {rep.spot_pct:+.2f}%, комиссии {rep.fees_pct:.2f}% — "
            f"разница меньше стоимости обменов.")
        return rep

    if rep.net_pct <= 0 and rep.tax_pct > 0 and (
            rep.spot_pct - rep.fees_pct - rep.slip_pct > 0):
        rep.stage = "tax"
        rep.verdict = (
            f"Спот {rep.spot_pct:+.2f}%, комиссии {rep.fees_pct:.2f}%, "
            f"проскальзывание {rep.slip_pct:.2f}% — связка выжила бы, но "
            f"налог контрактов {rep.tax_pct:.2f}% забирает остаток. Этот "
            f"процент удерживает сам токен при переводе, в цене пула его "
            f"не видно.")
        return rep

    if rep.net_pct <= 0:
        rep.stage = "slippage"
        rep.verdict = (
            f"Спот {rep.spot_pct:+.2f}%, комиссии {rep.fees_pct:.2f}%, "
            f"проскальзывание по модели {rep.slip_pct:.2f}% — в ноль уводит "
            f"именно проскальзывание. Проверьте оценку глубины: если своп "
            f"на самом деле прошёл лучше, значит множитель глубины V3 занижен.")
        return rep

    rep.stage = "ok"
    rep.verdict = (
        f"Связка считается: {rep.net_pct:+.2f}% чистыми с "
        f"{trade_usd:,.0f} USDT".replace(",", " ") + ".")
    return rep


def legs_frame(rep: ChainReport) -> pd.DataFrame:
    """Плечи таблицей — для страницы."""
    rows = []
    for i, l in enumerate(rep.legs, 1):
        rows.append({
            "№": i,
            "Обмен": f"{l.frm} → {l.to}",
            "Площадка": l.venue or "—",
            "Курс": l.price,
            "Возраст, мин": (round(l.age_sec / 60, 1)
                             if l.age_sec is not None else None),
            "Ликвидность, $": l.liquidity_usd,
            "Глубина, $": l.depth_usd,
            "V3": "да" if l.concentrated else ("нет" if l.found and
                                               l.venue_kind == "dex" else ""),
            "Комиссия, %": round(l.fee_pct, 3) if l.found else None,
            "Проскальзывание, %": round(l.slip_pct, 3) if l.found else None,
            "Налог, %": round(l.tax_pct, 3) if l.found else None,
            "Итого издержки, %": round(l.cost_pct, 3) if l.found else None,
            "Замечание": l.note,
        })
    return pd.DataFrame(rows)
