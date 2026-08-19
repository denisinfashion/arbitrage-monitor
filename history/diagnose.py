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
from .rates import cex_slippage_factor, dex_depth_usd, dex_slippage_factor


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
    alternatives: int = 0
    note: str = ""

    @property
    def age_sec(self) -> Optional[float]:
        return None if self.ts is None else max(0.0, time.time() - self.ts)

    @property
    def cost_pct(self) -> float:
        """Полная издержка плеча в процентах: комиссия плюс проскальзывание."""
        return self.fee_pct + self.slip_pct


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
    verdict: str = ""
    stage: str = ""
    """Короткий код шага, на котором связка выпала."""
    missing: List[str] = field(default_factory=list)
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


def _best_row(quotes: pd.DataFrame, a: str, b: str) -> Optional[pd.Series]:
    """Самая свежая котировка пары в любую сторону, приведённая к a -> b.

    Направление не важно: цена пары обратима, и хранить обе стороны
    было бы удвоением данных. Важно только не перепутать её при развороте.
    """
    fwd = quotes[(quotes["base"] == a) & (quotes["quote"] == b)]
    bwd = quotes[(quotes["base"] == b) & (quotes["quote"] == a)]
    if fwd.empty and bwd.empty:
        return None

    rows = []
    if not fwd.empty:
        r = fwd.sort_values("ts").iloc[-1].copy()
        r["_rate"] = float(r["close"])
        rows.append(r)
    if not bwd.empty:
        r = bwd.sort_values("ts").iloc[-1].copy()
        price = float(r["close"])
        r["_rate"] = 1.0 / price if price > 0 else 0.0
        rows.append(r)
    rows.sort(key=lambda r: int(r["ts"]))
    best = rows[-1]
    best["_alternatives"] = len(fwd) + len(bwd)
    return best


def diagnose(chain: List[str], quotes: pd.DataFrame, trade_usd: float = 1000.0,
             window_h: float = 6.0, settings=SETTINGS) -> ChainReport:
    """Проходит связку плечо за плечом и объясняет результат."""
    rep = ChainReport(tickers=list(chain), trade_usd=trade_usd,
                      window_h=window_h, quotes_seen=len(quotes))
    if len(chain) < 3:
        rep.stage, rep.verdict = "input", "Нужно хотя бы два перехода."
        return rep

    product = 1.0
    net_product = 1.0
    for a, b in zip(chain, chain[1:]):
        leg = LegReport(frm=a, to=b)
        row = _best_row(quotes, a, b) if not quotes.empty else None
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

        liq = row.get("liquidity_usd")
        leg.liquidity_usd = float(liq) if liq == liq and liq is not None else None
        vol = row.get("volume")
        price = float(row["close"]) if row["close"] == row["close"] else 0.0
        leg.volume_usd = (float(vol) * price
                          if vol == vol and vol is not None else None)

        if leg.venue_kind == "dex":
            leg.fee_pct = dex_fee_pct(leg.venue)
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

        product *= leg.price
        net_product *= leg.price * (1.0 - leg.fee_pct / 100.0) * slip
        rep.legs.append(leg)

    rep.fees_pct = sum(l.fee_pct for l in rep.legs if l.found)
    rep.slip_pct = sum(l.slip_pct for l in rep.legs if l.found)

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
            "Итого издержки, %": round(l.cost_pct, 3) if l.found else None,
            "Замечание": l.note,
        })
    return pd.DataFrame(rows)
