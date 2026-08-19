"""История с централизованных бирж через ccxt.

Почему ccxt, а не собственные адаптеры из exchanges.py:
  - fetch_ohlcv даёт ровно то, чего не было — историю, до 1000 свечей за запрос;
  - единый формат комиссий, лимитов и точности по 100+ биржам;
  - встроенный rateLimit на каждую биржу, ничего не надо угадывать.

Исходный exchanges.py остаётся нетронутым: он даёт стакан для live-калькулятора,
здесь нужен другой срез данных.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import (CEX_MAX_HISTORY_CANDLES, CEX_MAX_WORKERS, CEX_TAKER_PCT,
                      DEFAULT_CEX_WORKERS, SETTINGS, is_leveraged_token, norm_asset)
from ..store import Candle, get_last_ts, set_state, write_candles, write_fees

log = logging.getLogger(__name__)

try:
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover
    ccxt = None


MAX_CANDLES_PER_CALL = 1000


class CexSource:
    """Сборщик OHLCV с одной биржи."""

    kind = "cex"

    def __init__(self, exchange_id: str, settings=SETTINGS):
        if ccxt is None:
            raise RuntimeError("ccxt не установлен: pip install ccxt")
        self.name = exchange_id
        self.s = settings
        cls = getattr(ccxt, exchange_id, None)
        if cls is None:
            raise ValueError(f"ccxt не знает биржу {exchange_id!r}")
        self.ex = cls({"enableRateLimit": True, "timeout": 20000})
        self._symbols: List[str] = []
        self._markets_loaded = False
        self._depth_capped = False

    # ---------------------------------------------------------------- utils

    def _load_markets(self) -> None:
        if not self._markets_loaded:
            self.ex.load_markets()
            self._markets_loaded = True

    def _taker_pct(self) -> float:
        """Тейкерская комиссия в процентах."""
        fallback = CEX_TAKER_PCT.get(self.name, 0.10)
        try:
            fee = self.ex.fees.get("trading", {}).get("taker")
            if fee is not None:
                return float(fee) * 100.0
        except Exception:
            pass
        return fallback

    # ------------------------------------------------------------- discover

    def discover(self) -> int:
        """Отбирает пары к наблюдению: спот, активные, с нужной котировкой,
        отсортированные по суточному объёму."""
        self._load_markets()
        write_fees(self.name, "cex", self._taker_pct(), self._taker_pct())

        wanted_quotes = {q.upper() for q in self.s.cex_quote_assets}
        candidates = [
            m for m in self.ex.markets.values()
            if m.get("spot") and m.get("active", True)
            and (m.get("quote") or "").upper() in wanted_quotes
        ]
        if not candidates:
            self._symbols = []
            return 0

        if self.s.spot_only:
            candidates = self._drop_leveraged(candidates)

        ranked = self._rank_by_volume(candidates)
        # У части площадок квота не выдерживает общего потолка:
        # Bitfinex на 79 парах отдавал 54 отказа из-за лимита.
        from ..config import CEX_SYMBOL_LIMITS
        cap = min(self.s.cex_symbol_limit,
                  CEX_SYMBOL_LIMITS.get(self.name, self.s.cex_symbol_limit))
        self._symbols = ranked[:cap]
        log.info("%s: отобрано %d пар", self.name, len(self._symbols))
        return len(self._symbols)

    def _drop_leveraged(self, markets):
        """Убирает токены с плечом.

        Справочник обычных активов собирается из тех же рынков биржи:
        маркер плеча засчитывается, только если остаток имени сам
        торгуется здесь же. Так BTC3L отсеивается, а токен, чьё имя
        просто оканчивается на UP, остаётся.
        """
        known = {(m.get("base") or "").upper() for m in self.ex.markets.values()
                 if m.get("spot")}
        kept, dropped = [], []
        for m in markets:
            if is_leveraged_token((m.get("base") or "").upper(), known):
                dropped.append(m["symbol"])
            else:
                kept.append(m)
        if dropped:
            log.info("%s: отсеяно токенов с плечом: %d (%s%s)",
                     self.name, len(dropped), ", ".join(sorted(dropped)[:5]),
                     "…" if len(dropped) > 5 else "")
        return kept

    def _rank_by_volume(self, markets: Sequence[dict]) -> List[str]:
        """Сортирует пары по объёму за 24 ч. Если биржа не отдаёт тикеры
        пачкой — падаем на алфавитный порядок с приоритетом USDT."""
        symbols = [m["symbol"] for m in markets]
        try:
            if self.ex.has.get("fetchTickers"):
                tickers = self.ex.fetch_tickers()
                def vol(sym: str) -> float:
                    t = tickers.get(sym) or {}
                    return float(t.get("quoteVolume") or t.get("baseVolume") or 0.0)
                return sorted(symbols, key=vol, reverse=True)
        except Exception as exc:
            log.debug("%s: fetch_tickers не сработал (%s)", self.name, exc)
        prio = {"USDT": 0, "USDC": 1, "BTC": 2, "ETH": 3, "BNB": 4}
        return sorted(symbols, key=lambda s: (prio.get(s.split("/")[-1], 9), s))

    @property
    def symbols(self) -> List[str]:
        if not self._symbols:
            self.discover()
        return self._symbols

    # ------------------------------------------------------------- fetching

    def _fetch_symbol(self, symbol: str, since_ms: int, until_ms: int) -> List[Candle]:
        """Тянет свечи по одной паре постранично от since до until."""
        tf = self.s.timeframe
        tf_ms = self.s.timeframe_seconds() * 1000
        out: List[Candle] = []
        cursor = since_ms
        taker = self._taker_pct()
        market = self.ex.markets.get(symbol, {})
        base = norm_asset(market.get("base") or symbol.split("/")[0])
        quote = norm_asset(market.get("quote") or symbol.split("/")[-1])

        guard = 0
        while cursor < until_ms and guard < 200:
            guard += 1
            batch = self._fetch_batch(symbol, tf, cursor)
            if batch is None:
                break
            if not batch:
                break
            for ts_ms, o, h, l, c, v in batch:
                if c is None or c <= 0 or ts_ms > until_ms:
                    continue
                out.append(Candle(
                    ts=ts_ms // 1000, venue=self.name, venue_kind="cex", chain="",
                    base=base, quote=quote,
                    open=o, high=h, low=l, close=float(c), volume=v,
                ))
            last = batch[-1][0]
            if last <= cursor:
                break
            cursor = last + tf_ms
            if len(batch) < MAX_CANDLES_PER_CALL:
                break
        return out

    def _fetch_batch(self, symbol: str, tf: str, cursor: int):
        """Один запрос свечей с повторами при отказе по лимиту.

        Троттлинг ccxt рассчитан на последовательные вызовы: несколько
        потоков на одном объекте биржи сверяются с общей меткой времени
        и стреляют одновременно. OKX на живом прогоне отдал 50011
        (Too Many Requests) по 75 парам из 200. Помимо снижения
        параллелизма нужен и повтор с нарастающей паузой — квота
        восстанавливается за секунды.

        Возвращает список свечей, либо None, если пару надо пропустить
        (историю глубже биржа не отдаёт).
        """
        attempts = 4
        delay = 1.0
        for attempt in range(attempts):
            try:
                return self.ex.fetch_ohlcv(symbol, tf, since=cursor,
                                           limit=MAX_CANDLES_PER_CALL)
            except Exception as exc:
                name = type(exc).__name__
                text = str(exc)

                # Биржа сообщает, что запрошенная глубина недоступна.
                # Запоминаем ограничение, чтобы не долбиться по каждой паре.
                if self._is_depth_error(text):
                    self._note_depth_limit()
                    return None

                if name in ("RateLimitExceeded", "DDoSProtection"):
                    if attempt + 1 < attempts:
                        time.sleep(delay)
                        delay *= 2.5
                        continue
                    raise RuntimeError(
                        f"{symbol}: лимит запросов биржи не отпустил "
                        f"за {attempts} попытки"
                    ) from exc

                raise RuntimeError(f"{symbol}: {name} {text}") from exc

        raise RuntimeError(f"{symbol}: не удалось получить свечи")

    @staticmethod
    def _is_depth_error(text: str) -> bool:
        t = text.lower()
        return ("too long ago" in t or "maximum" in t and "points" in t
                or "too old" in t)

    def _note_depth_limit(self) -> None:
        """Запоминает, что биржа не отдаёт запрошенную глубину."""
        if self._depth_capped:
            return
        self._depth_capped = True
        log.warning("%s: глубина ограничена самой биржей, "
                    "следующие запросы укоротим", self.name)

    def _max_history_candles(self) -> Optional[int]:
        cap = CEX_MAX_HISTORY_CANDLES.get(self.name)
        if self._depth_capped:
            # Известного значения нет, но биржа отказала — берём безопасное.
            cap = min(cap or 9_800, 9_800)
        return cap

    def _clamp_since(self, since_ms: int, until_ms: int) -> int:
        """Не просим глубже, чем биржа готова отдать."""
        cap = self._max_history_candles()
        if not cap:
            return since_ms
        tf_ms = self.s.timeframe_seconds() * 1000
        earliest = until_ms - cap * tf_ms
        if since_ms < earliest:
            log.info("%s: глубина урезана до %d свечей (%.1f дн) — ограничение биржи",
                     self.name, cap, cap * tf_ms / 86_400_000)
            return earliest
        return since_ms

    # -------------------------------------------------------------- backfill

    def _workers(self, requested: Optional[int]) -> int:
        if requested is not None:
            return requested
        return CEX_MAX_WORKERS.get(self.name, DEFAULT_CEX_WORKERS)

    def backfill(self, days: Optional[float] = None,
                 max_workers: Optional[int] = None) -> int:
        """Первичная загрузка истории заданной глубины."""
        days = days if days is not None else self.s.history_days
        until_ms = int(time.time() * 1000)
        since_ms = self._clamp_since(until_ms - int(days * 86400 * 1000), until_ms)
        return self._run(since_ms, until_ms, self._workers(max_workers),
                         mode="backfill")

    def update(self, max_workers: Optional[int] = None) -> int:
        """Инкрементальная докачка от последней собранной свечи."""
        until_ms = int(time.time() * 1000)
        default_since = self._clamp_since(
            until_ms - int(self.s.history_days * 86400 * 1000), until_ms)
        return self._run(None, until_ms, self._workers(max_workers),
                         mode="update", default_since_ms=default_since)

    def _run(self, since_ms: Optional[int], until_ms: int, max_workers: int,
             mode: str, default_since_ms: Optional[int] = None) -> int:
        syms = self.symbols
        if not syms:
            set_state("cex:" + self.name, "*", ok=False, error="нет подходящих пар")
            return 0

        tf_ms = self.s.timeframe_seconds() * 1000
        total = 0
        errors = 0

        def job(symbol: str) -> Tuple[str, int, Optional[str]]:
            start = since_ms
            if start is None:
                last = get_last_ts("cex:" + self.name, symbol)
                start = (last * 1000 + tf_ms) if last else (default_since_ms or 0)
                start = self._clamp_since(start, until_ms)
            if start >= until_ms:
                return symbol, 0, None
            try:
                candles = self._fetch_symbol(symbol, start, until_ms)
            except Exception as exc:
                return symbol, 0, str(exc)[:300]
            n = write_candles(candles)
            if candles:
                set_state("cex:" + self.name, symbol,
                          last_ts=max(c.ts for c in candles), ok=True, rows=n)
            return symbol, n, None

        # ccxt сам держит rateLimit, поэтому параллелизм умеренный
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(job, s): s for s in syms}
            for f in as_completed(futures):
                symbol, n, err = f.result()
                total += n
                if err:
                    errors += 1
                    if errors <= 3:
                        log.warning("%s %s: %s", self.name, symbol, err)
                    set_state("cex:" + self.name, symbol, ok=False, error=err)

        status_ok = errors < len(syms)
        set_state("cex:" + self.name, "*", ok=status_ok,
                  error=f"{errors} пар с ошибкой из {len(syms)}" if errors else "",
                  rows=0)
        log.info("%s [%s]: %d свечей, ошибок %d/%d", self.name, mode, total, errors, len(syms))
        return total


def build_sources(settings=SETTINGS) -> List[CexSource]:
    """Создаёт сборщики по списку бирж из настроек, пропуская недоступные."""
    out: List[CexSource] = []
    for ex_id in settings.cex_venues:
        try:
            out.append(CexSource(ex_id, settings))
        except Exception as exc:
            log.warning("Биржа %s недоступна: %s", ex_id, exc)
    return out
