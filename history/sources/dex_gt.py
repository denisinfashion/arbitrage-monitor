"""История DEX через публичный API GeckoTerminal.

Бесплатный тариф: 30 запросов/мин, ключ не нужен. Из этого следует
двухуровневая схема сбора, иначе в лимит не уложиться:

  Уровень «живой» — эндпоинт top-pools отдаёт СРАЗУ 20 пулов за запрос,
  и в каждом уже есть текущая цена обоих токенов и резерв пула.
  100 пулов = 5 запросов = ~10 секунд. Это даёт свежесть.

  Уровень «история» — OHLCV запрашивается по одному пулу за раз
  (до 1000 свечей). 7 дней минутных свечей = 11 запросов на пул,
  на 100 пулов это ~1100 запросов ≈ 38 минут. Поэтому бэкфилл идёт
  фоном и по кругу, а не блокирует старт.

Важное про проскальзывание: OHLCV даёт только цену сделок без глубины.
Но top-pools отдаёт reserve_in_usd, и для пулов постоянного произведения
(V2) этого достаточно, чтобы восстановить price impact точно:
    amount_out = R_out * S / (R_in + S)
Для V3 концентрированная ликвидность около спота глубже, чем следует
из общего TVL, поэтому оценка получается консервативной — занижает
исполнимый объём. Для арбитража это безопасная сторона ошибки.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from ..config import (DEX_POOL_FEE_PCT, SETTINGS, load_watchlist,
                      norm_asset)
from ..http import HttpError, get_json
from ..store import Candle, get_last_ts, set_state, write_candles, write_pools

log = logging.getLogger(__name__)

# Три способа обращаться к одним и тем же данным. Без ключа лимит считается
# по IP, а раннеры GitHub делят адреса между тысячами проектов — квоту
# выбирает кто угодно, и исторические свечи становятся недоступны. Со своим
# ключом квота считается по нему, и общий адрес перестаёт мешать.
PUBLIC_API = "https://api.geckoterminal.com/api/v2"
DEMO_API = "https://api.coingecko.com/api/v3/onchain"
PRO_API = "https://pro-api.coingecko.com/api/v3/onchain"

HEADERS = {"Accept": "application/json;version=20230302"}


def endpoint(settings=SETTINGS) -> Tuple[str, dict]:
    """Адрес и заголовки под настроенный ключ."""
    key = (settings.coingecko_api_key or "").strip()
    if not key:
        return PUBLIC_API, dict(HEADERS)
    plan = (settings.coingecko_plan or "demo").strip().lower()
    if plan == "pro":
        return PRO_API, {**HEADERS, "x-cg-pro-api-key": key}
    return DEMO_API, {**HEADERS, "x-cg-demo-api-key": key}


# Совместимость со старым кодом и тестами
API = PUBLIC_API

MAX_FREE_PAGES = 10      # дальше требуется платный тариф
POOLS_PER_PAGE = 20
MAX_CANDLES = 1000

# ccxt-таймфрейм -> (timeframe, aggregate) в терминах GeckoTerminal
TIMEFRAME_MAP: Dict[str, Tuple[str, int]] = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "1h": ("hour", 1),
    "4h": ("hour", 4),
    "12h": ("hour", 12),
    "1d": ("day", 1),
}


def _f(value) -> Optional[float]:
    """GeckoTerminal отдаёт числа строками, иногда null."""
    if value is None or value == "":
        return None
    try:
        v = float(value)
        return v if v == v else None  # отсекаем NaN
    except (TypeError, ValueError):
        return None


def _parse_pool_name(name: str) -> Tuple[str, str]:
    """'WBNB / USDT 0.05%' -> ('WBNB', 'USDT'). Запасной путь, если
    include=base_token,quote_token не отработал."""
    if not name or "/" not in name:
        return "", ""
    left, _, right = name.partition("/")
    return left.strip().upper(), right.strip().split()[0].upper() if right.strip() else ""


class GeckoTerminalSource:
    """Сборщик котировок DEX по одной сети."""

    kind = "dex"

    def __init__(self, settings=SETTINGS):
        self.s = settings
        self.name = f"gt:{settings.chain}"
        self.chain = settings.chain
        self._pools: List[dict] = []
        self.last_backfill_complete = False
        """Очередь пулов дошла до конца и лимит не мешал."""
        self.api, self.headers = endpoint(settings)
        self.has_key = bool((settings.coingecko_api_key or "").strip())
        if self.has_key:
            log.info("%s: работаем с ключом CoinGecko (тариф %s)",
                     self.name, settings.coingecko_plan)

    # ------------------------------------------------------------- discover

    def discover(self) -> int:
        """Собирает пулы под наблюдение и снимает с них живые цены.

        Один запрос возвращает 20 пулов вместе с текущими ценами
        и резервами, поэтому этот же вызов работает как «живой» срез.

        Источников три, и второй с третьим появились не сразу.

        **Топ сети по обороту** — основа. Но сортировка не смотрит на
        площадку, а в BNB Chain оборот почти весь у PancakeSwap: в сотне
        лучших пулов посторонних площадок практически нет. Из-за этого
        все найденные связки исполнялись в одном месте, хотя разница цен
        между площадками — это ровно то, ради чего всё затевалось.

        **Топ каждой площадки отдельно** это чинит: по одному запросу на
        площадку, и у Biswap, Thena, SushiSwap появляются свои пулы.

        **Список наблюдения** — на случай, когда интересует конкретный
        токен, не проходящий по обороту ни в один топ. Ищется по тикеру,
        адрес указывать не нужно.
        """
        pools: List[dict] = []
        candles: List[Candle] = []
        now = int(time.time())
        tf_sec = self.s.timeframe_seconds()
        bucket = now - (now % tf_sec)
        seen: set = set()

        def take(payload) -> int:
            tokens = self._index_included(payload.get("included", []))
            items = payload.get("data", []) or []
            n = 0
            for item in items:
                parsed = self._parse_pool(item, tokens)
                if parsed is None or parsed["pool"] in seen:
                    continue
                seen.add(parsed["pool"])
                pools.append(parsed)
                snap = self._live_candle(parsed, bucket)
                if snap:
                    candles.append(snap)
                n += 1
            return n

        # --- 1. Топ сети ---------------------------------------------------
        need_pages = min(
            MAX_FREE_PAGES,
            max(1, -(-self.s.dex_pool_limit // POOLS_PER_PAGE)),
        )
        for page in range(1, need_pages + 1):
            try:
                payload = get_json(
                    f"{self.api}/networks/{self.chain}/pools",
                    params={"page": page, "include": "base_token,quote_token,dex",
                            "sort": "h24_volume_usd_desc"},
                    headers=self.headers,
                )
            except HttpError as exc:
                set_state(self.name, "discover", ok=False, error=str(exc))
                log.warning("discover страница %d: %s", page, exc)
                break
            if not take(payload):
                break

        # --- 2. Топ каждой площадки ----------------------------------------
        for dex in self.s.dex_venues or []:
            try:
                payload = get_json(
                    f"{self.api}/networks/{self.chain}/dexes/{dex}/pools",
                    params={"include": "base_token,quote_token,dex",
                            "sort": "h24_volume_usd_desc"},
                    headers=self.headers,
                )
            except HttpError as exc:
                # Площадки нет в этой сети или её переименовали — не повод
                # прерывать сбор: остальные источники от этого не страдают.
                log.debug("площадка %s: %s", dex, exc)
                continue
            got = take(payload)
            if got:
                log.debug("%s: +%d пулов площадки %s", self.name, got, dex)

        # --- 3. Список наблюдения ------------------------------------------
        # Пулы отсюда отмечаются: к ним применяется свой, низкий порог
        # ликвидности и они не вытесняются квотой. Иначе список наблюдения
        # был бы бесполезен ровно там, где он нужен: токен, не проходящий
        # в топ по обороту, обычно и по размеру пула не проходит общий порог.
        before_watch = len(pools)
        for token in load_watchlist(self.s):
            try:
                payload = get_json(
                    f"{self.api}/search/pools",
                    params={"query": token, "network": self.chain,
                            "include": "base_token,quote_token,dex"},
                    headers=self.headers,
                )
            except HttpError as exc:
                log.debug("поиск пулов %s: %s", token, exc)
                continue
            got = take(payload)
            log.info("%s: список наблюдения, %s — пулов найдено %d",
                     self.name, token, got)
        watched = {p["pool"] for p in pools[before_watch:]}

        # Отсев по ликвидности. Дальше — не просто «сто самых крупных»:
        # сначала каждой площадке отдаётся её квота, и лишь остаток лимита
        # разыгрывается по общему размеру. Иначе крупные пулы одной
        # площадки снова вытеснят всех остальных.
        floor_watch = float(getattr(self.s, "watch_min_reserve_usd",
                                    self.s.min_pool_reserve_usd))
        kept = []
        for p in pools:
            r = p["reserve_usd"] or 0
            limit = floor_watch if p["pool"] in watched else self.s.min_pool_reserve_usd
            if r >= limit:
                kept.append(p)
        dropped_watch = sum(1 for p in pools
                            if p["pool"] in watched and p not in kept)
        if dropped_watch:
            log.info("%s: из списка наблюдения отсеяно по ликвидности: %d "
                     "(порог $%s)", self.name, dropped_watch,
                     f"{floor_watch:,.0f}".replace(",", " "))
        pools = self._apply_venue_quota(kept, protected=watched)
        keep = {p["pool"] for p in pools}

        write_pools(pools)
        written = write_candles([c for c in candles if c.pool in keep])
        self._pools = pools

        set_state(self.name, "discover", ok=bool(pools),
                  error="" if pools else "пулы не получены", rows=written)
        by_dex: Dict[str, int] = {}
        for p in pools:
            by_dex[p["dex"]] = by_dex.get(p["dex"], 0) + 1
        top = sorted(by_dex.items(), key=lambda kv: -kv[1])[:6]
        log.info("%s: %d пулов под наблюдением, %d живых котировок; "
                 "площадок %d (%s)", self.name, len(pools), written, len(by_dex),
                 ", ".join(f"{d}:{n}" for d, n in top))
        return len(pools)

    def _apply_venue_quota(self, pools: List[dict],
                           protected: Optional[set] = None) -> List[dict]:
        """Оставляет каждой площадке её квоту, остаток — по размеру пула.

        Без этого шага отбор «сто самых крупных» отдаёт весь список одной
        площадке, и связки перестают быть межплощадочными.
        """
        quota = max(0, int(getattr(self.s, "dex_venue_quota", 0)))
        limit = int(self.s.dex_pool_limit)
        protected = protected or set()

        pools.sort(key=lambda p: p["reserve_usd"] or 0, reverse=True)

        # Пулы из списка наблюдения занимают места первыми: их запросили
        # явно, и обрезать их по размеру — значит проигнорировать просьбу.
        chosen: List[dict] = [p for p in pools if p["pool"] in protected]
        limit = max(limit, len(chosen))
        if not quota:
            rest = [p for p in pools if p["pool"] not in protected]
            return chosen + rest[: max(0, limit - len(chosen))]

        taken: Dict[str, int] = {}
        for p in chosen:
            taken[p["dex"]] = taken.get(p["dex"], 0) + 1
        for p in pools:
            if p["pool"] in protected:
                continue
            d = p["dex"]
            if taken.get(d, 0) < quota:
                taken[d] = taken.get(d, 0) + 1
                chosen.append(p)
            if len(chosen) >= limit:
                return chosen

        picked = {p["pool"] for p in chosen}
        for p in pools:
            if len(chosen) >= limit:
                break
            if p["pool"] not in picked:
                chosen.append(p)
        return chosen

    def _index_included(self, included: List[dict]) -> Dict[str, dict]:
        """id -> объект для секции included (токены и dex)."""
        return {obj.get("id", ""): obj for obj in included}

    def _parse_pool(self, item: dict, tokens: Dict[str, dict]) -> Optional[dict]:
        attrs = item.get("attributes", {}) or {}
        rel = item.get("relationships", {}) or {}
        address = attrs.get("address")
        if not address:
            return None

        def tok(side: str) -> Tuple[str, str, str]:
            ref = ((rel.get(side) or {}).get("data") or {}).get("id", "")
            obj = tokens.get(ref) or {}
            a = obj.get("attributes", {}) or {}
            sym = (a.get("symbol") or "").upper()
            addr = a.get("address") or (ref.split("_", 1)[-1] if "_" in ref else "")
            # Полное имя нужно для расшифровки тикеров в интерфейсе:
            # по «BTCB» не всякий сходу поймёт, что это Bitcoin в сети BNB.
            return sym, addr, (a.get("name") or "").strip()

        base_sym, base_addr, base_name = tok("base_token")
        quote_sym, quote_addr, quote_name = tok("quote_token")
        if not base_sym or not quote_sym:
            base_sym, quote_sym = _parse_pool_name(attrs.get("name", ""))
        if not base_sym or not quote_sym:
            return None

        dex_id = ((rel.get("dex") or {}).get("data") or {}).get("id", "") or "unknown"

        return {
            "chain": self.chain,
            "pool": address,
            "dex": dex_id,
            "base": norm_asset(base_sym),
            "quote": norm_asset(quote_sym),
            "base_addr": base_addr,
            "quote_addr": quote_addr,
            "base_name": base_name,
            "quote_name": quote_name,
            "reserve_usd": _f(attrs.get("reserve_in_usd")),
            "volume_24h": _f((attrs.get("volume_usd") or {}).get("h24")),
            "fee_pct": DEX_POOL_FEE_PCT.get(dex_id, DEX_POOL_FEE_PCT["default"]),
            # не пишется в таблицу pools, используется ниже
            "_base_usd": _f(attrs.get("base_token_price_usd")),
            "_quote_usd": _f(attrs.get("quote_token_price_usd")),
        }

    def _live_candle(self, pool: dict, ts: int) -> Optional[Candle]:
        """Текущая цена из top-pools как свеча текущего интервала."""
        b, q = pool.get("_base_usd"), pool.get("_quote_usd")
        if not b or not q or q <= 0:
            return None
        return Candle(
            ts=ts, venue=pool["dex"], venue_kind="dex", chain=self.chain,
            base=pool["base"], quote=pool["quote"],
            close=b / q, liquidity_usd=pool["reserve_usd"], pool=pool["pool"],
        )

    # ------------------------------------------------------------ pool list

    @property
    def pools(self) -> List[dict]:
        if not self._pools:
            from ..store import read_pools
            df = read_pools(self.chain, self.s.min_pool_reserve_usd)
            if not df.empty:
                self._pools = df.head(self.s.dex_pool_limit).to_dict("records")
            else:
                self.discover()
        return self._pools

    # -------------------------------------------------------------- OHLCV

    def _fetch_ohlcv(self, pool: dict, since_ts: int, until_ts: int) -> List[Candle]:
        """История одного пула. Пагинация идёт назад через before_timestamp."""
        tf, agg = TIMEFRAME_MAP.get(self.s.timeframe, ("minute", 1))
        tf_sec = self.s.timeframe_seconds()
        url = f"{self.api}/networks/{self.chain}/pools/{pool['pool']}/ohlcv/{tf}"

        out: List[Candle] = []
        cursor = until_ts
        guard = 0
        while cursor > since_ts and guard < 40:
            guard += 1
            payload = get_json(
                url,
                params={"aggregate": agg, "limit": MAX_CANDLES,
                        "before_timestamp": cursor, "currency": "usd", "token": "base"},
                headers=self.headers,
            )
            rows = (((payload.get("data") or {}).get("attributes") or {})
                    .get("ohlcv_list") or [])
            if not rows:
                break

            # currency=usd отдаёт цену базового токена в долларах.
            # Курс base->quote получаем делением на цену quote в долларах,
            # которую берём из последнего живого среза пула.
            quote_usd = pool.get("_quote_usd") or self._quote_usd_hint(pool)
            if not quote_usd or quote_usd <= 0:
                break

            oldest = cursor
            for ts, o, h, l, c, v in rows:
                ts = int(ts)
                oldest = min(oldest, ts)
                if ts < since_ts or not c or c <= 0:
                    continue
                out.append(Candle(
                    ts=ts, venue=pool["dex"], venue_kind="dex", chain=self.chain,
                    base=pool["base"], quote=pool["quote"],
                    open=(o / quote_usd) if o else None,
                    high=(h / quote_usd) if h else None,
                    low=(l / quote_usd) if l else None,
                    close=float(c) / quote_usd,
                    volume=v,
                    liquidity_usd=pool.get("reserve_usd"),
                    pool=pool["pool"],
                ))
            if oldest >= cursor or len(rows) < MAX_CANDLES:
                break
            cursor = oldest - tf_sec
        return out

    def _quote_usd_hint(self, pool: dict) -> Optional[float]:
        """Если котируемый токен — стейбл, его цена ≈ 1. Иначе пробуем
        взять последнюю известную цену из базы."""
        from ..config import USD_LIKE
        if pool.get("quote") in USD_LIKE:
            return 1.0
        from ..store import connect
        row = connect(read_only=True).execute(
            "SELECT close FROM quotes WHERE base=? AND quote IN ('USDT','USDC','BUSD') "
            "ORDER BY ts DESC LIMIT 1", (pool.get("quote"),)
        ).fetchone()
        return float(row["close"]) if row else None

    # ------------------------------------------------------------- backfill

    def backfill(self, days: Optional[float] = None, budget_requests: int = 120) -> int:
        """Дотягивает историю по пулам, у которых её меньше всего.

        budget_requests ограничивает число запросов за один заход,
        чтобы фоновый бэкфилл не съедал весь лимит и «живой» уровень
        продолжал обновляться.
        """
        days = days if days is not None else self.s.history_days
        until = int(time.time())
        since = until - int(days * 86400)

        pending = []
        for p in self.pools:
            last = get_last_ts(self.name + ":ohlcv", p["pool"])
            first = get_last_ts(self.name + ":ohlcv_first", p["pool"])
            if first and first <= since + self.s.timeframe_seconds() * 2:
                continue  # история набрана на нужную глубину
            pending.append((first or until, p))

        pending.sort(key=lambda x: -x[0])  # сперва те, у кого истории меньше
        total, spent = 0, 0
        throttled = 0
        self.last_backfill_complete = False

        for edge, pool in pending:
            if spent >= budget_requests:
                break
            try:
                candles = self._fetch_ohlcv(pool, since, int(edge))
            except HttpError as exc:
                set_state(self.name + ":ohlcv", pool["pool"], ok=False, error=str(exc))
                if exc.status == 429:
                    throttled += 1
                    # Без ключа лимит считается по общему IP раннера, и ждать
                    # почти всегда бессмысленно: на живом прогоне паузы
                    # 20+40+60 секунд не помогли, зато съели три с половиной
                    # минуты из восьми и оставили без данных биржу. Поэтому
                    # без ключа сдаёмся сразу, а с ключом квота своя —
                    # там отказ обычно временный и переждать стоит.
                    limit = 3 if self.has_key else 1
                    if throttled >= limit:
                        log.warning(
                            "GeckoTerminal держит лимит%s — история пулов "
                            "в этот прогон не собирается%s",
                            " даже с ключом" if self.has_key else " (общий IP раннера)",
                            "" if self.has_key else
                            ", живые срезы при этом работают",
                        )
                        break
                    wait = 20.0 * throttled
                    log.warning("лимит GeckoTerminal, пауза %.0f с", wait)
                    time.sleep(wait)
                    continue
                continue
            spent += max(1, len(candles) // MAX_CANDLES + 1)
            if not candles:
                set_state(self.name + ":ohlcv_first", pool["pool"], last_ts=since, ok=True)
                continue
            n = write_candles(candles)
            total += n
            set_state(self.name + ":ohlcv", pool["pool"],
                      last_ts=max(c.ts for c in candles), ok=True, rows=n)
            set_state(self.name + ":ohlcv_first", pool["pool"],
                      last_ts=min(c.ts for c in candles), ok=True)

        remaining = max(0, len(pending) - spent)
        self.last_backfill_complete = (remaining == 0 and throttled == 0)
        log.info("%s: бэкфилл %d свечей, %d пулов в очереди%s",
                 self.name, total, remaining,
                 f", отказов по лимиту: {throttled}" if throttled else "")
        return total

    def update(self) -> int:
        """Свежий срез: перечитывает top-pools. Дёшево — 5 запросов на 100 пулов."""
        return self.discover()
