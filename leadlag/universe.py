"""Отбор кандидатов в связки «фьючерс ↔ DEX».

Конвейер:
  1. Бессрочные USDT-фьючерсы со всех бирж из catalog.FUTURES (цена, объём, диапазон).
  2. Спот на крупных биржах — по каждой, чтобы исключение отбиралось галочками.
  3. Тикер фьючерса → монета CoinGecko. Тикеры совпадают у разных токенов,
     поэтому монета выбирается по цене: она должна совпасть с ценой фьючерса.
  4. Монета → адреса контрактов по сетям (CoinGecko platforms).
  5. Адрес → пулы на DEX (DexScreener), цена в пуле ещё раз сверяется с фьючерсом.

Результат — таблица без порогов: одна строка на пару «рынок фьючерса × сеть».
Пороги и галочки применяет страница, поэтому их смена не требует пересбора.

Запуск: python -m leadlag.universe
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from history.config import SETTINGS

from .catalog import CHAINS, FUTURES, SPOT, futures_url, norm_address
from .settings import LEADLAG_DIR

log = logging.getLogger(__name__)

try:
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover
    ccxt = None

UNIVERSE_PATH = LEADLAG_DIR / "universe.parquet"
META_PATH = LEADLAG_DIR / "universe_meta.json"

CG_BASE = "https://api.coingecko.com/api/v3"
DS_BASE = "https://api.dexscreener.com"
HTTP_TIMEOUT = 30
# Монета CoinGecko считается той же, что у фьючерса, если цены расходятся меньше
# чем на столько. Порог мягкий: цены снимаются в разные моменты.
IDENTITY_GAP = 0.10
USD_QUOTES = {"USDT", "USDC", "USD", "FDUSD", "USD1", "BUSD"}

# Токены традиционных активов: акции, фонды, сырьё. Их фьючерс идёт за фондовым
# рынком, а не за DEX, — связка «DEX ведёт фьючерс» к ним неприменима.
TRADFI_CATEGORIES = [
    "tokenized-stock", "tokenized-exchange-traded-funds-etfs", "tokenized-commodities",
    "tokenized-gold", "tokenized-pre-ipo-stocks", "xstocks-ecosystem",
    "backpack-securities-ecosystem", "bstocks-ecosystem",
]
_TRADFI_NAME = re.compile(r"\b(xStocks?|Securities|Tokenized|Stocks?|ETF|ETP)\b", re.I)

_MULT_RE = re.compile(r"^(1000000|100000|10000|1000|100|1M)(?=[A-Z])")


def split_multiplier(base: str) -> Tuple[str, float]:
    """1000PEPE → (PEPE, 1000): на фьючерсах мелкие монеты торгуют пачками."""
    m = _MULT_RE.match(base)
    if not m:
        return base, 1.0
    raw = m.group(1)
    mult = 1_000_000.0 if raw == "1M" else float(raw)
    return base[len(raw):], mult


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN → None


# --------------------------------------------------------------- 1. фьючерсы

def _exchange(ccxt_id: str, options: dict):
    cls = getattr(ccxt, ccxt_id)
    return cls({"enableRateLimit": True, "timeout": 30000, "options": dict(options)})


def fetch_futures(venue: str) -> List[dict]:
    v = FUTURES[venue]
    ex = _exchange(v.ccxt_id, v.options)
    markets = ex.load_markets()
    swaps = {s: m for s, m in markets.items()
             if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
             and m.get("active") is not False}
    tickers = ex.fetch_tickers(list(swaps)) if swaps else {}
    rows = []
    for sym, m in swaps.items():
        t = tickers.get(sym)
        if not t:
            continue
        last = _num(t.get("last")) or _num(t.get("close"))
        if not last:
            continue
        qv = _num(t.get("quoteVolume"))
        if qv is None:
            bv = _num(t.get("baseVolume"))
            qv = bv * last * float(m.get("contractSize") or 1) if bv else None
        high, low = _num(t.get("high")), _num(t.get("low"))
        coin, mult = split_multiplier(str(m["base"]).upper())
        rows.append({
            "fut_venue": venue,
            "fut_symbol": sym,
            "fut_id": m["id"],
            "fut_url": futures_url(venue, m["id"]),
            "coin": coin,
            "fut_mult": mult,
            "fut_price": last / mult,
            "fut_volume_usd": qv,
            "fut_range_pct": (high - low) / last * 100 if high and low else None,
            "fut_change_pct": _num(t.get("percentage")),
        })
    return rows


# ------------------------------------------------------------- 2. спот tier 1

def fetch_spot(venue: str) -> Dict[str, Optional[float]]:
    """Монета → цена в USD на споте этой биржи (None, если котировка не в USD)."""
    ex = _exchange(SPOT[venue], {"defaultType": "spot"})
    markets = ex.load_markets()
    spot = {s: m for s, m in markets.items() if m.get("spot") and m.get("active") is not False}
    usd = {s for s, m in spot.items() if m.get("quote") in USD_QUOTES}
    prices: Dict[str, Optional[float]] = {}
    try:
        # все тикеры одним запросом: список символов не влезает в URL у Binance
        tickers = {k: v for k, v in ex.fetch_tickers().items() if k in usd}
    except Exception as e:
        log.warning("спот %s: тикеры не получены (%s), сверка только по тикеру", venue, e)
        tickers = {}
    for s, m in spot.items():
        coin = str(m["base"]).upper()
        price = _num((tickers.get(s) or {}).get("last"))
        if price:
            prices[coin] = price
        else:
            prices.setdefault(coin, None)
    return prices


# ------------------------------------------------------------- 3–4. CoinGecko

class CoinGecko:
    def __init__(self):
        self.s = requests.Session()
        key = SETTINGS.coingecko_api_key
        if key:
            hdr = "x-cg-pro-api-key" if SETTINGS.coingecko_plan == "pro" else "x-cg-demo-api-key"
            self.s.headers[hdr] = key
        # без ключа — около 5–10 запросов в минуту
        self.pause = 2.5 if key else 12.0
        self._last = 0.0

    def get(self, path: str, **params):
        for attempt in range(5):
            wait = self.pause - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.s.get(CG_BASE + path, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"CoinGecko {path}: лимит запросов")

    def coins_with_platforms(self) -> List[dict]:
        return self.get("/coins/list", include_platform="true")

    def category_ids(self, category: str) -> set:
        ids, page = set(), 1
        while True:
            rows = self.get("/coins/markets", vs_currency="usd", category=category,
                            per_page=250, page=page)
            ids.update(r["id"] for r in rows)
            if len(rows) < 250 or page >= 8:
                return ids
            page += 1

    def prices(self, ids: List[str]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for i in range(0, len(ids), 200):
            chunk = ids[i:i + 200]
            for row in self.get("/coins/markets", vs_currency="usd", ids=",".join(chunk),
                                per_page=250, page=1):
                out[row["id"]] = row
        return out


def pick_coin(fut_price: float, candidates: List[dict]) -> Optional[dict]:
    """Монета CoinGecko, совпадающая с фьючерсом по цене (в пределах IDENTITY_GAP).

    Если по цене подходят несколько, берётся крупнейшая по капитализации:
    однотикерные копии почти всегда мелкие, а цены снимаются в разные моменты,
    так что «ближе по цене» само по себе ничего не доказывает.
    """
    fits = []
    for c in candidates:
        p = _num(c.get("current_price"))
        if p and abs(p - fut_price) / fut_price <= IDENTITY_GAP:
            fits.append((_num(c.get("market_cap")) or 0.0, -abs(p - fut_price), c))
    return max(fits, key=lambda t: t[:2])[2] if fits else None


# --------------------------------------------------------------- 5. DexScreener

def dex_pools(chain: str, address: str) -> List[dict]:
    c = CHAINS[chain]
    url = f"{DS_BASE}/token-pairs/v1/{c.dexscreener}/{address}"
    for attempt in range(5):
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        if r.status_code != 429:
            break
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    want = norm_address(chain, address)
    pools = []
    for p in r.json() or []:
        base = norm_address(chain, (p.get("baseToken") or {}).get("address", ""))
        quote = norm_address(chain, (p.get("quoteToken") or {}).get("address", ""))
        price_usd, native = _num(p.get("priceUsd")), _num(p.get("priceNative"))
        if base == want:
            price, other = price_usd, (p.get("quoteToken") or {}).get("symbol")
        elif quote == want and price_usd and native:
            price, other = price_usd / native, (p.get("baseToken") or {}).get("symbol")
        else:
            continue
        pools.append({
            "dex": p.get("dexId"),
            "pool": p.get("pairAddress"),
            "pool_url": p.get("url"),
            "pair_with": other,
            "price": price,
            "liquidity_usd": _num((p.get("liquidity") or {}).get("usd")) or 0.0,
            "volume_usd": _num((p.get("volume") or {}).get("h24")) or 0.0,
        })
    return pools


# ------------------------------------------------------------------ конвейер

@dataclass
class Funnel:
    futures_markets: int = 0
    futures_coins: int = 0
    coins_in_coingecko: int = 0
    coins_matched: int = 0
    coins_with_contract: int = 0
    coins_with_pools: int = 0


def build() -> Tuple[pd.DataFrame, dict]:
    if ccxt is None:
        raise RuntimeError("ccxt не установлен: pip install ccxt")
    started = time.time()
    errors: Dict[str, str] = {}
    funnel = Funnel()

    # 1–2: биржи параллельно, каждая со своим лимитом запросов
    def safe(fn, venue):
        try:
            return fn(venue)
        except Exception as e:
            errors[f"{fn.__name__}:{venue}"] = str(e)[:300]
            log.warning("%s %s: %s", fn.__name__, venue, e)
            return None

    # 4 потока, а не 20: справочники рынков ccxt тяжёлые, а сервер общий
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_jobs = {v: pool.submit(safe, fetch_futures, v) for v in FUTURES}
        spot_jobs = {v: pool.submit(safe, fetch_spot, v) for v in SPOT}
        fut_rows = [r for j in fut_jobs.values() for r in (j.result() or [])]
        spot = {v: j.result() for v, j in spot_jobs.items() if j.result() is not None}

    fut = pd.DataFrame(fut_rows)
    funnel.futures_markets = len(fut)
    if fut.empty:
        raise RuntimeError(f"ни одна биржа фьючерсов не ответила: {errors}")
    coin_price = fut.groupby("coin")["fut_price"].median()
    funnel.futures_coins = len(coin_price)
    log.info("фьючерсы: %d рынков, %d монет", len(fut), len(coin_price))

    # 3: тикер → монета CoinGecko по цене
    cg = CoinGecko()
    by_symbol: Dict[str, List[dict]] = {}
    for c in cg.coins_with_platforms():
        by_symbol.setdefault(c["symbol"].upper(), []).append(c)
    wanted = [i["id"] for coin in coin_price.index for i in by_symbol.get(coin, [])]
    funnel.coins_in_coingecko = sum(1 for coin in coin_price.index if coin in by_symbol)
    log.info("CoinGecko: %d кандидатов-монет на %d тикеров", len(wanted), funnel.coins_in_coingecko)
    markets = cg.prices(sorted(set(wanted)))

    matched: Dict[str, dict] = {}
    for coin, price in coin_price.items():
        cands = []
        for meta in by_symbol.get(coin, []):
            mk = markets.get(meta["id"])
            if mk:
                cands.append({**mk, "platforms": meta.get("platforms") or {}})
        best = pick_coin(float(price), cands)
        if best:
            matched[coin] = best
    funnel.coins_matched = len(matched)

    tradfi_ids = set()
    for cat in TRADFI_CATEGORIES:
        try:
            tradfi_ids |= cg.category_ids(cat)
        except Exception as e:
            errors[f"coingecko:category:{cat}"] = str(e)[:200]

    # 4–5: контракты по сетям → пулы
    platform_to_chain = {c.coingecko: c.key for c in CHAINS.values()}
    lookups: List[Tuple[str, str, str]] = []
    for coin, m in matched.items():
        for plat, addr in (m["platforms"] or {}).items():
            if addr and plat in platform_to_chain:
                lookups.append((coin, platform_to_chain[plat], addr.strip()))
    funnel.coins_with_contract = len({c for c, _, _ in lookups})

    def pools_for(item):
        coin, chain, addr = item
        try:
            return item, dex_pools(chain, addr)
        except Exception as e:
            errors[f"dexscreener:{chain}:{addr}"] = str(e)[:200]
            return item, []

    best_pool: Dict[Tuple[str, str], dict] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:  # ~300 запросов/мин у DexScreener
        for (coin, chain, addr), pools in pool.map(pools_for, lookups):
            if not pools:
                continue
            top = max(pools, key=lambda p: p["liquidity_usd"])
            best_pool[(coin, chain)] = {
                "chain": chain,
                "contract": addr,
                "dex": top["dex"],
                "dex_pool": top["pool"],
                "dex_url": top["pool_url"],
                "dex_pair_with": top["pair_with"],
                "dex_price": top["price"],
                "dex_pool_liquidity_usd": top["liquidity_usd"],
                "dex_liquidity_usd": sum(p["liquidity_usd"] for p in pools),
                "dex_volume_usd": sum(p["volume_usd"] for p in pools),
                "dex_pools": len(pools),
            }
    funnel.coins_with_pools = len({c for c, _ in best_pool})

    # сборка: рынок фьючерса × сеть
    rows = []
    fut_by_coin = {c: g for c, g in fut.groupby("coin")}
    for (coin, chain), dex in best_pool.items():
        m = matched[coin]
        tier1 = []
        for venue, prices in spot.items():
            if coin not in prices:
                continue
            p = prices[coin]
            # совпал тикер, но цена другая — на этой бирже другой токен
            if p and abs(p - coin_price[coin]) / coin_price[coin] > IDENTITY_GAP:
                continue
            tier1.append(venue)
        for _, f in fut_by_coin[coin].iterrows():
            gap = (abs(dex["dex_price"] - f["fut_price"]) / f["fut_price"] * 100
                   if dex["dex_price"] else None)
            rows.append({
                **f.to_dict(),
                "coingecko_id": m["id"],
                "coin_name": m.get("name"),
                "market_cap_usd": _num(m.get("market_cap")),
                "tradfi": m["id"] in tradfi_ids or bool(_TRADFI_NAME.search(m.get("name") or "")),
                "spot_on": ",".join(sorted(tier1)),
                **dex,
                "price_gap_pct": gap,
            })

    df = pd.DataFrame(rows)
    meta = {
        "built_at": int(time.time()),
        "duration_s": round(time.time() - started),
        "funnel": funnel.__dict__,
        "rows": len(df),
        "spot_checked": sorted(spot),
        "errors": errors,
    }
    return df, meta


def save(df: pd.DataFrame, meta: dict) -> None:
    LEADLAG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UNIVERSE_PATH.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(UNIVERSE_PATH)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load() -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    if not UNIVERSE_PATH.exists():
        return None, None
    df = pd.read_parquet(UNIVERSE_PATH)
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        meta = None
    return df, meta


def apply_filters(df: pd.DataFrame, s) -> pd.DataFrame:
    """Галочки и пороги из ScreenSettings поверх собранной таблицы."""
    if df is None or df.empty:
        return df
    tier1 = set(s.tier1_spot)
    on_tier1 = df["spot_on"].fillna("").map(lambda v: bool(tier1 & set(filter(None, v.split(",")))))
    mask = (
        df["fut_venue"].isin(s.futures_exchanges)
        & df["chain"].isin(s.chains)
        & ~on_tier1
        & ~(df["tradfi"].fillna(False).astype(bool) if s.exclude_tradfi and "tradfi" in df else False)
        & (df["fut_volume_usd"].fillna(0) >= s.min_fut_volume_usd)
        & (df["fut_range_pct"].fillna(0) >= s.min_fut_range_pct)
        & (df["dex_liquidity_usd"].fillna(0) >= s.min_dex_liquidity_usd)
        & (df["dex_volume_usd"].fillna(0) >= s.min_dex_volume_usd)
        & (df["price_gap_pct"].fillna(1e9) <= s.max_price_gap_pct)
    )
    return df[mask].sort_values(["fut_volume_usd"], ascending=False)


def run_once() -> None:
    df, meta = build()
    save(df, meta)
    log.info("готово за %s с: %d строк, воронка %s", meta["duration_s"], len(df), meta["funnel"])
    if meta["errors"]:
        log.warning("ошибки источников: %s", list(meta["errors"])[:20])


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Отбор кандидатов в связки фьючерс ↔ DEX")
    p.add_argument("--every-hours", type=float, default=0,
                   help="повторять каждые N часов (0 — один прогон)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    while True:
        started = time.time()
        try:
            run_once()
        except Exception:
            if not args.every_hours:
                raise
            log.exception("прогон упал, следующий по расписанию")
        if not args.every_hours:
            return
        time.sleep(max(60.0, args.every_hours * 3600 - (time.time() - started)))


if __name__ == "__main__":
    main()
