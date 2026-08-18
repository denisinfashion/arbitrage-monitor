"""Ссылки на обмен и человекочитаемые названия токенов.

Задача простая по сути: получив ногу «обменять A на B на площадке V
в сети C», выдать адрес страницы обмена, который можно открыть во
встроенном браузере кошелька и сразу свопнуть.

Тонкость в том, что почти всем биржам нужны не тикеры, а адреса
контрактов: `CAKE` они не понимают, `0x0E09...` понимают. Адреса
приходят вместе со списком пулов от GeckoTerminal и лежат в таблице
`pools` — отсюда и берутся.

Форматы адресов страниц у площадок меняются, поэтому они собраны
в одном месте: если какая-то ссылка перестанет открываться, править
нужно только словарь SWAP_URL.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Сети
# --------------------------------------------------------------------------

CHAIN_INFO: Dict[str, dict] = {
    "bsc": {"name": "BNB Chain", "id": 56, "gecko": "bsc",
            "native": "BNB", "explorer": "https://bscscan.com"},
    "eth": {"name": "Ethereum", "id": 1, "gecko": "eth",
            "native": "ETH", "explorer": "https://etherscan.io"},
    "arbitrum": {"name": "Arbitrum", "id": 42161, "gecko": "arbitrum",
                 "native": "ETH", "explorer": "https://arbiscan.io"},
    "base": {"name": "Base", "id": 8453, "gecko": "base",
             "native": "ETH", "explorer": "https://basescan.org"},
    "polygon_pos": {"name": "Polygon", "id": 137, "gecko": "polygon_pos",
                    "native": "POL", "explorer": "https://polygonscan.com"},
}


def chain_name(chain: str) -> str:
    return CHAIN_INFO.get(chain, {}).get("name", chain or "—")


def chain_id(chain: str) -> Optional[int]:
    return CHAIN_INFO.get(chain, {}).get("id")


# --------------------------------------------------------------------------
# Адреса страниц обмена
# --------------------------------------------------------------------------
# Подстановки: {chain} — слаг сети, {cid} — числовой идентификатор,
# {a} — адрес отдаваемого токена, {b} — адрес получаемого.

SWAP_URL: Dict[str, str] = {
    "pancakeswap": "https://pancakeswap.finance/swap"
                   "?chain={chain}&inputCurrency={a}&outputCurrency={b}",
    "uniswap": "https://app.uniswap.org/swap"
               "?chain={chain}&inputCurrency={a}&outputCurrency={b}",
    "sushiswap": "https://www.sushi.com/swap?chainId={cid}&token0={a}&token1={b}",
    "biswap": "https://exchange.biswap.org/#/swap"
              "?inputCurrency={a}&outputCurrency={b}",
    "thena": "https://thena.fi/swap?inputCurrency={a}&outputCurrency={b}",
    "apeswap": "https://apeswap.finance/swap?inputCurrency={a}&outputCurrency={b}",
}

# Слаг сети в адресах PancakeSwap и Uniswap отличается от нашего
CHAIN_SLUG = {
    "pancakeswap": {"bsc": "bsc", "eth": "eth", "arbitrum": "arb", "base": "base"},
    "uniswap": {"bsc": "bnb", "eth": "mainnet", "arbitrum": "arbitrum",
                "base": "base", "polygon_pos": "polygon"},
}

# Универсальный запасной вариант: агрегатор, который сам найдёт маршрут.
# Пригождается, когда площадка неизвестна или её формат сменился.
AGGREGATOR_URL = "https://app.1inch.io/#/{cid}/simple/swap/{a}/{b}"


def _family(venue: str) -> str:
    """pancakeswap_v3 -> pancakeswap. Версия протокола на адрес не влияет."""
    v = (venue or "").lower().replace("-", "_")
    for known in SWAP_URL:
        if v.startswith(known):
            return known
    return v


def swap_url(venue: str, chain: str, addr_in: str, addr_out: str) -> Optional[str]:
    """Страница обмена для одной ноги. None, если адресов нет."""
    if not addr_in or not addr_out:
        return None

    fam = _family(venue)
    cid = chain_id(chain)
    tpl = SWAP_URL.get(fam)

    if tpl:
        slug = CHAIN_SLUG.get(fam, {}).get(chain, chain)
        return tpl.format(chain=slug, cid=cid or "", a=addr_in, b=addr_out)

    # Площадка незнакомая — отдаём агрегатор: он подберёт маршрут сам.
    if cid:
        return AGGREGATOR_URL.format(cid=cid, a=addr_in, b=addr_out)
    return None


def pool_url(chain: str, pool: str) -> Optional[str]:
    """Страница пула на GeckoTerminal — посмотреть график и ликвидность."""
    slug = CHAIN_INFO.get(chain, {}).get("gecko", chain)
    return f"https://www.geckoterminal.com/{slug}/pools/{pool}" if pool else None


def token_url(chain: str, address: str) -> Optional[str]:
    """Страница контракта в обозревателе сети — проверить, что токен тот самый."""
    base = CHAIN_INFO.get(chain, {}).get("explorer")
    return f"{base}/token/{address}" if base and address else None


# --------------------------------------------------------------------------
# Справочник токенов
# --------------------------------------------------------------------------
# Названия для тикеров, которые встречаются повсеместно. Для остальных
# имя приходит от GeckoTerminal вместе со списком пулов и хранится
# в таблице pools.

WELL_KNOWN: Dict[str, str] = {
    "USDT": "Tether", "USDC": "USD Coin", "BUSD": "Binance USD",
    "DAI": "Dai", "FDUSD": "First Digital USD", "TUSD": "TrueUSD",
    "BNB": "BNB", "WBNB": "Wrapped BNB", "ETH": "Ethereum",
    "WETH": "Wrapped Ethereum", "BTC": "Bitcoin", "BTCB": "Bitcoin BEP-20",
    "WBTC": "Wrapped Bitcoin", "CAKE": "PancakeSwap", "XRP": "XRP",
    "ADA": "Cardano", "DOGE": "Dogecoin", "SOL": "Solana", "DOT": "Polkadot",
    "LINK": "Chainlink", "AVAX": "Avalanche", "MATIC": "Polygon",
    "POL": "Polygon", "UNI": "Uniswap", "LTC": "Litecoin", "TRX": "TRON",
    "ATOM": "Cosmos", "NEAR": "NEAR Protocol", "APT": "Aptos",
    "SUI": "Sui", "TON": "Toncoin", "SHIB": "Shiba Inu", "PEPE": "Pepe",
    "ARB": "Arbitrum", "OP": "Optimism", "INJ": "Injective",
    "FIL": "Filecoin", "ETC": "Ethereum Classic", "BCH": "Bitcoin Cash",
    "XLM": "Stellar", "ALGO": "Algorand", "VET": "VeChain",
    "AAVE": "Aave", "MKR": "Maker", "CRV": "Curve", "LDO": "Lido",
    "SNX": "Synthetix", "COMP": "Compound", "GRT": "The Graph",
    "SAND": "The Sandbox", "MANA": "Decentraland", "AXS": "Axie Infinity",
    "TWT": "Trust Wallet Token", "XVS": "Venus", "ALPACA": "Alpaca Finance",
    "BAKE": "BakeryToken", "BURGER": "BurgerCities", "SFP": "SafePal",
    "ONDO": "Ondo", "ENA": "Ethena", "JUP": "Jupiter", "WIF": "dogwifhat",
    "BONK": "Bonk", "FLOKI": "Floki", "TRUMP": "Official Trump",
}


def token_name(symbol: str, from_pools: Optional[Dict[str, str]] = None) -> str:
    """Читаемое название токена по тикеру.

    Порядок: сперва встроенный справочник (он надёжнее и короче),
    затем имя из данных о пулах, иначе пусто.
    """
    s = (symbol or "").upper()
    if s in WELL_KNOWN:
        return WELL_KNOWN[s]
    if from_pools:
        name = from_pools.get(s) or from_pools.get(symbol or "")
        if name and name.upper() != s:
            return name
    return ""


def describe_path(assets, from_pools: Optional[Dict[str, str]] = None,
                  skip_anchor: bool = True) -> str:
    """Расшифровка тикеров маршрута: 'CAKE — PancakeSwap · BTCB — Bitcoin BEP-20'.

    Стартовый актив пропускается: он один и тот же во всех строках таблицы,
    и повторять его в каждой строке — только тратить ширину экрана.
    """
    seen, parts = set(), []
    body = assets[1:-1] if skip_anchor and len(assets) > 2 else assets
    for a in body:
        if a in seen:
            continue
        seen.add(a)
        name = token_name(a, from_pools)
        if name and name.upper() != a.upper():
            parts.append(f"{a} — {name}")
    return " · ".join(parts)
