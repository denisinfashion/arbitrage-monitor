"""Справочник площадок: биржи фьючерсов, биржи спота, сети DEX.

Всё, что зависит от конкретной биржи или сети, собрано здесь, чтобы
добавление новой площадки было одной строкой.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FuturesVenue:
    key: str            # имя в интерфейсе и в данных
    ccxt_id: str        # класс ccxt
    options: dict       # опции ccxt: тип рынка по умолчанию и загрузка только фьючерсов —
                        # справочник спота и опционов занимает сотни мегабайт
    url: str            # шаблон ссылки на торговую страницу, {id} — id рынка


FUTURES: Dict[str, FuturesVenue] = {v.key: v for v in [
    FuturesVenue("binance", "binanceusdm", {}, "https://www.binance.com/en/futures/{id}"),
    FuturesVenue("bybit", "bybit", {"defaultType": "swap", "fetchMarkets": {"types": ["linear"]}}, "https://www.bybit.com/trade/usdt/{id}"),
    FuturesVenue("okx", "okx", {"defaultType": "swap", "fetchMarkets": {"types": ["swap"]}}, "https://www.okx.com/trade-swap/{id_lower}"),
    FuturesVenue("gate", "gate", {"defaultType": "swap", "fetchMarkets": {"types": ["swap"]}}, "https://www.gate.com/futures/USDT/{id}"),
    FuturesVenue("bitget", "bitget", {"defaultType": "swap", "fetchMarkets": {"types": ["swap"]}}, "https://www.bitget.com/futures/usdt/{id}"),
    FuturesVenue("mexc", "mexc", {"defaultType": "swap", "fetchMarkets": {"types": {"spot": False, "swap": {"linear": True, "inverse": False}}}}, "https://futures.mexc.com/exchange/{id}"),
    FuturesVenue("kucoin", "kucoinfutures", {}, "https://www.kucoin.com/futures/trade/{id}"),
    FuturesVenue("bingx", "bingx", {"defaultType": "swap"}, "https://bingx.com/en/perpetual/{id}/"),
]}

# Биржи, на которых торгуем, — по умолчанию. MEXC уходит с рынка РФ: данные по нему
# собираются (как ориентир цены), но в связки, оповещения и бумажную торговлю
# он не попадает, пока его не отметить в интерфейсе.
FUTURES_TRADABLE_DEFAULT = [k for k in FUTURES if k != "mexc"]


# Тейкерская комиссия фьючерсов, % — базовый уровень без VIP и скидок за токен биржи.
FUTURES_TAKER_PCT: Dict[str, float] = {
    "binance": 0.05, "bybit": 0.055, "okx": 0.05, "gate": 0.05,
    "bitget": 0.06, "mexc": 0.02, "kucoin": 0.06, "bingx": 0.05,
}


# Биржи, наличие спота на которых исключает токен («tier 1»). Отбор — галочками
# в интерфейсе; сборщик проверяет все, чтобы смена галочек не требовала пересбора.
SPOT: Dict[str, str] = {
    "binance": "binance",
    "coinbase": "coinbaseexchange",
    "okx": "okx",
    "bybit": "bybit",
    "upbit": "upbit",
    "kraken": "kraken",
    "bitget": "bitget",
    "gate": "gate",
    "kucoin": "kucoin",
    "mexc": "mexc",
    "htx": "htx",
    "bingx": "bingx",
}
SPOT_TIER1_DEFAULT = ["binance", "coinbase", "okx", "bybit", "upbit"]


@dataclass(frozen=True)
class Chain:
    key: str
    label: str
    coingecko: str      # ключ платформы в CoinGecko /coins/list?include_platform
    dexscreener: str    # chainId в DexScreener
    geckoterminal: str  # id сети в GeckoTerminal (лента сделок пула)
    block_s: float      # типичное время блока — точность метки времени свопа
    evm: bool = True    # EVM-адреса сравниваются без учёта регистра


CHAINS: Dict[str, Chain] = {c.key: c for c in [
    Chain("bsc", "BNB Chain", "binance-smart-chain", "bsc", "bsc", 0.75),
    Chain("solana", "Solana", "solana", "solana", "solana", 0.4, evm=False),
    Chain("base", "Base", "base", "base", "base", 2.0),
    Chain("arbitrum", "Arbitrum", "arbitrum-one", "arbitrum", "arbitrum", 0.25),
    Chain("tron", "Tron", "tron", "tron", "tron", 3.0, evm=False),
    Chain("sui", "Sui", "sui", "sui", "sui-network", 0.5, evm=False),
    Chain("ton", "TON", "the-open-network", "ton", "ton", 5.0, evm=False),
    Chain("avalanche", "Avalanche", "avalanche", "avalanche", "avax", 2.0),
    Chain("polygon", "Polygon", "polygon-pos", "polygon", "polygon_pos", 2.0),
    Chain("ethereum", "Ethereum", "ethereum", "ethereum", "eth", 12.0),
]}
# Ethereum по умолчанию выключен: блок раз в 12 с — лаг в несколько секунд не виден.
CHAINS_DEFAULT = ["bsc", "solana", "base", "arbitrum", "tron", "sui"]


def futures_url(venue: str, market_id: str) -> Optional[str]:
    v = FUTURES.get(venue)
    if v is None:
        return None
    return v.url.format(id=market_id, id_lower=market_id.lower())


def norm_address(chain: str, address: str) -> str:
    c = CHAINS.get(chain)
    return address.lower() if c is None or c.evm else address
