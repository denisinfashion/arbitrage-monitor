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
    options: dict       # опции ccxt (тип рынка по умолчанию)
    url: str            # шаблон ссылки на торговую страницу, {id} — id рынка


FUTURES: Dict[str, FuturesVenue] = {v.key: v for v in [
    FuturesVenue("binance", "binanceusdm", {}, "https://www.binance.com/en/futures/{id}"),
    FuturesVenue("bybit", "bybit", {"defaultType": "swap"}, "https://www.bybit.com/trade/usdt/{id}"),
    FuturesVenue("okx", "okx", {"defaultType": "swap"}, "https://www.okx.com/trade-swap/{id_lower}"),
    FuturesVenue("gate", "gate", {"defaultType": "swap"}, "https://www.gate.com/futures/USDT/{id}"),
    FuturesVenue("bitget", "bitget", {"defaultType": "swap"}, "https://www.bitget.com/futures/usdt/{id}"),
    FuturesVenue("mexc", "mexc", {"defaultType": "swap"}, "https://futures.mexc.com/exchange/{id}"),
    FuturesVenue("kucoin", "kucoinfutures", {}, "https://www.kucoin.com/futures/trade/{id}"),
    FuturesVenue("bingx", "bingx", {"defaultType": "swap"}, "https://bingx.com/en/perpetual/{id}/"),
]}


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
    block_s: float      # типичное время блока — точность метки времени свопа
    evm: bool = True    # EVM-адреса сравниваются без учёта регистра


CHAINS: Dict[str, Chain] = {c.key: c for c in [
    Chain("bsc", "BNB Chain", "binance-smart-chain", "bsc", 0.75),
    Chain("solana", "Solana", "solana", "solana", 0.4, evm=False),
    Chain("base", "Base", "base", "base", 2.0),
    Chain("arbitrum", "Arbitrum", "arbitrum-one", "arbitrum", 0.25),
    Chain("tron", "Tron", "tron", "tron", 3.0, evm=False),
    Chain("sui", "Sui", "sui", "sui", 0.5, evm=False),
    Chain("ton", "TON", "the-open-network", "ton", 5.0, evm=False),
    Chain("avalanche", "Avalanche", "avalanche", "avalanche", 2.0),
    Chain("polygon", "Polygon", "polygon-pos", "polygon", 2.0),
    Chain("ethereum", "Ethereum", "ethereum", "ethereum", 12.0),
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
