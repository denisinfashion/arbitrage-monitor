"""Настройки исторического сканнера.

Все параметры собраны здесь, чтобы охват и глубину можно было менять
в одном месте, не трогая код сборщиков.

Значения по умолчанию соответствуют согласованному объёму:
BNB Chain, топ-100 пулов, 7 дней истории с шагом 1 минута,
вход и выход в USDT, связки до 4 ног.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

# --------------------------------------------------------------------------
# Пути
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ARB_DATA_DIR", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "history.sqlite"
LOG_PATH = DATA_DIR / "collector.log"


# --------------------------------------------------------------------------
# Параметры сбора
# --------------------------------------------------------------------------


@dataclass
class Settings:
    """Конфигурация сбора и анализа."""

    # --- охват -------------------------------------------------------------
    chain: str = "bsc"
    """Сеть DEX. Идентификатор в терминах GeckoTerminal: bsc, eth, arbitrum..."""

    quote_asset: str = "USDT"
    """Стартовый и конечный актив всех связок."""

    dex_pool_limit: int = 100
    """Сколько топовых пулов сети держать под наблюдением."""

    min_pool_reserve_usd: float = 100_000.0
    """Пулы с ликвидностью ниже порога не берём: маржа съедается проскальзыванием."""

    cex_venues: List[str] = field(
        default_factory=lambda: ["binance", "okx", "bybit", "gate", "kucoin", "mexc"]
    )
    """Идентификаторы бирж в терминах ccxt."""

    cex_quote_assets: List[str] = field(default_factory=lambda: ["USDT", "USDC", "BTC", "ETH", "BNB"])
    """Котируемые валюты, по которым тянем пары с CEX."""

    cex_symbol_limit: int = 200
    """Максимум пар на биржу (отбор по объёму за 24 ч)."""

    spot_only: bool = True
    """Брать только обычные спотовые активы.

    Отсекает токены с плечом (BTC3L, ETHUP, BTCBULL и подобные). Биржи
    размещают их на спотовом рынке, но для арбитража они непригодны:
    внутри ежедневная перебалансировка, цена следует за NAV, а не за
    базовым активом, и разница цен между площадками неисполнима.
    """

    # --- глубина истории ---------------------------------------------------
    history_days: float = 7.0
    """Глубина истории от текущего момента назад, в днях."""

    timeframe: str = "1m"
    """Гранулярность свечей: 1m, 5m, 15m, 1h."""

    # --- частота обновления ------------------------------------------------
    cex_refresh_sec: int = 60
    """Как часто дотягивать свежие свечи с CEX."""

    dex_refresh_sec: int = 120
    """Как часто дотягивать свежие свечи с DEX."""

    pool_rediscover_sec: int = 3600
    """Как часто перечитывать список топовых пулов."""

    # --- анализ ------------------------------------------------------------
    max_legs: int = 4
    """Максимальная длина связки USDT -> ... -> USDT."""

    trade_size_usd: float = 1_000.0
    """Размер сделки, под который считается проскальзывание."""

    staleness_sec: int = 180
    """Котировка старше этого возраста считается протухшей и в связку не идёт.

    Это главный предохранитель от ложных срабатываний: forward-fill
    устаревшей цены фабрикует арбитраж, которого не было.
    """

    min_margin_pct: float = 0.0
    """Порог отсечения связок в таблице результатов."""

    # --- ключи (опционально) -----------------------------------------------
    graph_api_key: str = os.environ.get("GRAPH_API_KEY", "")
    """Ключ The Graph для сабграфов PancakeSwap. Без него работает
    приближённая модель проскальзывания по reserve_in_usd от GeckoTerminal."""

    def timeframe_seconds(self) -> int:
        unit = self.timeframe[-1]
        n = int(self.timeframe[:-1])
        return n * {"m": 60, "h": 3600, "d": 86400}[unit]

    def expected_candles(self) -> int:
        return int(self.history_days * 86400 / self.timeframe_seconds())


SETTINGS = Settings()


# --------------------------------------------------------------------------
# Комиссии площадок (тейкер, %)
# --------------------------------------------------------------------------
# CEX: базовые спотовые ставки без VIP-скидок. ccxt отдаёт их же в market['taker'],
# но не у всех бирж и не всегда актуально, поэтому держим свой словарь как fallback.

CEX_TAKER_PCT = {
    "binance": 0.10,
    "okx": 0.10,
    "bybit": 0.10,
    "gate": 0.09,
    "kucoin": 0.10,
    "mexc": 0.05,
    "bitget": 0.10,
    "htx": 0.20,
    "kraken": 0.40,
    "coinbase": 0.60,
}

# DEX: комиссия пула уже заложена в котировку/резервы, отдельно не начисляется.
# Здесь только типовые ставки протоколов для справки и для реконструкции V2.
DEX_POOL_FEE_PCT = {
    "pancakeswap_v2": 0.25,
    "pancakeswap-v2": 0.25,
    "pancakeswap_v3": 0.25,  # переменная, уточняется по fee-тиру пула
    "pancakeswap-v3": 0.25,
    "uniswap_v2": 0.30,
    "uniswap_v3": 0.30,
    "biswap": 0.10,
    "thena": 0.20,
    "default": 0.25,
}

# Стейблкоины: для них вход/выход считается по номиналу
USD_LIKE = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDD", "USDe", "FRAX"}

# Обёртки нативных монет: WBNB и BNB считаем одним активом
WRAPPED = {
    "WBNB": "BNB",
    "WETH": "ETH",
    "WBTC": "BTC",
    "BTCB": "BTC",
    "WMATIC": "MATIC",
    "WAVAX": "AVAX",
}


def norm_asset(symbol: str) -> str:
    """Приводит обёрнутые токены к базовому тикеру."""
    s = (symbol or "").strip().upper()
    return WRAPPED.get(s, s)


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


# --------------------------------------------------------------------------
# Токены с плечом
# --------------------------------------------------------------------------
# Биржи размещают их на спотовом рынке, и ccxt честно помечает такие пары
# как spot — отличить их по флагам нельзя. Между тем для арбитража они
# непригодны: внутри ежедневная перебалансировка позиции, цена привязана
# к NAV, а не к базовому активу, и видимый «спред» неисполним.
# Отсюда распознавание по имени.

_LEVERAGED_SUFFIX = re.compile(r"^(?P<base>[A-Z0-9]{2,12}?)(?P<tag>[2345](?:L|S)|UP|DOWN|BULL|BEAR|HEDGE|HALF)$")
"""BTC3L, ETH5S, BTCUP, ETHDOWN, BTCBULL — база плюс маркер плеча."""

_LEVERAGED_INFIX = re.compile(r"^[A-Z0-9]{2,12}(3X|5X)(LONG|SHORT)$")
"""Реже встречающийся вид: BTC3XLONG."""


def is_leveraged_token(symbol: str, known_assets: Optional[set] = None) -> bool:
    """Похож ли тикер на токен с плечом.

    known_assets, если передан, резко снижает число ложных срабатываний:
    маркер засчитывается, только когда остаток имени сам по себе торгуется.
    Без этой проверки под шаблон попал бы, например, обычный токен, чьё имя
    случайно кончается на UP или BEAR.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return False

    if _LEVERAGED_INFIX.match(s):
        return True

    m = _LEVERAGED_SUFFIX.match(s)
    if not m:
        return False

    base = m.group("base")
    if len(base) < 2:
        return False
    if known_assets is None:
        # Без справочника доверяем только однозначным маркерам:
        # цифра с буквой (3L, 5S) подделать сложно, слова — легко.
        return bool(re.fullmatch(r"[2345](L|S)", m.group("tag")))
    return base in known_assets


def filter_leveraged(symbols, known_assets: Optional[set] = None) -> list:
    """Оставляет только обычные спотовые тикеры."""
    return [s for s in symbols if not is_leveraged_token(s, known_assets)]
