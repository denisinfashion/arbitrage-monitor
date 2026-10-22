"""Настройки отбора кандидатов: значения по умолчанию и сохранение из интерфейса.

Пороги применяются к уже собранной таблице, поэтому их смена в интерфейсе
не требует нового прогона сборщика.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List

from history.config import DATA_DIR

from .catalog import CHAINS_DEFAULT, FUTURES, SPOT_TIER1_DEFAULT

log = logging.getLogger(__name__)

LEADLAG_DIR = DATA_DIR / "leadlag"
SETTINGS_PATH = LEADLAG_DIR / "settings.json"


@dataclass
class ScreenSettings:
    futures_exchanges: List[str] = field(default_factory=lambda: list(FUTURES))
    tier1_spot: List[str] = field(default_factory=lambda: list(SPOT_TIER1_DEFAULT))
    chains: List[str] = field(default_factory=lambda: list(CHAINS_DEFAULT))

    # Токенизированные акции, фонды и сырьё: их ведёт фондовый рынок, не DEX.
    exclude_tradfi: bool = True

    # Фьючерс: торгуемость. $1 млн в сутки — минимальный объём, на котором
    # вход-выход на несколько тысяч долларов не двигает цену.
    min_fut_volume_usd: float = 1_000_000
    # Диапазон high-low за 24 ч к цене. Меньше 5% — движений для отстающей
    # цены почти нет, зависимость нечем монетизировать.
    min_fut_range_pct: float = 5.0

    # DEX: пул должен формировать цену, а не повторять её.
    min_dex_liquidity_usd: float = 100_000
    min_dex_volume_usd: float = 50_000

    # Контроль сопоставления: цены фьючерса и пула — один и тот же актив.
    max_price_gap_pct: float = 3.0

    # Сбор сделок (ticks.py): сколько монет из прошедших фильтры вести
    # (крупнейшие по объёму фьючерса) и сколько дней хранить посекундные цены.
    track_max_coins: int = 40
    keep_days: float = 3.0

    # Спреды фьючерсов между биржами (spreads.py)
    spread_hold_hours: float = 8.0          # сколько держим связку — для расчёта funding
    spread_min_volume_usd: float = 500_000  # объём за 24 ч у каждой ноги
    spread_min_pnl_pct: float = 0.10        # итог за удержание после комиссий и funding
    spread_max_price_ratio: float = 1.30    # больше — скорее всего разные токены под одним тикером

    # Оповещения в Telegram (notify.py)
    alerts_enabled: bool = True
    alert_min_pnl_pct: float = 0.5          # итог связки за удержание
    alert_min_lag_hours: int = 12           # сколько часов лаг должен держаться

    # Поиск лага (lag.py)
    lag_window_hours: int = 24     # за сколько часов оценивать устойчивость
    max_lag_s: int = 60            # сдвиги от −max до +max секунд
    min_corr: float = 0.10         # пик корреляции слабее — связи нет

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "ScreenSettings":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except Exception as e:  # битый файл не должен ронять страницу
            log.warning("настройки не прочитаны (%s), беру по умолчанию", e)
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path = SETTINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                        encoding="utf-8")
