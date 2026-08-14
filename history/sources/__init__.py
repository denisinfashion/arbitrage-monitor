"""Источники исторических котировок.

Общий контракт: источник умеет
  - discover()  — определить, что наблюдать (список пар/пулов);
  - backfill()  — дотянуть историю до заданной глубины;
  - update()    — добрать свежие свечи с момента последнего сбора.

Все источники пишут в общий формат store.Candle, поэтому анализ
не знает, откуда пришли данные.
"""

from __future__ import annotations

from typing import List, Protocol

from ..store import Candle


class Source(Protocol):
    name: str
    kind: str  # 'cex' | 'dex'

    def discover(self) -> int: ...
    def backfill(self, days: float) -> int: ...
    def update(self) -> int: ...


__all__ = ["Source", "Candle"]
