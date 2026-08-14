"""Фоновый сборщик котировок.

Запускается отдельным процессом и живёт независимо от Streamlit:

    python -m history.collector

В исходном проекте «обновление» было устроено как time.sleep(15) + st.rerun()
внутри UI, то есть сбор шёл только пока открыта вкладка. Здесь наоборот:
сборщик пишет в SQLite, а UI только читает — можно закрыть браузер,
данные продолжат накапливаться.

Расписание внутри одного цикла:
  * живой срез DEX  — часто и дёшево (top-pools, 5 запросов на 100 пулов);
  * инкремент CEX   — раз в cex_refresh_sec;
  * бэкфилл истории — фоном, с бюджетом запросов, чтобы не выесть лимит;
  * обрезка глубины — раз в час.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import List

from . import store
from .config import LOG_PATH, SETTINGS, ensure_data_dir
from .sources.cex_ccxt import build_sources as build_cex
from .sources.dex_gt import GeckoTerminalSource

log = logging.getLogger("collector")

_STOP = False


def _handle_signal(signum, frame):  # pragma: no cover
    global _STOP
    _STOP = True
    log.info("получен сигнал %s, завершаюсь после текущего шага", signum)


def setup_logging(verbose: bool = False) -> None:
    ensure_data_dir()
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOG_PATH, encoding="utf-8")],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class Collector:
    def __init__(self, settings=SETTINGS, with_cex: bool = True, with_dex: bool = True):
        self.s = settings
        store.init()
        self.dex = GeckoTerminalSource(settings) if with_dex else None
        self.cex: List = build_cex(settings) if with_cex else []
        self._next_cex = 0.0
        self._next_dex = 0.0
        self._next_prune = 0.0
        self._backfill_done_cex = False

    # ------------------------------------------------------------------ шаги

    def bootstrap(self) -> None:
        """Первичная инициализация: список пулов и пар, затем история CEX.

        История CEX выкачивается быстро (1000 свечей за запрос), поэтому
        делается сразу. История DEX упирается в 30 запросов/мин и набирается
        фоном, поэтому здесь только запускается.
        """
        log.info("=== инициализация ===")
        if self.dex:
            try:
                n = self.dex.discover()
                log.info("DEX: %d пулов", n)
            except Exception as exc:
                log.error("DEX discover: %s", exc)

        for src in self.cex:
            try:
                n = src.discover()
                log.info("CEX %s: %d пар", src.name, n)
            except Exception as exc:
                log.error("CEX %s discover: %s", src.name, exc)
                store.set_state("cex:" + src.name, "*", ok=False, error=str(exc)[:300])

        log.info("=== бэкфилл CEX на %.1f дн ===", self.s.history_days)
        for src in self.cex:
            if _STOP:
                return
            try:
                n = src.backfill()
                log.info("CEX %s: %d свечей", src.name, n)
            except Exception as exc:
                log.error("CEX %s backfill: %s", src.name, exc)
        self._backfill_done_cex = True

    def step(self) -> None:
        now = time.monotonic()

        # живой срез DEX + постепенный бэкфилл истории
        if self.dex and now >= self._next_dex:
            self._next_dex = now + self.s.dex_refresh_sec
            try:
                self.dex.update()
            except Exception as exc:
                log.error("DEX update: %s", exc)
            try:
                self.dex.backfill(budget_requests=40)
            except Exception as exc:
                log.error("DEX backfill: %s", exc)

        # инкремент CEX
        if self.cex and now >= self._next_cex:
            self._next_cex = now + self.s.cex_refresh_sec
            for src in self.cex:
                if _STOP:
                    return
                try:
                    src.update()
                except Exception as exc:
                    log.error("CEX %s update: %s", src.name, exc)

        # обрезка до заданной глубины
        if now >= self._next_prune:
            self._next_prune = now + 3600
            cutoff = int(time.time() - self.s.history_days * 86400)
            deleted = store.prune(cutoff)
            if deleted:
                log.info("обрезано %d устаревших свечей", deleted)

    def run(self) -> None:
        self.bootstrap()
        log.info("=== рабочий цикл ===")
        while not _STOP:
            t0 = time.time()
            try:
                self.step()
            except Exception as exc:
                log.exception("сбой шага: %s", exc)
            st = store.stats()
            log.info("база: %d строк, %d площадок, %d пар, %.1f МБ",
                     st["rows"], st["venues"], st["pairs"], st["db_mb"])
            elapsed = time.time() - t0
            time.sleep(max(5.0, 15.0 - elapsed))
        log.info("остановлен")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Фоновый сборщик истории котировок")
    p.add_argument("--no-cex", action="store_true", help="не собирать CEX")
    p.add_argument("--no-dex", action="store_true", help="не собирать DEX")
    p.add_argument("--days", type=float, help="глубина истории в днях")
    p.add_argument("--timeframe", help="1m, 5m, 15m, 1h")
    p.add_argument("--pools", type=int, help="сколько пулов DEX наблюдать")
    p.add_argument("--once", action="store_true", help="один проход и выход")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    setup_logging(args.verbose)

    if args.days:
        SETTINGS.history_days = args.days
    if args.timeframe:
        SETTINGS.timeframe = args.timeframe
    if args.pools:
        SETTINGS.dex_pool_limit = args.pools

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (AttributeError, ValueError):
        pass

    c = Collector(SETTINGS, with_cex=not args.no_cex, with_dex=not args.no_dex)
    if args.once:
        c.bootstrap()
        c.step()
        print(store.stats())
    else:
        c.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
