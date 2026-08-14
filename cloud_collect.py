"""Один цикл сбора для одноразового раннера GitHub Actions.

Раннер живёт несколько минут и не имеет памяти между запусками, поэтому
накопленная история переносится файлом:

    скачать снимок -> развернуть в SQLite -> собрать новое -> выгрузить снимок

Файл снимка публикуется в GitHub Release с тегом `data`. Release выбран,
а не коммит в репозиторий, по двум причинам: история git не распухает от
бинарников, и приложение на Streamlit не передеплоивается каждые пятнадцать
минут — оно просто скачивает файл по постоянной ссылке.

Запуск:
    python cloud_collect.py --minutes 8

Ключ --minutes ограничивает время работы: раннер бесплатного тарифа не стоит
занимать надолго, а следующий запуск подхватит с того же места.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from history import store
from history.config import DATA_DIR, SETTINGS, ensure_data_dir
from history.snapshot import export_snapshot, import_snapshot

log = logging.getLogger("cloud")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def run_bounded(label: str, fn, seconds: float, default=0):
    """Выполняет шаг с жёстким ограничением по времени.

    Нужно потому, что недоступная биржа не возвращает ошибку сразу:
    ccxt внутри load_markets ретраит с собственными таймаутами и может
    висеть минутами. В одноразовом раннере это означает, что прогон
    упрётся в лимит задачи и снимок не будет выгружен вообще.

    Поток запускается фоновым (daemon), поэтому зависший шаг не мешает
    процессу завершиться: мы просто перестаём его ждать и идём дальше.
    """
    if seconds <= 0:
        log.warning("%s: время вышло, шаг пропущен", label)
        return default

    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 — логируем и продолжаем
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True, name=label)
    t0 = time.time()
    th.start()
    th.join(timeout=seconds)

    if th.is_alive():
        log.warning("%s: не уложился в %.0f с, брошен", label, seconds)
        return default
    if "error" in box:
        log.error("%s: %s", label, box["error"])
        return default
    log.debug("%s: готово за %.1f с", label, time.time() - t0)
    return box.get("value", default)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Один цикл сбора для CI")
    p.add_argument("--minutes", type=float, default=8.0,
                   help="сколько минут работать, прежде чем выгрузить снимок")
    p.add_argument("--days", type=float, help="глубина истории")
    p.add_argument("--pools", type=int, help="сколько пулов DEX наблюдать")
    p.add_argument("--timeframe", help="1m, 5m, 15m, 1h")
    p.add_argument("--no-cex", action="store_true")
    p.add_argument("--no-dex", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    setup_logging(args.verbose)

    if args.days:
        SETTINGS.history_days = args.days
    if args.pools:
        SETTINGS.dex_pool_limit = args.pools
    if args.timeframe:
        SETTINGS.timeframe = args.timeframe

    # Раннеры GitHub стоят в США: Binance и Bybit отвечают 451.
    # Оставляем только те биржи, которые оттуда доступны.
    env_venues = os.environ.get("ARB_CEX_VENUES", "").strip()
    if env_venues:
        SETTINGS.cex_venues = [v.strip() for v in env_venues.split(",") if v.strip()]
    log.info("биржи: %s", ", ".join(SETTINGS.cex_venues) or "нет")

    ensure_data_dir()
    deadline = time.time() + args.minutes * 60

    # --- 1. Разворачиваем накопленное ------------------------------------
    log.info("=== импорт снимка ===")
    import_snapshot()
    store.init()
    before = store.stats()
    log.info("в базе после импорта: %d строк", before["rows"])

    # --- 2. Собираем новое -----------------------------------------------
    log.info("=== сбор ===")
    from history.collector import Collector

    c = Collector(SETTINGS, with_cex=not args.no_cex, with_dex=not args.no_dex)

    def left() -> float:
        return deadline - time.time()

    # DEX идёт первым: он дешевле по времени и именно ради него всё затевалось.
    # Один вызов discover сразу даёт свежие цены всех наблюдаемых пулов.
    if c.dex:
        run_bounded("DEX discover", c.dex.discover, min(120.0, left()))

    # Биржи: discover каждой ограничен, чтобы недоступная площадка
    # не съела прогон целиком.
    alive = []
    for src in c.cex:
        n = run_bounded(f"CEX {src.name} discover", src.discover,
                        min(60.0, left()), default=0)
        if n:
            alive.append(src)
        else:
            log.warning("CEX %s пропущена", src.name)
    log.info("бирж доступно: %d из %d", len(alive), len(c.cex))

    # Порядок важен. На первом живом прогоне биржи съели почти всё время
    # (одна только KuCoin отдала 1.9 млн свечей за три минуты), и на DEX
    # осталось четырнадцать секунд — история пулов не собралась вовсе.
    # Между тем DEX и есть цель: сеть BNB, обмен без переводов. Биржи же
    # дотягиваются инкрементально за секунды на следующих прогонах.
    # Поэтому DEX получает гарантированную половину времени первым.
    dex_deadline = time.time() + (deadline - time.time()) * 0.5

    def collect_dex(until_ts: float) -> int:
        if not c.dex:
            return 0
        total = 0
        while time.time() < until_ts - 20:
            n = run_bounded("DEX backfill",
                            lambda: c.dex.backfill(budget_requests=25),
                            min(180.0, until_ts - time.time()), default=-1)
            if n < 0:
                break
            total += n
            log.info("DEX: +%d свечей (всего %d), осталось %.0f с",
                     n, total, until_ts - time.time())
            # Ноль свечей раньше трактовался как «история набрана», хотя это
            # же значение возвращается и когда нас придержали по лимиту.
            # Теперь источник сообщает об этом явно.
            if n == 0:
                if getattr(c.dex, "last_backfill_complete", False):
                    log.info("история пулов набрана на нужную глубину")
                else:
                    log.info("прогресса нет — вероятно лимит, "
                             "продолжим на следующем прогоне")
                break
        return total

    collect_dex(dex_deadline)

    # Инкремент по биржам: если истории нет, backfill наберёт её за раз,
    # если есть — доберётся только свежее.
    for src in alive:
        if left() <= 30:
            log.warning("время вышло, биржи собраны не полностью")
            break
        fn = src.update if before["rows"] else src.backfill
        n = run_bounded(f"CEX {src.name} сбор", fn, min(180.0, left()))
        log.info("CEX %s: +%d свечей", src.name, n)

    # Если после бирж время осталось — доберём ещё пулов.
    collect_dex(deadline)

    # --- 3. Обрезаем и выгружаем -----------------------------------------
    cutoff = int(time.time() - SETTINGS.history_days * 86400)
    deleted = store.prune(cutoff)
    if deleted:
        log.info("обрезано устаревших свечей: %d", deleted)

    after = store.stats()
    log.info("=== экспорт снимка ===")
    path = export_snapshot()

    grew = after["rows"] - before["rows"]
    log.info("итог: %d строк (%+d), %d площадок, %d пар, снимок %.1f МБ",
             after["rows"], grew, after["venues"], after["pairs"],
             path.stat().st_size / 1e6)

    if after["t0"] and after["t1"]:
        depth_h = (after["t1"] - after["t0"]) / 3600
        log.info("глубина истории: %.1f ч из целевых %.0f ч",
                 depth_h, SETTINGS.history_days * 24)

    # Сводка для шага summary в Actions
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### Сбор завершён\n\n")
            f.write(f"| Показатель | Значение |\n|---|---|\n")
            f.write(f"| Строк в базе | {after['rows']:,} |\n".replace(",", " "))
            f.write(f"| Прирост за прогон | {grew:+,} |\n".replace(",", " "))
            f.write(f"| Площадок | {after['venues']} |\n")
            f.write(f"| Пар | {after['pairs']} |\n")
            if after["t0"]:
                f.write(f"| Глубина | {(after['t1'] - after['t0']) / 3600:.1f} ч |\n")
            f.write(f"| Размер снимка | {path.stat().st_size / 1e6:.1f} МБ |\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
