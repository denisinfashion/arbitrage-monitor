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

CODE_VERSION = "2026-08-18.1"
# Отметка версии в логе. Нужна из-за реального случая: прогон в CI дважды
# шёл на старом коде — сперва потому, что коммит не был отправлен на сервер,
# затем потому, что вместо нового запуска был повтор прежнего (повтор берёт
# тот же коммит). По поведению это распознаётся не сразу, по строке в логе —
# мгновенно.


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


def _run_alerts() -> None:
    """Ищет связки по свежим данным и шлёт оповещения, если настроено."""
    from history import alerts

    if not alerts.configured():
        return

    import time as _t
    from history.paths import find_cycles
    from history.rates import build_grid

    window_h = float(os.environ.get("ALERT_WINDOW_HOURS", "6"))
    quotes = store.read_quotes(since_ts=int(_t.time() - window_h * 3600),
                               venue_kinds=["dex"])
    if quotes.empty:
        log.info("оповещения: нет свежих данных DEX")
        return

    s = SETTINGS
    # Шаг анализа соответствует частоте живых срезов DEX: снимаем раз
    # в пять минут — на пятиминутной сетке каждая точка заполнена, а на
    # минутной четыре из пяти были бы пустыми.
    s.analysis_timeframe = os.environ.get("ALERT_TIMEFRAME", "5m")
    s.staleness_sec = int(os.environ.get("ALERT_STALENESS", "900"))

    grid = build_grid(quotes, settings=s, venue_kinds=["dex"], max_assets=40)
    _, cycles = find_cycles(grid, anchor=s.quote_asset, max_legs=s.max_legs,
                            top=40, min_margin_pct=-100, settings=s)
    n = alerts.notify(cycles)
    log.info("оповещений отправлено: %d", n)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Один цикл сбора для CI")
    p.add_argument("--minutes", type=float, default=8.0,
                   help="сколько минут работать, прежде чем выгрузить снимок")
    p.add_argument("--days", type=float, help="глубина истории")
    p.add_argument("--pools", type=int, help="сколько пулов DEX наблюдать")
    p.add_argument("--dex-every", type=int,
                   help="как часто снимать живые цены пулов, секунд")
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
    if args.dex_every:
        SETTINGS.dex_pull_seconds = args.dex_every

    # Список наблюдения: тикеры, пулы которых добираются принудительно.
    # Заводится переменной окружения, чтобы добавить интересующую связку
    # можно было правкой одной строки в workflow, без выпуска кода.
    env_watch = os.environ.get("ARB_WATCH_TOKENS", "").strip()
    if env_watch:
        SETTINGS.watch_tokens = [w.strip().upper()
                                 for w in env_watch.split(",") if w.strip()]
        log.info("список наблюдения: %s", ", ".join(SETTINGS.watch_tokens))
    env_dex = os.environ.get("ARB_DEX_VENUES", "").strip()
    if env_dex:
        SETTINGS.dex_venues = [d.strip() for d in env_dex.split(",") if d.strip()]

    # Раннеры GitHub стоят в США: Binance и Bybit отвечают 451.
    # Оставляем только те биржи, которые оттуда доступны.
    env_venues = os.environ.get("ARB_CEX_VENUES", "").strip()
    if env_venues:
        SETTINGS.cex_venues = [v.strip() for v in env_venues.split(",") if v.strip()]
    sha = os.environ.get("GITHUB_SHA", "")
    log.info("версия кода: %s | коммит: %s | событие: %s",
             CODE_VERSION, sha[:7] or "локальный",
             os.environ.get("GITHUB_EVENT_NAME", "запуск вручную"))
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

    # Биржи и DEX работают одновременно, и это главное отличие от прежней
    # схемы. Раньше время делилось: сперва DEX, потом биржи. Но у этих двух
    # источников разная природа. Биржа отдаёт историю пачками по тысяче
    # свечей — ей нужен один заход, и прошлое она добирает целиком. У пулов
    # DEX прошлого нет: бесплатный API исторические свечи почти всегда
    # придерживает, и единственный надёжный источник — живой срез цен
    # прямо сейчас. Значит, для DEX важна не длительность захода,
    # а частота срезов.
    #
    # Поэтому биржи уходят в фоновый поток, а главный поток весь прогон
    # снимает живые цены пулов по расписанию — раз в dex_pull_seconds.
    pulse = max(30, int(SETTINGS.dex_pull_seconds))

    # Порядок бирж сдвигается от прогона к прогону. Причина конкретная:
    # KuCoin отвечает раннерам медленнее прочих и, стоя последней, два
    # прогона подряд упиралась в лимит времени и не отдавала ничего.
    if alive:
        shift = int(time.time() // 900) % len(alive)
        alive = alive[shift:] + alive[:shift]
        log.info("порядок бирж в этом прогоне: %s",
                 ", ".join(s.name for s in alive))

    def cex_worker() -> None:
        for i, src in enumerate(alive):
            if left() <= 30:
                log.warning("время вышло, биржи собраны не полностью")
                return
            # Остаток делится между оставшимися, но не меньше минуты:
            # фиксированные 180 с раньше означали, что первая биржа могла
            # съесть всё, а последняя не получала ничего.
            share = max(60.0, left() / max(1, len(alive) - i))
            fn = src.update if before["rows"] else src.backfill
            n = run_bounded(f"CEX {src.name} сбор", fn, min(share, left()))
            log.info("CEX %s: +%d свечей", src.name, n)

    cex_thread = None
    if alive:
        cex_thread = threading.Thread(target=cex_worker, daemon=True, name="cex")
        cex_thread.start()

    # Живые срезы DEX по расписанию. Первый уже сделан в discover выше,
    # поэтому отсчёт идёт от него.
    pulses = 1
    next_pulse = time.time() + pulse
    backfill_tried = False

    while c.dex and time.time() < deadline - 10:
        now = time.time()
        if now >= next_pulse:
            run_bounded("DEX срез", c.dex.discover, min(90.0, deadline - now))
            pulses += 1
            next_pulse = time.time() + pulse
            continue

        # Между срезами один раз пробуем дотянуть историю пулов. Чаще
        # незачем: бесплатный лимит GeckoTerminal общий на весь адрес
        # раннера, и повторные попытки в том же прогоне упираются в него же.
        if not backfill_tried:
            backfill_tried = True
            budget = min(120.0, next_pulse - now - 5.0, deadline - now)
            if budget > 20:
                n = run_bounded("DEX backfill",
                                lambda: c.dex.backfill(budget_requests=25),
                                budget, default=-1)
                if n and n > 0:
                    log.info("DEX история: +%d свечей", n)
                elif n == 0 and getattr(c.dex, "last_backfill_complete", False):
                    log.info("история пулов набрана на нужную глубину")
                else:
                    log.info("история пулов недоступна — работаем живыми срезами")
            continue

        time.sleep(min(5.0, max(0.5, min(next_pulse, deadline) - time.time())))

    log.info("живых срезов DEX за прогон: %d (раз в %d с)", pulses, pulse)

    if cex_thread is not None:
        cex_thread.join(timeout=max(0.0, left()))
        if cex_thread.is_alive():
            log.warning("биржи не успели к сроку — снимок выгружаем как есть")

    # --- 2б. Оповещения ---------------------------------------------------
    # Считаем связки по свежим данным и сообщаем о тех, что прибыльны
    # прямо сейчас. Шаг необязательный: без настроенного бота он молча
    # пропускается, а любая его ошибка не должна мешать выгрузке снимка.
    try:
        _run_alerts()
    except Exception as exc:
        log.error("оповещения: %s", exc)

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
            f.write(f"Версия кода: `{CODE_VERSION}`\n\n")
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
