"""Свежие цены: быстрый канал в обход снимка.

Задержка данных складывалась так. Срез цен берётся внутри прогона, но
публикуется только вместе со снимком — в самом конце, через несколько
минут. Дальше снимок лежит до следующего прогона. В итоге страница
показывала цены возрастом от одной минуты сразу после публикации
до пятнадцати перед следующей, в среднем около восьми. Для спреда,
который живёт пять минут, это бесполезно.

Причина не в архитектуре, а в размере посылки: снимок весит сорок
мегабайт, и чаще чем раз в прогон его не выгрузишь. Но для решения
«входить или нет» вся история не нужна — нужны последние цены. Это
двадцать килобайт, и их можно публиковать после каждого среза.

Отсюда два канала, оба в этом модуле:

**Быстрый файл.** Сборщик после каждого среза кладёт `live.json` рядом
со снимком. Страница читает его отдельно и подмешивает последней точкой
к истории. Свежесть — три-четыре минуты без участия человека.

**Прямой запрос.** Кнопка на странице: приложение само спрашивает цены
у источника, минуя и снимок, и сборщика. Сотня пулов — четыре запроса,
секунд десять. Свежесть — столько, сколько прошло с нажатия.

История при этом никуда не девается и остаётся тем, чем была: способом
понять, как часто окно открывается и сколько живёт. Решение о входе
принимается по свежей цене, а доверие к связке — по истории.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import DATA_DIR, SETTINGS, ensure_data_dir

log = logging.getLogger(__name__)

LIVE_NAME = "live.json"
ENV_LIVE_TTL = "ARB_LIVE_TTL"

COLUMNS = ["ts", "venue", "venue_kind", "chain", "base", "quote",
           "close", "volume", "liquidity_usd", "pool"]


# --------------------------------------------------------------------------
# Запись и публикация — сторона сборщика
# --------------------------------------------------------------------------


def write_live(rows: List[dict], chain: str = "", path: Optional[Path] = None) -> Optional[Path]:
    """Складывает последние цены в компактный файл рядом со снимком."""
    if not rows:
        return None
    ensure_data_dir()
    path = path or (DATA_DIR / LIVE_NAME)
    payload = {
        "ts": int(time.time()),
        "chain": chain or SETTINGS.chain,
        "rows": rows,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)
    return path


def publish_live(path: Optional[Path] = None, tag: str = "data") -> bool:
    """Выкладывает файл в тот же Release, где лежит снимок.

    Через `gh`, а не через API вручную: утилита есть на раннере, токен
    берётся из окружения, и один вызов занимает пару секунд. Это цена
    того, чтобы цены на странице обновлялись каждые три минуты, а не
    раз в прогон.
    """
    path = path or (DATA_DIR / LIVE_NAME)
    if not path.exists():
        return False
    if not os.environ.get("GH_TOKEN") and not os.environ.get("GITHUB_TOKEN"):
        log.debug("публикация живых цен пропущена: нет токена")
        return False
    try:
        res = subprocess.run(
            ["gh", "release", "upload", tag, str(path), "--clobber"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("живые цены не опубликованы: %s", exc)
        return False
    if res.returncode != 0:
        log.warning("живые цены не опубликованы: %s",
                    (res.stderr or "").strip()[:200])
        return False
    return True


def rows_from_candles(candles) -> List[dict]:
    """Свечи сборщика -> компактные строки для файла."""
    out = []
    for c in candles:
        out.append({
            "t": int(getattr(c, "ts", 0)),
            "v": getattr(c, "venue", ""),
            "c": getattr(c, "chain", ""),
            "b": getattr(c, "base", ""),
            "q": getattr(c, "quote", ""),
            "p": float(getattr(c, "close", 0.0) or 0.0),
            "l": getattr(c, "liquidity_usd", None),
            "a": getattr(c, "pool", None),
        })
    return out


# --------------------------------------------------------------------------
# Чтение — сторона приложения
# --------------------------------------------------------------------------


def live_url() -> str:
    """Адрес быстрого файла выводится из адреса снимка."""
    from .snapshot import SNAPSHOT_NAME, snapshot_url
    url = snapshot_url()
    return url.replace(SNAPSHOT_NAME, LIVE_NAME) if url else ""


def read_live(ttl: Optional[int] = None) -> Optional[dict]:
    """Последние опубликованные цены. None, если файла нет.

    Свой короткий срок кэша: файл маленький, скачивать его часто дёшево,
    и смысл всей затеи в свежести.
    """
    from .snapshot import cloud_mode

    if not cloud_mode():
        path = DATA_DIR / LIVE_NAME
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    url = live_url()
    if not url:
        return None
    ttl = ttl if ttl is not None else int(os.environ.get(ENV_LIVE_TTL, "30"))
    ensure_data_dir()
    local = DATA_DIR / f"remote_{LIVE_NAME}"
    if local.exists() and (time.time() - local.stat().st_mtime) < ttl:
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    from .http import HttpError, get_bytes
    try:
        data = get_bytes(url)
    except HttpError as exc:
        log.debug("живые цены недоступны: %s", exc)
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    try:
        local.write_bytes(data)
    except OSError:
        pass
    return payload


def as_frame(payload: Optional[dict]) -> pd.DataFrame:
    """Полезная нагрузка -> строки того же вида, что у истории."""
    if not payload or not payload.get("rows"):
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for r in payload["rows"]:
        rows.append({
            "ts": int(r.get("t") or payload.get("ts") or 0),
            "venue": r.get("v", ""),
            "venue_kind": "dex",
            "chain": r.get("c", "") or payload.get("chain", ""),
            "base": r.get("b", ""),
            "quote": r.get("q", ""),
            "close": float(r.get("p") or 0.0),
            "volume": None,
            "liquidity_usd": r.get("l"),
            "pool": r.get("a"),
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df[df["close"] > 0]


def age_seconds(payload: Optional[dict]) -> Optional[float]:
    if not payload or not payload.get("ts"):
        return None
    return max(0.0, time.time() - float(payload["ts"]))


# --------------------------------------------------------------------------
# Прямой запрос — самый быстрый путь
# --------------------------------------------------------------------------


def fetch_now(settings=SETTINGS, limit: int = 200) -> pd.DataFrame:
    """Спрашивает цены у источника прямо сейчас, минуя снимок и сборщик.

    Работает по тому же справочнику пулов, что и сборщик: адреса известны,
    поэтому нужен не поиск, а только текущие цены — до тридцати пулов
    за запрос.
    """
    from . import snapshot
    from .http import HttpError, get_json
    from .sources.dex_gt import MULTI_BATCH, endpoint

    try:
        pools = snapshot.pools(settings.chain)
    except Exception as exc:  # noqa: BLE001
        log.warning("справочник пулов недоступен: %s", exc)
        return pd.DataFrame(columns=COLUMNS)
    if pools is None or pools.empty:
        return pd.DataFrame(columns=COLUMNS)

    if "reserve_usd" in pools.columns:
        pools = pools.sort_values("reserve_usd", ascending=False)
    addrs = [str(a) for a in pools["pool"].tolist()[:limit] if a]

    api, headers = endpoint(settings)
    now = int(time.time())
    rows: List[dict] = []

    for i in range(0, len(addrs), MULTI_BATCH):
        batch = addrs[i:i + MULTI_BATCH]
        try:
            payload = get_json(
                f"{api}/networks/{settings.chain}/pools/multi/" + ",".join(batch),
                params={"include": "base_token,quote_token,dex"},
                headers=headers,
            )
        except HttpError as exc:
            log.warning("живые цены, пачка %d: %s", i // MULTI_BATCH + 1, exc)
            continue
        rows.extend(_parse(payload, settings.chain, now))

    df = pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)
    log.info("живые цены: %d котировок по %d пулам", len(df), len(addrs))
    return df


def _parse(payload: dict, chain: str, ts: int) -> List[dict]:
    """Ответ multi -> строки котировок. Разбор намеренно свой, а не общий
    с источником: здесь нужен DataFrame, а не запись в базу."""
    from .config import norm_asset
    from .sources.dex_gt import _f, _parse_pool_name

    tokens = {o.get("id", ""): o for o in payload.get("included", [])}
    out = []
    for item in payload.get("data", []) or []:
        attrs = item.get("attributes", {}) or {}
        rel = item.get("relationships", {}) or {}

        def side(name):
            ref = ((rel.get(name) or {}).get("data") or {}).get("id", "")
            a = (tokens.get(ref) or {}).get("attributes", {}) or {}
            return (a.get("symbol") or "").upper()

        base, quote = side("base_token"), side("quote_token")
        if not base or not quote:
            base, quote = _parse_pool_name(attrs.get("name", ""))
        if not base or not quote:
            continue

        # Курс самого пула, а не частное двух долларовых оценок: оценки
        # снимаются в разные моменты, и на четырёхногой связке их
        # расхождения складываются в маржу, которой нет.
        rate = _f(attrs.get("base_token_price_quote_token"))
        if not rate or rate <= 0:
            b_usd = _f(attrs.get("base_token_price_usd"))
            q_usd = _f(attrs.get("quote_token_price_usd"))
            if not b_usd or not q_usd or q_usd <= 0:
                continue
            rate = b_usd / q_usd

        out.append({
            "ts": ts,
            "venue": ((rel.get("dex") or {}).get("data") or {}).get("id", "?"),
            "venue_kind": "dex",
            "chain": chain,
            "base": norm_asset(base),
            "quote": norm_asset(quote),
            "close": rate,
            "volume": None,
            "liquidity_usd": _f(attrs.get("reserve_in_usd")),
            "pool": attrs.get("address"),
        })
    return out


def merge(history: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Подмешивает свежие цены последней точкой к истории.

    Именно подмешивает, а не заменяет: график и статистика окон нужны
    по-прежнему, а решение о входе принимается по последней точке.
    """
    if live is None or live.empty:
        return history
    if history is None or history.empty:
        return live
    cols = [c for c in COLUMNS if c in history.columns]
    return pd.concat([history[cols], live[cols]], ignore_index=True)
