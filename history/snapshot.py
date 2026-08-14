"""Перенос накопленной истории между машинами через Parquet.

Зачем это нужно. В облаке нет ни постоянного диска, ни возможности держать
фоновый процесс: Streamlit Community Cloud запускает один процесс и стирает
файловую систему при перезапуске. Поэтому сбор вынесен в GitHub Actions,
а результат кладётся одним сжатым файлом туда, откуда приложение может его
прочитать по обычной ссылке.

Раннер GitHub тоже одноразовый, так что цикл сбора выглядит так:

    скачать снимок -> развернуть в SQLite -> собрать новое -> выгрузить снимок

Parquet выбран вместо самого файла SQLite потому, что сжимает временные ряды
примерно в десять раз: миллион свечей укладывается в несколько мегабайт,
а такой файл уже не жалко перекладывать каждые пятнадцать минут.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from . import store
from .config import DATA_DIR, ensure_data_dir

log = logging.getLogger(__name__)

SNAPSHOT_NAME = "history.parquet"
POOLS_NAME = "pools.parquet"

# Переменные окружения, которыми настраивается облачный режим
ENV_SNAPSHOT_URL = "ARB_SNAPSHOT_URL"
"""Ссылка на снимок. Если задана — приложение читает данные оттуда."""

ENV_SNAPSHOT_TTL = "ARB_SNAPSHOT_TTL"
"""Сколько секунд держать скачанный снимок, прежде чем тянуть заново."""


# --------------------------------------------------------------------------
# Экспорт
# --------------------------------------------------------------------------


def export_snapshot(path: Optional[Path] = None, days: Optional[float] = None) -> Path:
    """Выгружает содержимое базы в Parquet.

    Возвращает путь к файлу. Пустая база тоже выгружается — так следующий
    запуск сборщика получит корректную схему вместо ошибки чтения.
    """
    ensure_data_dir()
    path = Path(path or DATA_DIR / SNAPSHOT_NAME)

    since = int(time.time() - days * 86400) if days else None
    df = store.read_quotes(since_ts=since)
    if "dt" in df.columns:
        df = df.drop(columns=["dt"])

    if df.empty:
        df = pd.DataFrame(columns=[
            "ts", "venue", "venue_kind", "chain", "base", "quote",
            "close", "volume", "liquidity_usd", "pool",
        ])

    df.to_parquet(path, compression="zstd", index=False)

    pools = _all_pools()
    pools.to_parquet(path.with_name(POOLS_NAME), compression="zstd", index=False)

    log.info("снимок: %d строк -> %s (%.1f МБ)",
             len(df), path, path.stat().st_size / 1e6)
    return path


def _all_pools() -> pd.DataFrame:
    conn = store.connect(read_only=True)
    try:
        return pd.read_sql_query("SELECT * FROM pools", conn)
    except Exception:
        return pd.DataFrame(columns=["chain", "pool", "dex", "base", "quote"])


# --------------------------------------------------------------------------
# Импорт
# --------------------------------------------------------------------------


def import_snapshot(path: Optional[Path] = None) -> int:
    """Заливает снимок в локальную базу. Возвращает число строк."""
    path = Path(path or DATA_DIR / SNAPSHOT_NAME)
    if not path.exists():
        log.info("снимка нет (%s) — начинаем с пустой базы", path)
        return 0

    store.init()
    df = pd.read_parquet(path)
    n = _write_quotes_frame(df)

    pools_path = path.with_name(POOLS_NAME)
    if pools_path.exists():
        pools = pd.read_parquet(pools_path)
        if not pools.empty:
            store.write_pools(pools.to_dict("records"))

    log.info("импортировано %d строк из снимка", n)
    return n


def _write_quotes_frame(df: pd.DataFrame) -> int:
    """Пишет DataFrame в таблицу quotes пачками.

    Идём через executemany, а не через to_sql: нужен апсерт по составному
    ключу, иначе повторный импорт того же снимка упадёт на конфликте.
    """
    if df.empty:
        return 0
    cols = ["ts", "venue", "venue_kind", "chain", "base", "quote",
            "open", "high", "low", "close", "volume", "liquidity_usd", "pool"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    rows = list(df[cols].itertuples(index=False, name=None))

    sql = """
    INSERT INTO quotes (ts, venue, venue_kind, chain, base, quote,
                        open, high, low, close, volume, liquidity_usd, pool)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT (ts, venue, chain, base, quote) DO UPDATE SET
        close=excluded.close, volume=excluded.volume,
        liquidity_usd=excluded.liquidity_usd, pool=excluded.pool
    """
    with store.transaction() as conn:
        for i in range(0, len(rows), 20_000):
            conn.executemany(sql, rows[i:i + 20_000])
    return len(rows)


# --------------------------------------------------------------------------
# Облачный режим: чтение снимка по ссылке
# --------------------------------------------------------------------------


def snapshot_url() -> str:
    """Ссылка на снимок из окружения или из секретов Streamlit."""
    url = os.environ.get(ENV_SNAPSHOT_URL, "").strip()
    if url:
        return url
    try:
        import streamlit as st
        return str(st.secrets.get("snapshot_url", "")).strip()
    except Exception:
        return ""


def cloud_mode() -> bool:
    """True, если приложение должно читать данные по ссылке, а не из базы."""
    return bool(snapshot_url())


def fetch_remote(url: Optional[str] = None, ttl: Optional[int] = None) -> Optional[Path]:
    """Скачивает снимок во временную папку, не чаще раза в ttl секунд.

    Возвращает путь к локальной копии или None, если скачать не удалось
    и копии ещё нет.
    """
    url = url or snapshot_url()
    if not url:
        return None

    ttl = ttl or int(os.environ.get(ENV_SNAPSHOT_TTL, "300"))
    ensure_data_dir()
    local = DATA_DIR / f"remote_{SNAPSHOT_NAME}"

    if local.exists() and (time.time() - local.stat().st_mtime) < ttl:
        return local

    from .http import HttpError, get_bytes
    try:
        data = get_bytes(url)
    except HttpError as exc:
        log.warning("снимок недоступен: %s", exc)
        return local if local.exists() else None

    tmp = local.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(local)
    log.info("снимок обновлён: %.1f МБ", len(data) / 1e6)
    return local


def load_remote_quotes(since_ts: Optional[int] = None,
                       venue_kinds: Optional[list] = None) -> pd.DataFrame:
    """Читает котировки из удалённого снимка — замена store.read_quotes
    в облачном режиме."""
    path = fetch_remote()
    if path is None or not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if df.empty:
        return df
    if since_ts:
        df = df[df["ts"] >= int(since_ts)]
    if venue_kinds:
        df = df[df["venue_kind"].isin(venue_kinds)]
    if not df.empty:
        df = df.copy()
        df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def remote_stats() -> dict:
    """Сводка по удалённому снимку — для страницы состояния."""
    path = fetch_remote()
    if path is None or not path.exists():
        return {"rows": 0, "t0": None, "t1": None, "venues": 0, "pairs": 0,
                "by_kind": {}, "db_mb": 0.0, "source": "remote", "ok": False}

    df = pd.read_parquet(path)
    if df.empty:
        return {"rows": 0, "t0": None, "t1": None, "venues": 0, "pairs": 0,
                "by_kind": {}, "db_mb": 0.0, "source": "remote", "ok": True}

    by_kind = {}
    for kind, grp in df.groupby("venue_kind"):
        by_kind[kind] = {"rows": len(grp), "venues": grp["venue"].nunique()}

    return {
        "rows": len(df),
        "t0": int(df["ts"].min()),
        "t1": int(df["ts"].max()),
        "venues": int(df["venue"].nunique()),
        "pairs": int((df["base"] + "/" + df["quote"]).nunique()),
        "by_kind": by_kind,
        "db_mb": round(path.stat().st_size / 1e6, 1),
        "source": "remote",
        "ok": True,
        "fetched_at": path.stat().st_mtime,
    }


def remote_pools() -> pd.DataFrame:
    path = fetch_remote()
    if path is None:
        return pd.DataFrame()
    pools_path = path.with_name(f"remote_{POOLS_NAME}")
    if not pools_path.exists():
        url = snapshot_url().replace(SNAPSHOT_NAME, POOLS_NAME)
        from .http import HttpError, get_bytes
        try:
            pools_path.write_bytes(get_bytes(url))
        except HttpError:
            return pd.DataFrame()
    try:
        return pd.read_parquet(pools_path)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Единая точка входа для страниц
# --------------------------------------------------------------------------


def read_quotes(since_ts: Optional[int] = None,
                venue_kinds: Optional[list] = None) -> pd.DataFrame:
    """Читает котировки из того источника, который настроен.

    Страницы вызывают именно её и не знают, локальная это база
    или снимок из облака.
    """
    if cloud_mode():
        return load_remote_quotes(since_ts=since_ts, venue_kinds=venue_kinds)
    return store.read_quotes(since_ts=since_ts, venue_kinds=venue_kinds)


def stats() -> dict:
    if cloud_mode():
        return remote_stats()
    s = store.stats()
    s["source"] = "local"
    s["ok"] = True
    return s


def pools(chain: str, min_reserve_usd: float = 0.0) -> pd.DataFrame:
    if cloud_mode():
        df = remote_pools()
        if df.empty:
            return df
        df = df[df["chain"] == chain]
        if min_reserve_usd:
            df = df[df["reserve_usd"].fillna(0) >= min_reserve_usd]
        return df.sort_values("reserve_usd", ascending=False)
    return store.read_pools(chain, min_reserve_usd)


def data_available() -> bool:
    if cloud_mode():
        return remote_stats().get("rows", 0) > 0
    return store.db_exists() and store.stats()["rows"] > 0


def coverage() -> pd.DataFrame:
    """Покрытие по площадкам — работает и на локальной базе, и на снимке."""
    if cloud_mode():
        path = fetch_remote()
        if path is None or not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if df.empty:
            return pd.DataFrame()
        agg = df.groupby(["venue", "venue_kind"]).agg(
            Свечей=("ts", "size"),
            t0=("ts", "min"),
            t1=("ts", "max"),
        ).reset_index()
        pairs = (df.assign(_p=df["base"] + "/" + df["quote"])
                   .groupby(["venue", "venue_kind"])["_p"].nunique()
                   .reset_index(name="Пар"))
        out = agg.merge(pairs, on=["venue", "venue_kind"])
        out = out.rename(columns={"venue": "Площадка", "venue_kind": "Тип"})
        return out.sort_values("Свечей", ascending=False)

    conn = store.connect(read_only=True)
    return pd.read_sql_query(
        """
        SELECT venue AS "Площадка", venue_kind AS "Тип",
               COUNT(*) AS "Свечей",
               COUNT(DISTINCT base || '/' || quote) AS "Пар",
               MIN(ts) AS t0, MAX(ts) AS t1
        FROM quotes GROUP BY venue, venue_kind ORDER BY "Свечей" DESC
        """,
        conn,
    )


def source_label() -> str:
    """Человекочитаемое описание источника данных — для подписи в интерфейсе."""
    if cloud_mode():
        url = snapshot_url()
        short = url.split("/download/")[-1] if "/download/" in url else url[-60:]
        return f"снимок из облака ({short})"
    return f"локальная база ({store.DB_PATH.name})"
