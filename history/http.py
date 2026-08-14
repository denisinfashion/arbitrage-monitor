"""Общий сетевой слой: переиспользуемая сессия, троттлинг по хосту, ретраи.

В исходном проекте троттлинг был только в dexes.py, а exchanges.py ходил
голым requests.get без Session — при периодическом опросе это выбивало
лимиты бирж и тратило время на TLS-handshake. Здесь один слой на всех.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 25
USER_AGENT = "arb-calculator/2.0"

# Минимальный интервал между запросами к одному хосту, секунды.
# GeckoTerminal: 30 запросов/мин на бесплатном тарифе -> 2.0 с с запасом.
MIN_INTERVAL: Dict[str, float] = {
    "api.geckoterminal.com": 2.6,   # лимит 30/мин по IP, а на раннерах GitHub адрес общий
    "pro-api.coingecko.com": 0.6,
    "api.coingecko.com": 2.5,
    "gateway.thegraph.com": 0.2,
    "open-api.openocean.finance": 1.15,
    "li.quest": 1.0,
    "api.paraswap.io": 0.35,
    "api.sushi.com": 0.25,
    "aggregator-api.kyberswap.com": 0.2,
    "api.dexpaprika.com": 0.15,
}
DEFAULT_MIN_INTERVAL = 0.05

_LOCKS: Dict[str, threading.Lock] = {}
_LAST: Dict[str, float] = {}
_REGISTRY_LOCK = threading.Lock()

_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


def session() -> requests.Session:
    """Единая сессия процесса: keep-alive и общий пул соединений."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
                adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
                s.mount("https://", adapter)
                s.mount("http://", adapter)
                _SESSION = s
    return _SESSION


def _host_lock(host: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        if host not in _LOCKS:
            _LOCKS[host] = threading.Lock()
        return _LOCKS[host]


def _throttle(host: str) -> None:
    """Выдерживает минимальный интервал между запросами к хосту."""
    gap = MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
    lock = _host_lock(host)
    with lock:
        last = _LAST.get(host, 0.0)
        wait = gap - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _LAST[host] = time.monotonic()


class HttpError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def get_json(
    url: str,
    params: Optional[dict] = None,
    *,
    attempts: int = 3,
    timeout: int = DEFAULT_TIMEOUT,
    headers: Optional[dict] = None,
) -> Any:
    """GET с троттлингом, ретраями на 429/5xx и понятной ошибкой."""
    host = urlparse(url).netloc
    last_exc: Optional[Exception] = None

    for i in range(attempts):
        _throttle(host)
        try:
            r = session().get(url, params=params, timeout=timeout, headers=headers)
        except requests.RequestException as exc:
            last_exc = exc
            if i + 1 < attempts:
                time.sleep(1.0 * (i + 1))
                continue
            raise HttpError(f"{host}: сеть недоступна ({type(exc).__name__})") from exc

        if r.status_code == 200:
            try:
                return r.json()
            except ValueError as exc:
                raise HttpError(f"{host}: ответ не JSON ({r.text[:120]})", 200) from exc

        if r.status_code in (429, 500, 502, 503, 504) and i + 1 < attempts:
            retry_after = r.headers.get("Retry-After")
            delay = float(retry_after) if (retry_after or "").isdigit() else 1.5 * (i + 1)
            log.debug("%s -> %s, пауза %.1f с", host, r.status_code, delay)
            time.sleep(min(delay, 30.0))
            continue

        raise HttpError(_explain(host, r.status_code, r.text), r.status_code)

    raise HttpError(f"{host}: не удалось получить ответ") from last_exc


def post_json(
    url: str,
    payload: dict,
    *,
    attempts: int = 3,
    timeout: int = DEFAULT_TIMEOUT,
    headers: Optional[dict] = None,
) -> Any:
    """POST JSON — нужен для GraphQL и JSON-RPC."""
    host = urlparse(url).netloc
    for i in range(attempts):
        _throttle(host)
        try:
            r = session().post(url, json=payload, timeout=timeout, headers=headers)
        except requests.RequestException as exc:
            if i + 1 < attempts:
                time.sleep(1.0 * (i + 1))
                continue
            raise HttpError(f"{host}: сеть недоступна ({type(exc).__name__})") from exc

        if r.status_code == 200:
            try:
                return r.json()
            except ValueError as exc:
                raise HttpError(f"{host}: ответ не JSON ({r.text[:120]})", 200) from exc

        if r.status_code in (429, 500, 502, 503, 504) and i + 1 < attempts:
            time.sleep(1.5 * (i + 1))
            continue

        raise HttpError(_explain(host, r.status_code, r.text), r.status_code)

    raise HttpError(f"{host}: не удалось получить ответ")


def get_bytes(url: str, *, attempts: int = 3, timeout: int = 60,
              headers: Optional[dict] = None) -> bytes:
    """Скачивает файл целиком. Нужен для снимков истории в облачном режиме."""
    host = urlparse(url).netloc
    for i in range(attempts):
        _throttle(host)
        try:
            r = session().get(url, timeout=timeout, headers=headers,
                              allow_redirects=True)
        except requests.RequestException as exc:
            if i + 1 < attempts:
                time.sleep(1.0 * (i + 1))
                continue
            raise HttpError(f"{host}: сеть недоступна ({type(exc).__name__})") from exc

        if r.status_code == 200:
            return r.content
        if r.status_code in (429, 500, 502, 503, 504) and i + 1 < attempts:
            time.sleep(1.5 * (i + 1))
            continue
        raise HttpError(_explain(host, r.status_code, ""), r.status_code)
    raise HttpError(f"{host}: не удалось скачать файл")


def _explain(host: str, status: int, body: str) -> str:
    """Человекочитаемая расшифровка типовых отказов."""
    if status == 451:
        return f"{host}: недоступно из региона сервера (451)"
    if status == 403:
        return f"{host}: доступ заблокирован (403)"
    if status == 429:
        return f"{host}: превышен лимит запросов (429)"
    if status in (400, 404):
        return f"{host}: ресурс не найден или неверные параметры ({status})"
    return f"{host}: HTTP {status} — {(body or '')[:160]}"
