"""Свопы пулов EVM-сетей (BNB Chain, Base, Arbitrum) прямо из блокчейна.

Вместо событий Swap, которые у каждого типа пула свои (V2, V3, Aerodrome…),
читаются переводы ERC-20 (Transfer) в пул и из пула: в каждом свопе пул
получает один токен и отдаёт другой. Это одинаково для любого пула с адресом.
Цена сделки = количество котируемого токена / количество нашего.

Публичный RPC требует фильтр по адресам токенов — поэтому у каждого пула
один раз читаются token0/token1 и decimals. Два запроса eth_getLogs на сеть
за опрос («из пула» и «в пул») покрывают все пулы сразу, без пропусков.

Время — blockTimestamp из лога (секунды). Цена переводится в USD: стейблкоин = 1,
WBNB/WETH — по посекундной цене фьючерса BNB/ETH на Binance (инструмент ref).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
SEL_TOKEN0, SEL_TOKEN1, SEL_DECIMALS = "0x0dfe1681", "0xd21220a7", "0x313ce567"
MAX_BLOCKS = 500            # блоков за один eth_getLogs
MAX_ADDRESSES = 10          # публичный узел publicnode принимает не больше 10 адресов
START_BACK_BLOCKS = 200     # с какого места начинать при старте
PENDING_MAX_S = 180         # сколько ждать цену BNB/ETH для пересчёта в USD


@dataclass(frozen=True)
class EvmChain:
    key: str
    rpc: str
    poll_s: float
    stables: frozenset          # адреса стейблкоинов (нижний регистр)
    native: Dict[str, str]      # адрес обёрнутой монеты → символ ref-фьючерса


EVM: Dict[str, EvmChain] = {
    "bsc": EvmChain(
        "bsc", os.environ.get("LEADLAG_RPC_BSC", "https://bsc-rpc.publicnode.com"), 12.0,
        frozenset({
            "0x55d398326f99059ff775485246999027b3197955",  # USDT
            "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",  # USDC
            "0x8d0d000ee44948fc98c9b98a4fa4921476f08b0d",  # USD1
            "0xc5f0f7b66764f6ec8c8dff7ba683102295e16409",  # FDUSD
            "0xe9e7cea3dedca5984780bafc599bd69add087d56",  # BUSD
        }),
        {"0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "BNB/USDT:USDT"},  # WBNB
    ),
    "base": EvmChain(
        "base", os.environ.get("LEADLAG_RPC_BASE", "https://base-rpc.publicnode.com"), 10.0,
        frozenset({
            "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",  # USDC
            "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2",  # USDT
        }),
        {"0x4200000000000000000000000000000000000006": "ETH/USDT:USDT"},  # WETH
    ),
    "arbitrum": EvmChain(
        "arbitrum", os.environ.get("LEADLAG_RPC_ARBITRUM", "https://arbitrum-one-rpc.publicnode.com"), 10.0,
        frozenset({
            "0xaf88d065e77c8cc2239327c5edb3a432268e5831",  # USDC
            "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",  # USDT
        }),
        {"0x82af49447d8a07e3bd95bd0d56f35241523fbab1": "ETH/USDT:USDT"},  # WETH
    ),
}

# Ref-фьючерсы для пересчёта в USD: всегда с Binance, у них сделка каждую секунду.
REF_VENUE = "binance"


def ref_symbols(chains) -> List[str]:
    return sorted({sym for c in chains if c in EVM for sym in EVM[c].native.values()})


SCHEMA = """
CREATE TABLE IF NOT EXISTS pool_tokens (
    pool TEXT PRIMARY KEY, chain TEXT, token0 TEXT, token1 TEXT, dec0 INTEGER, dec1 INTEGER
);
CREATE TABLE IF NOT EXISTS evm_cursor (chain TEXT PRIMARY KEY, block INTEGER);
"""


class Rpc:
    def __init__(self, url: str):
        self.url = url
        self.s = requests.Session()
        self.id = 0

    def call(self, method: str, params: list):
        self.id += 1
        for attempt in range(4):
            try:
                r = self.s.post(self.url, json={"jsonrpc": "2.0", "id": self.id,
                                                "method": method, "params": params}, timeout=30)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                j = r.json()
            except (requests.RequestException, ValueError) as e:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
                continue
            if "error" in j:
                raise RuntimeError(f"{method}: {j['error']}")
            return j["result"]
        raise RuntimeError(f"{method}: лимит запросов")

    def eth_call(self, to: str, data: str) -> str:
        return self.call("eth_call", [{"to": to, "data": data}, "latest"])


def _addr(word: str) -> str:
    return "0x" + word[-40:].lower()


def _topic(addr: str) -> str:
    return "0x" + "0" * 24 + addr[2:].lower()


@dataclass
class PoolMeta:
    pool: str           # нижний регистр
    inst: str
    token: str          # наш токен
    quote: str          # второй токен пула
    dec_token: int
    dec_quote: int


def group_pools(pools: List[PoolMeta], limit: int = MAX_ADDRESSES) -> List[List[PoolMeta]]:
    """Пулы → группы, в каждой не больше limit адресов токенов.
    Общие котируемые токены (WBNB, USDT) группы делят, поэтому групп немного."""
    groups: List[Tuple[set, List[PoolMeta]]] = []
    for p in sorted(pools, key=lambda x: x.quote):
        need = {p.token, p.quote}
        for tokens, members in groups:
            if len(tokens | need) <= limit:
                tokens |= need
                members.append(p)
                break
        else:
            groups.append((set(need), [p]))
    return [members for _, members in groups]


def net_amounts(logs: List[dict], pools: Dict[str, PoolMeta]) -> Dict[Tuple[str, str], dict]:
    """Логи Transfer → по (tx, пул): чистое движение каждого токена пула и время блока.
    Плюс — токен пришёл в пул, минус — ушёл из пула."""
    out: Dict[Tuple[str, str], dict] = {}
    seen = set()
    for lg in logs:
        key_log = (lg["transactionHash"], lg["logIndex"])
        if key_log in seen:  # запросы «из пула» и «в пул» пересекаются на переводах пул→пул
            continue
        seen.add(key_log)
        t = lg.get("topics") or []
        if len(t) < 3 or t[0] != TRANSFER:
            continue
        frm, to, token = _addr(t[1]), _addr(t[2]), lg["address"].lower()
        amount = int(lg["data"], 16) if lg.get("data") and lg["data"] != "0x" else 0
        ts = int(lg["blockTimestamp"], 16)
        for pool, sign in ((to, 1), (frm, -1)):
            p = pools.get(pool)
            if p is None or token not in (p.token, p.quote):
                continue
            rec = out.setdefault((lg["transactionHash"], pool), {"ts": ts, p.token: 0, p.quote: 0})
            rec[token] += sign * amount
    return out


def swaps_from_net(net: Dict[Tuple[str, str], dict], pools: Dict[str, PoolMeta]
                   ) -> List[Tuple[PoolMeta, str, int, float, float]]:
    """(пул, tx, ts, цена в котируемом токене, объём в котируемом токене).
    Добавление и изъятие ликвидности (оба токена в одну сторону) — не сделки."""
    out = []
    for (tx, pool), rec in net.items():
        p = pools[pool]
        dx, dq = rec[p.token], rec[p.quote]
        if dx == 0 or dq == 0 or (dx > 0) == (dq > 0):
            continue
        x = abs(dx) / 10 ** p.dec_token
        q = abs(dq) / 10 ** p.dec_quote
        if x <= 0:
            continue
        out.append((p, tx, rec["ts"], q / x, q))
    return out


class EvmPoller:
    """Опрос одной сети. Вызывается из своего потока в цикле."""

    def __init__(self, chain: str, con: sqlite3.Connection,
                 write: Callable, ref_price: Callable[[str, int], Optional[float]]):
        self.c = EVM[chain]
        self.chain = chain
        self.con = con
        self.con.executescript(SCHEMA)
        self.rpc = Rpc(self.c.rpc)
        self.write = write
        self.ref_price = ref_price
        self.pools: Dict[str, PoolMeta] = {}
        self.pending: List[Tuple[PoolMeta, str, int, float, float]] = []
        self.skipped_quote: set = set()

    # ---- пулы

    def _meta(self, pool: str) -> Optional[Tuple[str, str, int, int]]:
        row = self.con.execute("SELECT token0, token1, dec0, dec1 FROM pool_tokens WHERE pool=?",
                               (pool,)).fetchone()
        if row:
            return row
        t0 = _addr(self.rpc.eth_call(pool, SEL_TOKEN0))
        t1 = _addr(self.rpc.eth_call(pool, SEL_TOKEN1))
        d0 = int(self.rpc.eth_call(t0, SEL_DECIMALS), 16)
        d1 = int(self.rpc.eth_call(t1, SEL_DECIMALS), 16)
        self.con.execute("INSERT OR REPLACE INTO pool_tokens VALUES (?,?,?,?,?,?)",
                         (pool, self.chain, t0, t1, d0, d1))
        self.con.commit()
        return t0, t1, d0, d1

    def set_pools(self, items) -> None:
        """items — Instrument с kind=dex этой сети."""
        fresh: Dict[str, PoolMeta] = {}
        for i in items:
            pool = i.symbol.lower()
            if pool in self.pools:
                fresh[pool] = self.pools[pool]
                continue
            try:
                meta = self._meta(pool)
            except Exception as e:
                log.warning("%s %s: токены пула не прочитаны (%s)", self.chain, pool, str(e)[:150])
                continue
            t0, t1, d0, d1 = meta
            token = i.contract.lower()
            if token == t0:
                fresh[pool] = PoolMeta(pool, i.inst, t0, t1, d0, d1)
            elif token == t1:
                fresh[pool] = PoolMeta(pool, i.inst, t1, t0, d1, d0)
            else:
                log.warning("%s %s: токена %s нет в пуле", self.chain, pool, token)
                continue
            q = fresh[pool].quote
            if q not in self.c.stables and q not in self.c.native:
                log.warning("%s %s: котируемый токен %s не стейблкоин и не %s — пул пропущен",
                            self.chain, pool, q, "/".join(self.c.native.values()))
                del fresh[pool]
        self.pools = fresh

    # ---- курсы

    def _to_usd(self, p: PoolMeta, ts: int) -> Optional[float]:
        if p.quote in self.c.stables:
            return 1.0
        return self.ref_price(self.c.native[p.quote], ts)

    # ---- опрос

    def step(self) -> int:
        if not self.pools:
            return 0
        latest = int(self.rpc.call("eth_blockNumber", []), 16)
        row = self.con.execute("SELECT block FROM evm_cursor WHERE chain=?", (self.chain,)).fetchone()
        start = (row[0] + 1) if row else latest - START_BACK_BLOCKS
        start = max(start, latest - 20_000)  # после долгого простоя не догоняем бесконечно
        if start > latest:
            return 0
        end = min(latest, start + MAX_BLOCKS - 1)

        logs: List[dict] = []
        for group in group_pools(list(self.pools.values())):
            tokens = sorted({p.token for p in group} | {p.quote for p in group})
            topics = [_topic(p.pool) for p in group]
            base = {"fromBlock": hex(start), "toBlock": hex(end), "address": tokens}
            logs += self.rpc.call("eth_getLogs", [{**base, "topics": [TRANSFER, topics]}])
            logs += self.rpc.call("eth_getLogs", [{**base, "topics": [TRANSFER, None, topics]}])
        swaps = self.pending + swaps_from_net(net_amounts(logs, self.pools), self.pools)

        rows, keep = [], []
        now = time.time()
        for p, tx, ts, price_q, vol_q in swaps:
            usd = self._to_usd(p, ts)
            if usd is None:
                if now - ts < PENDING_MAX_S:
                    keep.append((p, tx, ts, price_q, vol_q))
                continue
            rows.append((p.inst, tx, ts, price_q * usd, vol_q * usd))
        self.pending = keep
        if rows:
            self.write(rows)
        self.con.execute("INSERT OR REPLACE INTO evm_cursor VALUES (?,?)", (self.chain, end))
        self.con.commit()
        return len(rows)
