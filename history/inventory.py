"""Что именно находится под наблюдением: токены, пулы, свежесть.

Вопрос «проверяется ли интересующий меня токен» до сих пор оставался без
ответа. Таблица связок показывает найденное, но молчит о ненайденном,
а у отсутствия связки есть две принципиально разные причины:

  токена нет в данных   — о нём никто не спрашивал, связка не проверялась;
  токен есть, связки нет — проверено, не сходится.

Разница решающая. В первом случае надо добавить токен в список
наблюдения, во втором — принять, что возможности нет. Модуль строит
таблицу, по которой это различается с одного взгляда.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Inventory:
    """Инвентарь наблюдаемого."""

    tokens: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: Dict[str, str] = field(default_factory=dict)
    watchlist: List[str] = field(default_factory=list)
    chain: str = ""
    window_h: float = 0.0

    def known(self, ticker: str) -> bool:
        if self.tokens.empty:
            return False
        return ticker.upper() in set(self.tokens["Тикер"])

    def row(self, ticker: str) -> Optional[dict]:
        if not self.known(ticker):
            return None
        sub = self.tokens[self.tokens["Тикер"] == ticker.upper()]
        return sub.iloc[0].to_dict()

    def verdict(self, ticker: str) -> dict:
        """Человеческий ответ на вопрос «проверяется ли этот токен».

        Возвращает статус, объяснение и совет — что сделать, чтобы
        токен начал участвовать в расчёте.
        """
        t = ticker.strip().upper()
        if not t:
            return {"status": "", "text": "", "advice": ""}

        if t in self.notes:
            return {
                "status": "отсеян",
                "text": f"{t} есть в данных, но исключён из расчёта: "
                        f"{self.notes[t]}.",
                "advice": "Если считаете отсев ошибочным, выключите "
                          "«Отсеивать недостоверные пулы» в боковой панели "
                          "страницы связок и посмотрите, что получится.",
            }

        r = self.row(t)
        # Тикер из списка наблюдения попадает в таблицу даже когда пулов
        # по нему не нашлось: пустая строка нагляднее отсутствия строки.
        # Но вердикт для неё — именно «нет в данных».
        if r is not None and not (r.get("Пар") or 0) and not (r.get("Пулов") or 0):
            r = None
        if r is None:
            in_watch = t in self.watchlist
            return {
                "status": "нет в данных",
                "text": f"{t} не собирается: в топ по обороту он не проходит"
                        + (", хотя и указан в списке наблюдения — значит "
                           "поиск пулов по этому тикеру ничего не вернул "
                           "или сбор с новым списком ещё не отработал"
                           if in_watch else
                           ", и в списке наблюдения его нет"),
                "advice": ("Проверьте, что сбор отработал после добавления, "
                           "и что тикер в сети написан именно так."
                           if in_watch else
                           f"Добавьте {t} в файл watchlist.txt — тогда его "
                           "пулы будут собираться независимо от оборота."),
            }

        if r["Пар"] < 2:
            return {
                "status": "мало пар",
                "text": f"{t} собирается, но прямой пул у него сейчас только "
                        f"с {r['Торгуется с']}. Цикл USDT → … → USDT через "
                        "него не замыкается: нужен как минимум второй пул.",
                "advice": "Это не значит, что токен нельзя обменять на что-то "
                          "ещё: кошелёк проложит путь через промежуточный "
                          "токен сам. Но такой обмен — две ноги с двумя "
                          "комиссиями, и в расчёте он так и выглядит. "
                          f"Посмотреть все пулы токена в сети: "
                          f"python probe_token.py {t}",
            }

        stale = r["Свежесть, мин"]
        if stale is not None and stale > 60:
            return {
                "status": "устарел",
                "text": f"{t} в данных есть, но последняя котировка получена "
                        f"{stale:.0f} мин назад — при пороге свежести "
                        "в расчёт она уже не идёт.",
                "advice": "Поднимите «Допустимый возраст котировки» в боковой "
                          "панели или проверьте, что сбор работает.",
            }

        # Формулировки без согласования с числом: «1 пулов» читается как
        # недоделка, а склонять числительные ради подписи не стоит.
        text = (f"{t} участвует в расчёте. Пулов: {r['Пулов']}, "
                f"площадок: {r['Площадок']}, торгуется с {r['Торгуется с']}")
        if stale is not None:
            text += f". Последняя котировка получена {stale:.0f} мин назад"
        return {
            "status": "проверяется",
            "text": text + ".",
            "advice": "Если связки через него нет — значит она проверена "
                      "и не сходится по издержкам, а не пропущена.",
        }


def build(chain: str, quotes: pd.DataFrame, pools: Optional[pd.DataFrame],
          notes: Optional[Dict[str, str]] = None,
          watchlist: Optional[List[str]] = None,
          window_h: float = 0.0, now: Optional[float] = None) -> Inventory:
    """Сводит котировки и справочник пулов в таблицу по токенам."""
    now = float(now if now is not None else time.time())
    inv = Inventory(notes=dict(notes or {}), watchlist=list(watchlist or []),
                    chain=chain, window_h=window_h)

    if quotes is None or quotes.empty:
        return inv

    # Разворачиваем пары в стороны: один токен — одна строка на наблюдение.
    sides = []
    for me, other in (("base", "quote"), ("quote", "base")):
        part = quotes[[me, other, "venue", "venue_kind", "ts"]].copy()
        part.columns = ["Тикер", "Против", "venue", "kind", "ts"]
        sides.append(part)
    long = pd.concat(sides, ignore_index=True)

    grouped = long.groupby("Тикер", sort=False)
    rows = []
    for token, grp in grouped:
        partners = sorted(set(grp["Против"]) - {token})
        venues = sorted(set(grp["venue"]))
        last_ts = float(grp["ts"].max())
        rows.append({
            "Тикер": str(token),
            "Пар": len(partners),
            "Торгуется с": ", ".join(partners[:6]) + ("…" if len(partners) > 6 else ""),
            "Площадок": len(venues),
            "Площадки": ", ".join(venues[:4]) + ("…" if len(venues) > 4 else ""),
            "Наблюдений": int(len(grp)),
            "Свежесть, мин": round((now - last_ts) / 60.0, 1),
        })
    df = pd.DataFrame(rows)

    # Данные справочника: имя, число пулов, ликвидность, оборот.
    if pools is not None and not pools.empty:
        p = pools.copy()
        for col in ("base", "quote", "base_name", "quote_name",
                    "reserve_usd", "volume_24h", "dex"):
            if col not in p.columns:
                p[col] = None
        parts = []
        for sym_c, name_c in (("base", "base_name"), ("quote", "quote_name")):
            q = p[[sym_c, name_c, "reserve_usd", "volume_24h", "dex"]].copy()
            q.columns = ["Тикер", "Имя", "reserve", "volume", "dex"]
            parts.append(q)
        tok = pd.concat(parts, ignore_index=True)
        tok["reserve"] = pd.to_numeric(tok["reserve"], errors="coerce")
        tok["volume"] = pd.to_numeric(tok["volume"], errors="coerce")
        agg = tok.groupby("Тикер", sort=False).agg(
            Пулов=("dex", "size"),
            Ликвидность=("reserve", "sum"),
            Оборот=("volume", "sum"),
        ).reset_index()
        names = (tok.dropna(subset=["Имя"]).groupby("Тикер")["Имя"]
                 .first().reset_index())
        agg = agg.merge(names, on="Тикер", how="left")
        df = df.merge(agg, on="Тикер", how="outer")

    for col, default in (("Пулов", 0), ("Пар", 0), ("Площадок", 0),
                         ("Наблюдений", 0)):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    if "Имя" not in df.columns:
        df["Имя"] = None
    for col in ("Ликвидность", "Оборот"):
        if col not in df.columns:
            df[col] = np.nan

    df["В расчёте"] = ~df["Тикер"].isin(set(inv.notes))
    df["В списке"] = df["Тикер"].isin({w.upper() for w in inv.watchlist})

    order = ["Тикер", "Имя", "В расчёте", "В списке", "Пар", "Площадок",
             "Пулов", "Ликвидность", "Оборот", "Свежесть, мин",
             "Торгуется с", "Площадки", "Наблюдений"]
    df = df[[c for c in order if c in df.columns]]
    df = df.sort_values(["В расчёте", "Пар", "Ликвидность"],
                        ascending=[False, False, False],
                        na_position="last").reset_index(drop=True)

    # Токены из списка наблюдения, которых в данных нет вовсе, — самый
    # важный случай: человек их ждёт, а их не собирают.
    missing = [w.upper() for w in inv.watchlist
               if w.upper() not in set(df["Тикер"])]
    if missing:
        extra = pd.DataFrame([{
            "Тикер": m, "Имя": None, "В расчёте": False, "В списке": True,
            "Пар": 0, "Площадок": 0, "Пулов": 0,
            "Ликвидность": np.nan, "Оборот": np.nan,
            "Свежесть, мин": None, "Торгуется с": "—", "Площадки": "—",
            "Наблюдений": 0,
        } for m in missing])
        df = pd.concat([df, extra], ignore_index=True)

    inv.tokens = df
    return inv


def summary(inv: Inventory) -> dict:
    df = inv.tokens
    if df.empty:
        return {"токенов": 0, "в расчёте": 0, "отсеяно": 0,
                "из списка наблюдения": 0, "не найдено из списка": 0}
    watch_found = int((df["В списке"] & (df["Пар"] > 0)).sum())
    return {
        "токенов": int(len(df)),
        "в расчёте": int(df["В расчёте"].sum()),
        "отсеяно": len(inv.notes),
        "из списка наблюдения": watch_found,
        "не найдено из списка": max(0, len(inv.watchlist) - watch_found),
    }
