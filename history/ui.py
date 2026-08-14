"""Мелкие помощники для интерфейса.

Главное здесь — совместимость по ширине виджетов. Streamlit объявил
`use_container_width` устаревшим и заменил на `width="stretch"`, но новый
параметр появился только в 1.49. Чтобы страницы работали и на старой,
и на новой версии без предупреждений в консоли, ширина передаётся
через распаковку словаря: `st.dataframe(df, **FULL)`.
"""

from __future__ import annotations

import streamlit as st


def _supports_width() -> bool:
    raw = getattr(st, "__version__", "0.0")
    try:
        major, minor = (int(p) for p in raw.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (1, 49)


FULL = {"width": "stretch"} if _supports_width() else {"use_container_width": True}
"""Растянуть виджет по ширине контейнера."""


def fmt_usd(value: float, digits: int = 0) -> str:
    """Денежный формат с неразрывными пробелами вместо запятых."""
    return f"${value:,.{digits}f}".replace(",", " ")


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")
