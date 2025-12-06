# utils.py
from __future__ import annotations

import math
from typing import Optional, Literal, Tuple

import pandas as pd


def mask_secret(secret: Optional[str], show: int = 4) -> str:
    if not secret:
        return "None"
    if len(secret) <= 2 * show:
        return "*" * len(secret)
    return secret[:show] + "*" * (len(secret) - 2 * show) + secret[-show:]


def round_step(value: float, step: float) -> float:
    return math.floor(value / step) * step


def find_two_swings(
    series: pd.Series,
    mode: Literal["low", "high"],
    lookback: int = 80,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Находим два последних локальных минимума/максимума за lookback баров.
    Возвращаем (старший_экстремум, младший_экстремум) или (None, None).
    """
    win = series[-lookback:]
    if mode == "low":
        mask = (win.shift(1) > win) & (win.shift(-1) > win)
    else:
        mask = (win.shift(1) < win) & (win.shift(-1) < win)

    idxs = win[mask].index
    if len(idxs) < 2:
        return None, None
    return idxs[-2], idxs[-1]
