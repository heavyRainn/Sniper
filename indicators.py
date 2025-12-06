# indicators.py
from __future__ import annotations

import pandas as pd
import ta

from config import (
    EMA_PERIOD,
    RSI_PERIOD,
    CCI_PERIOD,
    ATR_PERIOD,
    VOL_SMA_PERIOD,
)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в df индикаторы:
      ema200, rsi, cci, atr, vol_sma
    """
    df = df.copy()
    df = df.reset_index(drop=True)

    df["ema200"] = ema(df["close"], EMA_PERIOD)
    df["rsi"] = ta.momentum.rsi(df["close"], RSI_PERIOD).fillna(50.0)
    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], CCI_PERIOD).fillna(0.0)
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], ATR_PERIOD
    ).bfill()
    df["vol_sma"] = sma(df["volume"], VOL_SMA_PERIOD)
    return df
