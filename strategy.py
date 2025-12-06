# strategy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd

from config import (
    MIN_BARS,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    CCI_OVERSOLD,
    CCI_OVERBOUGHT,
    ATR_STOP_K,
    RR_TP,
)
from indicators import apply_indicators
from utils import find_two_swings


@dataclass
class Signal:
    side: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str

import logging
logger = logging.getLogger("trend_div_bot")

class TrendDivStrategy:

    """
    Стратегия:
      - Тренд по EMA200
      - Бычьи/медвежьи дивергенции по RSI и CCI
      - Фильтр по объёму
      - Стоп по ATR, TP = 2R
    """

    def generate_signal(self, raw_df: pd.DataFrame) -> Optional[Signal]:
        """
        raw_df: DataFrame со столбцами: open, high, low, close, volume, индекс = ts (int/Datetime)
        Возвращает Signal или None.
        """
        if len(raw_df) < MIN_BARS:
            return None

        df = apply_indicators(raw_df)
        last = df.iloc[-1]
        close = last["close"]
        high = last["high"]
        low = last["low"]
        ema200 = last["ema200"]
        rsi_val = last["rsi"]
        cci_val = last["cci"]
        atr_val = last["atr"]
        vol = last["volume"]
        vol_sma = last["vol_sma"]

        logger.info(
            "IND | close=%.6f ema200=%.6f rsi=%.1f cci=%.1f atr=%.6f vol=%.2f vol_sma=%.2f",
            float(close),
            float(ema200) if not pd.isna(ema200) else float("nan"),
            float(rsi_val) if not pd.isna(rsi_val) else float("nan"),
            float(cci_val) if not pd.isna(cci_val) else float("nan"),
            float(atr_val) if not pd.isna(atr_val) else float("nan"),
            float(vol),
            float(vol_sma) if not pd.isna(vol_sma) else float("nan"),
        )

        # если индикаторы ещё не готовы
        if any(map(pd.isna, [ema200, rsi_val, cci_val, atr_val, vol_sma])):
            return None

        # тренд
        if close > ema200 * 1.002:
            trend: Literal["bull", "bear", "flat"] = "bull"
        elif close < ema200 * 0.998:
            trend = "bear"
        else:
            trend = "flat"

        if trend == "flat":
            return None

        # объёмный фильтр
        if vol < 0.7 * vol_sma:
            return None

        if trend == "bull":
            return self._signal_long(df)
        elif trend == "bear":
            return self._signal_short(df)
        return None

    def _signal_long(self, df: pd.DataFrame) -> Optional[Signal]:
        last = df.iloc[-1]
        close = last["close"]
        low = last["low"]
        rsi_val = last["rsi"]
        cci_val = last["cci"]
        atr_val = last["atr"]

        if not (rsi_val < RSI_OVERSOLD and cci_val < CCI_OVERSOLD):
            return None

        idx1, idx2 = find_two_swings(df["low"], mode="low", lookback=80)
        if idx1 is None or idx2 is None:
            return None

        p1, p2 = df.loc[idx1, "low"], df.loc[idx2, "low"]
        r1, r2 = df.loc[idx1, "rsi"], df.loc[idx2, "rsi"]
        c1, c2 = df.loc[idx1, "cci"], df.loc[idx2, "cci"]

        bull_div_rsi = p2 < p1 and r2 > r1
        bull_div_cci = p2 < p1 and c2 > c1

        if not (bull_div_rsi and bull_div_cci):
            return None

        stop_loss = low - ATR_STOP_K * atr_val
        if stop_loss >= close:
            return None

        r_value = close - stop_loss
        take_profit = close + RR_TP * r_value

        return Signal(
            side="long",
            entry_price=float(close),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reason="Bull trend + bull div RSI/CCI",
        )

    def _signal_short(self, df: pd.DataFrame) -> Optional[Signal]:
        last = df.iloc[-1]
        close = last["close"]
        high = last["high"]
        rsi_val = last["rsi"]
        cci_val = last["cci"]
        atr_val = last["atr"]

        if not (rsi_val > RSI_OVERBOUGHT and cci_val > CCI_OVERBOUGHT):
            return None

        idx1, idx2 = find_two_swings(df["high"], mode="high", lookback=80)
        if idx1 is None or idx2 is None:
            return None

        p1, p2 = df.loc[idx1, "high"], df.loc[idx2, "high"]
        r1, r2 = df.loc[idx1, "rsi"], df.loc[idx2, "rsi"]
        c1, c2 = df.loc[idx1, "cci"], df.loc[idx2, "cci"]

        bear_div_rsi = p2 > p1 and r2 < r1
        bear_div_cci = p2 > p1 and c2 < c1

        if not (bear_div_rsi and bear_div_cci):
            return None

        stop_loss = high + ATR_STOP_K * atr_val
        if stop_loss <= close:
            return None

        r_value = stop_loss - close
        take_profit = close - RR_TP * r_value

        return Signal(
            side="short",
            entry_price=float(close),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reason="Bear trend + bear div RSI/CCI",
        )
