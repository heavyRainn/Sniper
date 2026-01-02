# strategy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd

from config import (
    PAIR,
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
from collections import Counter

@dataclass
class Signal:
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    reason: str

import logging
logger = logging.getLogger("trend_div_bot")

class TrendDivStrategy:

    def __init__(self):
        self.rej = Counter()

    """
    Стратегия:
      - Тренд по EMA200
      - Бычьи/медвежьи дивергенции по RSI и CCI
      - Фильтр по объёму
      - Стоп по ATR, TP = 2R
    """

    def generate_signal(self, raw_df: pd.DataFrame) -> Optional[Signal]:
        # --- 1) Минимум баров ---
        if len(raw_df) < MIN_BARS:
            self.rej["min_bars"] += 1
            return None

        # --- 2) Индикаторы ---
        df = apply_indicators(raw_df)
        last = df.iloc[-1]

        ema200 = last["ema200"]
        rsi = last["rsi"]
        cci = last["cci"]
        atr = last["atr"]
        vol_sma = last["vol_sma"]

        close = float(last["close"])
        volume = float(last["volume"])

        # (не обязательно, но удобно: лог раз в N баров чтобы не спамить)
        # if len(df) % 200 == 0:
        logger.info(
            "IND | close=%.8f ema200=%.8f rsi=%.2f cci=%.2f atr=%.8f vol=%.2f vol_sma=%.2f",
            close,
            float(ema200) if not pd.isna(ema200) else float("nan"),
            float(rsi) if not pd.isna(rsi) else float("nan"),
            float(cci) if not pd.isna(cci) else float("nan"),
            float(atr) if not pd.isna(atr) else float("nan"),
            volume,
            float(vol_sma) if not pd.isna(vol_sma) else float("nan"),
        )

        # --- 3) NaN индикаторов ---
        if any(map(pd.isna, [ema200, rsi, cci, atr, vol_sma])):
            self.rej["na"] += 1
            logger.debug("REJ[na] some indicators are NaN")
            return None

        ema200 = float(ema200)
        vol_sma = float(vol_sma)

        # --- 4) Тренд ---
        bull = close > ema200 * 1.002
        bear = close < ema200 * 0.998
        if not (bull or bear):
            self.rej["flat"] += 1
            logger.debug("REJ[flat] close=%.8f ema200=%.8f", close, ema200)
            return None

        # --- 5) Фильтр по объёму ---
        if volume < 0.4 * vol_sma:
            self.rej["low_vol"] += 1
            logger.debug(
                "REJ[low_vol] vol=%.2f < 0.7*vol_sma=%.2f", volume, 0.7 * vol_sma
            )
            return None

        # --- 6) Сигнал по направлению тренда ---
        if bull:
            sig = self._signal_long(df)
            if sig is None:
                self.rej["no_long"] += 1
            else:
                self.rej["signal_long"] += 1
                logger.info("SIGNAL LONG | %s", sig.reason)
            return sig

        # bear
        sig = self._signal_short(df)
        if sig is None:
            self.rej["no_short"] += 1
        else:
            self.rej["signal_short"] += 1
            logger.info("SIGNAL SHORT | %s", sig.reason)
        return sig

    def _signal_long(self, df: pd.DataFrame) -> Optional[Signal]:
        last = df.iloc[-1]
        close = float(last["close"])
        rsi = last["rsi"]
        cci = last["cci"]
        atr = last["atr"]

        if not (rsi < RSI_OVERSOLD or cci < CCI_OVERSOLD):
            self.rej["long_not_extreme"] += 1
            return None

        idx1, idx2 = find_two_swings(df["low"], mode="low", lookback=80)
        if idx1 is None or idx2 is None:
            self.rej["long_no_swings"] += 1
            return None

        p1, p2 = df.loc[idx1, "low"], df.loc[idx2, "low"]
        r1, r2 = df.loc[idx1, "rsi"], df.loc[idx2, "rsi"]
        c1, c2 = df.loc[idx1, "cci"], df.loc[idx2, "cci"]

        bull_div_rsi = p2 < p1 and r2 > r1
        bull_div_cci = p2 < p1 and c2 > c1

        if not (bull_div_rsi or bull_div_cci):
            self.rej["long_no_div"] += 1
            return None

        stop_loss = float(p2) - ATR_STOP_K * float(atr)
        if stop_loss >= close:
            self.rej["long_bad_stop"] += 1
            return None

        take_profit = close + RR_TP * (close - stop_loss)

        return Signal(
            symbol=PAIR,
            side="long",
            entry_price=close,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            atr=float(atr),
            reason="Bull trend + bull div RSI/CCI",
        )

    def _signal_short(self, df: pd.DataFrame) -> Optional[Signal]:
        last = df.iloc[-1]
        close = float(last["close"])
        rsi = last["rsi"]
        cci = last["cci"]
        atr = last["atr"]

        if not (rsi > RSI_OVERBOUGHT or cci > CCI_OVERBOUGHT):
            self.rej["short_not_extreme"] += 1
            return None

        idx1, idx2 = find_two_swings(df["high"], mode="high", lookback=80)
        if idx1 is None or idx2 is None:
            self.rej["short_no_swings"] += 1
            return None

        p1, p2 = float(df.loc[idx1, "high"]), float(df.loc[idx2, "high"])
        r1, r2 = float(df.loc[idx1, "rsi"]), float(df.loc[idx2, "rsi"])
        c1, c2 = float(df.loc[idx1, "cci"]), float(df.loc[idx2, "cci"])

        bear_div_rsi = (p2 > p1) and (r2 < r1)
        bear_div_cci = (p2 > p1) and (c2 < c1)

        # мягче:
        if not (bear_div_rsi or bear_div_cci):
            self.rej["short_no_div"] += 1
            return None

        stop_loss = p2 + ATR_STOP_K * float(atr)
        if stop_loss <= close:
            self.rej["short_bad_stop"] += 1
            return None

        take_profit = close - RR_TP * (stop_loss - close)

        self.rej["signal_short"] += 1
        return Signal(
            symbol=PAIR,
            side="short",
            entry_price=close,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            atr=float(atr),
            reason="Bear trend + bear div RSI/CCI",
        )

