from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional

import logging
import pandas as pd

from config import ATR_STOP_K, MIN_BARS, PAIR, RR_TP
from indicators import apply_indicators

logger = logging.getLogger("trend_div_bot")


@dataclass
class Signal:
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    reason: str


@dataclass
class PendingSetup:
    side: Literal["long", "short"]
    created_bar: int
    expires_bar: int
    trigger: float
    stop_loss: float
    atr: float
    reason: str
    signal_high: float
    signal_low: float
    score: float
    trigger_close_min: float
    trigger_vol_min: float


class TrendDivStrategy:
    """
    Long-biased trend continuation strategy for volatile perps.

    Current tuning goal:
      - keep the continuation logic from v2
      - loosen only the filters that over-compressed trade count
      - preserve compatibility with bot.py/backtest.py
    """

    def __init__(
        self,
        tf: str = "5m",
        cooldown_bars: int = 8,
        setup_ttl_bars: int = 7,
        fee_roundtrip: float = 0.0012,
        slippage: float = 0.0005,
        cost_mult: float = 1.15,
        enable_short: bool = False,
    ):
        self.tf = tf
        self.symbol = PAIR
        self.cooldown_bars = int(cooldown_bars)
        self.setup_ttl_bars = int(setup_ttl_bars)
        self.fee_roundtrip = float(fee_roundtrip)
        self.slippage = float(slippage)
        self.cost_mult = float(cost_mult)
        self.enable_short = bool(enable_short)

        self.rej = Counter()
        self.pending: Optional[PendingSetup] = None
        self._bar_counter = 0
        self._last_trade_bar = -10**18

    def generate_signal(self, raw_df: pd.DataFrame) -> Optional[Signal]:
        self._bar_counter += 1

        if len(raw_df) < MIN_BARS:
            self.rej["min_bars"] += 1
            return None

        df = apply_indicators(raw_df)
        if df.empty:
            self.rej["empty_df"] += 1
            return None

        df = self._add_fast_context(df)
        last = df.iloc[-1]

        required_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ema20",
            "ema50",
            "ema200",
            "rsi",
            "cci",
            "atr",
            "vol_sma",
        ]
        if any(pd.isna(last.get(col)) for col in required_cols):
            self.rej["na"] += 1
            return None

        if (self._bar_counter - self._last_trade_bar) < self.cooldown_bars:
            if self.pending is not None:
                self.rej["pending_dropped_cooldown"] += 1
                self.pending = None
            self.rej["cooldown"] += 1
            return None

        pending_signal, pending_still_active = self._evaluate_pending(df)
        if pending_signal is not None:
            self._last_trade_bar = self._bar_counter
            return pending_signal
        if pending_still_active:
            return None

        bias = self._market_bias(df)
        if bias == "bull":
            setup = self._build_setup_long(df)
            if setup is None:
                self.rej["no_long_setup"] += 1
                return None
            self.pending = setup
            self.rej["setup_long"] += 1
            logger.info(
                "SETUP LONG | trigger=%.8f stop=%.8f score=%.2f expires=%d | %s",
                setup.trigger,
                setup.stop_loss,
                setup.score,
                setup.expires_bar,
                setup.reason,
            )
            return None

        if bias == "bear" and self.enable_short:
            setup = self._build_setup_short(df)
            if setup is None:
                self.rej["no_short_setup"] += 1
                return None
            self.pending = setup
            self.rej["setup_short"] += 1
            return None

        if bias == "bear":
            self.rej["short_disabled"] += 1
        else:
            self.rej["flat"] += 1
        return None

    @staticmethod
    def _add_fast_context(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
        out["ret1"] = out["close"].pct_change()
        out["range"] = (out["high"] - out["low"]).clip(lower=0.0)
        out["body"] = (out["close"] - out["open"]).abs()
        return out

    def _market_bias(self, df: pd.DataFrame) -> Literal["bull", "bear", "flat"]:
        if not self._passes_regime_filters(df):
            return "flat"

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        ema200 = float(last["ema200"])

        bull = (
            close > ema200 * 1.0002
            and ema20 > ema50 > ema200
            and ema20 >= float(prev["ema20"])
            and ema50 >= float(prev["ema50"]) * 0.9999
        )
        bear = (
            close < ema200 * 0.9998
            and ema20 < ema50 < ema200
            and ema20 <= float(prev["ema20"])
            and ema50 <= float(prev["ema50"]) * 1.0001
        )

        if bull:
            return "bull"
        if bear:
            return "bear"
        return "flat"

    def _passes_regime_filters(self, df: pd.DataFrame) -> bool:
        if self.tf == "1m":
            slope_lb = 80
            slope_min = 0.00050
            atr_min = 0.0012
            atr_max = 0.0180
            dist_max = 0.038
            spread_min = 0.00028
        elif self.tf == "15m":
            slope_lb = 28
            slope_min = 0.00024
            atr_min = 0.0010
            atr_max = 0.0280
            dist_max = 0.075
            spread_min = 0.00055
        else:  # 5m / default
            slope_lb = 36
            slope_min = 0.00014
            atr_min = 0.0009
            atr_max = 0.0200
            dist_max = 0.065
            spread_min = 0.00028

        if len(df) <= slope_lb:
            self.rej["regime_not_enough_bars"] += 1
            return False

        ema_now = float(df["ema200"].iloc[-1])
        ema_prev = float(df["ema200"].iloc[-slope_lb])
        close_now = float(df["close"].iloc[-1])
        atr_now = float(df["atr"].iloc[-1])
        ema20_now = float(df["ema20"].iloc[-1])
        ema50_now = float(df["ema50"].iloc[-1])

        if ema_prev <= 0 or close_now <= 0:
            self.rej["bad_price"] += 1
            return False

        slope = (ema_now - ema_prev) / ema_prev
        if abs(slope) < slope_min:
            self.rej["ema_flat"] += 1
            return False

        atr_pct = atr_now / close_now
        if atr_pct < atr_min:
            self.rej["too_quiet"] += 1
            return False
        if atr_pct > atr_max:
            self.rej["too_volatile"] += 1
            return False

        dist = abs((close_now - ema_now) / ema_now)
        if dist > dist_max:
            self.rej["too_far_from_ema"] += 1
            return False

        spread = abs(ema20_now - ema50_now) / close_now
        if spread < spread_min:
            self.rej["ema_spread_too_small"] += 1
            return False

        return True

    def _build_setup_long(self, df: pd.DataFrame) -> Optional[PendingSetup]:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        open_ = float(last["open"])
        high = float(last["high"])
        low = float(last["low"])
        close = float(last["close"])
        volume = float(last["volume"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        ema200 = float(last["ema200"])
        atr = float(last["atr"])
        vol_sma = max(float(last["vol_sma"]), 1e-12)
        rsi = float(last["rsi"])
        cci = float(last["cci"])

        if close <= ema200:
            self.rej["long_below_ema200"] += 1
            return None
        if not (ema20 > ema50 > ema200):
            self.rej["long_bad_ema_stack"] += 1
            return None
        if volume < 0.25 * vol_sma:
            self.rej["low_vol"] += 1
            return None

        touched_ema20 = low <= ema20 * 1.0030
        touched_mid_zone = low <= ema20 * 1.0080 and low >= (ema50 - 0.55 * atr)
        touched_ema50 = low <= ema50 * 1.0060 and close >= ema50 * 0.9955
        if not (touched_ema20 or touched_mid_zone or touched_ema50):
            self.rej["long_no_pullback"] += 1
            return None
        if low < ema50 - 0.92 * atr:
            self.rej["long_pullback_too_deep"] += 1
            return None

        candle_range = max(high - low, 1e-12)
        body = abs(close - open_)
        close_location = (close - low) / candle_range
        body_ratio = body / candle_range

        score = 0.0
        reasons: list[str] = []

        score += 2.0
        reasons.append("ema_stack")

        if ema20 >= float(prev["ema20"]):
            score += 0.6
            reasons.append("ema20_up")
        if ema50 >= float(prev["ema50"]):
            score += 0.4
            reasons.append("ema50_up")

        if touched_ema20:
            score += 1.2
            reasons.append("pullback_ema20")
        elif touched_mid_zone:
            score += 0.95
            reasons.append("pullback_mid_zone")
        elif touched_ema50:
            score += 0.75
            reasons.append("pullback_ema50")

        if 34.0 <= rsi <= 60.0:
            score += 1.1
            reasons.append("rsi_good")
        elif 30.0 <= rsi <= 64.0:
            score += 0.6
            reasons.append("rsi_loose")
        else:
            self.rej["long_rsi_bad"] += 1
            return None

        if -120.0 <= cci <= 100.0:
            score += 0.5
            reasons.append("cci_ok")
        elif -160.0 <= cci <= 140.0:
            score += 0.2
            reasons.append("cci_loose")

        if body_ratio >= 0.34:
            score += 0.9
            reasons.append("body_strong")
        elif body_ratio < 0.18:
            self.rej["long_weak_body"] += 1
            return None
        else:
            score += 0.35
            reasons.append("body_ok")

        if close_location >= 0.64:
            score += 1.0
            reasons.append("close_high")
        elif close_location >= 0.54:
            score += 0.45
            reasons.append("close_ok")
        else:
            self.rej["long_weak_close"] += 1
            return None

        if close <= open_:
            self.rej["long_not_bullish_candle"] += 1
            return None

        prev_mid = (float(prev["open"]) + float(prev["close"])) / 2.0
        if close > float(prev["high"]):
            score += 0.6
            reasons.append("reclaim_prev_high")
        elif close > prev_mid or close >= float(prev["close"]) * 0.999:
            score += 0.25
            reasons.append("close_above_prev_mid")
        else:
            self.rej["long_no_reclaim"] += 1
            return None

        if volume >= 0.90 * vol_sma:
            score += 1.0
            reasons.append("volume_expansion")
        elif volume >= 0.50 * vol_sma:
            score += 0.45
            reasons.append("volume_ok")
        else:
            self.rej["long_volume_too_weak"] += 1
            return None

        if close > ema20 * 1.017:
            self.rej["long_overextended"] += 1
            return None

        if self._recent_bullish_divergence(df):
            score += 0.30
            reasons.append("bull_div_bonus")

        if score < 4.9:
            self.rej["long_low_score"] += 1
            return None

        pullback_lookback = 4 if self.tf != "15m" else 3
        pullback_low = float(df["low"].iloc[-pullback_lookback:].min())
        stop_buffer = max(0.24 * atr, 0.34 * ATR_STOP_K * atr)
        stop_loss = pullback_low - stop_buffer
        if stop_loss >= close:
            self.rej["long_bad_stop"] += 1
            return None

        risk_abs = close - stop_loss
        if risk_abs > 1.55 * atr:
            self.rej["long_risk_too_wide"] += 1
            return None
        if risk_abs < 0.16 * atr:
            self.rej["long_risk_too_tight"] += 1
            return None

        trigger_anchor = max(high, float(prev["high"]))
        trigger = trigger_anchor + 0.005 * atr
        trigger_close_min = max(high - 0.04 * atr, close)
        trigger_vol_min = max(0.70 * vol_sma, volume * 0.72)

        return PendingSetup(
            side="long",
            created_bar=self._bar_counter,
            expires_bar=self._bar_counter + self.setup_ttl_bars,
            trigger=float(trigger),
            stop_loss=float(stop_loss),
            atr=float(atr),
            reason="long_continuation_v3:" + ",".join(reasons),
            signal_high=float(high),
            signal_low=float(low),
            score=float(score),
            trigger_close_min=float(trigger_close_min),
            trigger_vol_min=float(trigger_vol_min),
        )

    def _build_setup_short(self, df: pd.DataFrame) -> Optional[PendingSetup]:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        open_ = float(last["open"])
        high = float(last["high"])
        low = float(last["low"])
        close = float(last["close"])
        volume = float(last["volume"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        ema200 = float(last["ema200"])
        atr = float(last["atr"])
        vol_sma = max(float(last["vol_sma"]), 1e-12)
        rsi = float(last["rsi"])

        if close >= ema200 or not (ema20 < ema50 < ema200):
            self.rej["short_bad_structure"] += 1
            return None
        if volume < 0.45 * vol_sma:
            self.rej["low_vol"] += 1
            return None
        if not (42.0 <= rsi <= 64.0):
            self.rej["short_rsi_bad"] += 1
            return None

        candle_range = max(high - low, 1e-12)
        close_location = (high - close) / candle_range
        if close_location < 0.68 or close >= open_:
            self.rej["short_weak_close"] += 1
            return None

        pullback_high = float(df["high"].iloc[-4:].max())
        stop_loss = pullback_high + max(0.32 * atr, 0.42 * ATR_STOP_K * atr)
        if stop_loss <= close:
            self.rej["short_bad_stop"] += 1
            return None
        if (stop_loss - close) > 1.20 * atr:
            self.rej["short_risk_too_wide"] += 1
            return None

        trigger = min(low, float(prev["low"])) - 0.02 * atr
        return PendingSetup(
            side="short",
            created_bar=self._bar_counter,
            expires_bar=self._bar_counter + self.setup_ttl_bars,
            trigger=float(trigger),
            stop_loss=float(stop_loss),
            atr=float(atr),
            reason="short_continuation_v3",
            signal_high=float(high),
            signal_low=float(low),
            score=5.5,
            trigger_close_min=float(min(low, float(prev["low"]))),
            trigger_vol_min=float(max(0.80 * vol_sma, volume * 0.80)),
        )

    def _evaluate_pending(self, df: pd.DataFrame) -> tuple[Optional[Signal], bool]:
        if self.pending is None:
            return None, False

        p = self.pending
        if self._bar_counter >= p.expires_bar:
            self.rej[f"setup_expired_{p.side}"] += 1
            self.pending = None
            return None, False

        last = df.iloc[-1]
        close = float(last["close"])
        high = float(last["high"])
        low = float(last["low"])
        volume = float(last["volume"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        ema200 = float(last["ema200"])
        candle_range = max(high - low, 1e-12)
        close_location = (close - low) / candle_range if p.side == "long" else (high - close) / candle_range

        if p.side == "long":
            if low <= p.stop_loss:
                self.rej["setup_invalid_long_stop"] += 1
                self.pending = None
                return None, False
            if close < ema50 * 0.995 or ema20 < ema50 or ema50 <= ema200:
                self.rej["setup_invalid_long_trend"] += 1
                self.pending = None
                return None, False
            if high < p.trigger:
                self.rej["setup_wait_long"] += 1
                return None, True
            if close < p.trigger_close_min:
                self.rej["trigger_long_weak_close"] += 1
                self.pending = None
                return None, False
            if close_location < 0.58:
                self.rej["trigger_long_not_near_high"] += 1
                self.pending = None
                return None, False
            if volume < p.trigger_vol_min:
                self.rej["trigger_long_low_vol"] += 1
                self.pending = None
                return None, False

            entry = close
            if p.stop_loss >= entry:
                self.rej["long_bad_stop_after_trigger"] += 1
                self.pending = None
                return None, False

            rr = self._dynamic_rr(score=p.score, side="long")
            take_profit = entry + rr * (entry - p.stop_loss)
            if not self._passes_cost_filter(entry, take_profit):
                self.rej["cost_filter_long"] += 1
                self.pending = None
                return None, False

            self.pending = None
            self.rej["signal_long"] += 1
            logger.info(
                "SIGNAL LONG | trigger=%.8f entry=%.8f stop=%.8f score=%.2f",
                p.trigger,
                entry,
                p.stop_loss,
                p.score,
            )
            return Signal(
                symbol=self.symbol,
                side="long",
                entry_price=float(entry),
                stop_loss=float(p.stop_loss),
                take_profit=float(take_profit),
                atr=float(p.atr),
                reason=f"{p.reason}|score={p.score:.2f}|rr={rr:.2f}",
            ), False

        if high >= p.stop_loss:
            self.rej["setup_invalid_short_stop"] += 1
            self.pending = None
            return None, False
        if close > ema50 * 1.003 or ema20 > ema50 or ema50 >= ema200:
            self.rej["setup_invalid_short_trend"] += 1
            self.pending = None
            return None, False
        if low > p.trigger:
            self.rej["setup_wait_short"] += 1
            return None, True
        if close > p.trigger_close_min:
            self.rej["trigger_short_weak_close"] += 1
            self.pending = None
            return None, False
        if close_location < 0.58:
            self.rej["trigger_short_not_near_low"] += 1
            self.pending = None
            return None, False
        if volume < p.trigger_vol_min:
            self.rej["trigger_short_low_vol"] += 1
            self.pending = None
            return None, False

        entry = close
        if p.stop_loss <= entry:
            self.rej["short_bad_stop_after_trigger"] += 1
            self.pending = None
            return None, False

        rr = self._dynamic_rr(score=p.score, side="short")
        take_profit = entry - rr * (p.stop_loss - entry)
        if not self._passes_cost_filter(entry, take_profit):
            self.rej["cost_filter_short"] += 1
            self.pending = None
            return None, False

        self.pending = None
        self.rej["signal_short"] += 1
        return Signal(
            symbol=self.symbol,
            side="short",
            entry_price=float(entry),
            stop_loss=float(p.stop_loss),
            take_profit=float(take_profit),
            atr=float(p.atr),
            reason=f"{p.reason}|score={p.score:.2f}|rr={rr:.2f}",
        ), False

    def _dynamic_rr(self, score: float, side: str) -> float:
        base = 1.70 if side == "long" else 1.50
        if score >= 7.5:
            rr = base + 0.25
        elif score >= 6.5:
            rr = base + 0.10
        else:
            rr = base
        return max(1.30, min(float(RR_TP), rr))

    def _passes_cost_filter(self, entry: float, take_profit: float) -> bool:
        cost = max(0.0, self.fee_roundtrip + self.slippage)
        if cost <= 0.0 or entry <= 0.0:
            return True
        expected_pct = abs(take_profit - entry) / entry
        return expected_pct >= self.cost_mult * cost

    def _recent_bullish_divergence(self, df: pd.DataFrame, lookback: int = 50) -> bool:
        lows = df["low"].iloc[-lookback:]
        pivots = self._pivot_indices(lows, mode="low")
        if len(pivots) < 2:
            return False
        i1, i2 = pivots[-2], pivots[-1]
        if (i2 - i1) < 3:
            return False
        p1 = float(df["low"].iloc[i1])
        p2 = float(df["low"].iloc[i2])
        r1 = float(df["rsi"].iloc[i1])
        r2 = float(df["rsi"].iloc[i2])
        c1 = float(df["cci"].iloc[i1])
        c2 = float(df["cci"].iloc[i2])
        atr2 = float(df["atr"].iloc[i2])
        return (p2 < p1 - 0.18 * atr2 and r2 > r1 + 1.0) or (p2 < p1 - 0.18 * atr2 and c2 > c1 + 5.0)

    def _recent_bearish_divergence(self, df: pd.DataFrame, lookback: int = 50) -> bool:
        highs = df["high"].iloc[-lookback:]
        pivots = self._pivot_indices(highs, mode="high")
        if len(pivots) < 2:
            return False
        i1, i2 = pivots[-2], pivots[-1]
        if (i2 - i1) < 3:
            return False
        p1 = float(df["high"].iloc[i1])
        p2 = float(df["high"].iloc[i2])
        r1 = float(df["rsi"].iloc[i1])
        r2 = float(df["rsi"].iloc[i2])
        c1 = float(df["cci"].iloc[i1])
        c2 = float(df["cci"].iloc[i2])
        atr2 = float(df["atr"].iloc[i2])
        return (p2 > p1 + 0.18 * atr2 and r2 < r1 - 1.0) or (p2 > p1 + 0.18 * atr2 and c2 < c1 - 5.0)

    @staticmethod
    def _pivot_indices(series: pd.Series, mode: Literal["low", "high"]) -> list[int]:
        s = series.reset_index(drop=True)
        if len(s) < 5:
            return []
        if mode == "low":
            mask = (s.shift(1) > s) & (s.shift(-1) > s)
        else:
            mask = (s.shift(1) < s) & (s.shift(-1) < s)
        return [int(i) for i, ok in mask.items() if bool(ok)]
