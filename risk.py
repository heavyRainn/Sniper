# risk.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional

from config import (
    EQUITY_PCT_PER_TRADE,
    DAILY_DD_LIMIT,
    MAX_QTY_PER_TRADE,
)

MAX_TRADES_PER_DAY = 3
MIN_STOP_DISTANCE_MULT = 0.2
MAX_STOP_DISTANCE_MULT = 5.0


@dataclass
class DayState:
    day: date
    equity_start: float
    trades: int = 0


class RiskManager:
    def __init__(self, bybit_client, logger):
        self.client = bybit_client
        self.logger = logger
        self.state: Optional[DayState] = None

    def _today_utc(self) -> date:
        return datetime.now(timezone.utc).date()

    def _get_equity_safe(self) -> float:
        """
        Берём реальный equity с биржи.
        Если API временно недоступен — fallback на виртуальный.
        """
        try:
            return float(self.client.get_equity())
        except Exception as e:
            self.logger.warning(
                "Risk: get_equity failed, fallback to virtual equity %.2f | err=%s",
                float(0), str(e)
            )
            return float(0)

    def _ensure_day_state(self):
        today = self._today_utc()
        if self.state is None or self.state.day != today:
            eq = self._get_equity_safe()
            self.state = DayState(day=today, equity_start=eq, trades=0)
            self.logger.info("Risk: new day state | day=%s | equity_start=%.2f", today, eq)

    def current_dd(self) -> float:
        self._ensure_day_state()
        eq_now = self._get_equity_safe()
        return (eq_now - self.state.equity_start) / self.state.equity_start

    def can_open_trade(self) -> bool:
        self._ensure_day_state()

        dd = self.current_dd()
        if dd <= DAILY_DD_LIMIT:
            self.logger.warning(
                "Risk: daily DD %.2f%% <= limit %.2f%%",
                dd * 100.0, DAILY_DD_LIMIT * 100.0
            )
            return False

        if self.state.trades >= MAX_TRADES_PER_DAY:
            self.logger.warning(
                "Risk: trades per day limit reached (%d)",
                self.state.trades
            )
            return False

        return True

    def register_trade_open(self):
        self._ensure_day_state()
        self.state.trades += 1

    def validate_stop_vs_atr(self, entry: float, stop: float, atr: float) -> bool:
        if atr <= 0:
            return True

        dist = abs(entry - stop)
        if dist < MIN_STOP_DISTANCE_MULT * atr:
            self.logger.info("Risk: stop too tight vs ATR | dist=%.8f atr=%.8f", dist, atr)
            return False

        if dist > MAX_STOP_DISTANCE_MULT * atr:
            self.logger.info("Risk: stop too wide vs ATR | dist=%.8f atr=%.8f", dist, atr)
            return False

        return True

    def calc_qty(self, entry: float, stop: float) -> float:
        """
        Риск на сделку = (реальный equity) * EQUITY_PCT_PER_TRADE
        qty = risk$ / |entry-stop|
        """
        per_unit_risk = abs(entry - stop)
        if per_unit_risk <= 0:
            return 0.0

        equity_now = self._get_equity_safe()              # ✅ весь баланс
        risk_usd = equity_now * float(EQUITY_PCT_PER_TRADE)

        qty = risk_usd / per_unit_risk

        if qty > MAX_QTY_PER_TRADE:
            qty = MAX_QTY_PER_TRADE

        return qty
