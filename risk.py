# risk.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional

from config import (
    ACCOUNT_EQUITY_VIRTUAL,
    EQUITY_PCT_PER_TRADE,
    DAILY_DD_LIMIT,
    MAX_QTY_PER_TRADE,
)

# Доп. ограничения (можешь вынести в config позже)
MAX_TRADES_PER_DAY = 3
MIN_STOP_DISTANCE_MULT = 0.2   # минимальная дистанция стопа = 0.2 * ATR (защита от микростопов)
MAX_STOP_DISTANCE_MULT = 5.0   # не лезем, если стоп слишком огромный относительно ATR


@dataclass
class DayState:
    day: date
    equity_start: float
    trades: int = 0


class RiskManager:
    """
    Отвечает за:
    - дневной лимит просадки (от реального equity на бирже)
    - лимит количества сделок в день
    - расчёт размера позиции по риску
    - жёсткий потолок qty
    - sanity-check стопа относительно ATR (защита от странных сигналов)
    """

    def __init__(self, bybit_client, logger):
        self.client = bybit_client
        self.logger = logger
        self.state: Optional[DayState] = None

    def _today_utc(self) -> date:
        return datetime.now(timezone.utc).date()

    def _ensure_day_state(self):
        today = self._today_utc()
        if self.state is None or self.state.day != today:
            eq = self.client.get_equity()
            self.state = DayState(day=today, equity_start=eq, trades=0)
            self.logger.info("Risk: new day state | day=%s | equity_start=%.2f", today, eq)

    def current_dd(self) -> float:
        """
        Текущая дневная просадка относительно equity_start сегодняшнего дня.
        """
        self._ensure_day_state()
        eq_now = self.client.get_equity()
        dd = (eq_now - self.state.equity_start) / self.state.equity_start
        return dd

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
        """
        Базовая sanity-проверка:
        - стоп не должен быть слишком близко к цене относительно ATR
        - и не должен быть экстремально далеко
        """
        if atr <= 0:
            return True

        dist = abs(entry - stop)
        if dist < MIN_STOP_DISTANCE_MULT * atr:
            self.logger.info(
                "Risk: stop too tight vs ATR | dist=%.8f atr=%.8f",
                dist, atr
            )
            return False

        if dist > MAX_STOP_DISTANCE_MULT * atr:
            self.logger.info(
                "Risk: stop too wide vs ATR | dist=%.8f atr=%.8f",
                dist, atr
            )
            return False

        return True

    def calc_qty(self, entry: float, stop: float) -> float:
        """
        Расчёт размера позиции:
        риск $ = виртуальный equity * EQUITY_PCT_PER_TRADE
        qty = риск$ / расстояние до стопа
        """
        per_unit_risk = abs(entry - stop)
        if per_unit_risk <= 0:
            return 0.0

        risk_usd = ACCOUNT_EQUITY_VIRTUAL * EQUITY_PCT_PER_TRADE
        qty = risk_usd / per_unit_risk

        # жёсткий потолок
        if qty > MAX_QTY_PER_TRADE:
            qty = MAX_QTY_PER_TRADE

        return qty
