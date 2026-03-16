from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from config import DAILY_DD_LIMIT, EQUITY_PCT_PER_TRADE, MAX_QTY_PER_TRADE

MAX_TRADES_PER_DAY = 3
MIN_STOP_DISTANCE_MULT = 0.2
MAX_STOP_DISTANCE_MULT = 5.0
MAX_EQUITY_STALENESS_SEC = 90


@dataclass
class DayState:
    day: date
    equity_start: float
    trades: int = 0
    provisional: bool = False


@dataclass
class EquitySnapshot:
    value: Optional[float]
    fresh: bool
    source: str
    age_sec: Optional[float] = None


class RiskManager:
    def __init__(self, bybit_client, logger):
        self.client = bybit_client
        self.logger = logger
        self.state: Optional[DayState] = None

        self.last_known_equity: Optional[float] = None
        self.last_equity_success_at: Optional[datetime] = None

    def _today_utc(self) -> date:
        return datetime.now(timezone.utc).date()

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _equity_age_sec(self) -> Optional[float]:
        if self.last_equity_success_at is None:
            return None
        return (self._now_utc() - self.last_equity_success_at).total_seconds()

    def _fetch_equity(self, allow_stale: bool = True) -> EquitySnapshot:
        try:
            equity = float(self.client.get_equity())
            if equity <= 0:
                raise ValueError(f"Non-positive equity returned: {equity}")

            self.last_known_equity = equity
            self.last_equity_success_at = self._now_utc()
            return EquitySnapshot(value=equity, fresh=True, source="api", age_sec=0.0)

        except Exception as exc:
            if allow_stale and self.last_known_equity is not None:
                age_sec = self._equity_age_sec()
                self.logger.warning(
                    "Risk: get_equity failed, using last known equity %.2f | age=%.1fs | err=%s",
                    float(self.last_known_equity),
                    float(age_sec or 0.0),
                    str(exc),
                )
                return EquitySnapshot(
                    value=float(self.last_known_equity),
                    fresh=False,
                    source="last_known",
                    age_sec=age_sec,
                )

            self.logger.warning(
                "Risk: get_equity failed and no last known equity available | err=%s",
                str(exc),
            )
            return EquitySnapshot(value=None, fresh=False, source="unavailable", age_sec=None)

    def _ensure_day_state(self, snapshot: Optional[EquitySnapshot] = None):
        today = self._today_utc()
        snapshot = snapshot or self._fetch_equity(allow_stale=True)

        if self.state is None or self.state.day != today:
            equity_start = float(snapshot.value) if snapshot.value is not None else 0.0
            provisional = not snapshot.fresh
            self.state = DayState(
                day=today,
                equity_start=equity_start,
                trades=0,
                provisional=provisional,
            )
            self.logger.info(
                "Risk: new day state | day=%s | equity_start=%.2f | provisional=%s | source=%s",
                today,
                equity_start,
                provisional,
                snapshot.source,
            )
            return

        if self.state.provisional and self.state.trades == 0 and snapshot.fresh and snapshot.value is not None:
            old_value = self.state.equity_start
            self.state.equity_start = float(snapshot.value)
            self.state.provisional = False
            self.logger.info(
                "Risk: re-based provisional day equity | old=%.2f -> new=%.2f",
                old_value,
                self.state.equity_start,
            )

    def current_dd(self) -> float:
        snapshot = self._fetch_equity(allow_stale=True)
        self._ensure_day_state(snapshot)

        if self.state is None or self.state.equity_start <= 0:
            return 0.0
        if snapshot.value is None:
            return 0.0

        return (float(snapshot.value) - self.state.equity_start) / self.state.equity_start

    def can_open_trade(self) -> bool:
        snapshot = self._fetch_equity(allow_stale=True)
        self._ensure_day_state(snapshot)

        if self.state is None:
            self.logger.warning("Risk: no day state -> block new trade")
            return False

        if snapshot.value is None:
            self.logger.warning("Risk: equity unavailable -> block new trade")
            return False

        if not snapshot.fresh:
            self.logger.warning(
                "Risk: blocking new trade because equity is stale | source=%s | age=%.1fs",
                snapshot.source,
                float(snapshot.age_sec or 0.0),
            )
            return False

        if snapshot.age_sec is not None and snapshot.age_sec > MAX_EQUITY_STALENESS_SEC:
            self.logger.warning(
                "Risk: blocking new trade because fresh equity is too old | age=%.1fs",
                float(snapshot.age_sec),
            )
            return False

        if self.state.provisional and self.state.trades == 0:
            self.state.equity_start = float(snapshot.value)
            self.state.provisional = False

        if self.state.equity_start <= 0:
            self.logger.warning("Risk: invalid equity_start %.2f -> block new trade", self.state.equity_start)
            return False

        dd = (float(snapshot.value) - self.state.equity_start) / self.state.equity_start
        if dd <= DAILY_DD_LIMIT:
            self.logger.warning(
                "Risk: daily DD %.2f%% <= limit %.2f%%",
                dd * 100.0,
                DAILY_DD_LIMIT * 100.0,
            )
            return False

        if self.state.trades >= MAX_TRADES_PER_DAY:
            self.logger.warning(
                "Risk: trades per day limit reached (%d)",
                self.state.trades,
            )
            return False

        return True

    def register_trade_open(self):
        self._ensure_day_state()
        if self.state is None:
            return
        self.state.trades += 1
        self.logger.info(
            "Risk: registered trade open | day=%s | trades_today=%d",
            self.state.day,
            self.state.trades,
        )

    def validate_stop_vs_atr(self, entry: float, stop: float, atr: float) -> bool:
        if atr <= 0:
            return True

        dist = abs(float(entry) - float(stop))
        if dist < MIN_STOP_DISTANCE_MULT * float(atr):
            self.logger.info("Risk: stop too tight vs ATR | dist=%.8f atr=%.8f", dist, atr)
            return False

        if dist > MAX_STOP_DISTANCE_MULT * float(atr):
            self.logger.info("Risk: stop too wide vs ATR | dist=%.8f atr=%.8f", dist, atr)
            return False

        return True

    def calc_qty(self, entry: float, stop: float) -> float:
        per_unit_risk = abs(float(entry) - float(stop))
        if per_unit_risk <= 0:
            return 0.0

        snapshot = self._fetch_equity(allow_stale=True)
        if snapshot.value is None:
            self.logger.warning("Risk: cannot calc qty without equity")
            return 0.0

        if not snapshot.fresh:
            self.logger.warning(
                "Risk: refusing qty calc on stale equity | source=%s | age=%.1fs",
                snapshot.source,
                float(snapshot.age_sec or 0.0),
            )
            return 0.0

        risk_usd = float(snapshot.value) * float(EQUITY_PCT_PER_TRADE)
        if risk_usd <= 0:
            return 0.0

        qty = risk_usd / per_unit_risk
        qty = min(qty, float(MAX_QTY_PER_TRADE))
        return float(qty)
