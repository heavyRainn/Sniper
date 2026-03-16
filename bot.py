from __future__ import annotations

import datetime as dt
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import pandas as pd

from bybit_client import BybitClient, ClosedPnlRecord, PositionSnapshot
from config import INTERVAL, MIN_BARS
from logger_setup import setup_logger
from risk import RiskManager
from strategy import Signal, TrendDivStrategy
from trade_recorder import TradeRecorder


@dataclass
class PendingEntry:
    signal: Signal
    source_bar_ts: int
    execute_on_bar_ts: int


class TrendDivBot:
    def __init__(self):
        self.logger = setup_logger("trend_div_bot")
        self.client = BybitClient()
        self.strategy = TrendDivStrategy()
        self.risk = RiskManager(self.client, self.logger)

        self.interval_ms = self._interval_to_ms(INTERVAL)
        self.bar_queue: Deque[pd.Series] = deque(maxlen=2000)
        self.last_bar_close: int | None = None
        self.pending_entry: Optional[PendingEntry] = None
        self.trades = TradeRecorder("logs/trades.csv")

        self.current_position: Optional[PositionSnapshot] = None
        self.last_closed_marker: Optional[str] = None

    # ── WebSocket callback ─────────────────────────────────
    def on_kline(self, msg: dict):
        if not msg or "data" not in msg or not msg["data"]:
            return

        bar = msg["data"][0]
        ts = int(bar["start"])

        self._maybe_execute_pending(ts, bar)

        if not bar.get("confirm"):
            return

        if ts == self.last_bar_close:
            return
        self.last_bar_close = ts

        s = pd.Series(
            {
                "ts": ts,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
            }
        )
        self.bar_queue.append(s)

        self._log_bar(ts, bar)
        self._sync_position_state(observed_bar_ts=ts)
        self._on_new_closed_bar(ts)

    def _log_bar(self, ts_ms: int, bar: dict):
        ts_sec = ts_ms // 1000
        utc_time = dt.datetime.fromtimestamp(ts_sec, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        self.logger.info(
            "BAR | %s UTC | o=%.6f h=%.6f l=%.6f c=%.6f v=%.2f | bars=%d",
            utc_time,
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
            float(bar["volume"]),
            len(self.bar_queue),
        )

    # ── состояние позиции ──────────────────────────────────
    def _sync_position_state(self, observed_bar_ts: Optional[int] = None):
        live_pos = self.client.get_position(force=True)

        if live_pos is not None:
            if self.current_position is None:
                self.current_position = live_pos
                self.logger.info(
                    "Position sync: OPEN detected | side=%s size=%.6f avg=%.8f",
                    live_pos.side,
                    live_pos.size,
                    live_pos.avg_price,
                )
                self.trades.append(
                    {
                        "event": "POSITION_SYNC_OPEN",
                        "symbol": getattr(self.strategy, "symbol", ""),
                        "side": live_pos.side,
                        "entry": live_pos.avg_price,
                        "fill_entry": live_pos.avg_price,
                        "filled_qty": live_pos.size,
                        "position_side": live_pos.side,
                        "position_size": live_pos.size,
                        "avg_price": live_pos.avg_price,
                        "reason": "position_detected_on_sync",
                        "status": "OPEN",
                        "exec_bar_ts": observed_bar_ts or "",
                        "source": "bybit_positions",
                        "raw": live_pos.raw,
                    }
                )
            else:
                self.current_position = live_pos
            return

        if self.current_position is None:
            return

        prev = self.current_position
        self.current_position = None
        closed = self.client.get_latest_closed_pnl(since_ms=(prev.updated_time_ms or 0) - self.interval_ms)
        marker = self._closed_marker(closed, observed_bar_ts)
        if marker == self.last_closed_marker:
            return
        self.last_closed_marker = marker

        self.logger.info(
            "Position sync: CLOSE detected | prev_side=%s prev_size=%.6f",
            prev.side,
            prev.size,
        )

        self.trades.append(
            {
                "event": "POSITION_SYNC_CLOSE",
                "symbol": getattr(self.strategy, "symbol", ""),
                "side": prev.side,
                "entry": prev.avg_price,
                "fill_entry": prev.avg_price,
                "exit_price": closed.avg_exit_price if closed else "",
                "filled_qty": prev.size,
                "position_side": prev.side,
                "position_size": prev.size,
                "avg_price": prev.avg_price,
                "closed_pnl": closed.closed_pnl if closed else "",
                "reason": "position_closed_on_sync",
                "status": "CLOSED",
                "exec_bar_ts": observed_bar_ts or "",
                "source": "bybit_closed_pnl" if closed else "bybit_positions",
                "raw": closed.raw if closed else prev.raw,
            }
        )

    @staticmethod
    def _closed_marker(closed: Optional[ClosedPnlRecord], observed_bar_ts: Optional[int]) -> str:
        if closed is None:
            return f"bar:{observed_bar_ts or 0}"
        t = closed.updated_time_ms or closed.created_time_ms or 0
        return f"{t}:{closed.side}:{closed.qty}:{closed.closed_pnl}"

    # ── обработка нового закрытого бара ────────────────────
    def _on_new_closed_bar(self, closed_bar_ts: int):
        if len(self.bar_queue) < MIN_BARS:
            return

        if self.pending_entry is not None:
            self.logger.info(
                "Signal already armed for next bar open | source_bar=%s",
                self.pending_entry.source_bar_ts,
            )
            return

        if self.current_position is not None or self.client.has_open_position():
            self.logger.info("Position already open, skip arming new signal")
            return

        df = pd.DataFrame(list(self.bar_queue)).set_index("ts").sort_index()
        signal = self.strategy.generate_signal(df)
        if signal is None:
            return

        execute_on_bar_ts = closed_bar_ts + self.interval_ms
        self.pending_entry = PendingEntry(
            signal=signal,
            source_bar_ts=closed_bar_ts,
            execute_on_bar_ts=execute_on_bar_ts,
        )

        self.logger.info(
            "ARM SIGNAL %s | signal_entry=%.6f SL=%.6f TP=%.6f | exec_on_bar=%s",
            signal.side.upper(),
            float(signal.entry_price),
            float(signal.stop_loss),
            float(signal.take_profit),
            execute_on_bar_ts,
        )

    # ── исполнение отложенного входа на открытии следующего бара ───────────
    def _maybe_execute_pending(self, bar_ts: int, bar: dict):
        pending = self.pending_entry
        if pending is None:
            return

        if bar_ts < pending.execute_on_bar_ts:
            return

        self.pending_entry = None

        if self.current_position is not None or self.client.has_open_position(force=True):
            self.logger.info(
                "Pending signal dropped: position already open before execution | source_bar=%s",
                pending.source_bar_ts,
            )
            return

        model_entry = float(bar["open"])
        self._maybe_open_position(
            signal=pending.signal,
            model_entry_price=model_entry,
            execution_bar_ts=bar_ts,
            signal_bar_ts=pending.source_bar_ts,
        )

    # ── открытие позиции ───────────────────────────────────
    def _maybe_open_position(
        self,
        signal: Signal,
        model_entry_price: float,
        execution_bar_ts: int,
        signal_bar_ts: int,
    ):
        if not self.risk.can_open_trade():
            self.logger.info("Risk blocked armed signal execution")
            return

        price = float(model_entry_price)
        stop_loss = self.client.normalize_price(float(signal.stop_loss))
        take_profit = self.client.normalize_price(float(signal.take_profit))

        if not self.risk.validate_stop_vs_atr(price, stop_loss, float(signal.atr)):
            self.logger.info("Risk: stop/ATR validation failed → skip trade")
            return

        qty = self.risk.calc_qty(entry=price, stop=stop_loss)
        qty = self.client.normalize_qty(qty)
        if qty <= 0:
            self.logger.warning("Qty=0 after normalization, skip")
            return

        self.logger.info(
            "EXECUTE %s | signal_entry=%.6f model_entry(next_open)=%.6f SL=%.6f TP=%.6f qty=%.6f reason=%s",
            signal.side.upper(),
            float(signal.entry_price),
            price,
            stop_loss,
            take_profit,
            qty,
            signal.reason,
        )

        resp = self.client.place_market_order(
            side=signal.side,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        ret_code: int | str = "EXC"
        ret_msg = "place_order exception/None"
        order_id = ""
        order_link_id = ""
        fill_entry = ""
        filled_qty = ""
        status = "SUBMIT_FAILED"
        source = "place_order"
        raw = resp or {}

        if resp is not None:
            ret_code = int(resp.get("retCode", -1))
            ret_msg = str(resp.get("retMsg", ""))
            result = resp.get("result", {}) or {}
            order_id = result.get("orderId", "")
            order_link_id = result.get("orderLinkId", "")

            if ret_code == 0:
                status = "ACCEPTED"
                self.risk.register_trade_open()
                live_pos = self.client.wait_for_position_open(expected_side=signal.side, timeout_sec=6.0, poll_sec=0.5)
                if live_pos is not None:
                    self.current_position = live_pos
                    fill_entry = live_pos.avg_price
                    filled_qty = live_pos.size
                    status = "FILLED"
                    source = "place_order+positions"
                else:
                    self.logger.warning("Order accepted but open position not detected during polling")
                    source = "place_order_no_fill_sync"
            else:
                self.logger.warning("Order rejected | retCode=%s retMsg=%s", ret_code, ret_msg)
                status = "REJECTED"

        ret_msg_full = (
            f"{ret_msg} | signal_bar_ts={signal_bar_ts} | exec_bar_ts={execution_bar_ts} "
            f"| signal_entry={float(signal.entry_price):.8f} | model_entry={price:.8f}"
        )

        self.trades.append(
            {
                "event": "ENTRY_ORDER",
                "symbol": signal.symbol,
                "side": signal.side,
                "signal_entry": float(signal.entry_price),
                "model_entry": price,
                "entry": price,
                "fill_entry": fill_entry,
                "sl": stop_loss,
                "tp": take_profit,
                "qty": qty,
                "filled_qty": filled_qty,
                "reason": signal.reason,
                "status": status,
                "retCode": ret_code,
                "retMsg": ret_msg_full,
                "orderId": order_id,
                "orderLinkId": order_link_id,
                "position_side": self.current_position.side if self.current_position else "",
                "position_size": self.current_position.size if self.current_position else "",
                "avg_price": self.current_position.avg_price if self.current_position else "",
                "signal_bar_ts": signal_bar_ts,
                "exec_bar_ts": execution_bar_ts,
                "source": source,
                "raw": raw,
            }
        )

    def warmup_history(self):
        target = max(MIN_BARS + 50, 400)

        self.logger.info("Warmup: fetching ~%d candles via HTTP...", target)
        df = self.client.get_klines_df(interval=INTERVAL, limit=min(target, 1000))

        if df.empty:
            self.logger.warning("Warmup: no data received")
            return

        self.bar_queue.clear()
        for ts, row in df.iterrows():
            self.bar_queue.append(
                pd.Series(
                    {
                        "ts": int(ts) * 1000,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                )
            )

        last_ts_sec = int(df.index[-1])
        self.last_bar_close = last_ts_sec * 1000

        self.logger.info(
            "Warmup done: loaded %d candles. last_bar_close=%s",
            len(df),
            self.last_bar_close,
        )

    # ── старт ──────────────────────────────────────────────
    def run(self):
        self.warmup_history()
        self._sync_position_state(observed_bar_ts=self.last_bar_close)
        self.client.subscribe_kline(INTERVAL, self.on_kline)
        self.logger.info(
            "Trend+Div bot started on MAINNET | interval=%s | balance=%.2f",
            INTERVAL,
            float(self.client.get_equity()),
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard exit")
            self.client.close_ws()

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        token = str(interval).strip().upper()
        if token.isdigit():
            return int(token) * 60_000
        mapping = {
            "D": 24 * 60 * 60 * 1000,
            "W": 7 * 24 * 60 * 60 * 1000,
        }
        if token in mapping:
            return mapping[token]
        raise ValueError(f"Unsupported INTERVAL for next-open execution: {interval}")
