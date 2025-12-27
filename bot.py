# bot.py
from __future__ import annotations

import datetime as dt
import time
from collections import deque
from typing import Deque

import pandas as pd

from bybit_client import BybitClient
from config import (
    INTERVAL
)
from config import MIN_BARS
from logger_setup import setup_logger
from risk import RiskManager
from strategy import TrendDivStrategy
from trade_recorder import TradeRecorder


class TrendDivBot:
    def __init__(self):
        self.logger = setup_logger("trend_div_bot")
        self.client = BybitClient()
        self.strategy = TrendDivStrategy()

        self.risk = RiskManager(self.client, self.logger)

        self.bar_queue: Deque[pd.Series] = deque(maxlen=2000)
        self.last_bar_close: int | None = None
        self.trades = TradeRecorder("logs/trades.csv")

    # ── WebSocket callback ─────────────────────────────────
    def on_kline(self, msg: dict):
        if not msg or "data" not in msg:
            return

        bar = msg["data"][0]
        if not bar.get("confirm"):
            return

        ts = int(bar["start"])
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
        self._on_new_bar()

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

    # ── обработка нового закрытого бара ────────────────────
    def _on_new_bar(self):
        if len(self.bar_queue) < MIN_BARS:
            return

        df = (
            pd.DataFrame(list(self.bar_queue))
            .set_index("ts")
            .sort_index()
        )

        signal = self.strategy.generate_signal(df)
        if signal is None:
            return

        # если уже есть позиция по инструменту — ничего не делаем
        if self.client.has_open_position():
            self.logger.info("Position already open, skip signals")
            return

        self._maybe_open_position(signal)

    # ── открытие позиции ───────────────────────────────────
    def _maybe_open_position(self, signal):
        # 1) общий риск-контроль
        if not self.risk.can_open_trade():
            return

        price = float(signal.entry_price)
        stop_loss = float(signal.stop_loss)
        take_profit = float(signal.take_profit)

        # 2) sanity-check стопа относительно ATR
        if not self.risk.validate_stop_vs_atr(price, stop_loss, float(signal.atr)):
            self.logger.info("Risk: stop/ATR validation failed → skip trade")
            return

        # 3) qty через risk manager (risk-based)
        qty = self.risk.calc_qty(entry=price, stop=stop_loss)
        qty = self.client.normalize_qty(qty)

        if qty <= 0:
            self.logger.warning("Qty=0, skip")
            return

        self.logger.info(
            "SIGNAL %s | entry=%.6f SL=%.6f TP=%.6f qty=%.6f reason=%s",
            signal.side.upper(),
            price, stop_loss, take_profit, qty, signal.reason,
        )

        # 4) отправляем ордер
        resp = self.client.place_market_order(
            side=signal.side,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # 5) пишем сделку в файл (в любом случае: успех/ошибка)
        if resp is None:
            self.trades.append({
                "symbol": signal.symbol,
                "side": signal.side,
                "entry": price,
                "sl": stop_loss,
                "tp": take_profit,
                "qty": qty,
                "reason": signal.reason,
                "retCode": "EXC",
                "retMsg": "place_order exception/None",
                "orderId": "",
                "orderLinkId": "",
            })
            return

        ret_code = int(resp.get("retCode", -1))
        ret_msg = resp.get("retMsg", "")
        result = resp.get("result", {}) or {}
        order_id = result.get("orderId", "")
        order_link_id = result.get("orderLinkId", "")

        self.trades.append({
            "symbol": signal.symbol,
            "side": signal.side,
            "entry": price,
            "sl": stop_loss,
            "tp": take_profit,
            "qty": qty,
            "reason": signal.reason,
            "retCode": ret_code,
            "retMsg": ret_msg,
            "orderId": order_id,
            "orderLinkId": order_link_id,
        })

        # 6) увеличиваем счётчик сделок только если биржа приняла ордер
        if ret_code == 0:
            self.risk.register_trade_open()
        else:
            self.logger.warning("Order rejected | retCode=%s retMsg=%s", ret_code, ret_msg)

    def warmup_history(self):
        """
        Подкачиваем историю по HTTP перед стартом WS,
        чтобы не ждать накопления MIN_BARS.
        """
        # Чуть больше минимума, чтобы индикаторы уверенно стартовали
        target = max(MIN_BARS + 50, 400)

        self.logger.info("Warmup: fetching ~%d candles via HTTP...", target)
        df = self.client.get_klines_df(interval=INTERVAL, limit=min(target, 1000))

        if df.empty:
            self.logger.warning("Warmup: no data received")
            return

        # Заполняем очередь баров в хронологическом порядке
        self.bar_queue.clear()
        for ts, row in df.iterrows():
            self.bar_queue.append(
                pd.Series(
                    {
                        "ts": int(ts) * 1000,  # В очереди мы храним ms-формат как раньше
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                )
            )

        # last_bar_close чтобы не повторить что уже есть
        last_ts_sec = int(df.index[-1])
        self.last_bar_close = last_ts_sec * 1000

        self.logger.info(
            "Warmup done: loaded %d candles. last_bar_close=%s",
            len(df),
            self.last_bar_close
        )

    # ── старт ──────────────────────────────────────────────
    def run(self):
        self.warmup_history()
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
