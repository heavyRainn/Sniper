# bybit_client.py
from __future__ import annotations

import pandas as pd
import logging
from typing import Callable


from pybit.unified_trading import HTTP, WebSocket

from config import BYBIT_KEY, BYBIT_SECRET, CATEGORY, PAIR, LEV

logger = logging.getLogger("bot")


class BybitClient:
    """
    Обёртка над HTTP/WebSocket-клиентом Bybit (mainnet, unified trading).
    """

    def __init__(self):
        if not BYBIT_KEY or not BYBIT_SECRET:
            raise RuntimeError("BYBIT_KEY / BYBIT_SECRET не заданы")

        self.http = HTTP(api_key=BYBIT_KEY, api_secret=BYBIT_SECRET)
        self.ws = WebSocket(
            testnet=False,
            channel_type=CATEGORY,
        )

    # --- HTTP helpers -------------------------------------------------
    def get_equity(self) -> float:
        bal = self.http.get_wallet_balance(accountType="UNIFIED")
        equity = float(bal["result"]["list"][0]["totalWalletBalance"])
        logger.info("Account equity: %.2f USDT", equity)
        return equity

    def has_open_position(self) -> bool:
        """
        Простейшая проверка: есть ли открытая позиция по инструменту.
        """
        try:
            resp = self.http.get_positions(category=CATEGORY, symbol=PAIR)
            lst = resp.get("result", {}).get("list", []) or []
            for pos in lst:
                size = float(pos.get("size", 0.0))
                if size != 0.0:
                    logger.info("Open position detected: side=%s size=%s", pos.get("side"), pos.get("size"))
                    return True
        except Exception as e:
            logger.exception("get_positions failed: %s", e)
        return False

    def place_market_order(self, side: str, qty: float, stop_loss: float, take_profit: float):
        bybit_side = "Buy" if side == "long" else "Sell"
        logger.info(
            "Sending order: %s %s qty=%.6f SL=%.6f TP=%.6f",
            bybit_side, PAIR, qty, stop_loss, take_profit
        )
        try:
            resp = self.http.place_order(
                category=CATEGORY,
                symbol=PAIR,
                side=bybit_side,
                orderType="Market",
                qty=str(qty),
                timeInForce="IOC",
                takeProfit=round(take_profit, 6),
                stopLoss=round(stop_loss, 6),
                leverage=LEV,
            )
            logger.info("ORDER RESP: %s", resp)
        except Exception as e:
            logger.exception("Order failed: %s", e)

    # --- WebSocket helpers --------------------------------------------
    def subscribe_kline(self, interval: str, callback: Callable[[dict], None]):
        self.ws.kline_stream(
            callback=callback,
            symbol=PAIR,
            interval=interval,
        )

    def get_klines_df(self, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Загружает последние свечи и возвращает DataFrame:
        index = ts (int, seconds)
        columns = open, high, low, close, volume
        """
        resp = self.http.get_kline(
            category=CATEGORY,
            symbol=PAIR,
            interval=interval,
            limit=min(max(1, limit), 1000),
        )

        lst = resp.get("result", {}).get("list", []) or []
        if not lst:
            return pd.DataFrame()

        # API отдаёт newest-first -> разворачиваем
        lst = list(reversed(lst))

        rows = []
        for k in lst:
            # kline: [startTime, open, high, low, close, volume, turnover...]
            ts_sec = int(k[0]) // 1000
            rows.append(
                {
                    "ts": ts_sec,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )

        df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
        df.set_index("ts", inplace=True)
        return df

    def close_ws(self):
        self.ws.exit()
