# bybit_client.py
from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd
from pybit.unified_trading import HTTP, WebSocket

from config import BYBIT_KEY, BYBIT_SECRET, CATEGORY, PAIR, LEV

logger = logging.getLogger("trend_div_bot")


class BybitClient:
    def __init__(self):
        if not BYBIT_KEY or not BYBIT_SECRET:
            raise RuntimeError("BYBIT_KEY / BYBIT_SECRET не заданы")

        # ✅ PUBLIC (без подписей)
        self.public_http = HTTP()

        # ✅ PRIVATE
        self.private_http = HTTP(
            api_key=BYBIT_KEY,
            api_secret=BYBIT_SECRET,
            recv_window=20000
        )

        self.ws = WebSocket(
            testnet=False,
            channel_type=CATEGORY,
        )

        self._pos_cache_value = False
        self._pos_cache_ts = 0.0
        self._pos_cache_ttl = 60.0

    # --- PRIVATE -------------------------------------------------
    def get_equity(self, recv_window: int = 20000, retries: int = 3) -> float:
        last_exc = None
        rw = recv_window

        for attempt in range(1, retries + 1):
            try:
                bal = self.private_http.get_wallet_balance(
                    accountType="UNIFIED",
                    recv_window=rw
                )
                equity = float(bal["result"]["list"][0]["totalWalletBalance"])
                logger.info("Account equity: %.2f USDT", equity)
                return equity

            except Exception as e:
                last_exc = e
                logger.warning("get_equity attempt %d/%d failed: %s", attempt, retries, str(e))
                rw += 2500
                time.sleep(0.2)

        raise last_exc

    def has_open_position(self) -> bool:
        now = time.time()

        # если кэш ещё свежий — отдаём его
        if now - self._pos_cache_ts < self._pos_cache_ttl:
            return self._pos_cache_value

        try:
            resp = self.private_http.get_positions(category=CATEGORY, symbol=PAIR,  recv_window=30000)
            lst = resp.get("result", {}).get("list", []) or []

            has_pos = False
            for pos in lst:
                size = float(pos.get("size", 0.0))
                if size != 0.0:
                    has_pos = True
                    break

            self._pos_cache_value = has_pos
            self._pos_cache_ts = now

            if has_pos:
                logger.info("Open position detected (fresh)")

            return has_pos


        except Exception as e:
            logger.warning("get_positions failed, using cached state: %s", str(e))

            # ✅ BACKOFF: считаем, что кэш "свежий" даже при ошибке
            self._pos_cache_ts = now
            return self._pos_cache_value

    def place_market_order(self, side: str, qty: float, stop_loss: float, take_profit: float):
        bybit_side = "Buy" if side == "long" else "Sell"
        logger.info(
            "Sending order: %s %s qty=%.6f SL=%.6f TP=%.6f",
            bybit_side, PAIR, qty, stop_loss, take_profit
        )
        try:
            resp = self.private_http.place_order(
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

    # --- PUBLIC -------------------------------------------------
    def get_klines_df(self, interval: str, limit: int = 500) -> pd.DataFrame:
        resp = self.public_http.get_kline(
            category=CATEGORY,
            symbol=PAIR,
            interval=interval,
            limit=min(max(1, limit), 1000),
        )

        lst = resp.get("result", {}).get("list", []) or []
        if not lst:
            return pd.DataFrame()

        lst = list(reversed(lst))

        rows = []
        for k in lst:
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

    # --- WebSocket ------------------------------------------------
    def subscribe_kline(self, interval: str, callback: Callable[[dict], None]):
        self.ws.kline_stream(
            callback=callback,
            symbol=PAIR,
            interval=interval,
        )

    def close_ws(self):
        self.ws.exit()
