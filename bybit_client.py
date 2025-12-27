# bybit_client.py
from __future__ import annotations

import logging
import time
from decimal import Decimal, ROUND_DOWN
from typing import Callable

import pandas as pd
from pybit.unified_trading import HTTP, WebSocket

from typing import Optional, Dict, Any
import uuid

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
            ping_interval=30,  # должно быть > ping_timeout
            ping_timeout=10,  # время ожидания pong
            restart_on_error=True,
            retries=10_000,
        )

        self._pos_cache_value = False
        self._pos_cache_ts = 0.0
        self._pos_cache_ttl = 2.0

        self._filters_loaded = False
        self.tick_size = None
        self.qty_step = None
        self.min_qty = None

        self._load_symbol_filters()
        self._ensure_leverage()

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

    def invalidate_position_cache(self):
        self._pos_cache_ts = 0.0

    def has_open_position(self, force: bool = False) -> bool:
        now = time.time()
        if (not force) and (now - self._pos_cache_ts < self._pos_cache_ttl):
            return self._pos_cache_value

        try:
            resp = self.private_http.get_positions(category=CATEGORY, symbol=PAIR, recv_window=30000)
            lst = resp.get("result", {}).get("list", []) or []

            has_pos = any(float(pos.get("size", 0.0)) != 0.0 for pos in lst)

            self._pos_cache_value = has_pos
            self._pos_cache_ts = now

            if has_pos:
                logger.info("Open position detected (fresh)")
            return has_pos

        except Exception as e:
            logger.warning("get_positions failed, keeping previous cached state: %s", str(e))
            # НЕ делаем _pos_cache_ts = now, чтобы не “заморозить” ложное состояние
            return self._pos_cache_value

    def place_market_order(self, side: str, qty: float, stop_loss: float, take_profit: float) -> Optional[
        Dict[str, Any]]:
        bybit_side = "Buy" if side == "long" else "Sell"

        qty_n = self.normalize_qty(qty)
        sl_n = self.normalize_price(stop_loss)
        tp_n = self.normalize_price(take_profit)

        if qty_n <= 0:
            logger.warning("Normalized qty is 0 -> skip order")
            return None

        order_link_id = f"sniper-{uuid.uuid4().hex[:16]}"

        logger.info(
            "Sending MARKET order: %s %s qty=%.10f SL=%.10f TP=%.10f",
            bybit_side, PAIR, qty_n, sl_n, tp_n
        )

        try:
            resp = self.private_http.place_order(
                category=CATEGORY,
                symbol=PAIR,
                side=bybit_side,
                orderType="Market",
                qty=str(qty_n),
                timeInForce="IOC",
                takeProfit=str(tp_n),
                stopLoss=str(sl_n),
                orderLinkId=order_link_id,  # ✅ идемпотентность
            )
            logger.info("ORDER RESP: %s", resp)

            self.invalidate_position_cache()
            return resp

        except Exception as e:
            logger.exception("Order failed: %s", e)
            return None

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

    def _round_step(self, value: float, step: float) -> float:
        if step is None or step <= 0:
            return float(value)
        v = Decimal(str(value))
        s = Decimal(str(step))
        return float((v / s).to_integral_value(rounding=ROUND_DOWN) * s)

    def _load_symbol_filters(self):
        try:
            info = self.public_http.get_instruments_info(category=CATEGORY, symbol=PAIR)
            item = info["result"]["list"][0]

            tick = float(item["priceFilter"]["tickSize"])
            step = float(item["lotSizeFilter"]["qtyStep"])
            minq = float(item["lotSizeFilter"].get("minOrderQty", 0.0))

            self.tick_size = tick
            self.qty_step = step
            self.min_qty = minq
            self._filters_loaded = True

            logger.info("Loaded filters | tick=%.10f qty_step=%.10f min_qty=%.10f", tick, step, minq)

        except Exception as e:
            logger.warning("Failed to load instruments filters, fallback to naive rounding: %s", e)
            self._filters_loaded = False

    def normalize_price(self, price: float) -> float:
        return self._round_step(price, self.tick_size or 0.0)

    def normalize_qty(self, qty: float) -> float:
        q = self._round_step(qty, self.qty_step or 0.0)
        if self.min_qty and q < self.min_qty:
            return 0.0
        return q

    def _ensure_leverage(self):
        try:
            # для linear обычно нужны buyLeverage/sellLeverage
            self.private_http.set_leverage(
                category=CATEGORY,
                symbol=PAIR,
                buyLeverage=str(LEV),
                sellLeverage=str(LEV),
            )
            logger.info("Leverage set to %s for %s", LEV, PAIR)
        except Exception as e:
            logger.warning("Failed to set leverage (maybe already set / permissions): %s", e)

    def close_ws(self):
        self.ws.exit()
