from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pybit.unified_trading import HTTP, WebSocket

from config import BYBIT_KEY, BYBIT_SECRET, CATEGORY, LEV, PAIR

logger = logging.getLogger("trend_div_bot")


@dataclass
class PositionSnapshot:
    side: str
    size: float
    avg_price: float
    mark_price: float
    unrealized_pnl: float
    position_value: float
    updated_time_ms: Optional[int]
    raw: Dict[str, Any]


@dataclass
class ClosedPnlRecord:
    side: str
    qty: float
    avg_entry_price: float
    avg_exit_price: float
    closed_pnl: float
    created_time_ms: Optional[int]
    updated_time_ms: Optional[int]
    raw: Dict[str, Any]


class BybitClient:
    def __init__(self):
        if not BYBIT_KEY or not BYBIT_SECRET:
            raise RuntimeError("BYBIT_KEY / BYBIT_SECRET не заданы")

        self.public_http = HTTP()
        self.private_http = HTTP(
            api_key=BYBIT_KEY,
            api_secret=BYBIT_SECRET,
            recv_window=20000,
        )

        self.ws = WebSocket(
            testnet=False,
            channel_type=CATEGORY,
            ping_interval=30,
            ping_timeout=10,
            restart_on_error=True,
            retries=10_000,
        )

        self._pos_cache_value = False
        self._pos_cache_ts = 0.0
        self._pos_cache_ttl = 2.0
        self._position_snapshot_cache: Optional[PositionSnapshot] = None

        self._filters_loaded = False
        self.tick_size: Optional[float] = None
        self.qty_step: Optional[float] = None
        self.min_qty: Optional[float] = None

        self._load_symbol_filters()
        self._ensure_leverage()

    # --- helpers -------------------------------------------------
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        try:
            if value in (None, ""):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _round_step(self, value: float, step: float) -> float:
        if step is None or step <= 0:
            return float(value)
        v = Decimal(str(value))
        s = Decimal(str(step))
        return float((v / s).to_integral_value(rounding=ROUND_DOWN) * s)

    # --- equity / account ---------------------------------------
    def get_equity(self, recv_window: int = 20000, retries: int = 3) -> float:
        last_exc = None
        rw = recv_window

        for attempt in range(1, retries + 1):
            try:
                bal = self.private_http.get_wallet_balance(
                    accountType="UNIFIED",
                    recv_window=rw,
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

    # --- position tracking --------------------------------------
    def invalidate_position_cache(self):
        self._pos_cache_ts = 0.0
        self._position_snapshot_cache = None

    def get_position(self, force: bool = False) -> Optional[PositionSnapshot]:
        now = time.time()
        if (not force) and (now - self._pos_cache_ts < self._pos_cache_ttl):
            return self._position_snapshot_cache

        try:
            resp = self.private_http.get_positions(
                category=CATEGORY,
                symbol=PAIR,
                recv_window=30000,
            )
            lst = resp.get("result", {}).get("list", []) or []
            live_positions = [p for p in lst if self._safe_float(p.get("size", 0.0)) > 0.0]

            if not live_positions:
                self._pos_cache_value = False
                self._position_snapshot_cache = None
                self._pos_cache_ts = now
                return None

            pos = max(live_positions, key=lambda x: self._safe_float(x.get("size", 0.0)))
            raw_side = str(pos.get("side", "")).lower()
            side = "long" if raw_side == "buy" else "short"

            snap = PositionSnapshot(
                side=side,
                size=self._safe_float(pos.get("size", 0.0)),
                avg_price=self._safe_float(pos.get("avgPrice", 0.0)),
                mark_price=self._safe_float(pos.get("markPrice", 0.0)),
                unrealized_pnl=self._safe_float(pos.get("unrealisedPnl", 0.0)),
                position_value=self._safe_float(pos.get("positionValue", 0.0)),
                updated_time_ms=self._safe_int(pos.get("updatedTime")),
                raw=dict(pos),
            )

            self._pos_cache_value = True
            self._position_snapshot_cache = snap
            self._pos_cache_ts = now
            return snap

        except Exception as e:
            logger.warning("get_positions failed, keeping previous cached state: %s", str(e))
            return self._position_snapshot_cache

    def has_open_position(self, force: bool = False) -> bool:
        return self.get_position(force=force) is not None

    def wait_for_position_open(
        self,
        expected_side: Optional[str] = None,
        timeout_sec: float = 6.0,
        poll_sec: float = 0.5,
    ) -> Optional[PositionSnapshot]:
        deadline = time.time() + max(timeout_sec, poll_sec)
        while time.time() < deadline:
            snap = self.get_position(force=True)
            if snap is not None and (expected_side is None or snap.side == expected_side):
                return snap
            time.sleep(max(poll_sec, 0.1))
        return self.get_position(force=True)

    # --- orders / fills -----------------------------------------
    def place_market_order(
        self,
        side: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        order_link_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        bybit_side = "Buy" if side == "long" else "Sell"

        qty_n = self.normalize_qty(qty)
        sl_n = self.normalize_price(stop_loss)
        tp_n = self.normalize_price(take_profit)

        if qty_n <= 0:
            logger.warning("Normalized qty is 0 -> skip order")
            return None

        order_link_id = order_link_id or f"sniper-{uuid.uuid4().hex[:16]}"

        logger.info(
            "Sending MARKET order: %s %s qty=%.10f SL=%.10f TP=%.10f orderLinkId=%s",
            bybit_side,
            PAIR,
            qty_n,
            sl_n,
            tp_n,
            order_link_id,
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
                orderLinkId=order_link_id,
            )
            logger.info("ORDER RESP: %s", resp)
            self.invalidate_position_cache()
            return resp
        except Exception as e:
            logger.exception("Order failed: %s", e)
            return None

    def get_recent_closed_pnls(self, limit: int = 20) -> List[ClosedPnlRecord]:
        method = getattr(self.private_http, "get_closed_pnl", None)
        if method is None:
            logger.warning("pybit client has no get_closed_pnl() method")
            return []

        try:
            resp = method(
                category=CATEGORY,
                symbol=PAIR,
                limit=max(1, min(int(limit), 100)),
                recv_window=30000,
            )
            lst = resp.get("result", {}).get("list", []) or []
            out: List[ClosedPnlRecord] = []
            for item in lst:
                raw_side = str(item.get("side", "")).lower()
                side = "long" if raw_side == "buy" else ("short" if raw_side == "sell" else raw_side)
                out.append(
                    ClosedPnlRecord(
                        side=side,
                        qty=self._safe_float(item.get("qty", 0.0)),
                        avg_entry_price=self._safe_float(item.get("avgEntryPrice", 0.0)),
                        avg_exit_price=self._safe_float(item.get("avgExitPrice", 0.0)),
                        closed_pnl=self._safe_float(item.get("closedPnl", 0.0)),
                        created_time_ms=self._safe_int(item.get("createdTime")),
                        updated_time_ms=self._safe_int(item.get("updatedTime")),
                        raw=dict(item),
                    )
                )
            return out
        except Exception as e:
            logger.warning("get_closed_pnl failed: %s", str(e))
            return []

    def get_latest_closed_pnl(self, since_ms: Optional[int] = None) -> Optional[ClosedPnlRecord]:
        records = self.get_recent_closed_pnls(limit=20)
        if since_ms is not None:
            records = [
                r for r in records
                if (r.updated_time_ms or r.created_time_ms or 0) >= int(since_ms)
            ]
        if not records:
            return None
        return max(records, key=lambda r: (r.updated_time_ms or r.created_time_ms or 0))

    # --- market data --------------------------------------------
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

    # --- websocket ----------------------------------------------
    def subscribe_kline(self, interval: str, callback: Callable[[dict], None]):
        self.ws.kline_stream(
            callback=callback,
            symbol=PAIR,
            interval=interval,
        )

    # --- exchange filters ---------------------------------------
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
