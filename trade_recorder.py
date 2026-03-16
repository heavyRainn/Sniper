from __future__ import annotations

import csv
import json
import pathlib
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


class TradeRecorder:
    def __init__(self, path: str = "logs/trades.csv"):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._fields = [
            "ts_utc",
            "event",
            "symbol",
            "side",
            "signal_entry",
            "model_entry",
            "entry",
            "fill_entry",
            "exit_price",
            "sl",
            "tp",
            "qty",
            "filled_qty",
            "reason",
            "status",
            "retCode",
            "retMsg",
            "orderId",
            "orderLinkId",
            "position_side",
            "position_size",
            "avg_price",
            "closed_pnl",
            "signal_bar_ts",
            "exec_bar_ts",
            "source",
            "raw",
        ]

        self._ensure_schema()

    def _ensure_schema(self) -> None:
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
            return

        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])

        if existing_header == self._fields:
            return

        rows: List[Dict[str, Any]] = []
        if existing_header:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields)
            writer.writeheader()
            for row in rows:
                normalized = {k: row.get(k, "") for k in self._fields}
                writer.writerow(normalized)

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value

    def append(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("ts_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        row.setdefault("event", "TRADE_EVENT")
        row.setdefault("status", "")

        normalized = {
            k: self._normalize_value(row.get(k, ""))
            for k in self._fields
        }

        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fields).writerow(normalized)

    def append_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fields)
                for row in rows:
                    row = dict(row)
                    row.setdefault("ts_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
                    row.setdefault("event", "TRADE_EVENT")
                    row.setdefault("status", "")
                    normalized = {
                        k: self._normalize_value(row.get(k, ""))
                        for k in self._fields
                    }
                    writer.writerow(normalized)
