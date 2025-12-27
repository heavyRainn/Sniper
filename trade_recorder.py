# trade_recorder.py
from __future__ import annotations

import csv
import pathlib
import threading
from datetime import datetime, timezone
from typing import Any, Dict


class TradeRecorder:
    def __init__(self, path: str = "logs/trades.csv"):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._fields = [
            "ts_utc",
            "symbol",
            "side",
            "entry",
            "sl",
            "tp",
            "qty",
            "reason",
            "retCode",
            "retMsg",
            "orderId",
            "orderLinkId",
        ]

        # если файла нет — создаём с заголовком
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._fields)
                w.writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("ts_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))

        # гарантируем наличие всех полей
        normalized = {k: row.get(k, "") for k in self._fields}

        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._fields)
                w.writerow(normalized)
