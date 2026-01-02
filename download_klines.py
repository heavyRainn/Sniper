from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from pybit.unified_trading import HTTP


# =========================
# НАСТРОЙКИ (просто меняй тут)
# =========================
CATEGORY = "linear"         # "linear" | "spot" | "inverse"
SYMBOL = "1000PEPEUSDT"
INTERVAL = "15"             # "15" для 15-минуток

DAYS_BACK = 30              # месяц назад
END_DT = datetime.now(timezone.utc)
START_DT = END_DT - timedelta(days=DAYS_BACK)

OUT_PATH = f"data/{SYMBOL}_{INTERVAL}m_{DAYS_BACK}d.csv"

LIMIT = 1000                # максимум у Bybit на запрос
SLEEP_S = 0.2               # пауза между запросами
# =========================


def interval_to_ms(interval: str) -> int:
    if interval.isdigit():
        return int(interval) * 60_000
    if interval == "D":
        return 24 * 60 * 60_000
    if interval == "W":
        return 7 * 24 * 60 * 60_000
    raise ValueError(f"Unsupported interval for ms calc: {interval}")


def fetch_klines(
    session: HTTP,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    cur_end = end_ms
    step_ms = interval_to_ms(interval)

    last_earliest: Optional[int] = None

    while cur_end > start_ms:
        resp = session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            start=start_ms,
            end=cur_end,
            limit=limit,
        )

        lst = (resp.get("result", {}) or {}).get("list", []) or []
        if not lst:
            break

        # Bybit отдаёт в обратном порядке (сначала новые), приводим к хронологическому
        lst = list(reversed(lst))

        for k in lst:
            ts = int(k[0])
            all_rows.append(
                {
                    "ts": ts,                 # ms UTC
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "turnover": float(k[6]),
                }
            )

        earliest = int(lst[0][0])
        if last_earliest == earliest:
            break
        last_earliest = earliest

        # двигаем окно дальше в прошлое
        cur_end = earliest - step_ms
        time.sleep(sleep_s)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates("ts").sort_values("ts")
    df.set_index("ts", inplace=True)
    return df


def main():
    start_ms = int(START_DT.timestamp() * 1000)
    end_ms = int(END_DT.timestamp() * 1000)

    Path("data").mkdir(parents=True, exist_ok=True)

    print(f"Downloading {SYMBOL} {INTERVAL}m | {START_DT.isoformat()} -> {END_DT.isoformat()} (UTC)")
    session = HTTP(testnet=False)

    df = fetch_klines(
        session=session,
        category=CATEGORY,
        symbol=SYMBOL,
        interval=INTERVAL,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=min(max(1, LIMIT), 1000),
        sleep_s=max(0.0, SLEEP_S),
    )

    if df.empty:
        print("No data fetched. Проверь SYMBOL/CATEGORY/период.")
        return

    df.to_csv(OUT_PATH)
    print(f"Saved {len(df)} candles -> {OUT_PATH}")
    print(f"ts range: {df.index.min()} .. {df.index.max()} (ms)")
    # Можно показать человекочитаемые даты:
    print("First candle UTC:", datetime.fromtimestamp(df.index.min() / 1000, tz=timezone.utc))
    print("Last  candle UTC:", datetime.fromtimestamp(df.index.max() / 1000, tz=timezone.utc))


if __name__ == "__main__":
    main()
