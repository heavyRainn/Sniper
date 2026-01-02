# backtest.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd

from strategy import TrendDivStrategy
from trade_recorder import TradeRecorder
from config import (
    MIN_BARS,
    EQUITY_PCT_PER_TRADE,
    MAX_QTY_PER_TRADE,
    DAILY_DD_LIMIT,
)

# =========================
# НАСТРОЙКИ (меняй здесь)
# =========================
INPUT_CSV = "data/1000PEPEUSDT_15m_30d.csv"
OUTPUT_TRADES_CSV = "logs/backtest_trades.csv"

SYMBOL = "1000PEPEUSDT"

# Bybit шаги (можешь руками поставить под символ)
QTY_STEP = 100.0      # у тебя для 1000PEPEUSDT было qty_step=100
MIN_QTY = 100.0       # min_qty=100
TICK_SIZE = 0.000001  # tickSize, если хочешь округлять SL/TP/entry

START_EQUITY = 100.0

# Модель исполнения
ENTRY_MODE = "next_open"   # "next_open" (честнее) или "close" (агрессивнее)
FEE_RATE = 0.0006          # taker fee (пример)
SLIPPAGE = 0.0002          # 0.02% проскальзывание для market входа/выхода (упрощённо)

# Если в одной свече могли задеть и SL и TP:
INTRABAR_RULE = "SL_FIRST"  # "SL_FIRST" (консервативно) или "TP_FIRST"
# =========================


@dataclass
class DayState:
    day: datetime.date
    equity_start: float
    trades: int = 0


@dataclass
class Position:
    side: str  # "long"|"short"
    entry: float
    sl: float
    tp: float
    qty: float
    entry_ts: int


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return (value // step) * step


def normalize_qty(qty: float) -> float:
    q = _round_step(float(qty), QTY_STEP)
    if q < MIN_QTY:
        return 0.0
    return q


def normalize_price(price: float) -> float:
    if TICK_SIZE <= 0:
        return float(price)
    return _round_step(float(price), TICK_SIZE)


def fee(notional: float) -> float:
    return float(notional) * float(FEE_RATE)


def slippage_price(side: str, price: float, is_entry: bool) -> float:
    """
    Упрощённо:
    - для long: вход хуже (выше), выход хуже (ниже)
    - для short: вход хуже (ниже), выход хуже (выше)
    """
    p = float(price)
    s = float(SLIPPAGE)
    if side == "long":
        return p * (1 + s) if is_entry else p * (1 - s)
    else:
        return p * (1 - s) if is_entry else p * (1 + s)


def candle_hits(pos: Position, bar: pd.Series) -> Tuple[bool, bool]:
    """
    Возвращает (hit_sl, hit_tp) по свече.
    """
    if pos.side == "long":
        hit_sl = float(bar["low"]) <= pos.sl
        hit_tp = float(bar["high"]) >= pos.tp
    else:
        hit_sl = float(bar["high"]) >= pos.sl
        hit_tp = float(bar["low"]) <= pos.tp
    return hit_sl, hit_tp


def choose_exit_reason(hit_sl: bool, hit_tp: bool) -> Optional[str]:
    if not hit_sl and not hit_tp:
        return None
    if hit_sl and hit_tp:
        return "SL" if INTRABAR_RULE == "SL_FIRST" else "TP"
    return "SL" if hit_sl else "TP"


def calc_qty_from_risk(equity: float, entry: float, sl: float) -> float:
    per_unit_risk = abs(float(entry) - float(sl))
    if per_unit_risk <= 0:
        return 0.0
    risk_usd = float(equity) * float(EQUITY_PCT_PER_TRADE)
    qty = risk_usd / per_unit_risk
    #qty = min(qty, float(MAX_QTY_PER_TRADE))
    return normalize_qty(qty)


def ts_to_utc_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def backtest(df: pd.DataFrame) -> Dict[str, Any]:
    Path("logs").mkdir(parents=True, exist_ok=True)
    rec = TradeRecorder(OUTPUT_TRADES_CSV)
    strat = TrendDivStrategy()

    equity = float(START_EQUITY)
    pos: Optional[Position] = None
    day_state: Optional[DayState] = None

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    trades_closed = 0

    peak = equity
    max_dd = 0.0

    def ensure_day_state(ts_ms: int):
        nonlocal day_state, equity
        d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
        if day_state is None or day_state.day != d:
            day_state = DayState(day=d, equity_start=equity, trades=0)

    def can_open_trade(ts_ms: int) -> bool:
        ensure_day_state(ts_ms)
        assert day_state is not None
        if day_state.equity_start <= 0:
            return False
        dd = (equity - day_state.equity_start) / day_state.equity_start
        if dd <= float(DAILY_DD_LIMIT):
            return False
        return True

    df = df.sort_index()
    if df.index.max() < 10_000_000_000:  # похоже на секунды
        df.index = (df.index.astype("int64") * 1000)

    for i in range(MIN_BARS, len(df) - 1):
        window = df.iloc[: i + 1]
        bar = df.iloc[i]
        ts_ms = int(df.index[i])

        ensure_day_state(ts_ms)

        # 1) если есть позиция — проверяем SL/TP на текущем баре
        if pos is not None:
            hit_sl, hit_tp = candle_hits(pos, bar)
            reason = choose_exit_reason(hit_sl, hit_tp)

            if reason is not None:
                exit_price = pos.sl if reason == "SL" else pos.tp
                exit_price = normalize_price(exit_price)
                exit_price = slippage_price(pos.side, exit_price, is_entry=False)

                pnl_per_unit = (exit_price - pos.entry) if pos.side == "long" else (pos.entry - exit_price)
                gross_pnl = pnl_per_unit * pos.qty

                fees = fee(pos.entry * pos.qty) + fee(exit_price * pos.qty)
                net_pnl = gross_pnl - fees
                equity += net_pnl

                trades_closed += 1
                if net_pnl >= 0:
                    wins += 1
                    gross_profit += net_pnl
                else:
                    losses += 1
                    gross_loss += -net_pnl

                if equity > peak:
                    peak = equity
                dd_cur = (peak - equity) / peak if peak > 0 else 0.0
                if dd_cur > max_dd:
                    max_dd = dd_cur

                rec.append({
                    "ts_utc": ts_to_utc_str(ts_ms),
                    "symbol": SYMBOL,
                    "side": pos.side,
                    "entry": pos.entry,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "qty": pos.qty,
                    "reason": f"EXIT_{reason}",
                    "retCode": 0,
                    "retMsg": f"net_pnl={net_pnl:.6f}; equity={equity:.2f}",
                    "orderId": "",
                    "orderLinkId": "",
                })

                pos = None

        # 2) если позиции нет — ищем сигнал и входим
        if pos is None:
            if not can_open_trade(ts_ms):
                continue

            signal = strat.generate_signal(window)
            if signal is None:
                continue

            if ENTRY_MODE == "close":
                entry_raw = float(bar["close"])
                entry_ts = ts_ms
            else:
                next_bar = df.iloc[i + 1]
                entry_raw = float(next_bar["open"])
                entry_ts = int(df.index[i + 1])

            entry = normalize_price(entry_raw)
            entry = slippage_price(signal.side, entry, is_entry=True)

            sl = normalize_price(float(signal.stop_loss))
            tp = normalize_price(float(signal.take_profit))

            qty = calc_qty_from_risk(equity, entry, sl)
            if qty <= 0:
                continue

            pos = Position(
                side=signal.side,
                entry=entry,
                sl=sl,
                tp=tp,
                qty=qty,
                entry_ts=entry_ts,
            )

            assert day_state is not None
            day_state.trades += 1

            rec.append({
                "ts_utc": ts_to_utc_str(entry_ts),
                "symbol": SYMBOL,
                "side": signal.side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "reason": f"ENTRY_{signal.reason}",
                "retCode": 0,
                "retMsg": f"equity={equity:.2f}",
                "orderId": "",
                "orderLinkId": "",
            })

    for k, v in strat.rej.most_common():
        print(f"{k:20s} {v}")

    return {
        "final_equity": equity,
        "trades_closed": trades_closed,
        "wins": wins,
        "losses": losses,
        "winrate": (wins / trades_closed) if trades_closed else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown": max_dd,
    }



def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ожидаем колонку ts (ms)
    if "ts" not in df.columns:
        raise RuntimeError("CSV должен содержать колонку 'ts' (timestamp в ms или sec).")

    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f"CSV должен содержать колонку '{c}'.")

    df["ts"] = df["ts"].astype("int64")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=cols)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df = df.set_index("ts")[cols]
    return df


def main():
    print(f"Loading: {INPUT_CSV}")
    df = load_csv(INPUT_CSV)
    print(f"Loaded candles: {len(df)}")
    print("First UTC:", datetime.fromtimestamp(df.index.min() / 1000, tz=timezone.utc))
    print("Last  UTC:", datetime.fromtimestamp(df.index.max() / 1000, tz=timezone.utc))

    stats = backtest(df)

    print("\n=== BACKTEST RESULT ===")
    print(f"Final equity:    {stats['final_equity']:.2f}")
    print(f"Trades closed:   {stats['trades_closed']}")
    print(f"Winrate:         {stats['winrate']*100:.2f}% ({stats['wins']}/{stats['trades_closed']})")
    print(f"Profit factor:   {stats['profit_factor']:.3f}")
    print(f"Max drawdown:    {stats['max_drawdown']*100:.2f}%")
    print(f"Trades CSV:      {OUTPUT_TRADES_CSV}")
    print("\n=== REJECT STATS ===")


if __name__ == "__main__":
    main()
