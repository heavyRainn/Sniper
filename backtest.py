from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config import DAILY_DD_LIMIT, EQUITY_PCT_PER_TRADE, MAX_QTY_PER_TRADE, MIN_BARS
from risk import MAX_TRADES_PER_DAY
from strategy import Signal, TrendDivStrategy
from trade_recorder import TradeRecorder

# =========================
# НАСТРОЙКИ (меняй здесь)
# =========================
INPUT_CSV = "data/1000PEPEUSDT_5m_60d.csv"
OUTPUT_TRADES_CSV = "logs/backtest_trades.csv"

SYMBOL = "1000PEPEUSDT"

QTY_STEP = 100.0
MIN_QTY = 100.0
TICK_SIZE = 0.000001

START_EQUITY = 100.0

ENTRY_MODE = "next_open"   # "next_open" | "signal_close"
FEE_RATE = 0.0006
SLIPPAGE = 0.0002
INTRABAR_RULE = "SL_FIRST"  # "SL_FIRST" | "TP_FIRST"

# Консервативный BE: активируется только со следующего бара после close >= trigger.
BREAKEVEN_TRIGGER_R = 0.80
BREAKEVEN_LOCK_R = 0.05
# =========================


@dataclass
class DayState:
    day: datetime.date
    equity_start: float
    trades: int = 0


@dataclass
class Position:
    side: str
    entry: float
    sl: float
    tp: float
    qty: float
    entry_ts: int
    entry_fee: float
    signal_reason: str
    initial_risk: float
    be_armed: bool = False
    be_activate_on_index: Optional[int] = None


@dataclass
class PendingEntry:
    signal: Signal
    execute_on_index: int
    source_bar_ts: int


@dataclass
class ExecutionModel:
    entry_mode: str
    fee_rate: float
    slippage: float
    intrabar_rule: str


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    v = Decimal(str(value))
    s = Decimal(str(step))
    return float((v / s).to_integral_value(rounding=ROUND_DOWN) * s)


def normalize_qty(qty: float) -> float:
    q = _round_step(float(qty), QTY_STEP)
    if q < MIN_QTY:
        return 0.0
    return q


def normalize_price(price: float) -> float:
    return _round_step(float(price), TICK_SIZE)


def fee(notional: float, model: ExecutionModel) -> float:
    return float(notional) * float(model.fee_rate)


def slippage_price(side: str, price: float, is_entry: bool, model: ExecutionModel) -> float:
    p = float(price)
    s = float(model.slippage)
    if side == "long":
        return p * (1 + s) if is_entry else p * (1 - s)
    return p * (1 - s) if is_entry else p * (1 + s)


def candle_hits(pos: Position, bar: pd.Series) -> tuple[bool, bool]:
    if pos.side == "long":
        return float(bar["low"]) <= pos.sl, float(bar["high"]) >= pos.tp
    return float(bar["high"]) >= pos.sl, float(bar["low"]) <= pos.tp


def choose_exit_reason(hit_sl: bool, hit_tp: bool, model: ExecutionModel) -> Optional[str]:
    if not hit_sl and not hit_tp:
        return None
    if hit_sl and hit_tp:
        return "SL" if model.intrabar_rule == "SL_FIRST" else "TP"
    return "SL" if hit_sl else "TP"


def calc_qty_from_risk(equity: float, entry: float, sl: float) -> float:
    per_unit_risk = abs(float(entry) - float(sl))
    if per_unit_risk <= 0:
        return 0.0
    risk_usd = float(equity) * float(EQUITY_PCT_PER_TRADE)
    qty = risk_usd / per_unit_risk
    qty = min(qty, float(MAX_QTY_PER_TRADE))
    return normalize_qty(qty)


def ts_to_utc_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def mark_to_market_equity(equity: float, pos: Optional[Position], mark_price: float, model: ExecutionModel) -> float:
    if pos is None:
        return equity
    gross_unreal = (mark_price - pos.entry) * pos.qty if pos.side == "long" else (pos.entry - mark_price) * pos.qty
    exit_fee_est = fee(mark_price * pos.qty, model)
    return equity + gross_unreal - exit_fee_est


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
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


def backtest(df: pd.DataFrame) -> Dict[str, Any]:
    model = ExecutionModel(
        entry_mode=ENTRY_MODE,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,
        intrabar_rule=INTRABAR_RULE,
    )

    Path("logs").mkdir(parents=True, exist_ok=True)
    output_path = Path(OUTPUT_TRADES_CSV)
    if output_path.exists():
        output_path.unlink()

    rec = TradeRecorder(str(output_path))
    strat = TrendDivStrategy(fee_roundtrip=2 * FEE_RATE, slippage=SLIPPAGE)

    equity = float(START_EQUITY)
    peak = equity
    max_dd = 0.0

    pos: Optional[Position] = None
    pending_entry: Optional[PendingEntry] = None
    day_state: Optional[DayState] = None

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    trades_closed = 0

    def ensure_day_state(ts_ms: int, equity_snapshot: float) -> DayState:
        nonlocal day_state
        d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
        if day_state is None or day_state.day != d:
            day_state = DayState(day=d, equity_start=float(equity_snapshot), trades=0)
        return day_state

    def update_drawdown(equity_snapshot: float):
        nonlocal peak, max_dd
        if equity_snapshot > peak:
            peak = equity_snapshot
        dd_cur = (peak - equity_snapshot) / peak if peak > 0 else 0.0
        if dd_cur > max_dd:
            max_dd = dd_cur

    def can_open_trade(ts_ms: int, equity_snapshot: float) -> bool:
        state = ensure_day_state(ts_ms, equity_snapshot)
        if state.equity_start <= 0:
            return False
        dd = (equity_snapshot - state.equity_start) / state.equity_start
        if dd <= float(DAILY_DD_LIMIT):
            return False
        if state.trades >= MAX_TRADES_PER_DAY:
            return False
        return True

    def close_position(exit_bar: pd.Series, ts_ms: int, reason: str):
        nonlocal pos, equity, trades_closed, wins, losses, gross_profit, gross_loss
        assert pos is not None

        exit_price = pos.sl if reason == "SL" else pos.tp
        exit_price = normalize_price(exit_price)
        exit_price = slippage_price(pos.side, exit_price, is_entry=False, model=model)

        pnl_per_unit = (exit_price - pos.entry) if pos.side == "long" else (pos.entry - exit_price)
        gross_trade_pnl = pnl_per_unit * pos.qty
        exit_fee = fee(exit_price * pos.qty, model)
        trade_net_pnl = gross_trade_pnl - pos.entry_fee - exit_fee

        equity += gross_trade_pnl - exit_fee
        trades_closed += 1

        if trade_net_pnl >= 0:
            wins += 1
            gross_profit += trade_net_pnl
        else:
            losses += 1
            gross_loss += -trade_net_pnl

        rec.append(
            {
                "ts_utc": ts_to_utc_str(ts_ms),
                "symbol": SYMBOL,
                "side": pos.side,
                "entry": pos.entry,
                "sl": pos.sl,
                "tp": pos.tp,
                "qty": pos.qty,
                "reason": f"EXIT_{reason}",
                "retCode": 0,
                "retMsg": (
                    f"gross_pnl={gross_trade_pnl:.6f}; net_pnl={trade_net_pnl:.6f}; "
                    f"entry_fee={pos.entry_fee:.6f}; equity={equity:.2f}"
                ),
                "orderId": "",
                "orderLinkId": "",
            }
        )

        pos = None

    def maybe_activate_breakeven(i: int, ts_ms: int):
        nonlocal pos
        if pos is None or not pos.be_armed or pos.be_activate_on_index != i:
            return
        new_sl = normalize_price(pos.entry + BREAKEVEN_LOCK_R * pos.initial_risk)
        if new_sl > pos.sl:
            pos.sl = new_sl
            rec.append(
                {
                    "ts_utc": ts_to_utc_str(ts_ms),
                    "symbol": SYMBOL,
                    "side": pos.side,
                    "entry": pos.entry,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "qty": pos.qty,
                    "reason": "BE_ACTIVATE",
                    "retCode": 0,
                    "retMsg": f"be_trigger_r={BREAKEVEN_TRIGGER_R:.2f}; lock_r={BREAKEVEN_LOCK_R:.2f}",
                    "orderId": "",
                    "orderLinkId": "",
                }
            )
        pos.be_activate_on_index = None

    def maybe_arm_breakeven(i: int):
        nonlocal pos
        if pos is None or pos.be_armed:
            return
        if pos.side != "long":
            return
        trigger_price = pos.entry + BREAKEVEN_TRIGGER_R * pos.initial_risk
        # Консервативно: arm only from bar close. Activation starts next bar.
        bar = df.iloc[i]
        if float(bar["close"]) >= trigger_price:
            pos.be_armed = True
            pos.be_activate_on_index = i + 1

    def open_position(signal: Signal, entry_price_raw: float, ts_ms: int):
        nonlocal pos, equity, pending_entry

        entry = normalize_price(entry_price_raw)
        entry = slippage_price(signal.side, entry, is_entry=True, model=model)
        sl = normalize_price(float(signal.stop_loss))
        tp = normalize_price(float(signal.take_profit))
        qty = calc_qty_from_risk(equity, entry, sl)
        if qty <= 0:
            pending_entry = None
            return

        entry_fee = fee(entry * qty, model)
        equity -= entry_fee
        initial_risk = abs(entry - sl)

        pos = Position(
            side=signal.side,
            entry=entry,
            sl=sl,
            tp=tp,
            qty=qty,
            entry_ts=ts_ms,
            entry_fee=entry_fee,
            signal_reason=signal.reason,
            initial_risk=initial_risk,
        )

        state = ensure_day_state(ts_ms, equity)
        state.trades += 1

        rec.append(
            {
                "ts_utc": ts_to_utc_str(ts_ms),
                "symbol": SYMBOL,
                "side": signal.side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "reason": f"ENTRY_{signal.reason}",
                "retCode": 0,
                "retMsg": (
                    f"equity_after_entry_fee={equity:.2f}; signal_entry={float(signal.entry_price):.8f}; "
                    f"entry_fee={entry_fee:.6f}; initial_risk={initial_risk:.8f}"
                ),
                "orderId": "",
                "orderLinkId": "",
            }
        )

        pending_entry = None

    df = df.sort_index()
    if int(df.index.max()) < 10_000_000_000:
        df.index = df.index.astype("int64") * 1000

    for i in range(MIN_BARS, len(df)):
        bar = df.iloc[i]
        ts_ms = int(df.index[i])

        # 1) Activate BE scheduled from previous close.
        maybe_activate_breakeven(i, ts_ms)

        # 2) Service open position on current bar.
        if pos is not None:
            hit_sl, hit_tp = candle_hits(pos, bar)
            exit_reason = choose_exit_reason(hit_sl, hit_tp, model)
            if exit_reason is not None:
                close_position(bar, ts_ms, exit_reason)

        # 3) Execute pending signal on open of current bar.
        if pending_entry is not None and pending_entry.execute_on_index == i and pos is None:
            if can_open_trade(ts_ms, equity):
                open_position(pending_entry.signal, float(bar["open"]), ts_ms)
                if pos is not None:
                    hit_sl, hit_tp = candle_hits(pos, bar)
                    exit_reason = choose_exit_reason(hit_sl, hit_tp, model)
                    if exit_reason is not None:
                        close_position(bar, ts_ms, exit_reason)
            else:
                pending_entry = None

        # 4) Arm BE from current bar close for NEXT bar only.
        if pos is not None:
            maybe_arm_breakeven(i)

        # 5) Honest MTM equity for max drawdown.
        mtm_equity = mark_to_market_equity(equity, pos, float(bar["close"]), model)
        ensure_day_state(ts_ms, mtm_equity)
        update_drawdown(mtm_equity)

        # 6) Generate new signal on bar close.
        window = df.iloc[: i + 1]
        signal = strat.generate_signal(window)
        if signal is None:
            continue

        if pos is not None or pending_entry is not None:
            continue

        if model.entry_mode == "signal_close":
            if can_open_trade(ts_ms, equity):
                open_position(signal, float(bar["close"]), ts_ms)
            continue

        if model.entry_mode != "next_open":
            raise ValueError(f"Unsupported ENTRY_MODE: {model.entry_mode}")

        if i + 1 >= len(df):
            continue

        pending_entry = PendingEntry(signal=signal, execute_on_index=i + 1, source_bar_ts=ts_ms)

    if pos is not None:
        last_bar = df.iloc[-1]
        last_ts = int(df.index[-1])
        last_close = slippage_price(pos.side, normalize_price(float(last_bar["close"])), is_entry=False, model=model)
        gross_trade_pnl = (last_close - pos.entry) * pos.qty if pos.side == "long" else (pos.entry - last_close) * pos.qty
        exit_fee = fee(last_close * pos.qty, model)
        trade_net_pnl = gross_trade_pnl - pos.entry_fee - exit_fee
        equity += gross_trade_pnl - exit_fee
        trades_closed += 1

        if trade_net_pnl >= 0:
            wins += 1
            gross_profit += trade_net_pnl
        else:
            losses += 1
            gross_loss += -trade_net_pnl

        rec.append(
            {
                "ts_utc": ts_to_utc_str(last_ts),
                "symbol": SYMBOL,
                "side": pos.side,
                "entry": pos.entry,
                "sl": pos.sl,
                "tp": pos.tp,
                "qty": pos.qty,
                "reason": "EXIT_EOD",
                "retCode": 0,
                "retMsg": (
                    f"gross_pnl={gross_trade_pnl:.6f}; net_pnl={trade_net_pnl:.6f}; "
                    f"entry_fee={pos.entry_fee:.6f}; equity={equity:.2f}"
                ),
                "orderId": "",
                "orderLinkId": "",
            }
        )
        update_drawdown(equity)
        pos = None

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


def main():
    print(f"Loading: {INPUT_CSV}")
    df = load_csv(INPUT_CSV)
    print(f"Loaded candles: {len(df)}")

    if int(df.index.max()) < 10_000_000_000:
        first_ts = int(df.index.min())
        last_ts = int(df.index.max())
    else:
        first_ts = int(df.index.min() / 1000)
        last_ts = int(df.index.max() / 1000)

    print("First UTC:", datetime.fromtimestamp(first_ts, tz=timezone.utc))
    print("Last  UTC:", datetime.fromtimestamp(last_ts, tz=timezone.utc))

    stats = backtest(df)

    print("\n=== BACKTEST RESULT ===")
    print(f"Final equity:    {stats['final_equity']:.2f}")
    print(f"Trades closed:   {stats['trades_closed']}")
    print(f"Winrate:         {stats['winrate'] * 100:.2f}% ({stats['wins']}/{stats['trades_closed']})")
    print(f"Profit factor:   {stats['profit_factor']:.3f}")
    print(f"Max drawdown:    {stats['max_drawdown'] * 100:.2f}%")
    print(f"Trades CSV:      {OUTPUT_TRADES_CSV}")
    print("\n=== REJECT STATS ===")


if __name__ == "__main__":
    main()
