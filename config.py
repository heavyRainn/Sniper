# config.py
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- API-ключи (mainnet) ---
BYBIT_KEY = os.getenv("BYBIT_KEY")
BYBIT_SECRET = os.getenv("BYBIT_SECRET")

# --- Инструмент и биржа ---
PAIR = "1000PEPEUSDT"
CATEGORY = "linear"       # линейные USDT-перпеты
INTERVAL = "5"           # kline интервал для WS: '1','3','5','15','30','60','240','D',...

# --- Риск / капитал ---
EQUITY_PCT_PER_TRADE = 0.02     # 2% на сделку (≈0.4$ риска)
DAILY_DD_LIMIT = -0.04          # стоп-день: -4% от equity_start
LEV = 2                         # плечо (проверь, что на бирже выставлено такое же)

MAX_QTY_PER_TRADE = 1_000_000       # жёсткий лимит размера позиции

# --- Параметры индикаторов / стратегии ---
EMA_PERIOD = 200
RSI_PERIOD = 14
CCI_PERIOD = 20
ATR_PERIOD = 14
VOL_SMA_PERIOD = 20
MIN_BARS = 300                  # минимум баров для нормальной работы индикаторов

ATR_STOP_K = 1.0                # стоп ≈ 0.4 * ATR
RR_TP = 3.0                     # тейк = 2R

RSI_OVERSOLD = 25
RSI_OVERBOUGHT = 75
CCI_OVERSOLD = -150
CCI_OVERBOUGHT = 150