"""
Faithful port of indicators.js - formulas copied from the deployed JS
source, not re-derived. Every function here mirrors its JS counterpart
exactly.
"""
import numpy as np
import pandas as pd


def ema(closes, period):
    """Standard EMA, seeded with an SMA of the first `period` values."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period:
        return np.array([])
    out = np.zeros(len(closes) - period + 1)
    k = 2 / (period + 1)
    out[0] = closes[:period].mean()
    for i in range(1, len(out)):
        out[i] = closes[period - 1 + i] * k + out[i - 1] * (1 - k)
    return out


def rsi(closes, period=14):
    """Wilder-style RSI. Matches indicators.js: returns 100 if avgLoss==0
    (a real edge case on perfectly monotonic data, matches source exactly)."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(closes, fast=12, slow=26, signal=9):
    """Returns dict with macd, signal, histogram (last value only)."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < slow + signal:
        return None
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    # align lengths (ema_fast is longer since fast period < slow)
    offset = len(ema_fast) - len(ema_slow)
    macd_line = ema_fast[offset:] - ema_slow
    signal_line = ema(macd_line, signal)
    macd_val = macd_line[-1]
    signal_val = signal_line[-1]
    return {"macd": macd_val, "signal": signal_val, "histogram": macd_val - signal_val}


def macd_series(closes, fast=12, slow=26, signal=9):
    """Full histogram series (needed for macd_turn detection)."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < slow + signal:
        return np.array([])
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    offset = len(ema_fast) - len(ema_slow)
    macd_line = ema_fast[offset:] - ema_slow
    signal_line = ema(macd_line, signal)
    off2 = len(macd_line) - len(signal_line)
    return macd_line[off2:] - signal_line


def macd_histogram_turn(closes):
    """1 = turned up, -1 = turned down, 0 = none. Matches indicators.js."""
    hist = macd_series(closes)
    if len(hist) < 3:
        return 0
    prev_prev, prev, latest = hist[-3], hist[-2], hist[-1]
    if prev_prev > prev and prev < latest and latest > 0:
        return 1
    if prev_prev < prev and prev > latest and latest < 0:
        return -1
    return 0


def atr_wilder(candles, period=14):
    """Wilder-smoothed ATR, matches stopLossCalculator.ts's calculateATR."""
    if len(candles) < period + 1:
        return 0
    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values
    trs = []
    for i in range(1, len(candles)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    trs = np.array(trs)
    atr = trs[:period].mean()
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


def atr_simple(candles, period=14):
    """Simple-average ATR, matches multiTimeframeAnalysis.ts's calculateATR
    (used for atr_ratio, distinct from the Wilder version used for stops)."""
    if len(candles) < period + 1:
        return 0
    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values
    trs = []
    for i in range(1, len(candles)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    trs = np.array(trs)
    if len(trs) < period:
        return 0
    return trs[-period:].mean()


def atr_ratio(candles, period=14):
    """Current ATR(14) vs ATR(14) from 20 candles ago."""
    current = atr_simple(candles, period)
    if len(candles) >= period + 20 + 1:
        historical = atr_simple(candles.iloc[:-20], period)
    else:
        historical = current
    return current / historical if historical != 0 else 1.0


def bollinger_bands(closes, period=20, std_dev=2):
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period:
        return {"upper": 0, "middle": 0, "lower": 0}
    recent = closes[-period:]
    middle = recent.mean()
    std = recent.std()
    return {"upper": middle + std_dev * std, "middle": middle, "lower": middle - std_dev * std}
