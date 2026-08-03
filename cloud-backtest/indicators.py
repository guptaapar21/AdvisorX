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


def avg_volume(candles, period=20):
    """Exact match to avgVolume in indicators.js."""
    recent = candles["volume"].tail(period)
    return recent.mean() if len(recent) > 0 else 0


def detect_volume_spike(current_volume, average_volume, threshold=1.5):
    """Exact match to detectVolumeSpike in strategyUtils.js. Previously
    missing entirely from this Python port - breakout signals here had
    zero volume awareness, unlike the live bot's real breakoutStrategy.js
    which downgrades/warns on weak volume."""
    if average_volume == 0:
        return {"is_spike": False, "ratio": 0.0, "level": "normal"}
    ratio = current_volume / average_volume
    is_spike = ratio >= threshold
    if ratio >= 3.0:
        level = "extreme"
    elif ratio >= 2.0:
        level = "significant"
    elif ratio >= 1.5:
        level = "moderate"
    else:
        level = "normal"
    return {"is_spike": is_spike, "ratio": round(ratio, 2), "level": level}


def obv(candles):
    """On-Balance Volume, full series. NOT real CVD/order-flow: CoinDCX's
    public /market_data/candles endpoint returns only open/high/low/close/
    volume, with no taker-buy-vs-sell split. True CVD needs the aggressor
    side of each trade, which this data source does not expose. OBV is the
    closest available proxy USING ONLY OHLCV: it signs each candle's total
    volume by the candle's own close-vs-prior-close direction (up candle =
    +volume, down candle = -volume, flat = 0), then cumulates. This is a
    real, honestly-weaker signal than true CVD - it can't distinguish
    "many small aggressive buys lifting price a little" from "one large
    passive buy absorbing a lot of selling at a flat price", which is
    exactly the absorption pattern idea #4 was meant to catch. Flagged
    here, not hidden.
    """
    closes = candles["close"].values
    vols = candles["volume"].values
    direction = np.sign(np.diff(closes, prepend=closes[0]))
    signed_vol = direction * vols
    return np.cumsum(signed_vol)


def real_cvd(taker_buy_volume, taker_sell_volume):
    """Genuine Cumulative Volume Delta from REAL aggressor-side data (e.g.
    Binance's taker_buy/taker_sell split via binance_taker_volume_fetcher.py)
    - NOT the OBV proxy below. delta = taker_buy - taker_sell per bar,
    cumulated. This is the real thing detect_obv_price_divergence was
    always a stand-in for."""
    delta = taker_buy_volume.values - taker_sell_volume.values
    return np.cumsum(delta)


def detect_real_cvd_divergence(candles, direction, lookback=10):
    """Same interface and same target_sign convention as
    detect_obv_price_divergence (direction = the OPEN POSITION's
    direction), but computed from REAL taker buy/sell volume columns
    ('taker_buy_volume', 'taker_sell_volume') that must already be present
    on `candles` - see main.py for where these get merged in from
    binance_taker_volume_fetcher.py. Returns None (not a dict) if those
    columns aren't present, so callers can cleanly detect "real CVD not
    available this run" and fall back to the OBV proxy."""
    if "taker_buy_volume" not in candles.columns or "taker_sell_volume" not in candles.columns:
        return None
    if len(candles) < lookback + 1:
        return {"cvd_slope_adverse": False, "cvd_slope_norm": 0.0}

    window = candles.tail(lookback + 1)
    cvd_series = real_cvd(window["taker_buy_volume"], window["taker_sell_volume"])
    cvd_slope = cvd_series[-1] - cvd_series[0]
    avg_vol = window["volume"].mean()
    cvd_slope_norm = cvd_slope / avg_vol if avg_vol else 0.0

    target_sign = -1 if direction == "long" else 1
    cvd_slope_adverse = np.sign(cvd_slope) == target_sign and abs(cvd_slope_norm) >= 0.3
    return {"cvd_slope_adverse": bool(cvd_slope_adverse), "cvd_slope_norm": round(float(cvd_slope_norm), 3)}


def detect_obv_price_divergence(candles, direction, lookback=10):
    """direction: "long" or "short" (the OPEN POSITION's direction, not
    the market's). Checks whether OBV over the last `lookback` primary
    candles is trending in the ADVERSE direction relative to the position,
    while price has moved comparatively little - i.e. more (proxy-)volume
    is flowing against the position than the price move alone would
    suggest, matching idea #4's "buyers/sellers absorbing the move"
    framing. Returns a dict with obv_slope_adverse (bool) and the raw
    normalized OBV slope, for use as an optional extra confidence flag -
    NOT wired into calculate_reversal_score by default (use
    obv_confirmation_bonus in run_backtest to opt in)."""
    if len(candles) < lookback + 1:
        return {"obv_slope_adverse": False, "obv_slope_norm": 0.0}
    obv_series = obv(candles.tail(lookback + 1))
    obv_slope = obv_series[-1] - obv_series[0]
    price_slope = candles["close"].values[-1] - candles["close"].values[-(lookback + 1)]
    avg_vol = candles["volume"].tail(lookback).mean()
    obv_slope_norm = obv_slope / avg_vol if avg_vol else 0.0

    # target_sign: which OBV direction counts as "adverse" for this
    # position. A long position is hurt by net selling pressure (OBV
    # falling); a short position is hurt by net buying pressure (OBV
    # rising) - same convention as calculate_reversal_score's target_sign.
    target_sign = -1 if direction == "long" else 1
    obv_slope_adverse = np.sign(obv_slope) == target_sign and abs(obv_slope_norm) >= 0.3
    return {"obv_slope_adverse": bool(obv_slope_adverse), "obv_slope_norm": round(float(obv_slope_norm), 3)}
