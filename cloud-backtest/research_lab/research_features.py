from __future__ import annotations
import numpy as np
import pandas as pd
from .research_time import canonical_minute


def _rsi(close, n=14):
    d = close.diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff(); dn = -l.diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return adx, pdi, mdi


def feature_snapshot(candles: pd.DataFrame, feature_time) -> dict:
    t = canonical_minute(feature_time)
    c = candles.copy(); c.index = pd.to_datetime(c.index, utc=True)
    c = c.loc[c.index <= t].sort_index()
    if c.empty: raise ValueError("No candles at or before feature_time")
    for col in ["open", "high", "low", "close", "volume"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["open","high","low","close","volume"])
    close, high, low, vol = c["close"], c["high"], c["low"], c["volume"]
    row = c.iloc[-1]

    def ret(n):
        if len(c) <= n: return None
        x = close.iloc[-1] / close.iloc[-1-n] - 1
        return float(x) if np.isfinite(x) else None

    def vol_std(n):
        x = close.pct_change().rolling(n).std().iloc[-1]
        return float(x) if np.isfinite(x) else None

    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr = float(atr14.iloc[-1]) if np.isfinite(atr14.iloc[-1]) else None
    rng = max(float(row["high"] - row["low"]), 0.0)
    body = abs(float(row["close"] - row["open"]))
    upper = float(row["high"] - max(row["open"], row["close"]))
    lower = float(min(row["open"], row["close"]) - row["low"])
    avg20 = vol.rolling(20).mean().shift(1).iloc[-1] if len(c) > 20 else np.nan
    avg60 = vol.rolling(60).mean().shift(1).iloc[-1] if len(c) > 60 else np.nan
    ema9 = close.ewm(span=9, adjust=False, min_periods=9).mean().iloc[-1]
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
    adx, pdi, mdi = _adx(c)
    vw_num = ((high+low+close)/3 * vol).rolling(20).sum().iloc[-1]
    vw_den = vol.rolling(20).sum().iloc[-1]
    vwap = vw_num / vw_den if np.isfinite(vw_den) and vw_den else np.nan

    result = {
        "feature_time": t.isoformat(), "close": float(close.iloc[-1]),
        "return_1m": ret(1), "return_3m": ret(3), "return_5m": ret(5), "return_10m": ret(10),
        "return_15m": ret(15), "return_30m": ret(30), "return_60m": ret(60),
        "volatility_5m": vol_std(5), "volatility_15m": vol_std(15), "volatility_30m": vol_std(30), "volatility_60m": vol_std(60),
        "rvol20": None if not np.isfinite(avg20) or avg20 == 0 else float(vol.iloc[-1]/avg20),
        "rvol60": None if not np.isfinite(avg60) or avg60 == 0 else float(vol.iloc[-1]/avg60),
        "atr14_pct": None if atr is None or close.iloc[-1] == 0 else float(atr/close.iloc[-1]),
        "candle_range_pct": None if close.iloc[-1] == 0 else float(rng/close.iloc[-1]),
        "body_to_range": None if rng == 0 else float(body/rng),
        "upper_wick_to_range": None if rng == 0 else float(upper/rng),
        "lower_wick_to_range": None if rng == 0 else float(lower/rng),
        "close_location": None if rng == 0 else float((row["close"]-row["low"])/rng),
        "ema9_gap_pct": None if not np.isfinite(ema9) else float(close.iloc[-1]/ema9-1),
        "ema20_gap_pct": None if not np.isfinite(ema20) else float(close.iloc[-1]/ema20-1),
        "ema50_gap_pct": None if not np.isfinite(ema50) else float(close.iloc[-1]/ema50-1),
        "ema9_20_gap_pct": None if not (np.isfinite(ema9) and np.isfinite(ema20)) else float(ema9/ema20-1),
        "rsi14": None if not np.isfinite(_rsi(close).iloc[-1]) else float(_rsi(close).iloc[-1]),
        "adx14": None if not np.isfinite(adx.iloc[-1]) else float(adx.iloc[-1]),
        "plus_di14": None if not np.isfinite(pdi.iloc[-1]) else float(pdi.iloc[-1]),
        "minus_di14": None if not np.isfinite(mdi.iloc[-1]) else float(mdi.iloc[-1]),
        "adx_slope_5m": None if len(adx) < 6 or not np.isfinite(adx.iloc[-6]) else float(adx.iloc[-1]-adx.iloc[-6]),
        "vwap_gap_pct": None if not np.isfinite(vwap) else float(close.iloc[-1]/vwap-1),
        "distance_from_30m_high": None if len(c) < 30 else float(close.iloc[-1]/high.iloc[-30:].max()-1),
        "distance_from_30m_low": None if len(c) < 30 else float(close.iloc[-1]/low.iloc[-30:].min()-1),
        "distance_from_60m_high": None if len(c) < 60 else float(close.iloc[-1]/high.iloc[-60:].max()-1),
        "distance_from_60m_low": None if len(c) < 60 else float(close.iloc[-1]/low.iloc[-60:].min()-1),
        "efficiency_20": None,
    }
    if len(c) > 20:
        net = abs(float(close.iloc[-1]-close.iloc[-21])); gross = float(close.diff().abs().iloc[-20:].sum())
        result["efficiency_20"] = float(net/gross) if gross > 0 else 0.0
    return result
