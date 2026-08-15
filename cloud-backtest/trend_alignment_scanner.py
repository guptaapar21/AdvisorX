"""
Trend-structure + regime-aware Gemini scanner for CoinDCX futures.

The scanner deliberately does NOT pre-decide long/short. Python computes
deterministic market facts and sends every fresh coin to Gemini, including:
- multi-timeframe EMA/ADX/VWAP/RSI/volatility context;
- confirmed meaningful swing highs/lows and their HH/HL/LH/LL structure;
- exact structural break (BOS) / change-of-character (CHoCH) events;
- trend phase (continuation, expansion, exhaustion, transition);
- range boundaries, range position, touches and failed breaks;
- liquidity sweeps/rejections, distance to key levels and room-to-target;
- candle/volume/RVOL context and prior-call context.

Gemini chooses among TREND_UP, TREND_DOWN, RANGE, BREAKOUT_TRANSITION,
BREAKDOWN_TRANSITION, EXHAUSTION, or UNCLEAR and then decides TAKE/SKIP.
For RANGE it may consider swing trades from range boundaries; it should
avoid middle-of-range entries unless a specific structural edge exists.
Python remains the final geometry/risk gate.

The live loop is resilient to partial data: a stale coin no longer blocks
all other fresh coins from reaching Gemini. Each coin is tracked as processed
independently for the current 3m candle and stale coins are retried on the
next run.
"""

import argparse
import json
import os
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import requests

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from gemini_advisor import get_trade_suggestions_batch, TAKER_FEE_RATE, USDT_INR_RATE

VALID_POSITION_ACTIONS = {"hold", "exit_now", "tighten_stop", "move_target"}

STATE_FILE = "trend_scanner_state.json"
RVOL_PERCENTILE_FILE = "rvol_percentiles.json"
DAILY_CANDLES_FILE = "daily_candles_30d.json"
# Trimmed from 150 to 100 15m candles after measuring the real fetch
# cost at 18 coins - 150 candles needed 3 API requests/coin (2250 min
# of 1m data), 100 needs only 2 (1500 min), saving ~14.4s across all 18
# coins. Still 10x the bare minimum (10) actually checked in the code
# below, real margin for EMA21/ADX14 to be stable, not just non-error.
# Trimmed to 65 15m candles (975 min) - fits in exactly 1 API request
# per coin instead of 2, still 4.6x ADX14's min_periods(14) warmup
# requirement and 6.5x the hard minimum (10) checked in code below.
# NOTE: the 24h multi-timeframe Gemini context does NOT need this
# widened - it's fetched separately, only for reversal-stage coins,
# in main()'s enrichment block below, keeping this routine per-cycle
# fetch (which runs for all coins, every cycle) unaffected.
FETCH_MINUTES_BACK = 24 * 60
MESSAGE_INTERVAL_MINUTES = 1


STATE_BACKUP_FILE = STATE_FILE + ".bak"

def load_state():
    for path in (STATE_FILE, STATE_BACKUP_FILE):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                state=json.load(f)
            state.setdefault("coins", {})
            state.setdefault("call_history", {})
            state.setdefault("ledger", [])
            state.setdefault("last_processed_candle_time", None)
            state.setdefault("last_sent_candle_time", None)
            state.setdefault("pending_telegram", None)
            return state
        except (json.JSONDecodeError,OSError) as e:
            print(f"  State WARNING: unable to load {path}: {e}")
    return {"coins":{},"last_sent_at":None,"last_content_signature":[],"last_processed_candle_time":None,"last_sent_candle_time":None,"pending_telegram":None,"last_stale_notice_candle_time":None,"call_history":{},"ledger":[]}


def save_state(state):
    directory=os.path.dirname(os.path.abspath(STATE_FILE)) or "."
    fd,temp_path=tempfile.mkstemp(prefix=".trend_scanner_state_",suffix=".tmp",dir=directory,text=True)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(state,f,indent=2,default=str)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp_path,STATE_FILE)
        try:
            dir_fd=os.open(directory,os.O_DIRECTORY)
            try: os.fsync(dir_fd)
            finally: os.close(dir_fd)
        except (AttributeError,OSError):
            pass
    except Exception:
        try: os.unlink(temp_path)
        except OSError: pass
        raise

def utc_datetime(value):
    """Parse persisted timestamps and normalize them to timezone-aware UTC."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        # Legacy state was written as UTC-naive; interpret it as UTC.
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def now_utc():
    return datetime.now(timezone.utc)


def candle_key(value):
    """Canonical UTC key for candle timestamps.

    CoinDCX/resampling can return timezone-naive UTC indexes while Python
    datetime objects here are timezone-aware. Comparing str(datetime) values
    directly therefore makes the same candle look different (e.g.
    `2026-08-12 15:30:00` vs `2026-08-12 15:30:00+00:00`).
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("min").strftime("%Y-%m-%dT%H:%M:%SZ")


def send_telegram(text, reply_markup=None, parse_mode="HTML"):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


def drop_still_forming_bucket(candles, now, bar_minutes):
    if len(candles) == 0:
        return candles
    # CoinDCX resampled indexes are UTC-naive. Keep persisted/application
    # timestamps timezone-aware, but compare against this data index as
    # UTC-naive at the boundary.
    now_naive_utc = utc_datetime(now).replace(tzinfo=None)
    last_bucket_start = candles.index[-1]
    if last_bucket_start + pd.Timedelta(minutes=bar_minutes) > now_naive_utc:
        return candles.iloc[:-1]
    return candles


def compute_raw_stats(candles):
    """Deliberately NOT the old compute_indicators - no ADX, no EMA
    cross, no VWAP position. Those ARE the strategy-specific trend
    logic this scanner used to pre-filter with, which is exactly what
    the person asked to stop feeding Gemini - it should decide
    everything itself from raw, generic, strategy-agnostic numbers.
    Keeps ATR (a standard volatility measure, same formula as before -
    it's a generic stat, not a directional verdict) and RVOL (volume
    relative to recent average, also generic). Adds plain momentum:
    raw % price change over a few lookback windows in candle counts,
    not an indicator, just arithmetic."""
    candles = candles.copy()

    high, low, close = candles["high"], candles["low"], candles["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    candles["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    candles["avg_volume_20"] = candles["volume"].rolling(20).mean().shift(1)
    candles["rvol"] = candles["volume"] / candles["avg_volume_20"].replace(0, float("nan"))

    for n in (5, 20, 60):
        candles[f"momentum_pct_{n}"] = (candles["close"] / candles["close"].shift(n) - 1) * 100
        candles[f"momentum_accel_{n}"] = candles[f"momentum_pct_{n}"] - candles[f"momentum_pct_{n}"].shift(3)

    # Candle anatomy: body/wick/range geometry and close location.
    candle_range = (candles["high"] - candles["low"]).replace(0, np.nan)
    candles["candle_range_pct"] = candle_range / candles["open"].replace(0, np.nan) * 100
    candles["upper_wick_pct"] = (candles["high"] - candles[["open", "close"]].max(axis=1)) / candles["open"].replace(0, np.nan) * 100
    candles["lower_wick_pct"] = (candles[["open", "close"]].min(axis=1) - candles["low"]) / candles["open"].replace(0, np.nan) * 100
    candles["body_to_range"] = (candles["close"] - candles["open"]).abs() / candle_range
    candles["close_location_in_range"] = (candles["close"] - candles["low"]) / candle_range
    candles["volume_acceleration_3"] = candles["volume"] / candles["volume"].rolling(3).mean().shift(1).replace(0, np.nan)

    return candles



def _safe_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _ema(series, span):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _adx14(df):
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return adx, plus_di, minus_di


def _rolling_vwap(df, window=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.rolling(window).sum() / df["volume"].rolling(window).sum().replace(0, np.nan)



def _rsi14(close):
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); avg_loss=loss.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    return 100-(100/(1+rs))

def _efficiency_ratio(close, window=20):
    if len(close) <= window:
        return None
    net = abs(float(close.iloc[-1] - close.iloc[-1-window]))
    gross = float(close.diff().abs().iloc[-window:].sum())
    return round(net / gross, 3) if gross > 0 else 0.0


def _confirmed_pivots(df, left=2, right=2, max_points=12):
    """Return confirmed local pivots. A pivot is usable only after `right`
    closed candles exist, so the reported confirmation time never uses future
    candles that were not yet known at decision time."""
    if len(df) < left + right + 5:
        return {"highs": [], "lows": []}
    highs, lows = [], []
    h = df["high"].to_numpy(); l = df["low"].to_numpy()
    idx = list(df.index)
    for i in range(left, len(df)-right):
        if h[i] >= max(h[i-left:i]) and h[i] > max(h[i+1:i+right+1]):
            highs.append({"pivot_time": str(idx[i]), "confirm_time": str(idx[i+right]), "price": float(h[i])})
        if l[i] <= min(l[i-left:i]) and l[i] < min(l[i+1:i+right+1]):
            lows.append({"pivot_time": str(idx[i]), "confirm_time": str(idx[i+right]), "price": float(l[i])})
    return {"highs": highs[-max_points:], "lows": lows[-max_points:]}


def _meaningful_pivots(df, pivots, atr_col="atr14", min_atr=0.6, max_points=8):
    """Suppress tiny same-side pivots while preserving the more extreme level.
    This is descriptive structure extraction, not an entry filter."""
    out = {"highs": [], "lows": []}
    for side in ("highs", "lows"):
        seq = []
        for p in pivots.get(side, []):
            ts = pd.Timestamp(p["pivot_time"])
            pos = df.index.get_indexer([ts])[0]
            atr = _safe_float(df.iloc[pos].get(atr_col)) if pos >= 0 else None
            threshold = (atr or 0.0) * min_atr
            if not seq:
                seq.append(p); continue
            if threshold <= 0 or abs(p["price"] - seq[-1]["price"]) >= threshold:
                seq.append(p)
            else:
                # Same-side micro swing: retain the more extreme pivot.
                if side == "highs" and p["price"] > seq[-1]["price"]:
                    seq[-1] = p
                elif side == "lows" and p["price"] < seq[-1]["price"]:
                    seq[-1] = p
        out[side] = seq[-max_points:]
    return out


def _label_structure(pivots):
    highs = pivots.get("highs", [])
    lows = pivots.get("lows", [])
    labelled_highs=[]; labelled_lows=[]
    for i,p in enumerate(highs):
        q=dict(p); q["label"] = "HH" if i and p["price"] > highs[i-1]["price"] else ("LH" if i else "H")
        labelled_highs.append(q)
    for i,p in enumerate(lows):
        q=dict(p); q["label"] = "HL" if i and p["price"] > lows[i-1]["price"] else ("LL" if i else "L")
        labelled_lows.append(q)
    recent = sorted([(x["pivot_time"], x["label"], x["price"]) for x in labelled_highs+labelled_lows], key=lambda x:x[0])[-8:]
    high_labels=[x["label"] for x in labelled_highs[-3:]]
    low_labels=[x["label"] for x in labelled_lows[-3:]]
    if len(high_labels)>=2 and len(low_labels)>=2 and high_labels[-1]=="HH" and low_labels[-1]=="HL":
        bias="bullish"
    elif len(high_labels)>=2 and len(low_labels)>=2 and high_labels[-1]=="LH" and low_labels[-1]=="LL":
        bias="bearish"
    else:
        bias="neutral"
    return {"highs": labelled_highs, "lows": labelled_lows, "recent_sequence": recent, "bias": bias}


def _break_events(df, labelled, lookback=80):
    """Find the first closed-candle close beyond each confirmed swing.
    The event is only reported after the pivot confirmation time."""
    events=[]
    work=df.tail(lookback)
    for side, points in (("bullish", labelled.get("highs", [])), ("bearish", labelled.get("lows", []))):
        for p in points:
            confirm=pd.Timestamp(p["confirm_time"])
            after=work[work.index > confirm]
            if after.empty: continue
            hit = after[after["close"] > p["price"]] if side=="bullish" else after[after["close"] < p["price"]]
            if not hit.empty:
                r=hit.iloc[0]
                events.append({"direction":side,"type":"BOS","level":round(p["price"],8),"pivot_time":p["pivot_time"],"confirm_time":p["confirm_time"],"break_time":str(hit.index[0]),"break_close":round(float(r["close"]),8)})
    events.sort(key=lambda x:x["break_time"])
    return events[-8:]


def _break_followthrough(df, events, bars=5):
    """Measure what happened AFTER each confirmed structural break.
    This is descriptive only: it never creates a trade direction.
    """
    out=[]
    for ev in events[-5:]:
        bt=pd.Timestamp(ev["break_time"])
        after=df[df.index>bt].head(bars)
        if after.empty:
            ev2=dict(ev); ev2.update({"followthrough_bars":0,"followthrough_close_pct":None,"followthrough_volume_avg_ratio":None})
            out.append(ev2); continue
        entry=float(ev["break_close"])
        direction=ev["direction"]
        last_close=float(after["close"].iloc[-1])
        close_move=(last_close-entry)/entry*100
        if direction=="bearish": close_move=-close_move
        prior_vol=df.loc[df.index<=bt,"volume"].tail(20).mean()
        avg_follow_vol=float(after["volume"].mean()) if len(after) else None
        ev2=dict(ev); ev2.update({
            "followthrough_bars":int(len(after)),
            "followthrough_close_pct":round(close_move,3),
            "followthrough_volume_avg_ratio":round(avg_follow_vol/prior_vol,2) if prior_vol and avg_follow_vol else None,
        })
        out.append(ev2)
    return out


def _range_context(df, labelled):
    if len(df) < 30:
        return {"candidate":False}
    w=df.tail(40); close=float(w["close"].iloc[-1]); atr=_safe_float(w["atr14"].iloc[-1]) or float(w["high"].sub(w["low"]).mean())
    hi=float(w["high"].max()); lo=float(w["low"].min()); width=hi-lo; mid=(hi+lo)/2
    pos=((close-lo)/width*100) if width>0 else 50.0
    tol=max(atr*0.75, width*0.03)
    high_touches=int((w["high"] >= hi-tol).sum()); low_touches=int((w["low"] <= lo+tol).sum())
    er=_efficiency_ratio(w["close"],20)
    # Boundary levels from confirmed swings are preferable to arbitrary extrema.
    sh=[p["price"] for p in labelled.get("highs",[])][-4:]; sl=[p["price"] for p in labelled.get("lows",[])][-4:]
    if sh: hi=max(sh+[hi])
    if sl: lo=min(sl+[lo])
    width=hi-lo; pos=((close-lo)/width*100) if width>0 else 50.0
    return {
        "candidate": bool(width>0 and high_touches>=2 and low_touches>=2 and (er is None or er<0.45)),
        "range_high":round(hi,8),"range_low":round(lo,8),"range_mid":round((hi+lo)/2,8),
        "range_width":round(width,8),"range_width_pct":round(width/mid*100,3) if mid else None,
        "position_pct":round(max(0,min(100,pos)),1),"high_touches":high_touches,"low_touches":low_touches,
        "efficiency_ratio":er,
        "near_high":pos>=80,"near_low":pos<=20,"middle":20<pos<80,
    }


def _liquidity_sweeps(df, labelled, lookback=30):
    w=df.tail(lookback); out=[]
    highs=labelled.get("highs",[]); lows=labelled.get("lows",[])
    levels=[("high",p["price"]) for p in highs[-3:]]+[ ("low",p["price"]) for p in lows[-3:]]
    for kind,level in levels:
        if kind=="high":
            hit=w[(w["high"]>level) & (w["close"]<level)]
            if not hit.empty:
                r=hit.iloc[-1]; out.append({"type":"sweep_high_rejection","level":round(level,8),"time":str(hit.index[-1]),"close":round(float(r["close"]),8)})
        else:
            hit=w[(w["low"]<level) & (w["close"]>level)]
            if not hit.empty:
                r=hit.iloc[-1]; out.append({"type":"sweep_low_reclaim","level":round(level,8),"time":str(hit.index[-1]),"close":round(float(r["close"]),8)})
    return out[-5:]



def _failed_breaks(df, labelled, lookback=60):
    w=df.tail(lookback); out=[]
    for p in labelled.get("highs", [])[-4:]:
        level=p["price"]; after=w[w.index>pd.Timestamp(p["confirm_time"])]
        if after.empty: continue
        hit=after[after["high"]>level]
        if not hit.empty:
            first=hit.iloc[0]; t=hit.index[0]
            if float(first["close"])<level:
                out.append({"type":"failed_break_high","level":round(level,8),"time":str(t),"close":round(float(first["close"]),8)})
            else:
                later=after[after.index>t]
                back=later[later["close"]<level]
                if not back.empty:
                    out.append({"type":"breakout_failed_after_close","level":round(level,8),"time":str(back.index[0]),"close":round(float(back.iloc[0]["close"]),8)})
    for p in labelled.get("lows", [])[-4:]:
        level=p["price"]; after=w[w.index>pd.Timestamp(p["confirm_time"])]
        if after.empty: continue
        hit=after[after["low"]<level]
        if not hit.empty:
            first=hit.iloc[0]; t=hit.index[0]
            if float(first["close"])>level:
                out.append({"type":"failed_break_low","level":round(level,8),"time":str(t),"close":round(float(first["close"]),8)})
            else:
                later=after[after.index>t]
                back=later[later["close"]>level]
                if not back.empty:
                    out.append({"type":"breakdown_failed_after_close","level":round(level,8),"time":str(back.index[0]),"close":round(float(back.iloc[0]["close"]),8)})
    out.sort(key=lambda x:x["time"])
    return out[-6:]

def _tf_structure(df, label, pivot_left=2, pivot_right=2):
    d=df.copy()
    prev_close=d["close"].shift(1)
    tr=pd.concat([(d["high"]-d["low"]),(d["high"]-prev_close).abs(),(d["low"]-prev_close).abs()],axis=1).max(axis=1)
    d["atr14"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d["ema9"]=_ema(d["close"],9); d["ema21"]=_ema(d["close"],21)
    d["ema50"]=_ema(d["close"],50)
    d["adx14"],d["plus_di"],d["minus_di"]=_adx14(d)
    d["vwap20"]=_rolling_vwap(d,20); d["rsi14"]=_rsi14(d["close"]); d["vol_ratio20"]=d["volume"]/d["volume"].rolling(20).mean().shift(1).replace(0,np.nan)
    piv=_meaningful_pivots(d,_confirmed_pivots(d,pivot_left,pivot_right),max_points=8)
    st=_label_structure(piv); events=_break_events(d,st)
    for ev in events:
        if st["bias"]=="bullish" and ev["direction"]=="bearish": ev["type"]="CHoCH"
        elif st["bias"]=="bearish" and ev["direction"]=="bullish": ev["type"]="CHoCH"
    events = _break_followthrough(d, events, bars=5)
    last=d.iloc[-1]
    ema9=float(last["ema9"]) if not pd.isna(last["ema9"]) else None
    ema21=float(last["ema21"]) if not pd.isna(last["ema21"]) else None
    ema50=float(last["ema50"]) if not pd.isna(last["ema50"]) else None
    adx=float(last["adx14"]) if not pd.isna(last["adx14"]) else None
    slope9=((ema9-float(d["ema9"].iloc[-4]))/float(d["ema9"].iloc[-4])*100) if ema9 is not None and len(d)>=4 and d["ema9"].iloc[-4] else None
    close=float(last["close"]); vwap=float(last["vwap20"]) if not pd.isna(last["vwap20"]) else None
    adx_slope=((adx-float(d["adx14"].iloc[-4])) if adx is not None and len(d)>=4 and not pd.isna(d["adx14"].iloc[-4]) else None)
    rsi=float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None
    vol_ratio=float(last["vol_ratio20"]) if not pd.isna(last["vol_ratio20"]) else None
    prior20_high=float(d["high"].iloc[-21:-1].max()) if len(d)>=21 else None
    prior20_low=float(d["low"].iloc[-21:-1].min()) if len(d)>=21 else None
    exhaustion=[]
    if st["bias"]=="bullish" and prior20_high is not None and close>=prior20_high and adx_slope is not None and adx_slope<0: exhaustion.append("new_high_with_falling_adx")
    if st["bias"]=="bearish" and prior20_low is not None and close<=prior20_low and adx_slope is not None and adx_slope<0: exhaustion.append("new_low_with_falling_adx")
    if st["bias"]=="bullish" and rsi is not None and rsi>70 and slope9 is not None and slope9<0: exhaustion.append("bullish_structure_but_ema9_slope_fading")
    if st["bias"]=="bearish" and rsi is not None and rsi<30 and slope9 is not None and slope9>0: exhaustion.append("bearish_structure_but_ema9_slope_fading")
    phase="neutral"
    if st["bias"]=="bullish" and ema9 and ema21 and ema9>ema21 and (adx is None or adx>=20): phase="bull_continuation"
    elif st["bias"]=="bearish" and ema9 and ema21 and ema9<ema21 and (adx is None or adx>=20): phase="bear_continuation"
    if events:
        last_event=events[-1]
        if st["bias"]=="bullish" and last_event["direction"]=="bearish": phase="bearish_transition"
        elif st["bias"]=="bearish" and last_event["direction"]=="bullish": phase="bullish_transition"
    return {
        "timeframe":label,"close":round(close,8),"ema9":_safe_float(ema9),"ema21":_safe_float(ema21),"ema50":_safe_float(ema50),
        "ema9_slope_pct_3bars":round(slope9,3) if slope9 is not None else None,"adx14":round(adx,2) if adx is not None else None,
        "plus_di":round(float(last["plus_di"]),2) if not pd.isna(last["plus_di"]) else None,
        "minus_di":round(float(last["minus_di"]),2) if not pd.isna(last["minus_di"]) else None,
        "vwap20":_safe_float(vwap),"distance_vwap_pct":round((close-vwap)/vwap*100,3) if vwap else None,
        "structure_bias":st["bias"],"phase":phase,"recent_structure":st["recent_sequence"],
        "swing_highs":st["highs"][-5:],"swing_lows":st["lows"][-5:],"break_events":events[-5:],
        "latest_break":events[-1] if events else None,"failed_breaks":_failed_breaks(d,st),"liquidity_sweeps":_liquidity_sweeps(d,st),
        "efficiency_ratio_20":_efficiency_ratio(d["close"],20),
        "rsi14":round(rsi,2) if rsi is not None else None,
        "volume_ratio_vs_prior20":round(vol_ratio,2) if vol_ratio is not None else None,
        "adx_slope_3bars":round(adx_slope,2) if adx_slope is not None else None,
        "exhaustion_flags":exhaustion,
        "momentum_acceleration_5":_safe_float(d["momentum_accel_5"].iloc[-1]) if "momentum_accel_5" in d else None,
        "momentum_acceleration_20":_safe_float(d["momentum_accel_20"].iloc[-1]) if "momentum_accel_20" in d else None,
        "candle_body_pct":round(abs(float(last["close"])-float(last["open"])) / float(last["open"])*100,3) if float(last["open"]) else None,
        "candle_range_pct":round((float(last["high"])-float(last["low"])) / float(last["open"])*100,3) if float(last["open"]) else None,
        "upper_wick_pct":round((float(last["high"])-max(float(last["open"]),float(last["close"]))) / float(last["open"])*100,3) if float(last["open"]) else None,
        "lower_wick_pct":round((min(float(last["open"]),float(last["close"]))-float(last["low"])) / float(last["open"])*100,3) if float(last["open"]) else None,
        "body_to_range":_safe_float(d["body_to_range"].iloc[-1]) if "body_to_range" in d else None,
        "close_location_in_range":_safe_float(d["close_location_in_range"].iloc[-1]) if "close_location_in_range" in d else None,
        "volume_acceleration_3":_safe_float(d["volume_acceleration_3"].iloc[-1]) if "volume_acceleration_3" in d else None,
    }


def _market_structure_snapshot(c3,c15,c1):
    s3=_tf_structure(c3,"3m"); s15=_tf_structure(c15,"15m"); s1=_tf_structure(c1,"1h") if len(c1)>=22 else {"timeframe":"1h","insufficient_history":True}
    rng=_range_context(c3,_label_structure(_meaningful_pivots(c3,_confirmed_pivots(c3),max_points=8)))
    latest_close=float(c3["close"].iloc[-1]); atr=_safe_float(c3["atr14"].iloc[-1]) if "atr14" in c3 else None
    recent_high = s3.get("swing_highs", [])[-1] if s3.get("swing_highs") else None
    recent_low = s3.get("swing_lows", [])[-1] if s3.get("swing_lows") else None
    s3["distance_to_recent_swing_high_pct"] = round((recent_high["price"]-latest_close)/latest_close*100,3) if recent_high else None
    s3["distance_to_recent_swing_low_pct"] = round((latest_close-recent_low["price"])/latest_close*100,3) if recent_low else None
    s3["distance_to_recent_swing_high_atr"] = round((recent_high["price"]-latest_close)/atr,2) if recent_high and atr and atr>0 else None
    s3["distance_to_recent_swing_low_atr"] = round((latest_close-recent_low["price"])/atr,2) if recent_low and atr and atr>0 else None
    # Build a compact phase statement without turning it into a trading signal.
    phases=[s3.get("phase"),s15.get("phase"),s1.get("phase")]
    bull=sum(p in ("bull_continuation","bullish_transition") for p in phases)
    bear=sum(p in ("bear_continuation","bearish_transition") for p in phases)
    if any(p=="bullish_transition" for p in phases): regime="BREAKOUT_TRANSITION"
    elif any(p=="bearish_transition" for p in phases): regime="BREAKDOWN_TRANSITION"
    elif bull>=2 and bear==0: regime="TREND_UP"
    elif bear>=2 and bull==0: regime="TREND_DOWN"
    elif rng.get("candidate") and bull<2 and bear<2: regime="RANGE"
    elif any(p in ("bull_continuation","bear_continuation") for p in phases): regime="EXHAUSTION_OR_TRANSITION"
    else: regime="UNCLEAR"
    return {"market_regime":regime,"current_price":round(latest_close,8),"atr14_3m":round(atr,8) if atr else None,
            "3m":s3,"15m":s15,"1h":s1,"range":rng}

def rvol_label(rvol):
    if rvol is None or pd.isna(rvol):
        return "unknown"
    if rvol < 1.0:
        return "weak"
    if rvol < 1.5:
        return "normal"
    if rvol < 2.0:
        return "strong"
    return "very strong (possible exhaustion)"


def load_rvol_percentiles():
    """Reads the compact per-coin percentile file the separate
    rvol_percentile_refresh.py script produces. Read-only here - the
    live scanner never writes this file, only the daily refresh job
    does, keeping every 1-minute run's git commit small regardless of
    how much history backs the percentiles."""
    if os.path.exists(RVOL_PERCENTILE_FILE):
        try:
            with open(RVOL_PERCENTILE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


LEDGER_MAX_AGE_HOURS = 26  # entries older than this are pruned regardless of status - covers the 24h scorecard window with 2h margin
EXPIRY_HOURS = 2  # a pending trade that hasn't hit target or stop within this window is closed at current market price

# Position-management telemetry / protection. These are deliberately conservative:
# Gemini remains the discretionary manager, while Python prevents a large amount
# of already-earned profit from silently evaporating.
PROFIT_PROTECT_MIN_MFE_R = 1.0
PROFIT_PROTECT_EARLY_GIVEBACK_PCT = 40.0
PROFIT_PROTECT_SEVERE_GIVEBACK_PCT = 65.0
PROFIT_PROTECT_LOCK_R = 0.25
PROFIT_PROTECT_HEALTH_SCORE = 2
PROFIT_PROTECT_SEVERE_HEALTH_SCORE = 3

def _position_gross_pnl(entry, price):
    ep = float(entry.get("entry_price") or 0)
    amt = float(entry.get("trade_amount_inr") or 0)
    if ep <= 0 or amt <= 0 or USDT_INR_RATE <= 0 or price is None:
        return None
    qty = amt / (ep * USDT_INR_RATE)
    if entry.get("direction") == "long":
        return qty * (float(price) - ep) * USDT_INR_RATE
    return qty * (ep - float(price)) * USDT_INR_RATE

def update_position_telemetry(ledger, fetched, now):
    """Update MFE/MAE from closed 1m candles before resolving stops/targets.

    This records what actually happened inside the 3m decision window, so a
    trade that reached +₹700 and later returned to +₹200 is presented to Gemini
    as a giveback rather than as a fresh +₹200 trade. Existing state is migrated
    lazily without requiring a manual reset.
    """
    for entry in ledger:
        if entry.get("status") != "pending":
            continue
        coin = entry.get("coin")
        data = fetched.get(coin)
        if not data:
            continue
        c1 = data[3] if len(data) >= 4 else None
        if c1 is None or c1.empty:
            continue
        ep = entry.get("entry_price")
        if not ep:
            continue
        entry.setdefault("peak_price", float(ep))
        entry.setdefault("trough_price", float(ep))
        entry.setdefault("mfe_pnl_inr", 0.0)
        entry.setdefault("mae_pnl_inr", 0.0)
        entry.setdefault("mfe_time", entry.get("time"))
        entry.setdefault("mae_time", entry.get("time"))
        call = utc_datetime(entry["time"])
        since = c1[c1.index > call.replace(tzinfo=None)]
        if since.empty:
            continue
        for idx, row in since.iterrows():
            high = float(row["high"]); low = float(row["low"])
            if entry.get("direction") == "long":
                if high > float(entry["peak_price"]):
                    entry["peak_price"] = high; entry["mfe_time"] = str(idx)
                if low < float(entry["trough_price"]):
                    entry["trough_price"] = low; entry["mae_time"] = str(idx)
                fav = _position_gross_pnl(entry, high)
                adverse = _position_gross_pnl(entry, low)
            else:
                if low < float(entry["trough_price"]):
                    entry["trough_price"] = low; entry["mfe_time"] = str(idx)
                if high > float(entry["peak_price"]):
                    entry["peak_price"] = high; entry["mae_time"] = str(idx)
                fav = _position_gross_pnl(entry, low)
                adverse = _position_gross_pnl(entry, high)
            if fav is not None and fav > float(entry.get("mfe_pnl_inr", 0.0)):
                entry["mfe_pnl_inr"] = round(fav, 2)
            if adverse is not None and adverse < float(entry.get("mae_pnl_inr", 0.0)):
                entry["mae_pnl_inr"] = round(adverse, 2)
        risk = float(entry.get("max_loss_this_trade_inr") or 0)
        mfe = float(entry.get("mfe_pnl_inr") or 0)
        entry["mfe_r"] = round(mfe / risk, 3) if risk > 0 else None

def _profit_health_from_fetched(entry, fetched):
    """Deterministic health score for an already-profitable runner.

    This is a safety overlay, not an entry strategy. It uses the scanner's
    existing 3m/15m structural machinery to detect deterioration.
    0-1 healthy, 2 weakening, 3+ severe.
    """
    data = fetched.get(entry.get("coin")) if fetched else None
    if not data:
        return {"score": 0, "reasons": [], "state": "unknown"}
    c3, c15 = data[0], data[1]
    direction = entry.get("direction")
    score = 0; reasons = []
    try:
        s3 = _tf_structure(c3, "3m") if len(c3) >= 55 else {}
        s15 = _tf_structure(c15, "15m") if len(c15) >= 55 else {}
    except Exception:
        s3 = {}; s15 = {}
    bullish = direction == "long"
    def add(cond, reason, points=1):
        nonlocal score
        if cond:
            score += points; reasons.append(reason)
    if s3:
        add(s3.get("structure_bias") == ("bearish" if bullish else "bullish"), "3m structure flipped", 2)
        add(s3.get("phase") == ("bearish_transition" if bullish else "bullish_transition"), "3m transition against position", 2)
        ema9=s3.get("ema9"); close=s3.get("close")
        add(ema9 is not None and close is not None and ((close < ema9) if bullish else (close > ema9)), "3m lost EMA9")
        adx_slope=s3.get("adx_slope_3bars")
        add(adx_slope is not None and adx_slope < 0, "ADX slope weakening")
        mom5=s3.get("momentum_acceleration_5")
        add(mom5 is not None and ((mom5 < 0) if bullish else (mom5 > 0)), "momentum acceleration reversed")
        upper=s3.get("upper_wick_pct") or 0; lower=s3.get("lower_wick_pct") or 0
        add((upper > lower*1.5) if bullish else (lower > upper*1.5), "rejection wick against position")
        add(bool(s3.get("failed_breaks")), "recent failed break")
    if s15:
        add(s15.get("structure_bias") == ("bearish" if bullish else "bullish"), "15m structure against position", 2)
        add(s15.get("phase") == ("bearish_transition" if bullish else "bullish_transition"), "15m transition against position", 2)
    state = "severe" if score >= PROFIT_PROTECT_SEVERE_HEALTH_SCORE else ("weakening" if score >= PROFIT_PROTECT_HEALTH_SCORE else "healthy")
    return {"score": score, "reasons": reasons[:8], "state": state}


def apply_python_profit_protection(ledger, current_prices, now, fetched=None):
    """Deterministic profit-protection floor after a substantial favorable move.

    The first protection layer activates when the trade has reached >=1R MFE,
    given back >=40% of that MFE, is still profitable, and the deterministic
    profit-health score is >=2. It tightens protection to lock approximately
    +0.25R where market geometry permits it.

    A stronger deterministic exit is used at >=65% MFE giveback when the
    profit-health score is >=3 and remaining profit is <=0.5R. That exit is
    recorded as ``python_profit_exit``. Gemini can still exit earlier or
    manage the trade normally.
    """
    summaries = []
    for entry in ledger:
        if entry.get("status") != "pending":
            continue
        coin = entry.get("coin"); cp = current_prices.get(coin)
        if cp is None:
            continue
        mfe = float(entry.get("mfe_pnl_inr") or 0)
        risk = float(entry.get("max_loss_this_trade_inr") or 0)
        current_gross = _position_gross_pnl(entry, cp)
        if risk <= 0 or mfe < risk * PROFIT_PROTECT_MIN_MFE_R or current_gross is None or current_gross <= 0:
            continue
        giveback = (mfe - current_gross) / mfe * 100 if mfe > 0 else 0
        entry["mfe_giveback_pct"] = round(max(0.0, giveback), 1)
        health = _profit_health_from_fetched(entry, fetched)
        entry["profit_health_score"] = health["score"]
        entry["profit_health_state"] = health["state"]
        entry["profit_health_reasons"] = health["reasons"]
        if giveback < PROFIT_PROTECT_EARLY_GIVEBACK_PCT or health["score"] < PROFIT_PROTECT_HEALTH_SCORE:
            continue
        ep = float(entry["entry_price"])
        current_r = current_gross / risk if risk > 0 else 0
        if (giveback >= PROFIT_PROTECT_SEVERE_GIVEBACK_PCT and health["score"] >= PROFIT_PROTECT_SEVERE_HEALTH_SCORE and current_r <= 0.50):
            fees = float(entry.get("trade_amount_inr") or 0) * TAKER_FEE_RATE * 2
            entry["status"] = "python_profit_exit"
            entry["resolved_pnl"] = round(current_gross - fees, 2)
            entry["resolved_time"] = now.isoformat()
            entry["exit_reasoning"] = f"Deterministic profit protection: {giveback:.0f}% MFE giveback, health score {health['score']} ({', '.join(health['reasons'][:4])})."
            summaries.append({"coin":entry.get("coin"),"direction":entry.get("direction"),"action":"profit_exit","pnl":entry["resolved_pnl"],"mfe_pnl_inr":mfe,"giveback_pct":giveback,"health_score":health["score"],"health_reasons":health["reasons"]})
            continue
        lock_move = risk * PROFIT_PROTECT_LOCK_R
        price_move = lock_move / (float(entry["trade_amount_inr"]) / ep) if entry.get("trade_amount_inr") else 0
        if entry.get("direction") == "long":
            desired = ep + price_move
            old = float(entry.get("stop_loss") or 0)
            if old < desired < float(cp):
                entry["stop_loss"] = desired
                entry["revised_at"] = now.isoformat()
                entry.setdefault("level_history", []).append({"from": now.isoformat(), "stop": desired, "target": entry.get("target_price"), "source": "python_profit_protection"})
                summaries.append({"coin":coin,"direction":"long","action":"profit_protection","new_stop_loss":desired,"mfe_pnl_inr":mfe,"current_pnl_inr":current_gross,"giveback_pct":giveback,"health_score":health["score"],"health_reasons":health["reasons"]})
        else:
            desired = ep - price_move
            old = float(entry.get("stop_loss") or 0)
            if float(cp) < desired < old:
                entry["stop_loss"] = desired
                entry["revised_at"] = now.isoformat()
                entry.setdefault("level_history", []).append({"from": now.isoformat(), "stop": desired, "target": entry.get("target_price"), "source": "python_profit_protection"})
                summaries.append({"coin":coin,"direction":"short","action":"profit_protection","new_stop_loss":desired,"mfe_pnl_inr":mfe,"current_pnl_inr":current_gross,"giveback_pct":giveback,"health_score":health["score"],"health_reasons":health["reasons"]})
    return summaries

def resolve_ledger(ledger, fetched, now):
    now_utc_value=utc_datetime(now)
    for entry in ledger:
        if entry.get("status")!="pending": continue
        ep=entry.get("entry_price"); amt=entry.get("trade_amount_inr")
        if not ep or ep<=0 or not amt or amt<=0 or USDT_INR_RATE<=0:
            entry["status"]="invalid"; entry["resolved_pnl"]=0.0; entry["resolved_time"]=now.isoformat(); continue
        coin=entry.get("coin")
        if coin not in fetched: continue
        c3,_,_,_=fetched[coin]; call=utc_datetime(entry["time"]); since=c3[c3.index>call.replace(tzinfo=None)]
        if since.empty: continue
        qty=amt/(ep*USDT_INR_RATE); fees=amt*TAKER_FEE_RATE*2; direction=entry["direction"]
        history=entry.get("level_history") or [{"from":entry["time"],"stop":entry.get("stop_loss"),"target":entry.get("target_price")}]
        history=sorted(history,key=lambda x:utc_datetime(x["from"]))
        for _,row in since.iterrows():
            candle_time=pd.Timestamp(row.name).to_pydatetime().replace(tzinfo=timezone.utc)
            active=history[0]
            for level in history:
                if utc_datetime(level["from"])<=candle_time: active=level
                else: break
            target=active.get("target"); stop=active.get("stop")
            if target is None or stop is None: continue
            if direction=="long": target_hit=row["high"]>=target; stop_hit=row["low"]<=stop
            else: target_hit=row["low"]<=target; stop_hit=row["high"]>=stop
            if target_hit and stop_hit:
                entry["status"]="stop_hit"; entry["resolved_pnl"]=round(-qty*abs(stop-ep)*USDT_INR_RATE-fees,2); entry["resolved_time"]=str(row.name); break
            if target_hit:
                entry["status"]="target_hit"; entry["resolved_pnl"]=round(qty*abs(target-ep)*USDT_INR_RATE-fees,2); entry["resolved_time"]=str(row.name); break
            if stop_hit:
                entry["status"]="stop_hit"; entry["resolved_pnl"]=round(-qty*abs(stop-ep)*USDT_INR_RATE-fees,2); entry["resolved_time"]=str(row.name); break
        else:
            age=(now_utc_value-call).total_seconds()/3600
            if age>=EXPIRY_HOURS:
                last=float(since["close"].iloc[-1]); pnl=qty*((last-ep) if direction=="long" else (ep-last))*USDT_INR_RATE-fees
                entry["status"]="expired"; entry["resolved_pnl"]=round(pnl,2); entry["resolved_time"]=str(since.index[-1])
    cutoff=now_utc_value-timedelta(hours=LEDGER_MAX_AGE_HOURS)
    return [e for e in ledger if e.get("status")=="pending" or utc_datetime(e["time"])>cutoff]

def compute_scorecard(ledger, now, window_hours=1, current_prices=None):
    """Real trades-in / target-hit / stop-hit / pending / P&L over the
    trailing window - computed from the ledger's actual resolved
    outcomes and current market prices, not asked of Gemini (which has
    no memory and no way to verify either of these itself).

    realized_pnl_inr: sum of P&L from trades that actually hit target
    or stop - money already made or lost, not an estimate.

    unrealized_pnl_inr: mark-to-market on still-PENDING positions -
    what they would be worth if closed right now at current price.
    This is a live estimate, not a locked-in outcome - a pending
    position showing positive unrealized P&L can still go on to hit
    its stop. Needs current_prices (a {coin: latest_close} dict, built
    from data already fetched this cycle - no extra API calls);
    without it, unrealized P&L simply is not computed rather than
    guessed."""
    now_utc_value = utc_datetime(now)
    cutoff = now_utc_value - timedelta(hours=window_hours)
    window = [e for e in ledger if utc_datetime(e["time"]) > cutoff]
    target_hit = [e for e in window if e["status"] == "target_hit"]
    stop_hit = [e for e in window if e["status"] == "stop_hit"]
    expired = [e for e in window if e["status"] == "expired"]
    # gemini_exit: closed early on Gemini's own hold/exit review of an
    # open position (thesis judged invalidated), not because target,
    # stop, or the 2h expiry clock was actually touched - a third,
    # distinct kind of resolution alongside target/stop/expiry.
    gemini_exit = [e for e in window if e["status"] == "gemini_exit"]
    profit_exit = [e for e in window if e["status"] == "python_profit_exit"]
    pending = [e for e in window if e["status"] == "pending"]
    abandoned_or_invalid = [e for e in window if e["status"] in ("abandoned", "invalid")]
    # Expired and gemini_exit trades' P&L counts toward realized (per
    # explicit instruction) - closed at mark-to-market (at the 2h
    # clock or at Gemini's exit call, respectively), so it's a real,
    # locked-in number, just not from genuinely touching the actual
    # target/stop price.
    realized_pnl = round(sum(e.get("resolved_pnl", 0) for e in target_hit + stop_hit + expired + gemini_exit + profit_exit), 2)

    unrealized_pnl = 0.0
    if current_prices:
        for entry in pending:
            current_price = current_prices.get(entry["coin"])
            entry_price = entry.get("entry_price")
            trade_amount = entry.get("trade_amount_inr")
            if not current_price or not entry_price or entry_price <= 0 or not trade_amount or USDT_INR_RATE <= 0:
                continue  # same validation standard as resolve_ledger - skip rather than guess on bad data
            quantity = trade_amount / (entry_price * USDT_INR_RATE) if USDT_INR_RATE > 0 else None
            if quantity is None:
                continue
            if entry["direction"] == "long":
                unrealized_pnl += quantity * (current_price - entry_price) * USDT_INR_RATE
            else:
                unrealized_pnl += quantity * (entry_price - current_price) * USDT_INR_RATE
    unrealized_pnl = round(unrealized_pnl, 2)

    return {
        "total": len(window), "target_hit": len(target_hit), "stop_hit": len(stop_hit),
        "expired": len(expired), "gemini_exit": len(gemini_exit), "profit_exit": len(profit_exit), "pending": len(pending),
        "abandoned_or_invalid": len(abandoned_or_invalid),
        "realized_pnl_inr": realized_pnl,
        "unrealized_pnl_inr": unrealized_pnl, "total_pnl_inr": round(realized_pnl + unrealized_pnl, 2),
    }


def build_open_position_context(ledger, current_prices, now, profit_health_by_coin=None):
    """Builds the 'open_positions' payload sent back to Gemini: every
    still-pending ledger entry whose coin has fresh data this cycle,
    with its original entry/stop/target/reasoning (so Gemini can judge
    whether ITS OWN thesis still holds) plus current price and live
    unrealized P&L. Skips entries missing a usable entry_price/
    trade_amount_inr, same validation standard as resolve_ledger -
    Gemini should never be asked to review a position built on
    already-invalid data."""
    now_utc_value = utc_datetime(now)
    positions = []
    for entry in ledger:
        if entry["status"] != "pending":
            continue
        coin = entry["coin"]
        current_price = current_prices.get(coin)
        entry_price = entry.get("entry_price")
        trade_amount = entry.get("trade_amount_inr")
        if not current_price or not entry_price or entry_price <= 0 or not trade_amount or USDT_INR_RATE <= 0:
            continue
        quantity = trade_amount / (entry_price * USDT_INR_RATE) if USDT_INR_RATE > 0 else None
        direction = entry["direction"]
        if quantity is None:
            continue
        # Update current giveback telemetry before sending the position to Gemini.
        mfe_now = float(entry.get("mfe_pnl_inr") or 0)
        current_gross = _position_gross_pnl(entry, current_price)
        if mfe_now > 0 and current_gross is not None:
            entry["mfe_giveback_pct"] = round(max(0.0, (mfe_now-current_gross)/mfe_now*100), 1)
        risk_now = float(entry.get("max_loss_this_trade_inr") or 0)
        if risk_now > 0:
            entry["mfe_r"] = round(mfe_now/risk_now, 3)
        if direction == "long":
            unrealized_pnl = quantity * (current_price - entry_price) * USDT_INR_RATE
        else:
            unrealized_pnl = quantity * (entry_price - current_price) * USDT_INR_RATE
        call_time = utc_datetime(entry["time"])
        minutes_open = round((now_utc_value - call_time).total_seconds() / 60, 1)
        positions.append({
            "coin": coin,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": entry.get("stop_loss"),
            "target_price": entry.get("target_price"),
            "original_reasoning": entry.get("reasoning"),
            "minutes_open": minutes_open,
            "current_price": current_price,
            "unrealized_pnl_inr": round(unrealized_pnl, 2),
            "mfe_pnl_inr": round(float(entry.get("mfe_pnl_inr") or 0), 2),
            "mae_pnl_inr": round(float(entry.get("mae_pnl_inr") or 0), 2),
            "mfe_r": entry.get("mfe_r"),
            "peak_price": entry.get("peak_price"),
            "trough_price": entry.get("trough_price"),
            "mfe_time": entry.get("mfe_time"),
            "mae_time": entry.get("mae_time"),
            "mfe_giveback_pct": entry.get("mfe_giveback_pct"),
            "profit_health": (profit_health_by_coin or {}).get(coin),
        })
    return positions


def apply_position_updates(ledger, position_updates, current_prices, now):
    """Applies Gemini's hold/exit_now/tighten_stop/move_target
    decisions to the matching pending ledger entries in place. Returns
    a list of {coin, action, reasoning, ...} summaries for the
    Telegram message - separate from the ledger mutation itself so the
    message-building code doesn't need to re-derive what changed.
    A coin with no update returned (missing from position_updates)
    defaults to holding - same behavior as an explicit "hold", just
    silent, since a missing entry is a Gemini-response gap, not a
    signal."""
    now_utc_value = utc_datetime(now)
    summaries = []
    for entry in ledger:
        if entry["status"] != "pending":
            continue
        coin = entry["coin"]
        update = position_updates.get(coin)
        if not update:
            continue
        action = update.get("action")
        update_direction = update.get("direction")
        if update_direction and update_direction != entry.get("direction"):
            print(f"  Gemini: position update for {coin} rejected - direction mismatch "
                  f"({update_direction} vs open {entry.get('direction')})")
            continue
        if action not in VALID_POSITION_ACTIONS:
            print(f"  Gemini: unrecognized position action '{action}' for {coin}, treating as hold")
            continue
        if action == "hold":
            continue

        if action == "exit_now":
            current_price = current_prices.get(coin)
            entry_price = entry.get("entry_price")
            trade_amount = entry.get("trade_amount_inr")
            if not current_price or not entry_price or entry_price <= 0 or not trade_amount or USDT_INR_RATE <= 0:
                print(f"  Gemini: exit_now for {coin} skipped - missing/invalid price, trade data, or FX rate")
                continue
            quantity = trade_amount / (entry_price * USDT_INR_RATE)
            round_trip_fee = trade_amount * TAKER_FEE_RATE * 2
            if entry["direction"] == "long":
                pnl = quantity * (current_price - entry_price) * USDT_INR_RATE
            else:
                pnl = quantity * (entry_price - current_price) * USDT_INR_RATE
            entry["status"] = "gemini_exit"
            entry["resolved_pnl"] = round(pnl - round_trip_fee, 2)
            entry["resolved_time"] = now.isoformat()
            entry["exit_reasoning"] = update.get("reasoning")
            summaries.append({"coin": coin, "direction": entry["direction"], "action": "exit_now",
                               "reasoning": update.get("reasoning"), "pnl": entry["resolved_pnl"]})
            print(f"  Gemini: {coin} {entry['direction'].upper()} closed early (exit_now) at {current_price}, "
                  f"pnl={entry['resolved_pnl']}")
            continue

        if action == "tighten_stop":
            new_stop = update.get("updated_stop_loss")
            if new_stop is None:
                print(f"  Gemini: tighten_stop for {coin} missing updated_stop_loss, ignoring")
                continue
            current_price = current_prices.get(coin)
            old_stop = entry.get("stop_loss")
            if not current_price or old_stop is None or new_stop <= 0:
                print(f"  Gemini: tighten_stop for {coin} rejected - invalid price/stop")
                continue
            if entry["direction"] == "long":
                valid = old_stop <= new_stop < current_price
            else:
                valid = current_price < new_stop <= old_stop
            if not valid:
                print(f"  Gemini: tighten_stop for {coin} rejected - new stop does not tighten safely")
                continue
            entry["stop_loss"] = new_stop
            entry["revised_at"] = now.isoformat()
            entry.setdefault("level_history", []).append({"from": now.isoformat(), "stop": new_stop, "target": entry.get("target_price")})
            summaries.append({"coin": coin, "direction": entry["direction"], "action": "tighten_stop",
                               "reasoning": update.get("reasoning"), "new_stop_loss": new_stop})
            continue

        if action == "move_target":
            new_target = update.get("updated_target_price")
            if new_target is None:
                print(f"  Gemini: move_target for {coin} missing updated_target_price, ignoring")
                continue
            current_price = current_prices.get(coin)
            old_target = entry.get("target_price")
            if not current_price or old_target is None or new_target <= 0:
                print(f"  Gemini: move_target for {coin} rejected - invalid price/target")
                continue
            if entry["direction"] == "long":
                valid = new_target > current_price
            else:
                valid = new_target < current_price
            if not valid:
                print(f"  Gemini: move_target for {coin} rejected - target is on wrong side of current price")
                continue
            entry["target_price"] = new_target
            entry["revised_at"] = now.isoformat()
            entry.setdefault("level_history", []).append({"from": now.isoformat(), "stop": entry.get("stop_loss"), "target": new_target})
            summaries.append({"coin": coin, "direction": entry["direction"], "action": "move_target",
                               "reasoning": update.get("reasoning"), "new_target_price": new_target})
            continue
    return summaries


def find_pending_same_direction(ledger, coin, direction, now):
    """Is there already an unresolved call on this exact coin+direction?
    Returns the actual ledger entry (so the caller can update it in
    place) or None. No time cutoff - "pending" status already means
    it hasn't hit target or stop yet, regardless of how long ago it
    was flagged. Confirmed directly as a real gap: an earlier version
    capped this at 60 minutes, which meant a genuinely still-open
    position from 90 minutes ago was missed entirely - defeating the
    whole point of this check for exactly the longest-running open
    positions, where it matters most. (Ledger entries are pruned after
    LEDGER_MAX_AGE_HOURS regardless of status, so this is naturally
    bounded without needing its own separate cutoff.)"""
    for entry in ledger:
        if entry["coin"] == coin and entry["direction"] == direction and entry["status"] == "pending":
            return entry
    return None


def candles_to_compact(candles):
    """Same compact OHLCV dict format used throughout - one place so
    every tier (3m/15m/1h/1d) serializes identically for the Gemini
    payload."""
    return [
        {"t": str(idx), "o": round(float(r["open"]), 8), "h": round(float(r["high"]), 8),
         "l": round(float(r["low"]), 8), "c": round(float(r["close"]), 8), "v": round(float(r["volume"]), 2)}
        for idx, r in candles.iterrows()
    ]


def load_daily_candles_30d():
    """Reads the compact per-coin 30-day daily-candle file the daily
    refresh job produces (same script/schedule as RVOL percentiles,
    extended to also fetch this). Read-only here, same reasoning as
    load_rvol_percentiles - a 30-day historical pull is far too heavy
    for the 1-minute live loop to do itself."""
    if os.path.exists(DAILY_CANDLES_FILE):
        try:
            with open(DAILY_CANDLES_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def rvol_percentile_rank(rvol, coin_percentiles):
    """Estimates which percentile of a coin's OWN historical RVOL
    distribution the current value falls in, via linear interpolation
    against the stored breakpoint grid. Returns None if this coin has
    no stored history yet (backfill hasn't run for it) - caller should
    fall back to the fixed-scale rvol_label in that case, not error."""
    if coin_percentiles is None or rvol is None or pd.isna(rvol):
        return None
    grid = coin_percentiles.get("grid")
    breakpoints = coin_percentiles.get("breakpoints")
    if not grid or not breakpoints:
        return None
    # np.interp clamps at the edges: a value below the historical
    # minimum reads as 0th percentile, above the historical maximum
    # reads as 100th - not extrapolated beyond the observed range,
    # which is the honest behavior for a value genuinely outside
    # anything seen in the backfill window.
    return round(float(np.interp(rvol, breakpoints, grid)), 1)


def build_coin_snapshot(coin, candles_3m, candles_15m, candles_1h=None, rvol_percentiles=None):
    row = candles_3m.iloc[-1]
    candle_time = candle_key(candles_3m.index[-1])
    coin_percentiles = (rvol_percentiles or {}).get(coin)
    rvol = row.get("rvol")
    structure = _market_structure_snapshot(candles_3m, candles_15m, candles_1h if candles_1h is not None else pd.DataFrame())
    snapshot = {
        "coin": coin,"candle_time": candle_time,"close": round(float(row["close"]),8),
        "atr14_3m": _safe_float(row.get("atr14")),"rvol": _safe_float(rvol),"rvol_label":rvol_label(rvol),
        "rvol_percentile":rvol_percentile_rank(rvol,coin_percentiles),
        "momentum_pct_5_3m":_safe_float(row.get("momentum_pct_5")),"momentum_pct_20_3m":_safe_float(row.get("momentum_pct_20")),"momentum_pct_60_3m":_safe_float(row.get("momentum_pct_60")),
        "momentum_acceleration_5_3m":_safe_float(row.get("momentum_accel_5")),"momentum_acceleration_20_3m":_safe_float(row.get("momentum_accel_20")),
        "candle_body_pct":round(abs(float(row["close"])-float(row["open"]))/float(row["open"])*100,3) if float(row["open"]) else None,
        "candle_range_pct":round((float(row["high"])-float(row["low"]))/float(row["open"])*100,3) if float(row["open"]) else None,
        "upper_wick_pct":round((float(row["high"])-max(float(row["open"]),float(row["close"]))) / float(row["open"])*100,3) if float(row["open"]) else None,
        "lower_wick_pct":round((min(float(row["open"]),float(row["close"]))-float(row["low"])) / float(row["open"])*100,3) if float(row["open"]) else None,
        "body_to_range":_safe_float(row.get("body_to_range")),
        "close_location_in_range":_safe_float(row.get("close_location_in_range")),
        "volume_acceleration_3":_safe_float(row.get("volume_acceleration_3")),
        "candle_direction":"green" if float(row["close"])>float(row["open"]) else ("red" if float(row["close"])<float(row["open"]) else "doji"),
        "market_structure":structure,
        "ctx_3m":candles_to_compact(candles_3m.tail(40)),
        "ctx_15m":candles_to_compact(candles_15m.tail(32)),
        "ctx_1h":candles_to_compact(candles_1h.tail(24)) if candles_1h is not None else [],
    }
    return snapshot

def _flush_pending_telegram(state):
    pending=state.get("pending_telegram")
    if not pending: return True
    try:
        send_telegram(pending["text"],pending.get("reply_markup"))
        state["pending_telegram"]=None; state["last_sent_at"]=now_utc().isoformat(); return True
    except Exception as e:
        print(f"Queued Telegram delivery failed: {e}"); return False


def _format_pnl(v): return f"+₹{v}" if v>=0 else f"-₹{abs(v)}"


def _build_message(candle_start_utc,now,scorecard,scan_stats=None,stale=None,position_summaries=None,flagged=None):
    import html
    ist=candle_start_utc+timedelta(hours=5,minutes=30); end=ist+timedelta(minutes=3); det=now+timedelta(hours=5,minutes=30)
    lines=[f"📊 {ist.strftime('%H:%M')}→{end.strftime('%H:%M')} IST (detected {det.strftime('%H:%M:%S')})"]
    lines.append(f"Last 24h: {scorecard['total']} calls — {scorecard['target_hit']} hit target, {scorecard['stop_hit']} hit stop, {scorecard['expired']} expired, {scorecard.get('gemini_exit',0)} closed by Gemini, {scorecard['pending']} pending")
    lines.append(f"Realized {_format_pnl(scorecard['realized_pnl_inr'])} | Unrealized {_format_pnl(scorecard['unrealized_pnl_inr'])} | Total {_format_pnl(scorecard['total_pnl_inr'])}")
    if scan_stats:
        lines.append(f"\n🔎 Gemini scan: {scan_stats['scanned']} coins | proposals {scan_stats['proposals']} (LONG {scan_stats.get('long_proposals',0)} / SHORT {scan_stats.get('short_proposals',0)}) | TAKE {scan_stats['take']} | Python risk rejected {scan_stats['risk_rejected']}")
        reviewed=scan_stats.get('position_reviewed',0); missing=scan_stats.get('position_missing',0); acts=scan_stats.get('position_actions',{})
        if reviewed or missing or scan_stats.get('gemini_review_failed'):
            lines.append(f"📋 Position review: {reviewed} reviewed | HOLD {acts.get('hold',0)} | EXIT {acts.get('exit_now',0)} | TIGHTEN {acts.get('tighten_stop',0)} | TARGET {acts.get('move_target',0)} | missing {missing}")
            if scan_stats.get('gemini_review_failed'):
                lines.append('🛑 Gemini review FAILED — no missing position was treated as HOLD; Python protection remains active.')
        for reason,count in sorted(scan_stats['risk_reasons'].items(),key=lambda x:-x[1])[:4]: lines.append(f"• Risk reject: {html.escape(reason)} ({count})")
        if scan_stats.get('short_proposals'):
            short_rej=sum(1 for g in (flagged or {}).values() if str(g.get('direction','')).lower()=='short' and g.get('risk_validation_error') and not g.get('take_trade'))
            lines.append(f"↘️ Short diagnostics: Gemini proposed {scan_stats.get('short_proposals',0)} SHORT | Python rejected {short_rej}")
    if stale: lines.append(f"\n🟡 Stale/missing data: {', '.join(sorted(set(stale)))} — Gemini not called for this candle.")
    if position_summaries:
        lines.append("\n📋 Position updates:")
        for ps in position_summaries:
            d=(ps.get('direction') or '?').upper(); a=ps.get('action')
            if a=='exit_now': lines.append(f"🚪 <b>{html.escape(ps['coin'])} {d} — CLOSED</b> ({_format_pnl(ps.get('pnl',0))})")
            elif a=='profit_protection': lines.append(f"🛡 <b>{html.escape(ps['coin'])} {d}</b> — profit protected at {ps.get('new_stop_loss')} | MFE {_format_pnl(ps.get('mfe_pnl_inr',0))} | giveback {ps.get('giveback_pct',0):.0f}% | health {ps.get('health_score','?')}")
            elif a=='profit_exit': lines.append(f"🚪 <b>{html.escape(ps['coin'])} {d} — PROFIT EXIT</b> ({_format_pnl(ps.get('pnl',0))}) | MFE {_format_pnl(ps.get('mfe_pnl_inr',0))} | giveback {ps.get('giveback_pct',0):.0f}% | health {ps.get('health_score','?')}")
            elif a=='tighten_stop': lines.append(f"🔻 <b>{html.escape(ps['coin'])} {d}</b> — stop tightened to {ps.get('new_stop_loss')}")
            elif a=='move_target': lines.append(f"🎯 <b>{html.escape(ps['coin'])} {d}</b> — target moved to {ps.get('new_target_price')}")
            if ps.get('reasoning'): lines.append(html.escape(ps['reasoning']))
    if flagged:
        for coin,g in flagged.items():
            verdict='TAKE' if g.get('take_trade') else 'SKIP'; d=(g.get('direction') or '?').upper(); em='✅' if verdict=='TAKE' else '⚪'
            lines.append(f"\n{em} <b>{html.escape(coin)} {d} — {verdict}</b> (conviction {g.get('conviction')}/10)")
            lines.append(html.escape(g.get('reasoning','-')))
            if g.get('risk_validation_error'): lines.append(f"🛡 Risk gate: {html.escape(g['risk_validation_error'])}")
            if g.get('market_regime'): lines.append(f"Regime {html.escape(str(g.get('market_regime')))} | Type {html.escape(str(g.get('trade_type','')))} | Location {html.escape(str(g.get('market_location','')))}")
            if g.get('key_level_used') is not None: lines.append(f"Key level {g.get('key_level_used')} | Invalidation: {html.escape(str(g.get('invalidation_reason','-')))}")
            lines.append(f"Entry {g.get('entry_price')} | SL {g.get('stop_loss')} | Target {g.get('target_price')}")
            if g.get('take_trade'): lines.append(f"Amount ₹{g.get('trade_amount_inr')} (risk ₹{g.get('max_loss_this_trade_inr')})")
    if not flagged and not position_summaries and not stale: lines.append("\nNo new signals this cycle.")
    return "\n".join(lines)


def build_market_context(snapshots):
    """Cross-coin context built only from already-fetched snapshots.
    No extra API calls. BTC is treated as the reference market only, never as
    an automatic trade direction. Breadth describes how broad the current move is.
    """
    btc=next((s for s in snapshots if s.get("coin")=="BTC"),None)
    if btc:
        btc_m5=btc.get("momentum_pct_5_3m"); btc_m20=btc.get("momentum_pct_20_3m"); btc_m60=btc.get("momentum_pct_60_3m")
        btc_regime=btc.get("market_structure",{}).get("market_regime")
    else:
        btc_m5=btc_m20=btc_m60=None; btc_regime=None
    valid=[s for s in snapshots if s.get("momentum_pct_20_3m") is not None]
    positive=sum(1 for s in valid if s["momentum_pct_20_3m"]>0)
    negative=sum(1 for s in valid if s["momentum_pct_20_3m"]<0)
    return {
        "btc_reference": {"regime":btc_regime,"momentum_pct_5_3m":btc_m5,"momentum_pct_20_3m":btc_m20,"momentum_pct_60_3m":btc_m60},
        "breadth": {"coins_with_positive_20c_momentum":positive,"coins_with_negative_20c_momentum":negative,"coins_measured":len(valid),"positive_pct":round(positive/len(valid)*100,1) if valid else None},
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--coins',type=str,required=True); args=parser.parse_args()
    coins=[c.strip().upper() for c in args.coins.split(',') if c.strip()]; now=now_utc(); state=load_state()
    if state.get("pending_telegram"):
        if not _flush_pending_telegram(state): save_state(state); return
        save_state(state)

    forming=now.replace(second=0,microsecond=0)-timedelta(minutes=now.minute%3)
    expected=forming-timedelta(minutes=3); expected_key=candle_key(expected)
    processed_map=state.setdefault("processed_candle_coins",{})
    processed=set(processed_map.get(expected_key,[]))
    if len(processed)>=len(coins):
        return

    # Resolve existing ledger positions using every coin that can be fetched;
    # this remains independent of new-signal processing.
    rvol_percentiles=load_rvol_percentiles(); daily_candles=load_daily_candles_30d()
    ledger=state.get('ledger',[])
    ledger_coins={e.get('coin') for e in ledger if e.get('status')=='pending'}
    pending_coins=[c for c in coins if c not in processed]
    fetch_coins=sorted(set(pending_coins)|ledger_coins)
    start_fetch=now-timedelta(minutes=FETCH_MINUTES_BACK)
    def fetch_one(coin):
        c1=fetch_coindcx_klines(coin,'1m',start_fetch.isoformat(),now.isoformat(),stagger_delay=False)
        return coin,drop_still_forming_bucket(resample_candles(c1,3),now,3),drop_still_forming_bucket(resample_candles(c1,15),now,15),drop_still_forming_bucket(resample_candles(c1,60),now,60),drop_still_forming_bucket(c1,now,1)
    fetched={}; fetch_errors={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(fetch_one,c):c for c in fetch_coins}
        for f in as_completed(fs):
            c=fs[f]
            try: _,a,b,d,c1=f.result(); fetched[c]=(a,b,d,c1)
            except Exception as e: fetch_errors[c]=str(e); print(f"  {c}: fetch failed ({e})")

    update_position_telemetry(ledger,fetched,now)
    ledger=resolve_ledger(ledger,fetched,now); state['ledger']=ledger
    current_prices={c:float(v[3]['close'].iloc[-1]) if len(v)>=4 and not v[3].empty else float(v[0]['close'].iloc[-1]) for c,v in fetched.items() if len(v[0])}
    scorecard=compute_scorecard(ledger,now,window_hours=24,current_prices=current_prices)

    snapshots=[]; stale=[]; stale_reasons={}; fresh_coins=[]
    for coin in pending_coins:
        if coin not in fetched:
            stale.append(coin); stale_reasons[coin]=f"fetch failed: {fetch_errors.get(coin,'no data returned')}"; continue
        c3,c15,c60,c1=fetched[coin]
        if len(c3)<65:
            stale.append(coin); stale_reasons[coin]=f"only {len(c3)} closed 3m candles"; continue
        try:
            c3=compute_raw_stats(c3)
            snap=build_coin_snapshot(coin,c3,c15,c60,rvol_percentiles)
            if snap.get('candle_time')!=expected_key:
                stale.append(coin); stale_reasons[coin]=f"last 3m={snap.get('candle_time')} expected={expected_key}"; continue
            hist=state.get('call_history',{}).get(coin,[])
            if hist:
                last=hist[-1]; mins=(now-utc_datetime(last['time'])).total_seconds()/60
                if mins<=180 and last.get('entry_price'):
                    ch=round((snap['close']-last['entry_price'])/last['entry_price']*100,3); fav=ch>0 if last.get('direction')=='long' else ch<0
                    snap['prior_call']={'direction':last.get('direction'),'take_trade':last.get('take_trade'),'minutes_ago':round(mins,1),'price_change_pct_since':ch,'moved_favorably':fav}
            snap['ctx_daily_30d']=daily_candles.get(coin,[])[-30:]
            snapshots.append(snap); fresh_coins.append(coin)
        except Exception as e:
            print(f"  {coin}: snapshot error ({e})"); stale.append(coin); stale_reasons[coin]=f"snapshot error: {e}"

    # Build cross-market context from this cycle's already-fetched universe.
    # Feed it to Gemini for every coin, while keeping direction discretionary.
    market_context=build_market_context(snapshots)
    btc_ref=market_context.get("btc_reference",{})
    for snap in snapshots:
        m5=snap.get("momentum_pct_5_3m"); m20=snap.get("momentum_pct_20_3m"); m60=snap.get("momentum_pct_60_3m")
        snap["relative_strength_vs_btc"]={
            "momentum_diff_5":round(m5-btc_ref["momentum_pct_5_3m"],3) if m5 is not None and btc_ref.get("momentum_pct_5_3m") is not None else None,
            "momentum_diff_20":round(m20-btc_ref["momentum_pct_20_3m"],3) if m20 is not None and btc_ref.get("momentum_pct_20_3m") is not None else None,
            "momentum_diff_60":round(m60-btc_ref["momentum_pct_60_3m"],3) if m60 is not None and btc_ref.get("momentum_pct_60_3m") is not None else None,
        }
        snap["market_context"]=market_context

    # Process fresh coins immediately. Stale coins are NOT allowed to block them.
    profit_health_by_coin = {}
    for p in ledger:
        if p.get("status") == "pending" and p.get("coin") in fetched:
            profit_health_by_coin[p.get("coin")] = _profit_health_from_fetched(p, fetched)
    open_positions=build_open_position_context(ledger,current_prices,now,profit_health_by_coin)
    gemini_review_failed=False
    if snapshots:
        ok,flagged,position_updates=get_trade_suggestions_batch(snapshots,scorecard,open_positions)
        if not ok:
            flagged={}; position_updates={}
            gemini_review_failed=True
            print('Gemini unavailable/invalid; fresh coins remain unprocessed; Python protection continues')
        else:
            for c in fresh_coins: processed.add(c)
    else:
        flagged={}; position_updates={}

    state['processed_candle_coins'][expected_key]=sorted(processed)
    # Keep bounded history; each value is just the list of coin symbols.
    for k in sorted(list(state['processed_candle_coins']))[:-8]:
        state['processed_candle_coins'].pop(k,None)

    position_summaries=apply_position_updates(ledger,position_updates,current_prices,now)
    protection_summaries=apply_python_profit_protection(ledger,current_prices,now,fetched=fetched)
    position_summaries.extend(protection_summaries)
    call_history=state.get('call_history',{})
    review_actions = {}
    for coin, upd in position_updates.items():
        a = (upd.get('action') or 'hold').lower()
        review_actions[a] = review_actions.get(a, 0) + 1
    expected_review = {p['coin'] for p in open_positions}
    missing_reviews = sorted(expected_review - set(position_updates))
    long_proposals=sum(1 for g in flagged.values() if str(g.get('direction','')).lower()=='long')
    short_proposals=sum(1 for g in flagged.values() if str(g.get('direction','')).lower()=='short')
    stats={'scanned':len(snapshots),'proposals':len(flagged),'take':sum(1 for g in flagged.values() if g.get('take_trade')),'risk_rejected':0,'risk_reasons':{},'fresh':len(fresh_coins),'stale':len(stale),'position_reviewed':len(position_updates),'position_missing':len(missing_reviews),'position_actions':review_actions,'gemini_review_failed':gemini_review_failed,'long_proposals':long_proposals,'short_proposals':short_proposals}
    for coin,g in flagged.items():
        if g.get('risk_validation_error') and not g.get('take_trade'):
            stats['risk_rejected']+=1; r=g['risk_validation_error']; stats['risk_reasons'][r]=stats['risk_reasons'].get(r,0)+1
        h=call_history.get(coin,[])
        if h:
            mins=(now-utc_datetime(h[-1]['time'])).total_seconds()/60
            if mins<=15 and h[-1].get('direction')!=g.get('direction'):
                g['whipsaw_warning']=f"reverses {h[-1].get('direction','?').upper()} call from {int(mins)} min ago"
        h.append({'direction':g.get('direction'),'entry_price':g.get('entry_price'),'take_trade':g.get('take_trade'),'time':now.isoformat()}); call_history[coin]=h[-5:]
        if g.get('take_trade'):
            opposite='short' if g.get('direction')=='long' else 'long'; old=find_pending_same_direction(ledger,coin,opposite,now)
            if old is not None:
                cp=current_prices.get(coin); ep=old.get('entry_price'); amt=old.get('trade_amount_inr')
                if cp and ep and amt and USDT_INR_RATE>0:
                    qty=amt/(ep*USDT_INR_RATE); fees=amt*TAKER_FEE_RATE*2; pnl=qty*((cp-ep) if old['direction']=='long' else (ep-cp))*USDT_INR_RATE-fees
                    old.update({'status':'gemini_exit','resolved_pnl':round(pnl,2),'resolved_time':now.isoformat(),'exit_reasoning':'Closed because Gemini reversed direction on the same coin.'})
            existing=find_pending_same_direction(ledger,coin,g.get('direction'),now)
            if existing is not None: existing['conviction']=g.get('conviction')
            else:
                ledger.append({'coin':coin,'direction':g.get('direction'),'entry_price':g.get('entry_price'),'stop_loss':g.get('stop_loss'),'target_price':g.get('target_price'),'trade_amount_inr':g.get('trade_amount_inr'),'quantity':g.get('quantity'),'max_loss_this_trade_inr':g.get('max_loss_this_trade_inr'),'conviction':g.get('conviction'),'status':'pending','time':now.isoformat(),'reasoning':g.get('reasoning'),'level_history':[{'from':now.isoformat(),'stop':g.get('stop_loss'),'target':g.get('target_price')}],'peak_price':g.get('entry_price'),'trough_price':g.get('entry_price'),'mfe_pnl_inr':0.0,'mae_pnl_inr':0.0,'mfe_r':0.0,'mfe_time':now.isoformat(),'mae_time':now.isoformat(),'mfe_giveback_pct':0.0})
    state['call_history']=call_history; state['ledger']=ledger
    state['last_processed_candle_time']=expected_key if len(processed)>=len(coins) else state.get('last_processed_candle_time')
    scorecard=compute_scorecard(ledger,now,window_hours=24,current_prices=current_prices)

    # Telegram: report useful scan progress even when only some coins were fresh.
    msg=_build_message(expected,now,scorecard,scan_stats=stats,position_summaries=position_summaries,flagged=flagged,stale=stale)
    if stale:
        details="\n".join(f"• {c}: {stale_reasons[c]}" for c in sorted(stale))
        msg += "\n\n🛠 Data diagnostics:\n"+details
    if not flagged and not position_summaries and not stale and not snapshots:
        msg += "\n\nNo new signals this cycle."
    chart_coins=list(flagged)+[p['coin'] for p in position_summaries if p['coin'] not in flagged]
    markup={'inline_keyboard':[[{'text':f'📈 {c} chart','url':f'https://coindcx.com/futures/B-{c}_USDT'}] for c in chart_coins]} if chart_coins else None
    state['pending_telegram']={'text':msg,'reply_markup':markup}
    save_state(state); _flush_pending_telegram(state); save_state(state)


if __name__ == "__main__":
    main()
