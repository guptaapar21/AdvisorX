"""
Trend-alignment + reversal-candle RVOL scanner (separate from Idea
#18F). This is a FOCUSED EXPERIMENT, not a complete trading strategy:
it only measures whether RVOL on a reversal candle carries useful
information - it does not enter, does not set a stop/target, does not
require a breakout, and does not track or act on anything after the
reversal candle.

STAGE 1 - trend qualification (checked every run, for every coin):
  Long requires ALL of:
    - 3m EMA9 > EMA21, and EMA9 rising (vs 3 candles back)
    - 3m candle closes above VWAP
    - 3m ADX14 >= 25, and rising
    - 15m EMA9 > EMA21, and EMA9 rising (vs 3 candles back on 15m)
  Short is the exact mirror.

STAGE 2 - pullback run (only while qualified):
  Watch for ONE OR MORE consecutive red candles (long case; green for
  short) - all of them together are the pullback/falling sequence. No
  requirement that they touch EMA9/EMA21.

STAGE 3 - reversal candle + RVOL:
  The FIRST candle that breaks the pullback run by closing the
  opposite color (green, for long) is the reversal candle. Its RVOL
  (this candle's volume / the average of the PRIOR 20 candles) is
  computed and reported, labeled per the standard scale (<1.0 weak,
  1.0-1.5 normal, 1.5-2.0 strong, >2.0 very strong/possible exhaustion)
  - informational only, never a filter, never an entry trigger.

DELIBERATELY NOT IMPLEMENTED, per explicit instruction: no EMA9/21
touch requirement on the pullback or reversal candles, no requirement
that the reversal candle break the pullback run's high/low, no
breakout tracking of any kind, no entry price, no stop-loss, no
target, no RVOL threshold, and nothing at all about the candle AFTER
the reversal candle - that one is for manual observation only.

All checks run every 1-minute poll, but the Telegram message itself is
only sent every 3 minutes, batching everything accumulated since the
last send. State persists across runs via the same git-committed JSON
pattern as live_signal_monitor_18f.py.

Usage:
  python3 trend_alignment_scanner.py --coins BTC,ETH,SOL,...
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import requests

from coindcx_fetcher import fetch_coindcx_klines, resample_candles

STATE_FILE = "trend_scanner_state.json"
RVOL_PERCENTILE_FILE = "rvol_percentiles.json"
# Trimmed from 150 to 100 15m candles after measuring the real fetch
# cost at 18 coins - 150 candles needed 3 API requests/coin (2250 min
# of 1m data), 100 needs only 2 (1500 min), saving ~14.4s across all 18
# coins. Still 10x the bare minimum (10) actually checked in the code
# below, real margin for EMA21/ADX14 to be stable, not just non-error.
# Trimmed to 65 15m candles (975 min) - fits in exactly 1 API request
# per coin instead of 2, still 4.6x ADX14's min_periods(14) warmup
# requirement and 6.5x the hard minimum (10) checked in code below.
FETCH_MINUTES_BACK = 15 * 65
MESSAGE_INTERVAL_MINUTES = 1


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"coins": {}, "last_sent_at": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def send_telegram(text, reply_markup=None):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


def drop_still_forming_bucket(candles, now, bar_minutes):
    if len(candles) == 0:
        return candles
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    last_bucket_start = candles.index[-1]
    if last_bucket_start + pd.Timedelta(minutes=bar_minutes) > now:
        return candles.iloc[:-1]
    return candles


def compute_indicators(candles):
    candles = candles.copy()
    candles["ema9"] = candles["close"].ewm(span=9, adjust=False).mean()
    candles["ema21"] = candles["close"].ewm(span=21, adjust=False).mean()

    typical = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    day = candles.index.floor("D")
    pv = typical * candles["volume"]
    candles["vwap"] = pv.groupby(day).cumsum() / candles["volume"].groupby(day).cumsum().replace(0, float("nan"))

    high, low, close = candles["high"], candles["low"], candles["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di14 = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr14.replace(0, float("nan"))
    minus_di14 = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr14.replace(0, float("nan"))
    dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, float("nan"))
    candles["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # RVOL: current candle's volume relative to the PRIOR 20-candle
    # average (shifted by 1 so the current candle's own volume never
    # inflates its own denominator).
    candles["avg_volume_20"] = candles["volume"].rolling(20).mean().shift(1)
    candles["rvol"] = candles["volume"] / candles["avg_volume_20"].replace(0, float("nan"))
    return candles


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


def check_trend_alignment(candles_3m, candles_15m, direction, trend_lookback=3):
    if len(candles_3m) < 30 or len(candles_15m) < trend_lookback + 5:
        return False, {}

    row3 = candles_3m.iloc[-1]
    ema9_prev3 = candles_3m["ema9"].iloc[-(trend_lookback + 1)]
    adx_prev1 = candles_3m["adx14"].iloc[-2]
    adx_prev2 = candles_3m["adx14"].iloc[-3]

    row15 = candles_15m.iloc[-1]
    ema9_15_prev3 = candles_15m["ema9"].iloc[-(trend_lookback + 1)]

    adx = row3["adx14"]
    adx15 = row15["adx14"]
    adx15_prev1 = candles_15m["adx14"].iloc[-2]
    if pd.isna(adx) or pd.isna(adx_prev1) or pd.isna(adx_prev2):
        return False, {}
    if pd.isna(adx15) or pd.isna(adx15_prev1):
        return False, {}

    # slope1_2: ADX must be rising vs BOTH the previous candle and the
    # one before it. Ported from live_signal_monitor_18f.py, where a
    # 1-bar-only check ("adx > adx_prev1") was found to pass at the
    # exact peak of the ADX arc - the peak bar is still higher than the
    # single bar before it. Requiring two consecutive rising bars
    # filters a single spurious uptick off a dip. NOTE: this does NOT
    # catch a clean, uninterrupted ADX climb that ends abruptly at its
    # top (confirmed against a real XRP case) - on a straight rise,
    # every bar beats both prior bars, so slope1_2 passes too.
    adx_rising_1_and_2 = bool(adx > adx_prev1 and adx > adx_prev2)

    checks = {
        "adx14": round(float(adx), 2),
        # Delta ADX: ADX(current) - ADX(previous), 1 bar - the raw
        # magnitude, shown alongside the absolute value in Telegram so
        # "ADX 31" can be read next to "rising sharply" vs "barely
        # ticking up". Separate from adx_rising_1_and_2 above, which
        # is the 2-bar slope1_2 gate used for qualify/disqualify.
        "adx14_delta": round(float(adx - adx_prev1), 2),
        # Acceleration: delta of delta, candle-to-candle, no smoothing.
        # acceleration = (ADX(now) - ADX(prev)) - (ADX(prev) - ADX(prev2))
        # ADX can still be RISING every candle while this goes negative
        # - each gain just gets smaller than the last one. That's a
        # real early-warning signal distinct from the delta itself:
        # e.g. deltas of +5, +6, +7, +4, +2 never once go negative, but
        # acceleration turns negative starting at the +4 candle,
        # flagging the stall before ADX itself ever turns down.
        "adx14_accel": round(float((adx - adx_prev1) - (adx_prev1 - adx_prev2)), 2),
        "adx_gate_passed": bool(adx >= 25 and adx_rising_1_and_2),
        # 15m ADX must be at/above 25 AND rising - floor matches the 3m
        # gate. ADX measures trend strength, not direction, so this is
        # direction-agnostic and applies to both long and short.
        # Single-bar slope comparison (vs the previous closed 15m bar),
        # not the 2-bar slope1_2 used on 3m - note this only updates
        # once every 15 minutes, so it holds the same value across five
        # consecutive 3m scans.
        "adx14_15m": round(float(adx15), 2),
        "15m_adx_gate_passed": bool(adx15 >= 25 and adx15 > adx15_prev1),
    }

    if direction == "long":
        checks["3m_ema9_above_ema21"] = bool(row3["ema9"] > row3["ema21"])
        checks["3m_ema9_rising"] = bool(row3["ema9"] > ema9_prev3)
        checks["3m_close_above_vwap"] = bool(row3["close"] > row3["vwap"])
        checks["15m_ema9_above_ema21"] = bool(row15["ema9"] > row15["ema21"])
        checks["15m_ema9_rising"] = bool(row15["ema9"] > ema9_15_prev3)
    else:
        checks["3m_ema9_below_ema21"] = bool(row3["ema9"] < row3["ema21"])
        checks["3m_ema9_falling"] = bool(row3["ema9"] < ema9_prev3)
        checks["3m_close_below_vwap"] = bool(row3["close"] < row3["vwap"])
        checks["15m_ema9_below_ema21"] = bool(row15["ema9"] < row15["ema21"])
        checks["15m_ema9_falling"] = bool(row15["ema9"] < ema9_15_prev3)

    all_passed = all(v for k, v in checks.items() if k not in ("adx14", "adx14_delta", "adx14_accel", "adx14_15m"))
    return all_passed, checks


def check_trend_broadly_intact(candles_3m, candles_15m, direction, adx_min=25, trend_lookback=3):
    """Middle-ground check used while already tracking a pullback run -
    re-verifies EVERYTHING the strict qualification check does (VWAP,
    15m EMA9>EMA21 AND rising, 3m EMA9>EMA21, ADX level) on every single
    candle, EXCLUDING TWO specific sub-conditions proven to spuriously
    fail on an ordinary, healthy pullback candle:
      - 3m EMA9 still rising (confirmed directly - a real pullback
        candle flattens short-term EMA9 slope for that one candle even
        in a genuinely valid uptrend)
      - ADX still rising (confirmed directly - ADX dips slightly on an
        ordinary pullback candle too, e.g. 97.01 -> 95.80 in one real
        test - completely normal noise, not genuine breakdown)
    Both were tested by actually re-including them and watching an
    ordinary pullback candle spuriously disqualify a valid trend before
    being excluded - not assumed safe to exclude.

    ADX's LEVEL (not slope) is still checked - that's what caught the
    real SOL case, where ADX collapsed well below the qualification
    threshold while EMA9 (a lagging indicator) hadn't caught up yet."""
    if len(candles_3m) < 30 or len(candles_15m) < trend_lookback + 5:
        return False

    row3 = candles_3m.iloc[-1]
    row15 = candles_15m.iloc[-1]
    ema9_15_prev3 = candles_15m["ema9"].iloc[-(trend_lookback + 1)]
    adx = row3["adx14"]

    # ADX-still-rising is deliberately EXCLUDED here, same reasoning as
    # 3m EMA9-rising - confirmed directly: ADX dips slightly on an
    # ordinary pullback candle (97.01 -> 95.80 in one real test), which
    # is completely normal noise, not genuine trend breakdown. The
    # LEVEL floor (below) is what actually caught the real SOL problem
    # and is kept; the SLOPE requirement is what caused this bug and is
    # excluded, just like EMA9's slope requirement was.
    if pd.isna(adx) or adx < adx_min:
        return False

    if direction == "long":
        return bool(row3["ema9"] > row3["ema21"] and row3["close"] > row3["vwap"]
                     and row15["ema9"] > row15["ema21"] and row15["ema9"] > ema9_15_prev3)
    return bool(row3["ema9"] < row3["ema21"] and row3["close"] < row3["vwap"]
                and row15["ema9"] < row15["ema21"] and row15["ema9"] < ema9_15_prev3)


def process_coin(coin, candles_3m, candles_15m, coin_state, rvol_percentiles=None):
    """Runs all 3 stages for one coin, mutating coin_state in place.
    Returns a dict of anything worth reporting this cycle, or None.
    rvol_percentiles: the full {coin: {...}} dict from
    load_rvol_percentiles(), or None to skip percentile ranking
    entirely and use only the fixed-scale label."""
    row = candles_3m.iloc[-1]
    candle_time = str(candles_3m.index[-1])
    report = {"coin": coin, "candle_time": candle_time}

    # Full strict check every single time, no exceptions - including
    # while a pullback run is already being tracked. Reverted from an
    # earlier looser design after direct evidence (a real chart showing
    # ADX rising through several consecutive red candles in a genuine
    # strong trend) that "still rising" is meaningful signal, not just
    # noise - if ADX or EMA9 genuinely stalls during a pullback, that's
    # treated as real evidence of weakening momentum, and the coin is
    # correctly disqualified rather than continuing to be tracked.
    long_ok, long_checks = check_trend_alignment(candles_3m, candles_15m, "long")
    short_ok, short_checks = check_trend_alignment(candles_3m, candles_15m, "short")
    new_direction = "long" if long_ok else ("short" if short_ok else None)

    current_direction = coin_state.get("qualified_direction")
    if new_direction != current_direction:
        # Trend context changed (including losing qualification) -
        # discard any in-progress pullback run rather than carry it
        # forward into a context it was never validated against.
        coin_state["qualified_direction"] = new_direction
        coin_state["in_pullback_run"] = False

    if new_direction is None:
        return None

    report["direction"] = new_direction
    report["adx14"] = (long_checks if new_direction == "long" else short_checks).get("adx14")
    report["adx14_delta"] = (long_checks if new_direction == "long" else short_checks).get("adx14_delta")
    report["adx14_accel"] = (long_checks if new_direction == "long" else short_checks).get("adx14_accel")
    report["newly_qualified"] = current_direction != new_direction

    # Stage 2 + 3, per the locked experiment spec: track a run of ONE
    # OR MORE consecutive red candles (long case; mirrored for short).
    # The first candle that breaks that run by closing the opposite
    # color (green, for long) is the REVERSAL candle - report its RVOL
    # and stop. Deliberately NOT implemented, per explicit instruction:
    # no EMA9/21 touch requirement on the red or reversal candles, no
    # breakout-of-high requirement, no entry/stop/target, no RVOL
    # threshold, and nothing at all about the candle AFTER the
    # reversal - that candle is for manual observation only.
    pullback_color = "red" if new_direction == "long" else "green"
    reversal_color = "green" if new_direction == "long" else "red"
    is_pullback_candle = (row["close"] < row["open"]) if pullback_color == "red" else (row["close"] > row["open"])
    is_reversal_candle = (row["close"] > row["open"]) if reversal_color == "green" else (row["close"] < row["open"])

    in_pullback_run = coin_state.get("in_pullback_run", False)

    if is_pullback_candle:
        coin_state["in_pullback_run"] = True
        report["pullback_run_active"] = True
        return report

    if in_pullback_run and is_reversal_candle:
        # ROBUST DEDUP: even if the in_pullback_run flag-reset below
        # somehow didn't land in time (a state-persistence timing
        # issue - confirmed as the likely real cause after tracing the
        # decision logic itself and finding it should already have
        # prevented this), refuse to re-report the exact same candle
        # twice. This check is independent of the flag reset, so it
        # stays correct even if that specific write gets lost.
        if coin_state.get("last_reversal_candle_time") == candle_time:
            return None
        rvol = row.get("rvol")
        report["reversal_candle"] = True
        report["rvol"] = round(float(rvol), 2) if rvol is not None and not pd.isna(rvol) else None
        report["rvol_label"] = rvol_label(rvol)
        coin_percentiles = (rvol_percentiles or {}).get(coin)
        report["rvol_percentile"] = rvol_percentile_rank(rvol, coin_percentiles)
        coin_state["in_pullback_run"] = False
        coin_state["last_reversal_candle_time"] = candle_time
        return report

    # Neither a pullback candle nor (if mid-run) a reversal candle -
    # e.g. a doji, or a candle matching neither color while not in a
    # run yet. Per spec, only a genuine color flip after 1+ pullback
    # candles counts as reversal - anything else just doesn't extend
    # or resolve a run.
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", type=str, required=True)
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(",")]
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=FETCH_MINUTES_BACK)
    state = load_state()
    state.setdefault("coins", {})
    rvol_percentiles = load_rvol_percentiles()

    print(f"Trend alignment + reversal RVOL scanner | coins={coins} | {now.isoformat()}")

    qualified_report = []
    pullback_report = []
    reversal_report = []

    def fetch_one(coin):
        """Fetch + resample + drop the still-forming bucket - read-only,
        no shared state touched, safe to run concurrently. Indicator
        computation and process_coin (which mutates shared state) stay
        sequential afterward - they're fast, local, and not worth the
        added complexity/race risk of threading."""
        candles_1m = fetch_coindcx_klines(coin, "1m", start.date().isoformat(), now.isoformat(), stagger_delay=False)
        candles_3m = resample_candles(candles_1m, 3)
        candles_15m = resample_candles(candles_1m, 15)
        candles_3m = drop_still_forming_bucket(candles_3m, now, 3)
        candles_15m = drop_still_forming_bucket(candles_15m, now, 15)
        return coin, candles_3m, candles_15m

    fetched = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_one, coin): coin for coin in coins}
        for future in as_completed(futures):
            coin = futures[future]
            try:
                _, candles_3m, candles_15m = future.result()
                fetched[coin] = (candles_3m, candles_15m)
            except Exception as e:
                print(f"  {coin}: fetch failed ({e}), skipping")

    for coin in coins:
        try:
            if coin not in fetched:
                continue
            candles_3m, candles_15m = fetched[coin]
            if len(candles_3m) < 30 or len(candles_15m) < 10:
                print(f"  {coin}: not enough closed candles yet, skipping")
                continue

            candles_3m = compute_indicators(candles_3m)
            candles_15m = compute_indicators(candles_15m)

            coin_state = state["coins"].setdefault(coin, {"qualified_direction": None, "in_pullback_run": False})
            result = process_coin(coin, candles_3m, candles_15m, coin_state, rvol_percentiles)

            if result is None:
                print(f"  {coin}: not qualified")
                continue

            print(f"  {coin}: {result}")
            if result.get("reversal_candle"):
                reversal_report.append(result)
            elif result.get("pullback_run_active"):
                pullback_report.append(result)
            else:
                qualified_report.append(result)

        except Exception as e:
            print(f"  {coin}: ERROR - {e}")

    # Send once per NEW 3-minute candle close, as long as there's any
    # content at all - even if it's identical to the previous candle's
    # report. Only skip sending if this candle genuinely has zero
    # content (nothing qualified/tracked/reversed at all). Gated on
    # candle_time specifically (not content equality) so a repeat,
    # unchanged signal across a fresh candle still gets reported - the
    # earlier content-equality design suppressed those on purpose,
    # which is the opposite of what's wanted here.
    all_reports = qualified_report + pullback_report + reversal_report
    candle_times_utc = [pd.Timestamp(r["candle_time"]) for r in all_reports if r.get("candle_time")]
    current_candle_time = str(max(candle_times_utc)) if candle_times_utc else None
    has_content = bool(all_reports)
    should_send = has_content and current_candle_time != state.get("last_sent_candle_time")

    if should_send:
        candle_start_utc = max(candle_times_utc)
        candle_start_ist = candle_start_utc + timedelta(hours=5, minutes=30)
        candle_close_ist = candle_start_ist + timedelta(minutes=3)
        detected_ist = now + timedelta(hours=5, minutes=30)

        lines = [f"\U0001F4CA {candle_start_ist.strftime('%H:%M')}\u2192{candle_close_ist.strftime('%H:%M')} IST "
                 f"(detected {detected_ist.strftime('%H:%M:%S')})"]
        if qualified_report:
            lines.append("\nQualified:")
            for r in qualified_report:
                accel = r.get("adx14_accel")
                accel_arrow = " \u2191" if accel and accel > 0.3 else (" \u2193" if accel and accel < -0.3 else "")
                lines.append(f"  {r['coin']} {r['direction'].upper()} - ADX {r['adx14']} ({r['adx14_delta']:+.1f}{accel_arrow})"
                              + (" \u2022 new" if r.get("newly_qualified") else ""))
        if pullback_report:
            lines.append("\nPullback:")
            for r in pullback_report:
                lines.append(f"  {r['coin']} {r['direction'].upper()}")
        if reversal_report:
            lines.append("\n\U0001F3AF Reversal:")
            for r in reversal_report:
                pct = r.get("rvol_percentile")
                rvol_display = f"{pct:.0f}th %ile" if pct is not None else r["rvol_label"]
                accel = r.get("adx14_accel")
                accel_arrow = " \u2191" if accel and accel > 0.3 else (" \u2193" if accel and accel < -0.3 else "")
                adx_str = f" - ADX {r['adx14']} ({r['adx14_delta']:+.1f}{accel_arrow})" if r.get("adx14") is not None else ""
                lines.append(f"  {r['coin']} {r['direction'].upper()} - RVOL {r['rvol']} ({rvol_display}){adx_str}")
        message = "\n".join(lines)

        # Per-coin "View Chart" buttons for reversal-stage coins - the
        # one genuinely interactive element available: Telegram Bot API
        # supports URL buttons with no bot process needed to handle
        # them (unlike callback buttons, which would need a persistent
        # listener - this scanner is a one-shot script, not a running
        # bot, so callback-driven buttons aren't feasible here).
        # Points to CoinDCX's own real futures page - confirmed live
        # via direct fetch (not guessed), same B-{coin}_USDT pair
        # naming already used internally by coindcx_fetcher.py.
        reply_markup = None
        if reversal_report:
            reply_markup = {"inline_keyboard": [
                [{"text": f"\U0001F4C8 {r['coin']} chart", "url": f"https://coindcx.com/futures/B-{r['coin']}_USDT"}]
                for r in reversal_report
            ]}

        print(f"\nSending Telegram message:\n{message}")
        send_telegram(message, reply_markup)
        state["last_sent_candle_time"] = current_candle_time
    else:
        print(f"\nNot sending (has_content={has_content}, "
              f"same_candle_already_sent={current_candle_time == state.get('last_sent_candle_time')})")

    save_state(state)


if __name__ == "__main__":
    main()
