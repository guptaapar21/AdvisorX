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
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from coindcx_fetcher import fetch_coindcx_klines, resample_candles

STATE_FILE = "trend_scanner_state.json"
# Trimmed from 150 to 100 15m candles after measuring the real fetch
# cost at 18 coins - 150 candles needed 3 API requests/coin (2250 min
# of 1m data), 100 needs only 2 (1500 min), saving ~14.4s across all 18
# coins. Still 10x the bare minimum (10) actually checked in the code
# below, real margin for EMA21/ADX14 to be stable, not just non-error.
FETCH_MINUTES_BACK = 15 * 100
MESSAGE_INTERVAL_MINUTES = 3


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"coins": {}, "last_sent_at": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
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


def check_trend_alignment(candles_3m, candles_15m, direction, trend_lookback=3):
    if len(candles_3m) < 30 or len(candles_15m) < trend_lookback + 5:
        return False, {}

    row3 = candles_3m.iloc[-1]
    ema9_prev3 = candles_3m["ema9"].iloc[-(trend_lookback + 1)]
    adx_prev1 = candles_3m["adx14"].iloc[-2]

    row15 = candles_15m.iloc[-1]
    ema9_15_prev3 = candles_15m["ema9"].iloc[-(trend_lookback + 1)]

    adx = row3["adx14"]
    if pd.isna(adx):
        return False, {}

    checks = {
        "adx14": round(float(adx), 2),
        "adx_gate_passed": bool(adx >= 25 and adx > adx_prev1),
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

    all_passed = all(v for k, v in checks.items() if k not in ("adx14",))
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


def process_coin(coin, candles_3m, candles_15m, coin_state):
    """Runs all 3 stages for one coin, mutating coin_state in place.
    Returns a dict of anything worth reporting this cycle, or None."""
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

    print(f"Trend alignment + reversal RVOL scanner | coins={coins} | {now.isoformat()}")

    qualified_report = []
    pullback_report = []
    reversal_report = []

    for coin in coins:
        try:
            candles_1m = fetch_coindcx_klines(coin, "1m", start.date().isoformat(), now.isoformat(), stagger_delay=False)
            candles_3m = resample_candles(candles_1m, 3)
            candles_15m = resample_candles(candles_1m, 15)
            candles_3m = drop_still_forming_bucket(candles_3m, now, 3)
            candles_15m = drop_still_forming_bucket(candles_15m, now, 15)
            if len(candles_3m) < 30 or len(candles_15m) < 10:
                print(f"  {coin}: not enough closed candles yet, skipping")
                continue

            candles_3m = compute_indicators(candles_3m)
            candles_15m = compute_indicators(candles_15m)

            coin_state = state["coins"].setdefault(coin, {"qualified_direction": None, "in_pullback_run": False})
            result = process_coin(coin, candles_3m, candles_15m, coin_state)

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

    # Decide whether it's time to actually send a Telegram message
    last_sent = state.get("last_sent_at")
    should_send = last_sent is None
    if last_sent is not None:
        last_sent_dt = datetime.fromisoformat(last_sent)
        if last_sent_dt.tzinfo is None:
            last_sent_dt = last_sent_dt.replace(tzinfo=timezone.utc)
        should_send = (now - last_sent_dt) >= timedelta(minutes=MESSAGE_INTERVAL_MINUTES)

    if should_send and (qualified_report or pullback_report or reversal_report):
        # All candle_time values in one scan cycle come from the same
        # fetch, so they're the same 3m boundary across coins - shown
        # once at the top instead of repeated per-line. Converted from
        # UTC (the raw candle index) to IST for direct readability -
        # previously shown as raw UTC with no label, which looked
        # "wrong" only because it was being silently compared against
        # Telegram's own local-time message stamp.
        all_reports = qualified_report + pullback_report + reversal_report
        candle_times_utc = [pd.Timestamp(r["candle_time"]) for r in all_reports if r.get("candle_time")]
        candle_time_ist = None
        if candle_times_utc:
            latest_utc = max(candle_times_utc)
            candle_time_ist = (latest_utc + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M IST")

        lines = ["\U0001F4CA Trend Alignment + Reversal RVOL Scanner"]
        if candle_time_ist:
            lines.append(f"As of candle: {candle_time_ist}\n")
        if qualified_report:
            lines.append("Qualified (trend-aligned, no pullback run yet):")
            for r in qualified_report:
                lines.append(f"  {r['coin']} {r['direction'].upper()} - ADX14: {r['adx14']}"
                              + (" (newly qualified)" if r.get("newly_qualified") else ""))
        if pullback_report:
            lines.append("\nPullback run in progress:")
            for r in pullback_report:
                lines.append(f"  {r['coin']} {r['direction'].upper()}")
        if reversal_report:
            lines.append("\n\U0001F3AF Reversal candle formed - RVOL (for observation only, not an entry signal):")
            for r in reversal_report:
                lines.append(f"  {r['coin']} {r['direction'].upper()} - RVOL: {r['rvol']} ({r['rvol_label']})")
        message = "\n".join(lines)
        print(f"\nSending Telegram message:\n{message}")
        send_telegram(message)
        state["last_sent_at"] = now.isoformat()
    else:
        print(f"\nNot sending yet (should_send={should_send}, "
              f"has_content={bool(qualified_report or pullback_report or reversal_report)})")

    save_state(state)


if __name__ == "__main__":
    main()
