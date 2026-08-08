"""
Live signal monitor for Idea #18F (rejection-structure + EMA/VWAP
slope + ADX). This is NOT a backtest and NOT an execution tool - it
only WATCHES live candles and pings Telegram when the exact same entry
conditions from run_ema_pullback_backtest_18f_rejection.py fire on a
real, just-closed candle. Advisory only, matching this project's
existing "advisory only" pattern (agent.yml, fastwatch.yml) - it never
places an order.

ENTRY LOGIC, ported faithfully from the uploaded 18F backtest (same
formulas, same structural stop/target, same ADX/DI computation) - not
a simplified approximation:
  - EMA9/EMA21 trend alignment + 3-bar slope confirmation
  - Daily-reset (UTC) session VWAP, price beyond it in trend direction
  - A falling-then-rejecting (or rising-then-rejecting) candle pair at
    the EMA9/21 zone
  - ADX14 threshold + selectable confirmation mode (level_only / di /
    slope1 / slope3 / full_slope1 / full_slope3)
  - Structural stop (extreme of the pullback candle) and structural
    target (prior N-bar high/low, excluding the signal bar itself)
  - Minimum stop-distance and minimum structural R:R filters

WHAT'S DIFFERENT FROM THE BACKTEST, BY NECESSITY: real entry happens
at the NEXT 3m candle's open, which doesn't exist yet at alert time.
This script estimates risk/reward/stop_pct/RR using the just-closed
signal candle's OWN close as a stand-in for the (unknown) next open -
a real, stated approximation, not a hidden one. The Telegram message
says so explicitly, and flags that real numbers may shift slightly by
the time of actual entry.

DUPLICATE-ALERT PREVENTION: each poll only evaluates the most recently
CLOSED 3m candle (using a real closed-bucket check, not just "the last
row" - a still-forming bucket is explicitly dropped, same discipline
used throughout this project's backtests). A small JSON state file
(committed back to the repo) tracks the last candle timestamp already
alerted per coin, so re-running before a new candle forms doesn't
re-send the same alert.

Usage:
  python3 live_signal_monitor_18f.py --coins BTC,ETH,SOL --adx-min 25 --adx-mode level_only
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from coindcx_fetcher import fetch_coindcx_klines, resample_candles

STATE_FILE = "live_monitor_state.json"
FETCH_MINUTES_BACK = 3 * 500  # enough 1m candles for 3m resample + full EMA21/ADX14 warmup with real margin

# BZ/BLESS/XAU/XAG/CL/NATGAS all failed under the default
# B-{symbol}_USDT format. PAXG (a real crypto token, gold-backed)
# succeeded under that same default format, which is real evidence
# XAU/XAG/CL/NATGAS aren't a symbol-naming issue - they're very likely
# a different CoinDCX product line entirely (synthetic commodity/CFD
# contracts, not standard crypto futures), so no attempt is made to
# guess an alternate for those four here.
#
# BZ/BLESS remain a more plausible symbol-format case since they're
# real crypto tokens - but I don't have a specific, well-grounded
# alternate string to propose (no live network access to verify
# against, and no real basis to prefer one guessed prefix over
# another). Rather than plug in a guess dressed up as a fix, this map
# is left empty and ready to use: once you have the correct pair
# string for either symbol (from CoinDCX support/docs, or by finding
# it some other way), add it here as e.g. {"BZ": "<correct pair
# string>"} and it will be used automatically.
PAIR_OVERRIDES = {}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
    resp.raise_for_status()


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
    candles["plus_di14"] = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr14.replace(0, float("nan"))
    candles["minus_di14"] = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr14.replace(0, float("nan"))
    dx = 100 * (candles["plus_di14"] - candles["minus_di14"]).abs() / (candles["plus_di14"] + candles["minus_di14"]).replace(0, float("nan"))
    candles["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return candles


def drop_still_forming_bucket(candles_3m, now):
    """Same closed-bucket discipline used throughout this project's
    backtests - a 3m bucket only counts as closed once bucket_start +
    3 minutes <= now, not merely because it's the last row returned.
    `now` must be tz-NAIVE to compare against the candle index, which
    is tz-naive throughout this project's convention - confirmed
    directly that comparing a tz-aware `now` against this index raises
    immediately ("can't compare offset-naive and offset-aware
    datetimes"), which would have crashed every single run."""
    if len(candles_3m) == 0:
        return candles_3m
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    last_bucket_start = candles_3m.index[-1]
    if last_bucket_start + pd.Timedelta(minutes=3) > now:
        return candles_3m.iloc[:-1]
    return candles_3m


def prior_structure_target(candles, signal_i, direction, lookback=10):
    start = max(0, signal_i - lookback)
    prior = candles.iloc[start:signal_i]
    if prior.empty:
        return None
    return float(prior["high"].max()) if direction == "long" else float(prior["low"].min())


def check_signal(candles, adx_min, adx_mode, trend_lookback=3,
                  min_stop_distance_pct=0.25, target_lookback=10, min_structural_rr=1.0):
    """Evaluates the signal on the LAST row of `candles` (the caller is
    responsible for ensuring this is a real, fully-closed candle - see
    drop_still_forming_bucket).

    Returns (signal_dict_or_None, diagnostics_dict). diagnostics always
    reports every condition checked and whether it passed, so a "no
    signal" outcome is never a black box - the exact blocking condition
    is always visible, not just inferred or asserted."""
    diag = {}
    i = len(candles) - 1
    if i < max(50, target_lookback + 2, trend_lookback + 2):
        diag["blocked_at"] = "not_enough_candles"
        return None, diag

    row = candles.iloc[i]
    prev = candles.iloc[i - 1]

    adx = row["adx14"]
    adx_ok = not pd.isna(adx) and adx >= adx_min
    diag["adx14"] = None if pd.isna(adx) else round(float(adx), 2)
    diag["adx_min_required"] = adx_min
    diag["adx_gate_passed"] = adx_ok
    if not adx_ok:
        diag["blocked_at"] = "adx_below_minimum"
        return None, diag

    plus_di, minus_di = row["plus_di14"], row["minus_di14"]
    adx_prev1 = candles["adx14"].iloc[i - 1]
    adx_prev2 = candles["adx14"].iloc[i - 2]
    adx_prev3 = candles["adx14"].iloc[i - 3]
    adx_rising_1 = adx > adx_prev1
    adx_rising_2 = adx > adx_prev2
    adx_rising_3 = adx > adx_prev3
    adx_rising_1_and_2 = adx_rising_1 and adx_rising_2
    diag["adx_rising_vs_1candle_ago"] = bool(adx_rising_1)
    diag["adx_rising_vs_2candles_ago"] = bool(adx_rising_2)
    diag["adx_rising_vs_3candles_ago"] = bool(adx_rising_3)

    ema9, ema21, vwap = row["ema9"], row["ema21"], row["vwap"]
    ema9_prev3 = candles["ema9"].iloc[i - 3]
    ema21_prev3 = candles["ema21"].iloc[i - 3]
    vwap_prev3 = candles["vwap"].iloc[i - 3]

    long_trend_checks = {
        "ema9_above_ema21": ema9 > ema21,
        "ema9_rising_vs_3candles_ago": ema9 > ema9_prev3,
        "ema21_not_falling_vs_3candles_ago": ema21 >= ema21_prev3,
        "vwap_not_falling_vs_3candles_ago": vwap >= vwap_prev3,
        "price_above_vwap": row["close"] > vwap,
    }
    short_trend_checks = {
        "ema9_below_ema21": ema9 < ema21,
        "ema9_falling_vs_3candles_ago": ema9 < ema9_prev3,
        "ema21_not_rising_vs_3candles_ago": ema21 <= ema21_prev3,
        "vwap_not_rising_vs_3candles_ago": vwap <= vwap_prev3,
        "price_below_vwap": row["close"] < vwap,
    }
    long_trend = all(long_trend_checks.values())
    short_trend = all(short_trend_checks.values())
    diag["long_trend_checks"] = long_trend_checks
    diag["short_trend_checks"] = short_trend_checks
    diag["long_trend_passed"] = long_trend
    diag["short_trend_passed"] = short_trend

    prev_falling = prev["close"] < prev["open"]
    prev_rising = prev["close"] > prev["open"]
    zone_lo, zone_hi = min(ema9, ema21), max(ema9, ema21)
    touched_zone = row["low"] <= zone_hi and row["high"] >= zone_lo
    diag["touched_ema_zone"] = bool(touched_zone)

    bullish_rejection_checks = {
        "touched_zone": touched_zone,
        "closed_green": row["close"] > row["open"],
        "closed_at_or_above_ema9": row["close"] >= ema9,
        "prior_candle_was_falling": prev_falling,
        "low_did_not_undercut_prior_low": row["low"] >= prev["low"],
    }
    bearish_rejection_checks = {
        "touched_zone": touched_zone,
        "closed_red": row["close"] < row["open"],
        "closed_at_or_below_ema9": row["close"] <= ema9,
        "prior_candle_was_rising": prev_rising,
        "high_did_not_overshoot_prior_high": row["high"] <= prev["high"],
    }
    bullish_rejection = all(bullish_rejection_checks.values())
    bearish_rejection = all(bearish_rejection_checks.values())
    diag["bullish_rejection_checks"] = bullish_rejection_checks
    diag["bearish_rejection_checks"] = bearish_rejection_checks

    direction = "long" if (long_trend and bullish_rejection) else ("short" if (short_trend and bearish_rejection) else None)
    diag["direction"] = direction
    if direction is None:
        diag["blocked_at"] = ("no_valid_trend" if not (long_trend or short_trend)
                               else "trend_ok_but_no_rejection_candle")
        return None, diag

    di_ok = (plus_di > minus_di) if direction == "long" else (minus_di > plus_di)
    diag["di_direction_agrees"] = bool(di_ok)
    mode_ok = {
        "level_only": True, "di": di_ok, "slope1": adx_rising_1, "slope3": adx_rising_3,
        "slope1_2": adx_rising_1_and_2, "full_slope1_2": di_ok and adx_rising_1_and_2,
        "full_slope1": di_ok and adx_rising_1, "full_slope3": di_ok and adx_rising_3,
    }.get(adx_mode)
    diag["adx_mode"] = adx_mode
    diag["adx_mode_gate_passed"] = bool(mode_ok)
    if not mode_ok:
        diag["blocked_at"] = "adx_mode_gate_failed"
        return None, diag

    stop = float(prev["low"]) if direction == "long" else float(prev["high"])
    target = prior_structure_target(candles, i, direction, target_lookback)
    diag["stop"] = stop
    diag["target"] = target
    if target is None:
        diag["blocked_at"] = "no_structural_target_found"
        return None, diag

    # Real entry executes at the NEXT candle's open, which doesn't exist
    # yet - this uses the signal candle's own close as an estimate,
    # stated explicitly rather than hidden. Real numbers may shift
    # slightly by actual entry time.
    estimated_entry = float(row["close"])
    risk = (estimated_entry - stop) if direction == "long" else (stop - estimated_entry)
    reward = (target - estimated_entry) if direction == "long" else (estimated_entry - target)
    stop_pct = risk / estimated_entry * 100 if estimated_entry else 0
    rr = reward / risk if risk > 0 else -1
    diag["estimated_entry"] = estimated_entry
    diag["risk"] = round(risk, 6)
    diag["reward"] = round(reward, 6)
    diag["stop_pct"] = round(stop_pct, 3)
    diag["rr"] = round(rr, 3)
    diag["min_stop_distance_pct_required"] = min_stop_distance_pct
    diag["min_structural_rr_required"] = min_structural_rr

    if risk <= 0 or reward <= 0:
        diag["blocked_at"] = "non_positive_risk_or_reward"
        return None, diag
    if stop_pct < min_stop_distance_pct:
        diag["blocked_at"] = "stop_too_tight"
        return None, diag
    if rr < min_structural_rr:
        diag["blocked_at"] = "rr_below_minimum"
        return None, diag

    diag["blocked_at"] = None
    return {
        "direction": direction, "signal_time": candles.index[i], "estimated_entry": estimated_entry,
        "stop": stop, "target": target, "stop_pct": round(stop_pct, 3), "rr": round(rr, 2),
        "adx14": round(float(adx), 1),
    }, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", type=str, default="BTC,ETH,BNB,SOL,XRP,DOGE,LTC,LINK,TRX,AVAX")
    parser.add_argument("--adx-min", type=float, default=25.0)
    parser.add_argument("--adx-mode", type=str, default="level_only",
                         choices=["level_only", "di", "slope1", "slope3", "full_slope1", "full_slope3",
                                  "slope1_2", "full_slope1_2"])
    parser.add_argument("--min-stop-pct", type=float, default=0.25)
    parser.add_argument("--target-lookback", type=int, default=10)
    parser.add_argument("--min-structural-rr", type=float, default=1.0)
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(",")]
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(minutes=FETCH_MINUTES_BACK)
    state = load_state()

    print(f"Idea #18F live monitor | coins={coins} | adx_min={args.adx_min}, mode={args.adx_mode} | {now.isoformat()}")

    for coin in coins:
        try:
            candles_1m = fetch_coindcx_klines(coin, "1m", start_date.date().isoformat(), now.isoformat(),
                                               stagger_delay=False, pair_override=PAIR_OVERRIDES.get(coin))
            candles_3m = resample_candles(candles_1m, 3)
            candles_3m = drop_still_forming_bucket(candles_3m, now)
            if len(candles_3m) < 60:
                print(f"  {coin}: not enough closed 3m candles yet ({len(candles_3m)}), skipping")
                continue

            candles_3m = compute_indicators(candles_3m)
            signal, diag = check_signal(candles_3m, args.adx_min, args.adx_mode, min_stop_distance_pct=args.min_stop_pct,
                                         target_lookback=args.target_lookback, min_structural_rr=args.min_structural_rr)

            if signal is None:
                print(f"  {coin}: no signal - blocked_at={diag.get('blocked_at')}")
                for key, value in diag.items():
                    if key == "blocked_at":
                        continue
                    print(f"    {key}: {value}")
                continue

            signal_key = f"{coin}_{signal['signal_time'].isoformat()}"
            if state.get(coin) == signal_key:
                print(f"  {coin}: signal already alerted for this candle, skipping duplicate")
                continue

            direction_word = "LONG" if signal["direction"] == "long" else "SHORT"
            message = (
                f"🎯 Idea #18F signal: {coin} {direction_word}\n"
                f"Signal candle closed: {signal['signal_time']}\n"
                f"Est. entry (~next candle open): {signal['estimated_entry']:.6g}\n"
                f"Stop: {signal['stop']:.6g} ({signal['stop_pct']}%)\n"
                f"Target: {signal['target']:.6g} (structural R:R {signal['rr']})\n"
                f"ADX14: {signal['adx14']}\n"
                f"Note: real entry executes at the NEXT 3m candle's open - these numbers "
                f"are estimated from the signal candle's own close and may shift slightly."
            )
            print(f"  {coin}: SIGNAL FIRED -> {direction_word}, sending Telegram alert")
            send_telegram(message)
            state[coin] = signal_key

        except Exception as e:
            print(f"  {coin}: ERROR - {e}")

    save_state(state)


if __name__ == "__main__":
    main()
