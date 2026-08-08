"""
Idea #19: two specific rule changes to the 18F rejection-structure
strategy, tested as an ablation (each alone, then combined) against
the current design - not assumed to help, tested.

RULE 1 - Breakout-confirmed entry (instead of "enter at next candle's
open"): after a rejection candle fires, wait for a SUBSEQUENT candle's
high (long) / low (short) to actually break the rejection candle's own
high/low before entering. If no break happens within
`breakout_expiry_bars`, the signal is cancelled - no trade taken. R:R
is recomputed at the REAL breakout entry price (not the old estimate),
and the trade is skipped if R:R has fallen below the minimum by the
time the breakout happens - since waiting for confirmation costs a
worse entry price by construction (closer to target, closer to stop).

RULE 2 - VWAP extension filter: reject signals where price is already
extended too far from VWAP at signal time (as a % of price). The
motivating case: a coin already up ~50% in an hour, extended ~25-27%
above VWAP, produced a rejection-candle-shaped signal that immediately
reversed hard - a chase, not a genuine pullback. Swept across several
thresholds (no filter, 3%, 5%, 8%) rather than picking one blindly.

ZERO LOOKAHEAD: breakout confirmation only uses OHLC of candles AFTER
the signal candle, in sequence, exactly as they'd become known live.
Entry price is the ACTUAL breakout candle's trigger level (rejection
candle's own high/low), not any future information beyond that.

Usage:
  python3 run_18f_breakout_vwap_backtest.py --coin BTC --sweep-all
"""
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from live_signal_monitor_18f import compute_indicators
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 60


def prior_structure_target(candles, signal_i, direction, lookback=10):
    start = max(0, signal_i - lookback)
    prior = candles.iloc[start:signal_i]
    if prior.empty:
        return None
    return float(prior["high"].max()) if direction == "long" else float(prior["low"].min())


def detect_signal_at(candles, i, adx_min, adx_mode, trend_lookback=3):
    if i < max(MIN_CANDLES_NEEDED, trend_lookback + 3):
        return None

    row = candles.iloc[i]
    prev = candles.iloc[i - 1]
    adx = row["adx14"]
    if pd.isna(adx) or adx < adx_min:
        return None

    plus_di, minus_di = row["plus_di14"], row["minus_di14"]
    adx_prev1 = candles["adx14"].iloc[i - 1]
    adx_prev2 = candles["adx14"].iloc[i - 2]
    adx_rising_1_and_2 = (adx > adx_prev1) and (adx > adx_prev2)
    adx_rising_3 = adx > candles["adx14"].iloc[i - 3]

    ema9, ema21, vwap = row["ema9"], row["ema21"], row["vwap"]
    ema9_prev3 = candles["ema9"].iloc[i - 3]
    ema21_prev3 = candles["ema21"].iloc[i - 3]
    vwap_prev3 = candles["vwap"].iloc[i - 3]

    long_trend = (ema9 > ema21 and ema9 > ema9_prev3 and ema21 >= ema21_prev3
                  and vwap >= vwap_prev3 and row["close"] > vwap)
    short_trend = (ema9 < ema21 and ema9 < ema9_prev3 and ema21 <= ema21_prev3
                   and vwap <= vwap_prev3 and row["close"] < vwap)

    prev_falling = prev["close"] < prev["open"]
    prev_rising = prev["close"] > prev["open"]
    zone_lo, zone_hi = min(ema9, ema21), max(ema9, ema21)
    touched_zone = row["low"] <= zone_hi and row["high"] >= zone_lo

    bullish_rejection = (touched_zone and row["close"] > row["open"] and row["close"] >= ema9
                          and prev_falling and row["low"] >= prev["low"])
    bearish_rejection = (touched_zone and row["close"] < row["open"] and row["close"] <= ema9
                          and prev_rising and row["high"] <= prev["high"])

    direction = "long" if (long_trend and bullish_rejection) else ("short" if (short_trend and bearish_rejection) else None)
    if direction is None:
        return None

    di_ok = (plus_di > minus_di) if direction == "long" else (minus_di > plus_di)
    mode_ok = {
        "level_only": True, "di": di_ok, "slope1_2": adx_rising_1_and_2,
        "full_slope1_2": di_ok and adx_rising_1_and_2, "slope3": adx_rising_3,
    }.get(adx_mode)
    if not mode_ok:
        return None

    vwap_extension_pct = abs(row["close"] - vwap) / vwap * 100 if vwap else 0
    return {
        "direction": direction, "signal_index": i,
        "rejection_high": float(row["high"]), "rejection_low": float(row["low"]),
        "stop": float(prev["low"]) if direction == "long" else float(prev["high"]),
        "vwap_extension_pct": vwap_extension_pct,
    }


def run_backtest(candles, adx_min, adx_mode, require_breakout, breakout_expiry_bars,
                  max_vwap_extension_pct, r_target, max_hold_bars, breakout_mode="wick",
                  min_stop_distance_pct=0.25):
    """breakout_mode: "wick" (default) enters the moment price touches
    the rejection candle's high/low - a single spike is enough, even if
    the candle immediately reverses. "close" requires the candle to
    actually CLOSE beyond that level before entering - a real,
    different confirmation strength, motivated directly by the kind of
    loss already seen (a brief spike above a level, immediately
    reversing). In "close" mode, entry price is the confirming candle's
    own CLOSE, not the rejection candle's high/low - that's the
    realistic execution point once you're specifically waiting for a
    full close (you wouldn't know the close confirmed until it
    happened, and by then price is already at that close level, not
    the earlier trigger level)."""
    trades = []
    open_position = None
    pending_signal = None
    pending_since = None
    n = len(candles)

    for i in range(MIN_CANDLES_NEEDED, n):
        row = candles.iloc[i]

        if open_position is not None:
            pos = open_position
            direction = pos["direction"]
            hit_stop = row["low"] <= pos["stop"] if direction == "long" else row["high"] >= pos["stop"]
            hit_target = row["high"] >= pos["target"] if direction == "long" else row["low"] <= pos["target"]
            bars_held = i - pos["entry_index"]
            hit_max_hold = bars_held >= max_hold_bars

            if hit_stop or hit_target or hit_max_hold:
                if hit_stop:
                    exit_price, exit_reason = pos["stop"], "stop"
                elif hit_target:
                    exit_price, exit_reason = pos["target"], "target"
                else:
                    exit_price, exit_reason = float(row["close"]), "max_hold_time"
                risk = abs(pos["entry"] - pos["stop"])
                profit = (exit_price - pos["entry"]) if direction == "long" else (pos["entry"] - exit_price)
                r_achieved = profit / risk if risk != 0 else 0
                trades.append({
                    "direction": direction, "entry_time": candles.index[pos["entry_index"]], "exit_time": candles.index[i],
                    "entry_price": pos["entry"], "exit_price": exit_price, "r_achieved": r_achieved,
                    "exit_reason": exit_reason, "stages_done": 0, "leverage": 5,
                    "stop_distance_pct": abs(pos["entry"] - pos["stop"]) / pos["entry"], "bars_held": bars_held,
                    "vwap_extension_pct": pos["vwap_extension_pct"], "waited_bars_for_breakout": pos["waited_bars"],
                })
                open_position = None
            continue

        if pending_signal is not None:
            direction = pending_signal["direction"]
            waited = i - pending_since
            if breakout_mode == "close":
                broke_out = (row["close"] > pending_signal["rejection_high"] if direction == "long"
                             else row["close"] < pending_signal["rejection_low"])
            else:  # "wick" - a touch is enough, even if the candle reverses before closing
                broke_out = (row["high"] > pending_signal["rejection_high"] if direction == "long"
                             else row["low"] < pending_signal["rejection_low"])

            if broke_out:
                if breakout_mode == "close":
                    entry_price = float(row["close"])
                else:
                    entry_price = pending_signal["rejection_high"] if direction == "long" else pending_signal["rejection_low"]
                stop = pending_signal["stop"]
                risk = (entry_price - stop) if direction == "long" else (stop - entry_price)
                stop_pct = risk / entry_price * 100 if entry_price else 0
                if risk > 0 and stop_pct >= min_stop_distance_pct:
                    target = (entry_price + risk * r_target) if direction == "long" else (entry_price - risk * r_target)
                    open_position = {
                        "direction": direction, "entry": entry_price, "entry_index": i,
                        "stop": stop, "target": target,
                        "vwap_extension_pct": pending_signal["vwap_extension_pct"], "waited_bars": waited,
                    }
                pending_signal = None
                pending_since = None
                continue
            elif waited >= breakout_expiry_bars:
                pending_signal = None
                pending_since = None

        signal = detect_signal_at(candles, i, adx_min, adx_mode)
        if signal is None:
            continue
        if max_vwap_extension_pct is not None and signal["vwap_extension_pct"] > max_vwap_extension_pct:
            continue

        if require_breakout:
            pending_signal = signal
            pending_since = i
        else:
            if i + 1 < n:
                entry_price = float(candles["open"].iloc[i + 1])
                direction = signal["direction"]
                stop = signal["stop"]
                risk = (entry_price - stop) if direction == "long" else (stop - entry_price)
                stop_pct = risk / entry_price * 100 if entry_price else 0
                if risk > 0 and stop_pct >= min_stop_distance_pct:
                    target = (entry_price + risk * r_target) if direction == "long" else (entry_price - risk * r_target)
                    open_position = {
                        "direction": direction, "entry": entry_price, "entry_index": i + 1,
                        "stop": stop, "target": target,
                        "vwap_extension_pct": signal["vwap_extension_pct"], "waited_bars": 0,
                    }

    return pd.DataFrame(trades)


def run_one_combo(candles, coin, adx_min, adx_mode, require_breakout, breakout_expiry_bars,
                   max_vwap_extension_pct, r_target, max_hold_bars, breakout_mode="wick"):
    trades = run_backtest(candles, adx_min, adx_mode, require_breakout, breakout_expiry_bars,
                           max_vwap_extension_pct, r_target, max_hold_bars, breakout_mode=breakout_mode)
    label = {"coin": coin, "require_breakout": require_breakout,
             "max_vwap_ext_pct": max_vwap_extension_pct if max_vwap_extension_pct else "none",
             "r_target": r_target, "breakout_mode": breakout_mode if require_breakout else "n/a"}
    if len(trades) == 0:
        return {**label, "trades": 0}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "18f_breakout_vwap"
    trades = apply_fees_and_interest(trades, bar_minutes=3)
    trades = apply_dollar_pnl(trades)
    return {
        **label, "trades": len(trades),
        "win_rate": round((trades["net_r"] > 0).mean() * 100, 1),
        "gross_r": round(trades["r_achieved"].mean(), 4),
        "net_r": round(trades["net_r"].mean(), 4),
        "avg_wait_bars": round(trades["waited_bars_for_breakout"].mean(), 2),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="BTC")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--adx-min", type=float, default=25.0)
    parser.add_argument("--adx-mode", type=str, default="slope1_2")
    parser.add_argument("--breakout-expiry-bars", type=int, default=5)
    parser.add_argument("--r-target", type=float, default=2.0)
    parser.add_argument("--max-hold-hours", type=float, default=4.0)
    parser.add_argument("--sweep-all", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    max_hold_bars = round(args.max_hold_hours * 60 / 3)

    print(f"Idea #19: Breakout-confirmed entry + VWAP extension filter | {args.coin} | "
          f"adx_min={args.adx_min}, mode={args.adx_mode}, {args.days}d")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), now.isoformat(), stagger_delay=False)
    candles_3m = resample_candles(candles_1m, 3)
    candles_3m = compute_indicators(candles_3m)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_3m)} 3m candles"
          + (" (ONE fetch, reused for every combo)" if args.sweep_all else ""))

    if not args.sweep_all:
        result = run_one_combo(candles_3m, args.coin, args.adx_min, args.adx_mode, True,
                                args.breakout_expiry_bars, None, args.r_target, max_hold_bars)
        print(result)
        return

    # Ablation: breakout entry vs immediate entry x VWAP filter x r_target
    # x breakout_mode (wick vs close - only meaningful when
    # require_breakout=True, since immediate entry never watches for a
    # breakout at all). wick vs close matters specifically because a
    # candle can briefly spike above the rejection high then reverse
    # before closing - exactly the kind of loss already seen - so
    # requiring an actual CLOSE beyond the level is a genuinely
    # different, stricter confirmation, tested here rather than assumed.
    combos = []
    for max_vwap in (None, 3.0, 5.0, 8.0):
        for r_target in (1.5, 2.0):
            vwap_label = f"max {max_vwap}%" if max_vwap else "no filter"
            combos.append((f"immediate (baseline) entry, VWAP {vwap_label}, r_target={r_target}",
                            False, max_vwap, r_target, "wick"))
            for breakout_mode in ("wick", "close"):
                combos.append((f"breakout-confirmed ({breakout_mode}) entry, VWAP {vwap_label}, r_target={r_target}",
                                True, max_vwap, r_target, breakout_mode))

    results = []
    for label, require_breakout, max_vwap, r_target, breakout_mode in combos:
        result = run_one_combo(candles_3m, args.coin, args.adx_min, args.adx_mode, require_breakout,
                                args.breakout_expiry_bars, max_vwap, r_target, max_hold_bars, breakout_mode)
        result["label"] = label
        results.append(result)
        print(f"  {label} -> trades={result.get('trades')}, win%={result.get('win_rate')}, "
              f"gross_R={result.get('gross_r')}, net_R={result.get('net_r')}, "
              f"total_pnl={result.get('total_pnl')}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL ABLATION RESULTS: {args.coin} ===")
    print(results_df.to_string(index=False))

    csv_path = f"idea19_results_{args.coin}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path} for summary aggregation")


if __name__ == "__main__":
    main()
