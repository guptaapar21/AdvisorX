"""
Idea #18: VWAP + 9/21 EMA pullback-rejection entries (from the
web-search trading guide, adapted for crypto), tested on BTC as
requested.

TIMING, DELIBERATELY DIFFERENT FROM EVERY PRIOR IDEA IN THIS PROJECT:
every prior idea decided AND entered using the same bar's own close
("decide at bar i's close, enter at bar i's close" - the established
zero-lookahead convention throughout this project). This strategy is
different by design: the guide specifies "enter on the OPEN of the
NEXT candle after a rejection candle forms" - a genuine one-candle
delay between signal and entry.

ZERO LOOKAHEAD, PRESERVED ACROSS THIS DELAY: the signal is detected
using ONLY bar i's own close/high/low/open (all known once bar i
closes). Entry then uses bar i+1's OWN OPEN PRICE ONLY - never bar
i+1's close, high, or low, which aren't known until bar i+1 itself
closes. Using a bar's own open the moment that bar begins is standard,
legitimate "enter at next candle's open" execution, not lookahead.

Usage:
  python3 run_ema_pullback_backtest.py --coin BTC --r-target 1.5 --sweep-all
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import detect_ema_pullback_rejection
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 40
LOOKBACK_WINDOW_BARS = 100


def run_backtest(candles, r_target=1.5, max_hold_bars=60, trend_lookback=3,
                  momentum_checkpoint_bars=None, momentum_mfe_r_threshold=0.0):
    """momentum_checkpoint_bars=None -> momentum-failure exit is fully
    disabled (this is the control / original strategy, unchanged).
    When set, a position still open at that many bars after entry is
    checked exactly ONCE: if its max favorable excursion so far (in R,
    the same R used for the 1.5R/2R target) never reached
    momentum_mfe_r_threshold, it's exited at that checkpoint's close.
    If it DID reach the threshold, nothing else changes - the trade
    keeps running under the existing SL / target / 3h max hold exactly
    as before. SL/TP hits always take precedence over the momentum
    check (checked first, below) - a trade that already stopped or
    hit target this bar never reaches the momentum check at all."""
    trades = []
    open_position = None
    pending_entry = None
    n = len(candles)

    for i in range(MIN_CANDLES_NEEDED, n):
        window = candles.iloc[max(0, i - LOOKBACK_WINDOW_BARS):i + 1]
        t = candles.index[i]
        bar_open = window["open"].iloc[-1]
        bar_close = window["close"].iloc[-1]

        if pending_entry is not None and open_position is None:
            direction = pending_entry["direction"]
            entry_price = bar_open
            stop_price = pending_entry["stop_price"]
            stop_distance = abs(entry_price - stop_price)
            if stop_distance > 0:
                target_price = (entry_price + stop_distance * r_target if direction == "long"
                                 else entry_price - stop_distance * r_target)
                open_position = {
                    "direction": direction, "entry": entry_price, "entry_time": t, "entry_index": i,
                    "stop": stop_price, "target": target_price,
                    "mfe_r": 0.0, "momentum_checked": False,
                }
            pending_entry = None
            # NO continue here on purpose: the position was just opened
            # at this candle's OWN open, so this same candle's high/low
            # (after the open) can still hit the fresh stop or target.
            # Falling through to the open_position block below tests
            # this entry candle too, instead of skipping straight to the
            # next candle and missing an intrabar SL/TP hit on entry day.

        if open_position is not None:
            pos = open_position
            direction = pos["direction"]
            bar_high = window["high"].iloc[-1]
            bar_low = window["low"].iloc[-1]
            risk = abs(pos["entry"] - pos["stop"])
            # INTRABAR touch, not close-only: a 3m candle can pierce the
            # SL or TP and still close back inside, which the old
            # close-only check would have missed entirely.
            if direction == "long":
                hit_stop = bar_low <= pos["stop"]
                hit_target = bar_high >= pos["target"]
                bar_mfe = bar_high - pos["entry"]
            else:
                hit_stop = bar_high >= pos["stop"]
                hit_target = bar_low <= pos["target"]
                bar_mfe = pos["entry"] - bar_low
            pos["mfe_r"] = max(pos["mfe_r"], bar_mfe / risk if risk != 0 else 0.0)

            bars_held = i - pos["entry_index"]
            hit_max_hold = bars_held >= max_hold_bars

            # MOMENTUM-FAILURE CHECKPOINT: only evaluated once, only on
            # a position that survived SL/TP this bar, only once bars_held
            # reaches the checkpoint. SL/TP precedence is enforced by the
            # "not hit_stop and not hit_target" guard below.
            hit_momentum_failure = False
            if (momentum_checkpoint_bars is not None
                    and not pos["momentum_checked"]
                    and bars_held >= momentum_checkpoint_bars
                    and not hit_stop and not hit_target):
                pos["momentum_checked"] = True
                if pos["mfe_r"] < momentum_mfe_r_threshold:
                    hit_momentum_failure = True

            if hit_stop or hit_target or hit_max_hold or hit_momentum_failure:
                # CONSERVATIVE TIEBREAK: 3m OHLC alone can't tell us
                # which level was touched first if both were hit inside
                # the same candle - assume the worse outcome (stop) hit
                # first. hit_stop is therefore checked ahead of hit_target.
                # Fill price is the actual stop/target level (not the
                # candle's close), matching how a real SL/TP order fills.
                if hit_stop:
                    exit_price = pos["stop"]
                    exit_reason = "stop"
                elif hit_target:
                    exit_price = pos["target"]
                    exit_reason = "target"
                elif hit_momentum_failure:
                    exit_price = bar_close
                    exit_reason = "momentum_failure"
                else:
                    exit_price = bar_close
                    exit_reason = "max_hold_time"
                profit = (exit_price - pos["entry"]) if direction == "long" else (pos["entry"] - exit_price)
                r_achieved = profit / risk if risk != 0 else 0
                trades.append({
                    "direction": direction, "entry_time": pos["entry_time"], "exit_time": t,
                    "entry_price": pos["entry"], "exit_price": exit_price,
                    "r_achieved": r_achieved, "exit_reason": exit_reason,
                    "stages_done": 0, "leverage": 5,
                    "stop_distance_pct": abs(pos["entry"] - pos["stop"]) / pos["entry"],
                    "bars_held": bars_held,
                })
                open_position = None
            continue

        for direction in ("long", "short"):
            if detect_ema_pullback_rejection(window, direction, trend_lookback=trend_lookback):
                stop_price = window["low"].iloc[-1] if direction == "long" else window["high"].iloc[-1]
                pending_entry = {"direction": direction, "stop_price": stop_price}
                break

    return pd.DataFrame(trades)


def run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback,
                   momentum_checkpoint_bars=None, momentum_mfe_r_threshold=0.0):
    trades = run_backtest(candles, r_target=r_target, max_hold_bars=max_hold_bars, trend_lookback=trend_lookback,
                           momentum_checkpoint_bars=momentum_checkpoint_bars,
                           momentum_mfe_r_threshold=momentum_mfe_r_threshold)
    if len(trades) == 0:
        return {"coin": coin, "r_target": r_target, "trend_lookback": trend_lookback,
                "trades": 0, "win_rate": None, "gross_expected_r": None, "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "ema_pullback_vwap"
    trades = apply_fees_and_interest(trades, bar_minutes=3)
    trades = apply_dollar_pnl(trades)
    return {
        "coin": coin, "r_target": r_target, "trend_lookback": trend_lookback,
        "trades": len(trades), "win_rate": round((trades["dollar_pnl"] > 0).mean() * 100, 1),
        "gross_expected_r": round(trades["r_achieved"].mean(), 4),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
    }


def _run_hold_minutes_task(args):
    """Top-level (picklable) worker for the parallel exit-mode sweep.
    r_target and trend_lookback are held FIXED here - the only thing
    varying across parallel workers is max_hold_minutes, so any P&L
    difference between runs is attributable to the hold time alone,
    not tangled up with other parameters or (not touched at all here)
    an RSI filter."""
    candles, coin, r_target, hold_minutes, trend_lookback = args
    max_hold_bars = round(hold_minutes / 3)  # 3-minute candles
    result = run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback)
    result["max_hold_minutes"] = hold_minutes
    return result


def _run_momentum_task(args):
    """Top-level (picklable) worker for the parallel momentum-failure
    ablation. r_target, trend_lookback, and max_hold_bars are held FIXED
    at the CLI defaults - the only things varying across workers are the
    checkpoint (in minutes) and the required MFE-in-R threshold. A
    checkpoint of None with threshold None is the CONTROL: momentum-
    failure exit fully disabled, i.e. the original strategy untouched."""
    candles, coin, r_target, max_hold_bars, trend_lookback, checkpoint_minutes, mfe_r_threshold = args
    checkpoint_bars = round(checkpoint_minutes / 3) if checkpoint_minutes is not None else None
    result = run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback,
                            momentum_checkpoint_bars=checkpoint_bars,
                            momentum_mfe_r_threshold=mfe_r_threshold if mfe_r_threshold is not None else 0.0)
    result["checkpoint_minutes"] = checkpoint_minutes
    result["mfe_r_threshold"] = mfe_r_threshold
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="BTC")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--max-hold-hours", type=float, default=3.0)
    parser.add_argument("--r-target", type=float, default=1.5)
    parser.add_argument("--trend-lookback", type=int, default=3)
    parser.add_argument("--sweep-all", action="store_true")
    parser.add_argument("--momentum-sweep", action="store_true",
                         help="Momentum-failure-exit ablation: control (no momentum check) vs "
                              "checkpoint in {3,6} minutes x required MFE in {>0R,0.1R,0.25R,0.5R}. "
                              "r_target/trend_lookback/max_hold_hours held fixed at their CLI values.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    mode = "momentum-sweep" if args.momentum_sweep else ("hold-sweep" if args.sweep_all else "single")
    print(f"Idea #18 [{mode}]: EMA Pullback + VWAP | {args.coin} | 3m | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_3m = resample_candles(candles_1m, 3)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_3m)} 3m candles"
          + (" (ONE fetch, reused for every combo below)" if mode != "single" else ""))

    if mode == "single":
        max_hold_bars = round(args.max_hold_hours * 60 / 3)
        result = run_one_combo(candles_3m, args.coin, args.r_target, max_hold_bars, args.trend_lookback)
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
              f"Gross expected R: {result['gross_expected_r']} | Total $ P&L: {result['total_pnl']}")
        print(f"Exit reason breakdown: {result['exit_breakdown']}")
        return

    if mode == "momentum-sweep":
        # MOMENTUM-FAILURE ABLATION: r_target, trend_lookback, and the 3h
        # max hold are all held FIXED at their CLI values - nothing about
        # the entry, target sizing, or final backstop hold changes. The
        # ONLY thing varying is whether/when/how strictly a still-open
        # position gets killed for failing to show favorable movement.
        # SL/TP always take precedence over this check (enforced inside
        # run_backtest, not here). Checkpoint restricted to {3,6} minutes
        # since 3m candles can't resolve a precise 5-minute mark. Control
        # (checkpoint=None) is the original strategy, completely unmodified -
        # included so every other variant is compared against it directly.
        # 0.25R is the primary hypothesis per the spec; the surrounding
        # thresholds (>0R, 0.1R, 0.5R) are there to see if there's a broad
        # relationship rather than cherry-picking whichever number wins.
        max_hold_bars = round(args.max_hold_hours * 60 / 3)
        checkpoints_minutes = [3, 6]
        mfe_r_thresholds = [0.0, 0.1, 0.25, 0.5]

        variants = [(None, None)]  # control: momentum-failure exit disabled
        variants += [(cp, thr) for cp in checkpoints_minutes for thr in mfe_r_thresholds]

        tasks = [
            (candles_3m, args.coin, args.r_target, max_hold_bars, args.trend_lookback, cp, thr)
            for cp, thr in variants
        ]

        results = []
        max_workers = min(len(tasks), os.cpu_count() or 1)
        print(f"Running {len(tasks)} momentum-failure variants in parallel ({max_workers} workers)...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_momentum_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                label = "CONTROL (no momentum check)" if result["checkpoint_minutes"] is None \
                    else f"checkpoint={result['checkpoint_minutes']}m, MFE>={result['mfe_r_threshold']}R"
                print(f"  {label} -> {result['trades']:5} trades | win_rate={result['win_rate']} | "
                      f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']} | "
                      f"exits={result['exit_breakdown']}")

        results_df = pd.DataFrame(results).sort_values(
            by=["checkpoint_minutes", "mfe_r_threshold"], na_position="first")
        print(f"\n=== MOMENTUM-FAILURE ABLATION: {args.coin} "
              f"(r_target={args.r_target}, trend_lookback={args.trend_lookback}, "
              f"max_hold={args.max_hold_hours}h all fixed; {len(tasks)} variants incl. control, 1 fetch) ===")
        print(results_df.to_string(index=False))

        valid = results_df.dropna(subset=["total_pnl"])
        if len(valid):
            best = valid.loc[valid["total_pnl"].idxmax()]
            best_label = "CONTROL" if pd.isna(best["checkpoint_minutes"]) \
                else f"checkpoint={best['checkpoint_minutes']}m, MFE>={best['mfe_r_threshold']}R"
            print(f"\nBest variant for {args.coin}: {best_label} -> ${best['total_pnl']:.2f} "
                  f"({best['trades']} trades, {best['win_rate']}% win rate, gross R={best['gross_expected_r']})")
        return

    # HOLD-TIME SWEEP (--sweep-all): a separate, simpler experiment from
    # the momentum-failure ablation above - this tests a hard timeout
    # ("exit everything still open after N minutes, regardless of
    # profitability"), not the momentum-failure rule. r_target and
    # trend_lookback held fixed; NOT mixed with RSI or any other change.
    hold_minutes_options = [3, 6, 9]

    tasks = [
        (candles_3m, args.coin, args.r_target, hold_minutes, args.trend_lookback)
        for hold_minutes in hold_minutes_options
    ]

    results = []
    max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"Running {len(tasks)} hold-time variants in parallel ({max_workers} workers)...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_hold_minutes_task, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  hold={result['max_hold_minutes']}m, r_target={result['r_target']}, "
                  f"trend_lookback={result['trend_lookback']} -> "
                  f"{result['trades']:5} trades | win_rate={result['win_rate']} | "
                  f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']}")

    results_df = pd.DataFrame(results).sort_values("max_hold_minutes")
    print(f"\n=== EXIT-MODE SWEEP RESULTS: {args.coin} "
          f"(r_target={args.r_target}, trend_lookback={args.trend_lookback} fixed; "
          f"{len(tasks)} hold-time variants, 1 fetch) ===")
    print(results_df.to_string(index=False))

    valid = results_df.dropna(subset=["total_pnl"])
    if len(valid):
        best = valid.loc[valid["total_pnl"].idxmax()]
        print(f"\nBest hold time for {args.coin}: {best['max_hold_minutes']}m -> "
              f"${best['total_pnl']:.2f} ({best['trades']} trades, "
              f"{best['win_rate']}% win rate, gross R={best['gross_expected_r']})")


if __name__ == "__main__":
    main()
