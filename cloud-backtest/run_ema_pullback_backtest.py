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
import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import detect_ema_pullback_rejection
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 40
LOOKBACK_WINDOW_BARS = 100


def run_backtest(candles, r_target=1.5, max_hold_bars=60, trend_lookback=3):
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
                }
            pending_entry = None
            continue

        if open_position is not None:
            pos = open_position
            direction = pos["direction"]
            hit_stop = bar_close <= pos["stop"] if direction == "long" else bar_close >= pos["stop"]
            hit_target = bar_close >= pos["target"] if direction == "long" else bar_close <= pos["target"]
            bars_held = i - pos["entry_index"]
            hit_max_hold = bars_held >= max_hold_bars

            if hit_stop or hit_target or hit_max_hold:
                exit_price = bar_close
                risk = abs(pos["entry"] - pos["stop"])
                profit = (exit_price - pos["entry"]) if direction == "long" else (pos["entry"] - exit_price)
                r_achieved = profit / risk if risk != 0 else 0
                exit_reason = "stop" if hit_stop else ("target" if hit_target else "max_hold_time")
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


def run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback):
    trades = run_backtest(candles, r_target=r_target, max_hold_bars=max_hold_bars, trend_lookback=trend_lookback)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="BTC")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--max-hold-hours", type=float, default=3.0)
    parser.add_argument("--r-target", type=float, default=1.5)
    parser.add_argument("--trend-lookback", type=int, default=3)
    parser.add_argument("--sweep-all", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    print(f"{'Idea #18 sweep' if args.sweep_all else 'Idea #18'}: EMA Pullback + VWAP | {args.coin} | 3m | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_3m = resample_candles(candles_1m, 3)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_3m)} 3m candles"
          + (" (ONE fetch, reused for every combo below)" if args.sweep_all else ""))

    if not args.sweep_all:
        max_hold_bars = round(args.max_hold_hours * 60 / 3)
        result = run_one_combo(candles_3m, args.coin, args.r_target, max_hold_bars, args.trend_lookback)
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
              f"Gross expected R: {result['gross_expected_r']} | Total $ P&L: {result['total_pnl']}")
        print(f"Exit reason breakdown: {result['exit_breakdown']}")
        return

    # Trimmed from an original 2x3x3=18-combo grid after measuring the
    # REAL per-candle cost of this strategy (~1.07ms/candle, confirmed
    # via direct O(n) scaling test) - at full-year/3m scale that's
    # ~187.5s PER COMBO, meaning 18 combos would take ~56 minutes,
    # risking the same timeout-margin mistake already made twice in
    # this project. This strategy is genuinely slower per-candle than
    # prior ideas (checks both directions every bar, calls VWAP+EMA
    # each time) - not a bug, just real, measured cost. Trimmed to the
    # two most informative dimensions (r_target, trend_lookback) at a
    # single fixed max_hold, keeping runtime to ~18-19 minutes.
    r_targets = [1.5, 2.0]
    trend_lookbacks = [2, 3, 5]
    max_hold_hours_options = [3.0]

    results = []
    for r_tgt, tl, max_hold_h in itertools.product(r_targets, trend_lookbacks, max_hold_hours_options):
        max_hold_bars = round(max_hold_h * 60 / 3)
        result = run_one_combo(candles_3m, args.coin, r_tgt, max_hold_bars, tl)
        result["max_hold_hours"] = max_hold_h
        results.append(result)
        print(f"  r_target={r_tgt}, trend_lookback={tl}, hold={max_hold_h}h -> "
              f"{result['trades']:5} trades | win_rate={result['win_rate']} | "
              f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL SWEEP RESULTS: {args.coin} (6 combos, 1 fetch) ===")
    print(results_df.to_string(index=False))

    valid = results_df.dropna(subset=["total_pnl"])
    if len(valid):
        best = valid.loc[valid["total_pnl"].idxmax()]
        print(f"\nBest combo for {args.coin}: r_target={best['r_target']}, trend_lookback={best['trend_lookback']}, "
              f"hold={best['max_hold_hours']}h -> ${best['total_pnl']:.2f} ({best['trades']} trades, "
              f"{best['win_rate']}% win rate, gross R={best['gross_expected_r']})")


if __name__ == "__main__":
    main()
