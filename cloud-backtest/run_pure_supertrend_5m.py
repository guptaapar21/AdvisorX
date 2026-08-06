"""
Idea #15: Pure SuperTrend, 5m timeframe, single coin. Uses SuperTrend's
OWN line for both entry and exit - the canonical "always in the market"
way this indicator is meant to be used, rather than the fixed R-target
bolted on top of it in ideas #12/#13.

Entry: SuperTrend flips (up -> long, down -> short) - same as before.
Exit: the SuperTrend line flips to the OPPOSITE direction. When that
happens while a position is open, the position closes AND a new one
opens immediately in the new direction (standard SuperTrend system
behavior - always in the market, no flat period between reversals).

HONEST CAVEAT, stated up front rather than silently worked around:
there is NO separate hard stop-loss here beyond the trend flip itself.
If price gaps hard against a position before SuperTrend's own band
catches up, this design has no independent floor on the loss for that
single trade - this is the direct, faithful implementation of "let
SuperTrend fully manage both stop and target", not a hidden safety net
added against that instruction.

Same bounded-window, zero-lookahead loop already proven across every
prior idea in this series.

Usage:
  python3 run_pure_supertrend_5m.py --coin SOL --period 10 --multiplier 3.0 --sweep-all
"""
import argparse
import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import calculate_supertrend
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 30
LOOKBACK_WINDOW_BARS = 100  # same fixed bound/reasoning as every prior idea's runner


def run_pure_supertrend(candles, period=10, multiplier=3.0):
    trades = []
    open_position = None
    n = len(candles)

    for i in range(MIN_CANDLES_NEEDED, n):
        window = candles.iloc[max(0, i - LOOKBACK_WINDOW_BARS):i + 1]
        t = candles.index[i]
        current_price = window["close"].iloc[-1]
        st = calculate_supertrend(window, period=period, multiplier=multiplier)
        if st["trend"] is None:
            continue

        if open_position is not None:
            pos = open_position
            direction = pos["direction"]
            opposite_flip = st["flipped"] and ((direction == "long" and st["trend"] == "down") or
                                                 (direction == "short" and st["trend"] == "up"))
            if opposite_flip:
                exit_price = current_price
                risk = abs(pos["entry"] - pos["initial_stop"])
                profit = (exit_price - pos["entry"]) if direction == "long" else (pos["entry"] - exit_price)
                r_achieved = profit / risk if risk != 0 else 0
                trades.append({
                    "direction": direction, "entry_time": pos["entry_time"], "exit_time": t,
                    "entry_price": pos["entry"], "exit_price": exit_price,
                    "r_achieved": r_achieved, "exit_reason": "supertrend_flip",
                    "stages_done": 0, "leverage": 5,
                    "stop_distance_pct": abs(pos["entry"] - pos["initial_stop"]) / pos["entry"],
                    "bars_held": i - pos["entry_index"],
                })
                new_direction = "long" if st["trend"] == "up" else "short"
                open_position = {
                    "direction": new_direction, "entry": current_price, "entry_time": t, "entry_index": i,
                    "initial_stop": st["value"],
                }
        else:
            if st["flipped"]:
                direction = "long" if st["trend"] == "up" else "short"
                open_position = {
                    "direction": direction, "entry": current_price, "entry_time": t, "entry_index": i,
                    "initial_stop": st["value"],
                }

    return pd.DataFrame(trades)


def run_one_combo(candles, coin, period, multiplier):
    trades = run_pure_supertrend(candles, period=period, multiplier=multiplier)
    if len(trades) == 0:
        return {"coin": coin, "period": period, "multiplier": multiplier,
                "trades": 0, "win_rate": None, "gross_expected_r": None, "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "pure_supertrend_5m"
    trades = apply_fees_and_interest(trades, bar_minutes=5)
    trades = apply_dollar_pnl(trades)
    return {
        "coin": coin, "period": period, "multiplier": multiplier,
        "trades": len(trades), "win_rate": round((trades["dollar_pnl"] > 0).mean() * 100, 1),
        "gross_expected_r": round(trades["r_achieved"].mean(), 4),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="SOL")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--sweep-all", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    print(f"{'Idea #15 sweep' if args.sweep_all else 'Idea #15'}: Pure SuperTrend (5m, flip-to-flip) | {args.coin} | "
          f"{start_date} to {now.date()} ({args.days}d)")
    print("No separate fixed stop/target - SuperTrend's own line manages both entry and exit.")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_5m = resample_candles(candles_1m, 5)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_5m)} 5m candles"
          + (" (ONE fetch, reused for every combo below)" if args.sweep_all else ""))

    if not args.sweep_all:
        result = run_one_combo(candles_5m, args.coin, args.period, args.multiplier)
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
              f"Gross expected R: {result['gross_expected_r']} | Total $ P&L: {result['total_pnl']}")
        print(f"Exit reason breakdown: {result['exit_breakdown']}")
        return

    periods = [7, 10, 14]
    multipliers = [1.5, 2.0, 2.5, 3.0]

    results = []
    for period, mult in itertools.product(periods, multipliers):
        result = run_one_combo(candles_5m, args.coin, period, mult)
        results.append(result)
        print(f"  period={period}, mult={mult} -> {result['trades']:5} trades | win_rate={result['win_rate']} | "
              f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL SWEEP RESULTS: {args.coin} (12 combos, 1 fetch) ===")
    print(results_df.to_string(index=False))

    valid = results_df.dropna(subset=["total_pnl"])
    if len(valid):
        best = valid.loc[valid["total_pnl"].idxmax()]
        print(f"\nBest combo for {args.coin}: period={best['period']}, mult={best['multiplier']} -> "
              f"${best['total_pnl']:.2f} ({best['trades']} trades, {best['win_rate']}% win rate, "
              f"gross R={best['gross_expected_r']})")


if __name__ == "__main__":
    main()
