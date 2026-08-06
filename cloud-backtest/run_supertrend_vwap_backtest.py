"""
Idea #16: back to a FIXED stop/target (idea #15's pure flip-to-flip on
5m performed worse than idea #12's fixed-target version - reverting to
what worked better), plus VWAP as a directional confirmation filter -
tested WITH and WITHOUT, as an ablation, not assumed to help.

Entry: SuperTrend flip on 5m (same as idea #12).
VWAP filter (when enabled): only take the long if current price is
above VWAP, only take the short if price is below VWAP - a real
directional agreement check, not a magnitude/strength gauge.
Stop: SuperTrend's own line at entry (structural, same as idea #12).
Target: fixed R-multiple, closed 100% at once (same as idea #12).

Single coin (SOL), 5m only, per instruction to keep this focused.

Usage:
  python3 run_supertrend_vwap_backtest.py --coin SOL --period 10 --multiplier 3.0 --r-target 1.5 --use-vwap --sweep-all
"""
import argparse
import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import calculate_supertrend, calculate_vwap
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 30
LOOKBACK_WINDOW_BARS = 100  # same fixed bound/reasoning as every prior idea's runner


def run_backtest(candles, period=10, multiplier=3.0, r_target=1.5, use_vwap=False, max_hold_bars=48):
    trades = []
    open_position = None
    n = len(candles)

    for i in range(MIN_CANDLES_NEEDED, n):
        window = candles.iloc[max(0, i - LOOKBACK_WINDOW_BARS):i + 1]
        t = candles.index[i]
        current_price = window["close"].iloc[-1]

        if open_position is not None:
            pos = open_position
            direction = pos["direction"]
            hit_stop = current_price <= pos["stop"] if direction == "long" else current_price >= pos["stop"]
            hit_target = current_price >= pos["target"] if direction == "long" else current_price <= pos["target"]
            bars_held = i - pos["entry_index"]
            hit_max_hold = bars_held >= max_hold_bars

            if hit_stop or hit_target or hit_max_hold:
                exit_price = current_price
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
        else:
            st = calculate_supertrend(window, period=period, multiplier=multiplier)
            if st["flipped"]:
                direction = "long" if st["trend"] == "up" else "short"

                if use_vwap:
                    vwap = calculate_vwap(window)
                    if vwap is None:
                        continue
                    vwap_agrees = (direction == "long" and current_price > vwap) or \
                                  (direction == "short" and current_price < vwap)
                    if not vwap_agrees:
                        continue

                stop_price = st["value"]
                stop_distance = abs(current_price - stop_price)
                if stop_distance > 0:
                    target_price = (current_price + stop_distance * r_target if direction == "long"
                                     else current_price - stop_distance * r_target)
                    open_position = {
                        "direction": direction, "entry": current_price, "entry_time": t, "entry_index": i,
                        "stop": stop_price, "target": target_price,
                    }

    return pd.DataFrame(trades)


def run_one_combo(candles, coin, period, multiplier, r_target, use_vwap, max_hold_bars):
    trades = run_backtest(candles, period=period, multiplier=multiplier, r_target=r_target,
                           use_vwap=use_vwap, max_hold_bars=max_hold_bars)
    if len(trades) == 0:
        return {"coin": coin, "period": period, "multiplier": multiplier, "r_target": r_target,
                "use_vwap": use_vwap, "trades": 0, "win_rate": None, "gross_expected_r": None,
                "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "supertrend_vwap"
    trades = apply_fees_and_interest(trades, bar_minutes=5)
    trades = apply_dollar_pnl(trades)
    return {
        "coin": coin, "period": period, "multiplier": multiplier, "r_target": r_target, "use_vwap": use_vwap,
        "trades": len(trades), "win_rate": round((trades["dollar_pnl"] > 0).mean() * 100, 1),
        "gross_expected_r": round(trades["r_achieved"].mean(), 4),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="SOL")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--max-hold-hours", type=float, default=4.0)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--r-target", type=float, default=1.5)
    parser.add_argument("--use-vwap", action="store_true")
    parser.add_argument("--sweep-all", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    print(f"{'Idea #16 sweep' if args.sweep_all else 'Idea #16'}: SuperTrend + VWAP filter (ablation) | {args.coin} | 5m | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_5m = resample_candles(candles_1m, 5)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_5m)} 5m candles"
          + (" (ONE fetch, reused for every combo below)" if args.sweep_all else ""))

    if not args.sweep_all:
        max_hold_bars = round(args.max_hold_hours * 60 / 5)
        result = run_one_combo(candles_5m, args.coin, args.period, args.multiplier, args.r_target,
                                args.use_vwap, max_hold_bars)
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
              f"Gross expected R: {result['gross_expected_r']} | Total $ P&L: {result['total_pnl']}")
        print(f"Exit reason breakdown: {result['exit_breakdown']}")
        return

    periods = [7, 10, 14]
    multipliers = [2.0, 3.0]
    r_targets = [1.5, 2.0]
    vwap_options = [False, True]
    max_hold_bars = round(args.max_hold_hours * 60 / 5)

    results = []
    for period, mult, r_tgt, use_vwap in itertools.product(periods, multipliers, r_targets, vwap_options):
        result = run_one_combo(candles_5m, args.coin, period, mult, r_tgt, use_vwap, max_hold_bars)
        results.append(result)
        print(f"  period={period}, mult={mult}, r_target={r_tgt}, vwap={use_vwap} -> "
              f"{result['trades']:5} trades | win_rate={result['win_rate']} | "
              f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL SWEEP RESULTS: {args.coin} (24 combos, 1 fetch) ===")
    print(results_df.to_string(index=False))

    valid = results_df.dropna(subset=["gross_expected_r"])
    no_vwap = valid[valid["use_vwap"] == False]["gross_expected_r"]
    with_vwap = valid[valid["use_vwap"] == True]["gross_expected_r"]
    print(f"\nVWAP ABLATION: avg gross R WITHOUT vwap = {no_vwap.mean():.4f} | WITH vwap = {with_vwap.mean():.4f}")

    valid_pnl = results_df.dropna(subset=["total_pnl"])
    if len(valid_pnl):
        best = valid_pnl.loc[valid_pnl["total_pnl"].idxmax()]
        print(f"\nBest combo for {args.coin}: period={best['period']}, mult={best['multiplier']}, "
              f"r_target={best['r_target']}, vwap={best['use_vwap']} -> ${best['total_pnl']:.2f} "
              f"({best['trades']} trades, {best['win_rate']}% win rate, gross R={best['gross_expected_r']})")


if __name__ == "__main__":
    main()
