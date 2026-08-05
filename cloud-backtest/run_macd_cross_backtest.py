"""
Idea #14: MACD zero-line crossover, native 1m timeframe.

Long: MACD line crosses ABOVE its signal line while the MACD line
itself is already above the zero line (histogram flips from <=0 to >0,
AND macd_line > 0 at that moment - not just any crossover, specifically
one confirming an already-positive MACD line, filtering out early
reversal attempts still in negative territory).
Short: mirrored - crosses below signal while MACD line is below zero.

Exit: fixed stop (ATR-based, same convention as ideas #10/#12) and a
single R-target, closed 100% at once.

Same bounded-window, zero-lookahead loop already proven across every
prior idea in this series (and the same O(n^2) mistake already fixed
once - not repeated here).

Usage:
  python3 run_macd_cross_backtest.py --coin SOL --atr-multiplier 1.5 --r-target 1.5 --sweep-all
"""
import argparse
import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines
from momentum_scalp import detect_macd_zero_cross, calculate_scalp_stop_target
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 40  # MACD needs slow(26)+signal(9)=35 minimum, +margin
LOOKBACK_WINDOW_BARS = 100  # same fixed bound/reasoning as every prior idea's runner


def run_macd_cross_backtest(candles, atr_multiplier=1.5, r_target=1.5, max_hold_bars=60):
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
            for direction in ("long", "short"):
                if detect_macd_zero_cross(window, direction):
                    sl = calculate_scalp_stop_target(window, direction, current_price,
                                                      atr_multiplier=atr_multiplier, r_target=r_target)
                    if sl["stop_distance"] > 0:
                        open_position = {
                            "direction": direction, "entry": current_price, "entry_time": t, "entry_index": i,
                            "stop": sl["stop_price"], "target": sl["target_price"],
                        }
                    break

    return pd.DataFrame(trades)


def run_one_combo(candles, coin, atr_multiplier, r_target, max_hold_bars):
    trades = run_macd_cross_backtest(candles, atr_multiplier=atr_multiplier, r_target=r_target,
                                      max_hold_bars=max_hold_bars)
    if len(trades) == 0:
        return {"coin": coin, "atr_multiplier": atr_multiplier, "r_target": r_target,
                "trades": 0, "win_rate": None, "gross_expected_r": None, "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "macd_zero_cross"
    trades = apply_fees_and_interest(trades, bar_minutes=1)
    trades = apply_dollar_pnl(trades)
    return {
        "coin": coin, "atr_multiplier": atr_multiplier, "r_target": r_target,
        "trades": len(trades), "win_rate": round((trades["dollar_pnl"] > 0).mean() * 100, 1),
        "gross_expected_r": round(trades["r_achieved"].mean(), 4),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, required=True)
    parser.add_argument("--days", type=int, default=90,
                        help="1m data is enormous - default 90d, override if needed.")
    parser.add_argument("--max-hold-hours", type=float, default=2.0)
    parser.add_argument("--atr-multiplier", type=float, default=1.5)
    parser.add_argument("--r-target", type=float, default=1.5)
    parser.add_argument("--sweep-all", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    print(f"{'Idea #14 sweep' if args.sweep_all else 'Idea #14'}: MACD Zero Cross | {args.coin} | 1m | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    print(f"{args.coin}: {len(candles)} 1m candles" + (" (ONE fetch, reused for every combo below)" if args.sweep_all else ""))

    if not args.sweep_all:
        max_hold_bars = round(args.max_hold_hours * 60)
        result = run_one_combo(candles, args.coin, args.atr_multiplier, args.r_target, max_hold_bars)
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
              f"Gross expected R: {result['gross_expected_r']} | Total $ P&L: {result['total_pnl']}")
        print(f"Exit reason breakdown: {result['exit_breakdown']}")
        return

    atr_multipliers = [1.0, 1.5, 2.0]
    r_targets = [1.5, 2.0]
    max_hold_hours_options = [1.0, 2.0, 4.0]

    results = []
    for atr_mult, r_tgt, max_hold_h in itertools.product(atr_multipliers, r_targets, max_hold_hours_options):
        max_hold_bars = round(max_hold_h * 60)
        result = run_one_combo(candles, args.coin, atr_mult, r_tgt, max_hold_bars)
        result["max_hold_hours"] = max_hold_h
        results.append(result)
        print(f"  atr={atr_mult}, r_target={r_tgt}, hold={max_hold_h}h -> "
              f"{result['trades']:5} trades | win_rate={result['win_rate']} | "
              f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL SWEEP RESULTS: {args.coin} (18 combos, 1 fetch) ===")
    print(results_df.to_string(index=False))

    valid = results_df.dropna(subset=["total_pnl"])
    if len(valid):
        best = valid.loc[valid["total_pnl"].idxmax()]
        print(f"\nBest combo for {args.coin}: atr={best['atr_multiplier']}, r_target={best['r_target']}, "
              f"hold={best['max_hold_hours']}h -> ${best['total_pnl']:.2f} ({best['trades']} trades, "
              f"{best['win_rate']}% win rate, gross R={best['gross_expected_r']})")


if __name__ == "__main__":
    main()
