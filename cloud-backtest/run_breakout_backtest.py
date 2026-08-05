"""
Standalone backtest for Idea #11 (Consolidation Breakout) - genuinely
different mechanism from Idea #10, per the diagnosis that idea #10's
lagging-confirmation entries showed ~breakeven gross expectancy at
every setting tested. This triggers AT the breakout moment, not after
a trend is already confirmed three ways.

Same bounded-window, zero-lookahead loop as idea #10's runner (already
tested at full-year scale: bounded window fixed the O(n^2) bug that
caused a real DOGE timeout) - candles.iloc[max(0, i-WINDOW):i+1], never
touching anything past the current closed bar.

Usage:
  python3 run_breakout_backtest.py --coin SOL --lookback 10 --range-atr-ratio 1.5 --r-target 1.5 --sweep-all
"""
import argparse
import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import detect_consolidation_breakout, calculate_breakout_stop_target
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 30
LOOKBACK_WINDOW_BARS = 100  # same fixed bound and same reasoning as idea #10's runner -
                            # see that file's comment for the full O(n^2) history this fixes.


def run_breakout_backtest(candles_5m, lookback=10, range_atr_ratio=1.5, volume_spike_threshold=1.5,
                           r_target=1.5, max_hold_bars=48):
    trades = []
    open_position = None
    n = len(candles_5m)

    for i in range(MIN_CANDLES_NEEDED, n):
        window = candles_5m.iloc[max(0, i - LOOKBACK_WINDOW_BARS):i + 1]
        t = candles_5m.index[i]
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
            signal = detect_consolidation_breakout(window, lookback=lookback, range_atr_ratio=range_atr_ratio,
                                                     volume_spike_threshold=volume_spike_threshold)
            if signal["action"] != "wait":
                direction = signal["action"]
                sl = calculate_breakout_stop_target(direction, current_price, signal["range_high"],
                                                     signal["range_low"], r_target=r_target)
                if sl["stop_distance"] > 0:
                    open_position = {
                        "direction": direction, "entry": current_price, "entry_time": t, "entry_index": i,
                        "stop": sl["stop_price"], "target": sl["target_price"],
                    }

    return pd.DataFrame(trades)


def run_one_combo(candles_5m, coin, lookback, range_atr_ratio, r_target, max_hold_bars):
    trades = run_breakout_backtest(candles_5m, lookback=lookback, range_atr_ratio=range_atr_ratio,
                                    r_target=r_target, max_hold_bars=max_hold_bars)
    if len(trades) == 0:
        return {"coin": coin, "lookback": lookback, "range_atr_ratio": range_atr_ratio, "r_target": r_target,
                "trades": 0, "win_rate": None, "gross_expected_r": None, "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "consolidation_breakout"
    trades = apply_fees_and_interest(trades, bar_minutes=5)
    trades = apply_dollar_pnl(trades)
    win_rate = (trades["dollar_pnl"] > 0).mean()
    gross_expected_r = trades["r_achieved"].mean()  # GROSS, pre-fee - the number idea #10's
                                                     # first report was missing, added here from the start
    return {
        "coin": coin, "lookback": lookback, "range_atr_ratio": range_atr_ratio, "r_target": r_target,
        "trades": len(trades), "win_rate": round(win_rate * 100, 1),
        "gross_expected_r": round(gross_expected_r, 4),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--max-hold-hours", type=float, default=6.0)
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--range-atr-ratio", type=float, default=1.5)
    parser.add_argument("--r-target", type=float, default=1.5)
    parser.add_argument("--sweep-all", action="store_true",
                         help="Fetch once, run every parameter combination in-memory.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()
    max_hold_bars = round(args.max_hold_hours * 60 / 5)

    print(f"{'Idea #11 sweep' if args.sweep_all else 'Idea #11'}: Consolidation Breakout | {args.coin} | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_5m = resample_candles(candles_1m, 5)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_5m)} 5m candles"
          + (" (ONE fetch, reused for every combo below)" if args.sweep_all else ""))

    if not args.sweep_all:
        result = run_one_combo(candles_5m, args.coin, args.lookback, args.range_atr_ratio, args.r_target, max_hold_bars)
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
              f"Gross expected R: {result['gross_expected_r']} | Total $ P&L: {result['total_pnl']}")
        print(f"Exit reason breakdown: {result['exit_breakdown']}")
        return

    lookbacks = [8, 10, 15]
    range_atr_ratios = [1.0, 1.5, 2.0]
    r_targets = [1.5, 2.0]

    results = []
    for lb, ratio, r_tgt in itertools.product(lookbacks, range_atr_ratios, r_targets):
        result = run_one_combo(candles_5m, args.coin, lb, ratio, r_tgt, max_hold_bars)
        results.append(result)
        print(f"  lookback={lb}, range_atr_ratio={ratio}, r_target={r_tgt} -> "
              f"{result['trades']:4} trades | win_rate={result['win_rate']} | "
              f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL SWEEP RESULTS: {args.coin} (18 combos, 1 fetch) ===")
    print(results_df.to_string(index=False))

    valid = results_df.dropna(subset=["total_pnl"])
    if len(valid):
        best = valid.loc[valid["total_pnl"].idxmax()]
        print(f"\nBest combo for {args.coin}: lookback={best['lookback']}, "
              f"range_atr_ratio={best['range_atr_ratio']}, r_target={best['r_target']} -> "
              f"${best['total_pnl']:.2f} ({best['trades']} trades, {best['win_rate']}% win rate, "
              f"gross R={best['gross_expected_r']})")


if __name__ == "__main__":
    main()
