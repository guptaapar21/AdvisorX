"""
Idea #12: SuperTrend flip entries, tested at native 1m/3m/15m resolution
(not 5m-resampled - genuinely fetched at that granularity), addressing
the "try faster timeframes" request directly rather than re-testing the
already-rejected 1m/5m/15m combo from the ORIGINAL system (that combo
was for the 15m/1h-based route_strategy signal - a completely different
mechanism from this).

Entry: SuperTrend flips direction (up->down or down->up) - simplest
possible trend-following trigger, deliberately different in kind from
both idea #10 (lagging conjunction of confirmations) and idea #11
(range-breakout). Stop: the SuperTrend line itself at entry (a
structural, not ATR-multiple, stop). Target: fixed R-multiple.

Same bounded-window, zero-lookahead loop pattern already proven for
ideas #10 and #11 (and the same O(n^2) mistake already fixed once -
not repeated here).

Usage:
  python3 run_supertrend_backtest.py --coin SOL --timeframe 3m --period 10 --multiplier 3.0 --r-target 1.5
"""
import argparse
import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import calculate_supertrend
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 30
LOOKBACK_WINDOW_BARS = 100  # same fixed bound as ideas #10/#11's runners - see those files
                            # for the full O(n^2) history this avoids repeating.

TIMEFRAME_MINUTES = {"1m": 1, "3m": 3, "15m": 15}


def run_supertrend_backtest(candles, period=10, multiplier=3.0, r_target=1.5, max_hold_bars=48):
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


def run_one_combo(candles, coin, timeframe, period, multiplier, r_target, max_hold_bars):
    trades = run_supertrend_backtest(candles, period=period, multiplier=multiplier, r_target=r_target,
                                      max_hold_bars=max_hold_bars)
    if len(trades) == 0:
        return {"coin": coin, "timeframe": timeframe, "period": period, "multiplier": multiplier,
                "r_target": r_target, "trades": 0, "win_rate": None, "gross_expected_r": None,
                "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "supertrend"
    bar_minutes = TIMEFRAME_MINUTES[timeframe]
    trades = apply_fees_and_interest(trades, bar_minutes=bar_minutes)
    trades = apply_dollar_pnl(trades)
    return {
        "coin": coin, "timeframe": timeframe, "period": period, "multiplier": multiplier, "r_target": r_target,
        "trades": len(trades), "win_rate": round((trades["dollar_pnl"] > 0).mean() * 100, 1),
        "gross_expected_r": round(trades["r_achieved"].mean(), 4),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, required=True)
    parser.add_argument("--days", type=int, default=180,
                        help="1m data for a full year is enormous - default 180d for this idea "
                             "specifically, override with --days if you want the full year anyway.")
    parser.add_argument("--timeframe", type=str, default="3m", choices=["1m", "3m", "15m"])
    parser.add_argument("--max-hold-hours", type=float, default=4.0)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--r-target", type=float, default=1.5)
    parser.add_argument("--sweep-all", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    print(f"{'Idea #12 sweep' if args.sweep_all else 'Idea #12'}: SuperTrend | {args.coin} | timeframe={args.timeframe} | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    if args.sweep_all:
        # sweep-all tests all 3 timeframes from the SAME 1m fetch, resampled -
        # one real fetch per coin, not three.
        print(f"{args.coin}: {len(candles_1m)} 1m candles fetched (ONE fetch, resampled for all 3 timeframes below)")
        candles_by_tf = {tf: (candles_1m if tf == "1m" else resample_candles(candles_1m, TIMEFRAME_MINUTES[tf]))
                          for tf in ["1m", "3m", "15m"]}

        periods = [7, 10, 14]
        multipliers = [2.0, 3.0]  # trimmed from 3 values to 2 after measuring real 1m-scale cost
                                  # (55.6s/combo at 1m/180d) - the full 3x3x2=18-per-timeframe grid
                                  # would push total runtime close enough to a 30-45min budget to
                                  # risk the same timeout-margin mistake already made twice in this
                                  # project. 3x2x2=12-per-timeframe (36 total) leaves real margin.
        r_targets = [1.5, 2.0]
        results = []
        for tf, period, mult, r_tgt in itertools.product(["1m", "3m", "15m"], periods, multipliers, r_targets):
            max_hold_bars = round(args.max_hold_hours * 60 / TIMEFRAME_MINUTES[tf])
            result = run_one_combo(candles_by_tf[tf], args.coin, tf, period, mult, r_tgt, max_hold_bars)
            results.append(result)
            print(f"  tf={tf}, period={period}, mult={mult}, r_target={r_tgt} -> "
                  f"{result['trades']:5} trades | win_rate={result['win_rate']} | "
                  f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']}")

        results_df = pd.DataFrame(results)
        print(f"\n=== FULL SWEEP RESULTS: {args.coin} (36 combos, 1 fetch) ===")
        print(results_df.to_string(index=False))
        valid = results_df.dropna(subset=["total_pnl"])
        if len(valid):
            best = valid.loc[valid["total_pnl"].idxmax()]
            print(f"\nBest combo for {args.coin}: tf={best['timeframe']}, period={best['period']}, "
                  f"mult={best['multiplier']}, r_target={best['r_target']} -> ${best['total_pnl']:.2f} "
                  f"({best['trades']} trades, {best['win_rate']}% win rate, gross R={best['gross_expected_r']})")
        return

    candles = candles_1m if args.timeframe == "1m" else resample_candles(candles_1m, TIMEFRAME_MINUTES[args.timeframe])
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles)} {args.timeframe} candles")
    max_hold_bars = round(args.max_hold_hours * 60 / TIMEFRAME_MINUTES[args.timeframe])
    result = run_one_combo(candles, args.coin, args.timeframe, args.period, args.multiplier, args.r_target, max_hold_bars)
    print(f"\n=== RESULTS: {args.coin} ({args.timeframe}) ===")
    print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
          f"Gross expected R: {result['gross_expected_r']} | Total $ P&L: {result['total_pnl']}")
    print(f"Exit reason breakdown: {result['exit_breakdown']}")


if __name__ == "__main__":
    main()
