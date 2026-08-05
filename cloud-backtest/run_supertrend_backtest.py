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

TIMEFRAME_MINUTES = {"1m": 1, "3m": 3, "15m": 15, "1h": 60}

HTF_PERIOD = 10       # fixed 1h SuperTrend settings used only for the alignment filter -
HTF_MULTIPLIER = 3.0  # not swept; the point is a coarse chop filter, not another tunable.

# SAFETY ASSUMPTION - VERIFY BEFORE TRUSTING htf=1h_align RESULTS:
# candles_1h.index[i] is assumed to be the bar's OPEN time (pandas resample's default,
# label="left"), meaning the bar's close/SuperTrend value isn't actually known until
# index[i] + 1h. If that's correct, this shift is required to avoid handing 3m bars an
# 1h trend value up to ~57 minutes before it was really available. If your fetcher/
# resample_candles instead labels bars by CLOSE time, set this to False - leaving it
# True in that case would just make the filter needlessly stale by an extra hour, not
# wrong, but check to be sure.
HTF_INDEX_IS_OPEN_TIME = True


def compute_htf_trend_series(candles_1h):
    """
    Walks the 1h candles once with the same bounded-window pattern as the main
    loop and records the SuperTrend trend ("up"/"down") as of each 1h close.
    No lookahead: the trend at 1h bar i only uses candles up to and including i.
    Returned as a Series indexed by the timestamp the value actually becomes
    available (see HTF_INDEX_IS_OPEN_TIME - shifted forward by 1h if the source
    index is open-time-labeled), for merge_asof'ing onto the entry timeframe's
    timestamps using only 1h bars already closed.
    """
    n = len(candles_1h)
    idx, trend = [], []
    for i in range(MIN_CANDLES_NEEDED, n):
        window = candles_1h.iloc[max(0, i - LOOKBACK_WINDOW_BARS):i + 1]
        st = calculate_supertrend(window, period=HTF_PERIOD, multiplier=HTF_MULTIPLIER)
        idx.append(candles_1h.index[i])
        trend.append(st["trend"])
    available_at = pd.DatetimeIndex(idx)
    if HTF_INDEX_IS_OPEN_TIME:
        available_at = available_at + pd.Timedelta(hours=1)
    return pd.Series(trend, index=available_at, name="htf_trend")


def run_supertrend_backtest(candles, period=10, multiplier=3.0, r_target=1.5, max_hold_bars=48,
                             htf_trend_asof=None):
    """
    htf_trend_asof: optional pandas Series (DatetimeIndex -> "up"/"down"), already
    merge_asof-aligned so that htf_trend_asof.loc[t] gives the most recently CLOSED
    1h SuperTrend direction as of entry timestamp t (or NaN before the first 1h bar
    closes). When provided, an entry is only taken if it agrees with this direction -
    i.e. long entries require 1h trend == "up", shorts require "down".
    """
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

                if htf_trend_asof is not None:
                    htf_dir = htf_trend_asof.loc[t] if t in htf_trend_asof.index else None
                    if htf_dir is None or pd.isna(htf_dir):
                        continue  # no closed 1h bar yet - skip rather than assume alignment
                    required = "up" if direction == "long" else "down"
                    if htf_dir != required:
                        continue  # HTF disagrees - filtered out

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


def run_one_combo(candles, coin, timeframe, period, multiplier, r_target, max_hold_bars,
                   htf_trend_asof=None, htf_label="off"):
    trades = run_supertrend_backtest(candles, period=period, multiplier=multiplier, r_target=r_target,
                                      max_hold_bars=max_hold_bars, htf_trend_asof=htf_trend_asof)
    if len(trades) == 0:
        return {"coin": coin, "timeframe": timeframe, "period": period, "multiplier": multiplier,
                "r_target": r_target, "htf_filter": htf_label, "trades": 0, "win_rate": None,
                "gross_expected_r": None, "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "supertrend"
    bar_minutes = TIMEFRAME_MINUTES[timeframe]
    trades = apply_fees_and_interest(trades, bar_minutes=bar_minutes)
    trades = apply_dollar_pnl(trades)
    return {
        "coin": coin, "timeframe": timeframe, "period": period, "multiplier": multiplier, "r_target": r_target,
        "htf_filter": htf_label,
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
    parser.add_argument("--sweep-tuning", action="store_true",
                        help="Path A grid: fixed 3m timeframe, period in [10,14], multiplier in "
                             "[3.0,4.0,5.0], r_target in [2.0,2.5,3.0,3.5], HTF filter off vs 1h-align "
                             "(48 combos). Widens stops/targets to shrink fees as a fraction of R.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    mode_label = "Idea #12 sweep" if args.sweep_all else ("Idea #12 tuning sweep" if args.sweep_tuning else "Idea #12")
    print(f"{mode_label}: SuperTrend | {args.coin} | timeframe={args.timeframe} | "
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

    if args.sweep_tuning:
        # Fixed 3m timeframe - grounded on the validated gross signal from the idea #12
        # sweep. Wider multiplier/r_target values shrink fees as a fraction of R; the
        # HTF filter is tested as an ablation on top, not swept in combination with itself.
        candles_3m = resample_candles(candles_1m, TIMEFRAME_MINUTES["3m"])
        candles_1h = resample_candles(candles_1m, TIMEFRAME_MINUTES["1h"])
        print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_3m)} 3m candles, "
              f"{len(candles_1h)} 1h candles (ONE fetch)")

        htf_trend_1h = compute_htf_trend_series(candles_1h)
        print(f"  HTF filter: assuming candles_1h index = "
              f"{'open' if HTF_INDEX_IS_OPEN_TIME else 'close'}-time "
              f"(HTF_INDEX_IS_OPEN_TIME={HTF_INDEX_IS_OPEN_TIME}) - "
              f"trend values shifted to their real availability time before merging.")
        # merge_asof onto the 3m index: for each 3m bar t, take the most recent 1h trend
        # value whose availability timestamp is <= t, so only already-closed (and, per the
        # shift above, already-available) 1h bars are used - no lookahead.
        htf_aligned = pd.merge_asof(
            pd.DataFrame({"t": candles_3m.index}),
            pd.DataFrame({"t": htf_trend_1h.index, "htf_trend": htf_trend_1h.values}),
            on="t", direction="backward",
        ).set_index("t")["htf_trend"]

        periods = [10, 14]
        multipliers = [3.0, 4.0, 5.0]
        r_targets = [2.0, 2.5, 3.0, 3.5]
        htf_options = [("off", None), ("1h_align", htf_aligned)]

        max_hold_bars = round(args.max_hold_hours * 60 / TIMEFRAME_MINUTES["3m"])
        results = []
        for period, mult, r_tgt, (htf_label, htf_series) in itertools.product(
                periods, multipliers, r_targets, htf_options):
            result = run_one_combo(candles_3m, args.coin, "3m", period, mult, r_tgt, max_hold_bars,
                                    htf_trend_asof=htf_series, htf_label=htf_label)
            results.append(result)
            print(f"  period={period}, mult={mult}, r_target={r_tgt}, htf={htf_label} -> "
                  f"{result['trades']:5} trades | win_rate={result['win_rate']} | "
                  f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']}")

        results_df = pd.DataFrame(results)
        print(f"\n=== TUNING SWEEP RESULTS: {args.coin} (48 combos, 1 fetch) ===")
        print(results_df.to_string(index=False))
        valid = results_df.dropna(subset=["total_pnl"])
        if len(valid):
            best = valid.loc[valid["total_pnl"].idxmax()]
            print(f"\nBest combo for {args.coin}: period={best['period']}, mult={best['multiplier']}, "
                  f"r_target={best['r_target']}, htf={best['htf_filter']} -> ${best['total_pnl']:.2f} "
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
