"""
Idea #17: 3m SuperTrend + VWAP (idea #16's best combo: period=14,
mult=3.0, r_target=2.0) + a Cost-to-Edge Ratio (CER) filter, swept
across thresholds - the next narrow, isolated experiment per the
refined proposal, tested before any cooldown/regime-classifier/
composite-score additions.

CER DEFINITION (the critical, zero-lookahead-safe methodological
choice, per the proposal's own flag that this is the most important
decision in the experiment):

  expected_move_pct = stop_distance_pct * r_target

  stop_distance_pct is already computed at entry (SuperTrend's own line
  vs entry price - the SAME quantity already used for the stop-loss,
  not a new estimate). r_target is a fixed, known parameter for this
  run. Multiplying gives the DISTANCE TO THE FIXED TARGET in percentage
  terms - fully known at entry, using nothing about what actually
  happens after the trade opens.

  This is NOT circular despite being derived from r_target: stop_distance_pct
  itself varies trade-to-trade with CURRENT VOLATILITY (SuperTrend's ATR-
  based band width changes constantly), so expected_move_pct is a real,
  varying-per-trade quantity, not a constant scaling factor - a trade
  entered during high volatility gets a wider stop and thus a bigger
  absolute expected move, exactly the intent.

  round_trip_cost_pct = 2 x EFFECTIVE_TAKER_FEE (from fee_model.py,
  CoinDCX's own published rate, already used throughout this whole
  project - not a new arbitrary number). Margin interest deliberately
  excluded from this pre-trade estimate since fee_model.py's own
  documentation notes it's "a much smaller effect than fees for any
  trade under a few days" - the ACTUAL post-hoc cost (fee + interest)
  still gets applied for real via apply_fees_and_interest() when
  computing net R; CER only uses the fee portion as a pre-trade proxy.

  CER = expected_move_pct / round_trip_cost_pct

Reports exactly the metrics requested: Trades, Gross R/trade,
Cost/trade, Net R/trade, Win rate, Avg winner R, Avg loser R, PF,
Max DD, Total net P&L - per threshold, so the key question (does net
expectancy improve MONOTONICALLY as the CER threshold rises) can be
checked directly rather than just picking whichever threshold
maximizes final P&L.

Usage:
  python3 run_supertrend_cer_backtest.py --coin SOL --sweep-all
"""
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import calculate_supertrend, calculate_vwap
from fee_model import apply_fees_and_interest, apply_dollar_pnl, EFFECTIVE_TAKER_FEE

MIN_CANDLES_NEEDED = 30
LOOKBACK_WINDOW_BARS = 100
ROUND_TRIP_COST_PCT = 2 * EFFECTIVE_TAKER_FEE  # entry + exit, both taker legs


def run_backtest(candles, period=14, multiplier=3.0, r_target=2.0, cer_threshold=None, max_hold_bars=48):
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
                    "bars_held": bars_held, "cer_at_entry": pos["cer"],
                })
                open_position = None
        else:
            st = calculate_supertrend(window, period=period, multiplier=multiplier)
            if st["flipped"]:
                direction = "long" if st["trend"] == "up" else "short"

                vwap = calculate_vwap(window)
                if vwap is None:
                    continue
                vwap_agrees = (direction == "long" and current_price > vwap) or \
                              (direction == "short" and current_price < vwap)
                if not vwap_agrees:
                    continue

                stop_price = st["value"]
                stop_distance = abs(current_price - stop_price)
                if stop_distance <= 0:
                    continue
                stop_distance_pct = stop_distance / current_price
                expected_move_pct = stop_distance_pct * r_target
                cer = expected_move_pct / ROUND_TRIP_COST_PCT

                if cer_threshold is not None and cer < cer_threshold:
                    continue  # the CER filter itself - skip low-quality-relative-to-cost setups

                target_price = (current_price + stop_distance * r_target if direction == "long"
                                 else current_price - stop_distance * r_target)
                open_position = {
                    "direction": direction, "entry": current_price, "entry_time": t, "entry_index": i,
                    "stop": stop_price, "target": target_price, "cer": cer,
                }

    return pd.DataFrame(trades)


def max_drawdown(dollar_pnl_series):
    cumulative = dollar_pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    return float(drawdown.min())  # most negative point = max drawdown (as a negative $ figure)


def run_one_threshold(candles, coin, period, multiplier, r_target, cer_threshold, max_hold_bars):
    trades = run_backtest(candles, period=period, multiplier=multiplier, r_target=r_target,
                           cer_threshold=cer_threshold, max_hold_bars=max_hold_bars)
    label = "no_filter" if cer_threshold is None else f">={cer_threshold}"
    if len(trades) == 0:
        return {"coin": coin, "cer_threshold": label, "trades": 0}

    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "supertrend_vwap_cer"
    trades = apply_fees_and_interest(trades, bar_minutes=3)
    trades = apply_dollar_pnl(trades)

    wins = trades[trades["net_r"] > 0]
    losses = trades[trades["net_r"] <= 0]
    gross_profit = trades[trades["dollar_pnl"] > 0]["dollar_pnl"].sum()
    gross_loss = abs(trades[trades["dollar_pnl"] <= 0]["dollar_pnl"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "coin": coin, "cer_threshold": label,
        "trades": len(trades),
        "gross_r_per_trade": round(trades["r_achieved"].mean(), 4),
        "cost_per_trade": round(trades["fee_interest_r_cost"].mean(), 4),
        "net_r_per_trade": round(trades["net_r"].mean(), 4),
        "win_rate": round((trades["net_r"] > 0).mean() * 100, 1),
        "avg_winner_r": round(wins["net_r"].mean(), 4) if len(wins) else None,
        "avg_loser_r": round(losses["net_r"].mean(), 4) if len(losses) else None,
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "max_drawdown": round(max_drawdown(trades["dollar_pnl"]), 2),
        "total_net_pnl": round(trades["dollar_pnl"].sum(), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="SOL")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--max-hold-hours", type=float, default=4.0)
    parser.add_argument("--period", type=int, default=14)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--r-target", type=float, default=2.0)
    parser.add_argument("--sweep-all", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    print(f"Idea #17: 3m SuperTrend + VWAP + CER filter | {args.coin} | "
          f"{start_date} to {now.date()} ({args.days}d)")
    print(f"Base settings (idea #16's best): period={args.period}, mult={args.multiplier}, r_target={args.r_target}, VWAP=True")
    print(f"Round-trip cost used for CER: {ROUND_TRIP_COST_PCT*100:.4f}% (2x CoinDCX taker fee, fee_model.py)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_3m = resample_candles(candles_1m, 3)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_3m)} 3m candles\n")

    max_hold_bars = round(args.max_hold_hours * 60 / 3)
    # Thresholds widened from the proposal's example (1.5-4.0) after
    # checking the REAL CER distribution this strategy produces: median
    # ~16, range ~10-34. CoinDCX's actual fee rate (0.059%/leg) is simply
    # too low relative to typical 3m ATR-based stop distances (~1% of
    # price) for thresholds in the 1.5-4 range to ever bind - every
    # trade already clears that bar, making the filter a silent no-op.
    # Shipping the proposal's literal example range would have produced
    # an uninformative flat line (identical results at every threshold).
    thresholds = [None, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0]

    results = []
    for cer_threshold in thresholds:
        result = run_one_threshold(candles_3m, args.coin, args.period, args.multiplier, args.r_target,
                                    cer_threshold, max_hold_bars)
        results.append(result)
        print(f"  CER {result['cer_threshold']:>10} -> trades={result.get('trades')}, "
              f"gross_R={result.get('gross_r_per_trade')}, cost_R={result.get('cost_per_trade')}, "
              f"net_R={result.get('net_r_per_trade')}, win%={result.get('win_rate')}, "
              f"PF={result.get('profit_factor')}, maxDD=${result.get('max_drawdown')}, "
              f"total_pnl=${result.get('total_net_pnl')}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL RESULTS TABLE: {args.coin} ===")
    print(results_df.to_string(index=False))

    print("\n=== MONOTONICITY CHECK: does net R/trade rise as CER threshold rises? ===")
    valid = results_df.dropna(subset=["net_r_per_trade"])
    net_r_values = valid["net_r_per_trade"].tolist()
    is_monotonic = all(net_r_values[i] <= net_r_values[i + 1] for i in range(len(net_r_values) - 1))
    print("Net R/trade sequence:", net_r_values)
    print("Monotonically non-decreasing:", is_monotonic)


if __name__ == "__main__":
    main()
