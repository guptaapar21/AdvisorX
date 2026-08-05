"""
Standalone backtest for Idea #10 (5m Momentum Scalp). Deliberately NOT
using backtest_engine.py's loop - a fully separate, much simpler
bar-by-bar loop, specifically so the zero-lookahead property is trivial
to verify by reading this file alone, without needing to also hold the
main engine's more complex staged-exit/multi-idea logic in your head.

ZERO LOOKAHEAD, explicitly: at bar i, the ONLY data touched is
candles.iloc[:i+1] - candle i itself (already closed, its close price is
"now") and everything before it. Nothing from i+1 onward is ever read.
Entry decision and exit decision both happen strictly on this bounded
slice - confirmed by the single line `window = candles.iloc[:i+1]` that
every other calculation in this loop derives from.

Usage:
  python3 run_momentum_scalp_backtest.py --coin SOL --atr-multiplier 1.2 --r-target 1.5
"""
import argparse
import itertools
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import detect_momentum_scalp_signal, calculate_scalp_stop_target
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 30  # enough for EMA(13)+lookback with margin


def run_momentum_scalp(candles_5m, atr_multiplier=1.2, r_target=1.5, use_macd=False, use_rsi=False,
                        max_hold_bars=48):
    """candles_5m: full DataFrame, chronologically ordered, indexed by
    candle OPEN time (same convention as the rest of this project).
    Returns a trades DataFrame with the same core columns the rest of
    the project's fee_model.py expects (entry_time, exit_time,
    entry_price, exit_price, direction, r_achieved, symbol, strategy)."""
    trades = []
    open_position = None
    n = len(candles_5m)

    for i in range(MIN_CANDLES_NEEDED, n):
        window = candles_5m.iloc[:i + 1]   # <-- the ONE slice everything below derives from.
        t = candles_5m.index[i]            # Nothing past this row is ever touched in this loop.
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
                    "stages_done": 0, "leverage": 5, "stop_distance_pct": abs(pos["entry"] - pos["stop"]) / pos["entry"],
                    "bars_held": bars_held,
                })
                open_position = None
        else:
            signal = detect_momentum_scalp_signal(window, use_macd=use_macd, use_rsi=use_rsi)
            if signal["action"] != "wait":
                direction = signal["action"]
                sl = calculate_scalp_stop_target(window, direction, current_price,
                                                  atr_multiplier=atr_multiplier, r_target=r_target)
                if sl["stop_distance"] > 0:
                    open_position = {
                        "direction": direction, "entry": current_price, "entry_time": t, "entry_index": i,
                        "stop": sl["stop_price"], "target": sl["target_price"],
                    }

    return pd.DataFrame(trades)


def run_one_combo(candles_5m, coin, atr_multiplier, r_target, use_macd, use_rsi, max_hold_bars):
    """One parameter combo's full result, given already-fetched candles.
    Factored out so --sweep-all can call this 24 times per fetch instead
    of fetching 24 times."""
    trades = run_momentum_scalp(candles_5m, atr_multiplier=atr_multiplier, r_target=r_target,
                                 use_macd=use_macd, use_rsi=use_rsi, max_hold_bars=max_hold_bars)
    if len(trades) == 0:
        return {"coin": coin, "atr_multiplier": atr_multiplier, "r_target": r_target,
                "use_macd": use_macd, "use_rsi": use_rsi, "trades": 0,
                "win_rate": None, "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "momentum_scalp_5m"
    trades = apply_fees_and_interest(trades, bar_minutes=5)
    trades = apply_dollar_pnl(trades)
    return {
        "coin": coin, "atr_multiplier": atr_multiplier, "r_target": r_target,
        "use_macd": use_macd, "use_rsi": use_rsi, "trades": len(trades),
        "win_rate": round((trades["dollar_pnl"] > 0).mean() * 100, 1),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--max-hold-hours", type=float, default=4.0)
    parser.add_argument("--atr-multiplier", type=float, default=1.2)
    parser.add_argument("--r-target", type=float, default=1.5)
    parser.add_argument("--use-macd", action="store_true")
    parser.add_argument("--use-rsi", action="store_true")
    parser.add_argument("--sweep-all", action="store_true",
                         help="Fetch this coin's data ONCE, then run every parameter combination "
                              "in-memory instead of re-fetching per combo (24x fewer real API "
                              "requests per coin - the original per-combo-job design hammered "
                              "CoinDCX's public API with 12,480 requests per coin instead of 520).")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()
    max_hold_bars = round(args.max_hold_hours * 60 / 5)

    print(f"{'Idea #10 sweep' if args.sweep_all else 'Idea #10'}: 5m Momentum Scalp | {args.coin} | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_5m = resample_candles(candles_1m, 5)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_5m)} 5m candles "
          f"(ONE fetch, reused for every combo below)" if args.sweep_all else "")

    if not args.sweep_all:
        trades = run_momentum_scalp(candles_5m, atr_multiplier=args.atr_multiplier, r_target=args.r_target,
                                     use_macd=args.use_macd, use_rsi=args.use_rsi, max_hold_bars=max_hold_bars)
        print(f"{args.coin}: {len(trades)} trades")
        if len(trades) == 0:
            print("No trades - nothing further to report.")
            return
        trades["symbol"] = args.coin
        trades["strategy"] = "momentum_scalp_5m"
        trades = apply_fees_and_interest(trades, bar_minutes=5)
        trades = apply_dollar_pnl(trades)
        win_rate = (trades["dollar_pnl"] > 0).mean() * 100
        total_pnl = trades["dollar_pnl"].sum()
        exit_breakdown = trades["exit_reason"].value_counts().to_dict()
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {len(trades)} | Win rate: {win_rate:.1f}% | Total $ P&L: {total_pnl:.2f}")
        print(f"Exit reason breakdown: {exit_breakdown}")
        return

    atr_multipliers = [1.0, 1.25, 1.5]
    r_targets = [1.5, 2.0]
    confirmation_sets = [
        ("core3", False, False),
        ("core3_macd", True, False),
        ("core3_rsi", False, True),
        ("core3_macd_rsi", True, True),
    ]

    results = []
    for atr_mult, r_tgt, (conf_name, use_macd, use_rsi) in itertools.product(atr_multipliers, r_targets, confirmation_sets):
        result = run_one_combo(candles_5m, args.coin, atr_mult, r_tgt, use_macd, use_rsi, max_hold_bars)
        result["confirmations"] = conf_name
        results.append(result)
        print(f"  atr={atr_mult}, r_target={r_tgt}, conf={conf_name:15} -> "
              f"{result['trades']:4} trades | win_rate={result['win_rate']} | total_pnl={result['total_pnl']}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL SWEEP RESULTS: {args.coin} (24 combos, 1 fetch) ===")
    print(results_df.to_string(index=False))

    valid = results_df.dropna(subset=["total_pnl"])
    if len(valid):
        best = valid.loc[valid["total_pnl"].idxmax()]
        print(f"\nBest combo for {args.coin}: atr={best['atr_multiplier']}, r_target={best['r_target']}, "
              f"conf={best['confirmations']} -> ${best['total_pnl']:.2f} ({best['trades']} trades, "
              f"{best['win_rate']}% win rate)")


if __name__ == "__main__":
    main()
