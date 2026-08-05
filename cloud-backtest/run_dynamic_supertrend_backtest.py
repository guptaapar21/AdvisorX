"""
Idea #13: 3m SuperTrend (the one timeframe from idea #12 that showed a
real, consistent gross edge - 89% of combos positive) as the base
signal, with exits DYNAMICALLY sized by 1m confirmation strength
instead of a single fixed r_target.

At every 3m SuperTrend flip:
  - Look at the 1m EMA(9)/EMA(21) relationship, using ONLY 1m candles
    that have themselves already closed by the same moment the 3m bar
    closed (zero lookahead preserved across timeframes, not just within
    one).
  - "contradicting" -> skip the trade entirely. This directly targets
    idea #12's real problem (genuine edge, too many trades) by cutting
    volume at the entry gate, not just changing the exit.
  - "strong"         -> use a WIDER r_target (let real-conviction moves run).
  - "weak"           -> use a TIGHTER r_target (bank marginal-conviction
                        moves fast rather than risk giving them back).

Same bounded-window, zero-lookahead loop already proven across ideas
#10/#11/#12 - the only new wrinkle is aligning two DIFFERENT timeframes
(3m entries, 1m confirmation) at the same zero-lookahead reference
point, handled explicitly below.

Usage:
  python3 run_dynamic_supertrend_backtest.py --coin SOL --period 10 --multiplier 3.0 --sweep-all
"""
import argparse
import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import calculate_supertrend, classify_1m_confirmation
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 30
LOOKBACK_WINDOW_BARS = 100          # for the 3m SuperTrend window - same bound/reasoning as idea #12
CONFIRMATION_1M_WINDOW_MINUTES = 60  # how much 1m history to hand classify_1m_confirmation each time


def run_dynamic_backtest(candles_3m, candles_1m, period=10, multiplier=3.0,
                          r_target_strong=2.5, r_target_weak=1.2, max_hold_bars=80):
    trades = []
    open_position = None
    n = len(candles_3m)

    for i in range(MIN_CANDLES_NEEDED, n):
        window_3m = candles_3m.iloc[max(0, i - LOOKBACK_WINDOW_BARS):i + 1]
        t = candles_3m.index[i]
        current_price = window_3m["close"].iloc[-1]

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
                    "bars_held": bars_held, "confirmation_tier": pos["confirmation_tier"],
                })
                open_position = None
        else:
            st = calculate_supertrend(window_3m, period=period, multiplier=multiplier)
            if st["flipped"]:
                direction = "long" if st["trend"] == "up" else "short"

                # ZERO-LOOKAHEAD ACROSS TIMEFRAMES: the last 1m candle
                # included must be the one that closes AT OR BEFORE the
                # 3m bar's own close time, not the one merely LABELED
                # with that timestamp. A candle labeled with open time X
                # (this project's convention throughout) only closes at
                # X + interval - so the 1m candle labeled exactly at the
                # 3m bar's close time hasn't closed yet at that instant.
                # Confirmed directly: for a 3m bar closing at 00:06:00,
                # the 1m candle labeled 00:06:00 spans [00:06:00,
                # 00:07:00) and only closes at 00:07:00 - one minute
                # AFTER the 3m bar's own reference point. Subtracting one
                # minute excludes it correctly.
                bar_close_time = t + pd.Timedelta(minutes=3) - pd.Timedelta(minutes=1)
                window_1m = candles_1m.loc[:bar_close_time].tail(CONFIRMATION_1M_WINDOW_MINUTES)

                tier = classify_1m_confirmation(window_1m, direction) if len(window_1m) > 0 else "weak"
                if tier == "contradicting":
                    continue

                r_target = r_target_strong if tier == "strong" else r_target_weak
                stop_price = st["value"]
                stop_distance = abs(current_price - stop_price)
                if stop_distance > 0:
                    target_price = (current_price + stop_distance * r_target if direction == "long"
                                     else current_price - stop_distance * r_target)
                    open_position = {
                        "direction": direction, "entry": current_price, "entry_time": t, "entry_index": i,
                        "stop": stop_price, "target": target_price, "confirmation_tier": tier,
                    }

    return pd.DataFrame(trades)


def run_one_combo(candles_3m, candles_1m, coin, period, multiplier, r_target_strong, r_target_weak, max_hold_bars):
    trades = run_dynamic_backtest(candles_3m, candles_1m, period=period, multiplier=multiplier,
                                   r_target_strong=r_target_strong, r_target_weak=r_target_weak,
                                   max_hold_bars=max_hold_bars)
    if len(trades) == 0:
        return {"coin": coin, "period": period, "multiplier": multiplier, "r_target_strong": r_target_strong,
                "r_target_weak": r_target_weak, "trades": 0, "win_rate": None, "gross_expected_r": None,
                "total_pnl": None, "tier_breakdown": {}, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "dynamic_supertrend_3m"
    trades = apply_fees_and_interest(trades, bar_minutes=3)
    trades = apply_dollar_pnl(trades)
    return {
        "coin": coin, "period": period, "multiplier": multiplier,
        "r_target_strong": r_target_strong, "r_target_weak": r_target_weak,
        "trades": len(trades), "win_rate": round((trades["dollar_pnl"] > 0).mean() * 100, 1),
        "gross_expected_r": round(trades["r_achieved"].mean(), 4),
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "tier_breakdown": trades["confirmation_tier"].value_counts().to_dict(),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, required=True)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--max-hold-hours", type=float, default=4.0)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--r-target-strong", type=float, default=2.5)
    parser.add_argument("--r-target-weak", type=float, default=1.2)
    parser.add_argument("--sweep-all", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    print(f"{'Idea #13 sweep' if args.sweep_all else 'Idea #13'}: Dynamic 3m SuperTrend | {args.coin} | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_3m = resample_candles(candles_1m, 3)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_3m)} 3m candles"
          + (" (ONE fetch, reused for every combo below)" if args.sweep_all else ""))

    if not args.sweep_all:
        max_hold_bars = round(args.max_hold_hours * 60 / 3)
        result = run_one_combo(candles_3m, candles_1m, args.coin, args.period, args.multiplier,
                                args.r_target_strong, args.r_target_weak, max_hold_bars)
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
              f"Gross expected R: {result['gross_expected_r']} | Total $ P&L: {result['total_pnl']}")
        print(f"Confirmation tier breakdown: {result['tier_breakdown']}")
        print(f"Exit reason breakdown: {result['exit_breakdown']}")
        return

    periods = [7, 10, 14]
    multipliers = [2.0, 3.0]
    strong_weak_pairs = [(2.0, 1.0), (2.5, 1.2), (3.0, 1.5)]
    max_hold_hours_options = [4.0, 6.0]

    results = []
    for period, mult, (r_strong, r_weak), max_hold_h in itertools.product(periods, multipliers, strong_weak_pairs, max_hold_hours_options):
        max_hold_bars = round(max_hold_h * 60 / 3)
        result = run_one_combo(candles_3m, candles_1m, args.coin, period, mult, r_strong, r_weak, max_hold_bars)
        result["max_hold_hours"] = max_hold_h
        results.append(result)
        print(f"  period={period}, mult={mult}, r_strong={r_strong}, r_weak={r_weak}, hold={max_hold_h}h -> "
              f"{result['trades']:5} trades | win_rate={result['win_rate']} | "
              f"gross_R={result['gross_expected_r']} | total_pnl={result['total_pnl']} | "
              f"tiers={result['tier_breakdown']}")

    results_df = pd.DataFrame(results)
    print(f"\n=== FULL SWEEP RESULTS: {args.coin} (36 combos, 1 fetch) ===")
    print(results_df.drop(columns=["tier_breakdown", "exit_breakdown"]).to_string(index=False))

    valid = results_df.dropna(subset=["total_pnl"])
    if len(valid):
        best = valid.loc[valid["total_pnl"].idxmax()]
        print(f"\nBest combo for {args.coin}: period={best['period']}, mult={best['multiplier']}, "
              f"r_strong={best['r_target_strong']}, r_weak={best['r_target_weak']}, hold={best['max_hold_hours']}h -> "
              f"${best['total_pnl']:.2f} ({best['trades']} trades, {best['win_rate']}% win rate, "
              f"gross R={best['gross_expected_r']})")


if __name__ == "__main__":
    main()
