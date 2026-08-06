"""
Idea #17b: non-overlapping CER buckets, fixing the real methodological
gap in idea #17's cumulative-threshold design - "CER >= 20" silently
included the >=25 and >=30 trades too, making it impossible to tell
whether the real effect is concentrated in a 16-25 band or smeared
across everything above 20. Non-overlapping buckets isolate each range
cleanly.

Also: tested across 5 coins at once (BTC, ETH, SOL, XRP, DOGE - this
project's own original DEFAULT_COINS set from main.py), per the
"universality" standard - the same CER definition and same bucket
boundaries, unchanged, run on every coin. A real, transferable effect
should show a similar pattern across multiple independent coins, not
just one.

More efficient than idea #17's design too: ONE backtest pass per coin
(cer_threshold=None, so every SuperTrend+VWAP-qualifying trade is kept
regardless of CER), then bucketed post-hoc by each trade's own recorded
cer_at_entry - not seven separate re-runs.

Usage:
  python3 run_supertrend_cer_buckets.py --coin SOL
  python3 run_supertrend_cer_buckets.py --coin SOL --days 365
"""
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from run_supertrend_cer_backtest import run_backtest
from fee_model import apply_fees_and_interest, apply_dollar_pnl

BUCKET_EDGES = [0, 8, 12, 16, 20, 25, 30, float("inf")]
BUCKET_LABELS = ["0-8", "8-12", "12-16", "16-20", "20-25", "25-30", "30+"]


def bucket_metrics(trades_in_bucket):
    if len(trades_in_bucket) == 0:
        return {"n": 0, "gross_r": None, "cost_r": None, "net_r": None, "win_rate": None, "pf": None}
    gross_profit = trades_in_bucket[trades_in_bucket["dollar_pnl"] > 0]["dollar_pnl"].sum()
    gross_loss = abs(trades_in_bucket[trades_in_bucket["dollar_pnl"] <= 0]["dollar_pnl"].sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    return {
        "n": len(trades_in_bucket),
        "gross_r": round(trades_in_bucket["r_achieved"].mean(), 4),
        "cost_r": round(trades_in_bucket["fee_interest_r_cost"].mean(), 4),
        "net_r": round(trades_in_bucket["net_r"].mean(), 4),
        "win_rate": round((trades_in_bucket["net_r"] > 0).mean() * 100, 1),
        "pf": round(pf, 3) if pf != float("inf") else "inf",
    }


def run_coin(coin, days, period, multiplier, r_target, max_hold_bars):
    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=days)
    fetch_end_time = now.isoformat()

    print(f"=== {coin}: fetching 1m data ===")
    candles_1m = fetch_coindcx_klines(coin, "1m", str(start_date), fetch_end_time)
    candles_3m = resample_candles(candles_1m, 3)
    print(f"{coin}: {len(candles_1m)} 1m candles -> {len(candles_3m)} 3m candles")

    trades = run_backtest(candles_3m, period=period, multiplier=multiplier, r_target=r_target,
                           cer_threshold=None, max_hold_bars=max_hold_bars)
    print(f"{coin}: {len(trades)} total SuperTrend+VWAP-qualifying trades (all CER values, unfiltered)")
    if len(trades) == 0:
        return None

    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "supertrend_vwap_cer_bucket"
    trades = apply_fees_and_interest(trades, bar_minutes=3)
    trades = apply_dollar_pnl(trades)

    print(f"\n{coin} CER bucket results:")
    print(f"  {'bucket':>8} {'n':>5} {'gross_R':>9} {'cost_R':>8} {'net_R':>8} {'win%':>6} {'PF':>6}")
    rows = []
    for lo, hi, label in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:], BUCKET_LABELS):
        bucket_trades = trades[(trades["cer_at_entry"] >= lo) & (trades["cer_at_entry"] < hi)]
        m = bucket_metrics(bucket_trades)
        print(f"  {label:>8} {m['n']:>5} {str(m['gross_r']):>9} {str(m['cost_r']):>8} "
              f"{str(m['net_r']):>8} {str(m['win_rate']):>6} {str(m['pf']):>6}")
        rows.append({"coin": coin, "bucket": label, **m})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--max-hold-hours", type=float, default=4.0)
    parser.add_argument("--period", type=int, default=14)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--r-target", type=float, default=2.0)
    args = parser.parse_args()

    max_hold_bars = round(args.max_hold_hours * 60 / 3)
    print(f"Idea #17b: Non-overlapping CER buckets | {args.coin} | 3m SuperTrend+VWAP | "
          f"period={args.period}, mult={args.multiplier}, r_target={args.r_target}, {args.days}d\n")

    rows = run_coin(args.coin, args.days, args.period, args.multiplier, args.r_target, max_hold_bars)
    if rows:
        df = pd.DataFrame(rows)
        print(f"\n=== FULL BUCKET TABLE: {args.coin} ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
