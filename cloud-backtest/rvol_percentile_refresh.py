"""Builds/refreshes each coin's own RVOL percentile distribution from
real history, so the live scanner can rank a reversal candle's RVOL
against what's actually normal for THAT coin, instead of a fixed
1.0/1.5/2.0 scale applied uniformly across very different instruments.

Deliberately a SEPARATE script and workflow from the live 1-minute
scanner - this pulls ~15 API requests per coin (30 days of 1m data,
resampled to 3m), which is far too heavy to run every minute. Run this
once to bootstrap, then on a daily schedule to keep it current. The
live scanner only ever READS the small output file this produces - it
never does this heavier pull itself, so live run time is unaffected.

Stores only a compact percentile grid per coin (21 floats: the 0th,
5th, 10th, ... 100th percentile of that coin's RVOL distribution) -
NOT the raw history, which would grow the state file into hundreds of
thousands of numbers over time. Confirmed directly: this project's own
git-commit-per-run pattern is sensitive to file size overhead (a past
session found runs slowing down from state file growth), so keeping
this file tiny and rewritten (not appended-to) each refresh matters.

Usage:
  python3 rvol_percentile_refresh.py --coins BTC,ETH,SOL,... --days 30
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles

PERCENTILE_FILE = "rvol_percentiles.json"
DAILY_FILE = "daily_candles_30d.json"
PERCENTILE_GRID = list(range(0, 101, 5))  # 0,5,10,...,100 - 21 points


def compute_rvol_series(candles_3m):
    """Same RVOL formula as the live scanner's compute_indicators:
    current candle's volume / the prior 20-candle average (shifted by
    1 so a candle never inflates its own denominator)."""
    avg_volume_20 = candles_3m["volume"].rolling(20).mean().shift(1)
    rvol = candles_3m["volume"] / avg_volume_20.replace(0, float("nan"))
    return rvol.dropna()


def backfill_one_coin(coin, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    candles_1m = fetch_coindcx_klines(coin, "1m", start.date().isoformat(), now.isoformat(), stagger_delay=False)
    if len(candles_1m) < 100:
        return coin, None, f"only {len(candles_1m)} 1m candles returned, too little history"
    candles_3m = resample_candles(candles_1m, 3)
    rvol_series = compute_rvol_series(candles_3m)
    candles_1d = resample_candles(candles_1m, 1440)
    daily = []
    for idx, row in candles_1d.tail(days).iterrows():
        daily.append({"t": str(idx), "o": round(float(row["open"]), 8), "h": round(float(row["high"]), 8), "l": round(float(row["low"]), 8), "c": round(float(row["close"]), 8), "v": round(float(row["volume"]), 2)})
    if len(rvol_series) < 50:
        return coin, None, f"only {len(rvol_series)} valid RVOL samples, too few for a stable distribution"
    breakpoints = np.percentile(rvol_series.values, PERCENTILE_GRID).tolist()
    return coin, {
        "grid": PERCENTILE_GRID,
        "breakpoints": [round(float(b), 4) for b in breakpoints],
        "n_samples": int(len(rvol_series)),
        "computed_at": now.isoformat(),
        "days": days,
        "daily_candles": daily,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", type=str, required=True)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    coins = [c.strip().upper() for c in args.coins.split(",")]

    print(f"RVOL percentile refresh | coins={coins} | days={args.days}")

    result = {}
    daily_result = {}
    if os.path.exists(PERCENTILE_FILE):
        with open(PERCENTILE_FILE) as f:
            result = json.load(f)
    if os.path.exists(DAILY_FILE):
        with open(DAILY_FILE) as f:
            daily_result = json.load(f)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(backfill_one_coin, coin, args.days): coin for coin in coins}
        for future in as_completed(futures):
            coin = futures[future]
            try:
                _, data, err = future.result()
                if data is not None:
                    result[coin] = {k:v for k,v in data.items() if k != "daily_candles"}
                    daily_result[coin] = data.get("daily_candles", [])
                    print(f"  {coin}: {data['n_samples']} samples, p50={data['breakpoints'][10]}, "
                          f"p90={data['breakpoints'][18]}, p95={data['breakpoints'][19]}")
                else:
                    print(f"  {coin}: skipped - {err}")
            except Exception as e:
                print(f"  {coin}: ERROR - {e}")

    with open(PERCENTILE_FILE, "w") as f:
        json.dump(result, f, indent=2)
    with open(DAILY_FILE, "w") as f:
        json.dump(daily_result, f, indent=2)
    print(f"\nWrote {PERCENTILE_FILE} with {len(result)} coins")
    print(f"Wrote {DAILY_FILE} with {len(daily_result)} coins")


if __name__ == "__main__":
    main()
