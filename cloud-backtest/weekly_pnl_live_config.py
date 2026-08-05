"""
Weekly P&L for the CURRENT LIVE configuration, all 3 coins, last 365 days.

This intentionally does NOT read any BT_* env vars or sweep anything - it
hardcodes exactly what's deployed right now in the live JS bot, so the
numbers here answer "what would my actual current setup have made each
week for the last year", not a hypothetical variant.

Live config as of Aug 2026 (keep this in sync manually if you retune
anything live - same manual-sync caveat as the takeProfitManagement.js
preview message before it was fixed to read a shared table; there isn't
a live/backtest shared source of truth to import from here):

  SOL:  min_score=80, hold=18h,  drift stop-tighten ON (1.2x ATR)
  DOGE: min_score=79, hold=48h,  stage_fractions=15%/25%/60%
  ETH:  min_score=81, hold=60h,  OBV bonus=8, BTC trend bonus=5 (mag=20)

Usage:
  python3 weekly_pnl_live_config.py [--days 365]

Output:
  - Prints a weekly table per coin + combined total to console
  - Writes results/weekly_pnl_live_config_<timestamp>.csv
"""
import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from backtest_engine import run_backtest
from fee_model import apply_fees_and_interest, apply_dollar_pnl

LIVE_CONFIG = {
    "SOL": {
        "min_score": 80, "max_hold_hours": 18,
        "use_adverse_drift": True, "drift_stop_tighten_enabled": True,
        "drift_stop_tighten_atr_multiplier": 1.2,
    },
    "DOGE": {
        "min_score": 79, "max_hold_hours": 48,
        "stage_fractions": (0.15, 0.25, 0.60),
    },
    "ETH": {
        "min_score": 81, "max_hold_hours": 60,
        "obv_confirmation_bonus": 8,
        "use_btc_trend_bonus": True, "btc_trend_bonus": 5, "btc_min_score_magnitude": 20,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--coin", type=str, default=None,
                         help="Run just this one coin (SOL/DOGE/ETH) instead of all 3. "
                              "Used by the parallel per-coin workflow to avoid the sequential "
                              "4x-fetch timeout that killed the first full run at 60 minutes.")
    args = parser.parse_args()

    coins_to_run = {args.coin: LIVE_CONFIG[args.coin]} if args.coin else LIVE_CONFIG

    end_date = datetime.now(timezone.utc).date()  # display/labeling only - NOT what gets fetched
    fetch_end_time = datetime.now(timezone.utc).isoformat()  # the REAL current moment -
                                       # str(end_date) alone parses as midnight of today,
                                       # silently dropping every hour since midnight from the fetch
    start_date = end_date - timedelta(days=args.days)
    print(f"Weekly P&L, live config, {start_date} to {end_date} ({args.days}d)")
    print("Engine: POST-LOOKAHEAD-FIX (_closed_bucket_candles now uses a strict, zero-slack "
          "closed-bucket check - see backtest_engine.py). Every number below was computed after "
          "that fix, not before it.\n")

    # ETH needs BTC's own candles for the BTC trend bonus - fetch once,
    # shared, same as main.py does (not once per coin). Skipped entirely
    # when ETH isn't being run this call - saves ~9 minutes of fetch time
    # for a SOL-only or DOGE-only job, which matters now that each coin
    # runs as its own parallel matrix job.
    btc_renamed = None
    if "ETH" in coins_to_run:
        print("=== BTC: fetching 1m data (needed for ETH's BTC trend bonus) ===")
        btc_1m = fetch_coindcx_klines("BTC", "1m", str(start_date), fetch_end_time)
        btc_5m = resample_candles(btc_1m, 5)
        btc_renamed = btc_5m.rename(columns={
            "open": "btc_open", "high": "btc_high", "low": "btc_low",
            "close": "btc_close", "volume": "btc_volume",
        })
        print(f"  BTC: {len(btc_1m)} 1m candles -> {len(btc_5m)} 5m candles\n")

    all_trades = []
    baseline_totals = {}

    for coin, cfg in coins_to_run.items():
        print(f"=== {coin}: fetching 1m data ===")
        candles_1m = fetch_coindcx_klines(coin, "1m", str(start_date), fetch_end_time)
        candles_5m = resample_candles(candles_1m, 5)
        print(f"  {coin}: {len(candles_1m)} 1m candles -> {len(candles_5m)} 5m candles")

        if coin == "ETH":
            candles_5m = candles_5m.join(
                btc_renamed[["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]], how="left"
            )
            n_matched = candles_5m["btc_close"].notna().sum()
            print(f"  {coin}: BTC data merged - {n_matched}/{len(candles_5m)} bars matched")

        cfg = dict(cfg)  # don't mutate the shared module-level LIVE_CONFIG - .pop() below would
                         # break a second call to main() in the same process otherwise
        max_hold_bars = round(cfg.pop("max_hold_hours") * 60 / 5)
        min_score = cfg.pop("min_score")

        # BASELINE: same coin, same min_score/hold, but NONE of the
        # deployed features - proves the edge against a clean control
        # under the SAME corrected engine, not just reported in isolation.
        baseline_trades, _ = run_backtest(
            coin, candles_5m, strategy="conservative",
            min_score=min_score, max_hold_bars=max_hold_bars,
        )
        if len(baseline_trades):
            baseline_trades = apply_fees_and_interest(baseline_trades, bar_minutes=5)
            baseline_trades = apply_dollar_pnl(baseline_trades)
            baseline_total = baseline_trades["dollar_pnl"].sum()
        else:
            baseline_total = 0.0
        print(f"  {coin} BASELINE (no features, same engine): {len(baseline_trades)} trades, ${baseline_total:.2f}")
        baseline_totals[coin] = baseline_total

        trades, _ = run_backtest(
            coin, candles_5m, strategy="conservative",
            min_score=min_score, max_hold_bars=max_hold_bars,
            **cfg,
        )
        print(f"  {coin} DEPLOYED CONFIG: {len(trades)} trades\n")

        if len(trades) == 0:
            continue

        trades = apply_fees_and_interest(trades, bar_minutes=5)
        trades = apply_dollar_pnl(trades)
        trades["coin"] = coin
        all_trades.append(trades)

    if not all_trades:
        print("No trades produced by any coin - nothing to break down.")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    combined["exit_time"] = pd.to_datetime(combined["exit_time"])
    combined["week"] = combined["exit_time"].dt.to_period("W-MON").apply(lambda p: p.start_time.date())

    print("\n" + "=" * 70)
    print("PER-COIN WEEKLY BREAKDOWN")
    print("=" * 70)
    for coin in coins_to_run:
        coin_trades = combined[combined["coin"] == coin]
        if coin_trades.empty:
            continue
        weekly = coin_trades.groupby("week").agg(
            trades=("dollar_pnl", "count"),
            dollar_pnl=("dollar_pnl", "sum"),
            win_rate=("dollar_pnl", lambda x: round((x > 0).mean() * 100, 1)),
        ).reset_index()
        weekly["dollar_pnl"] = weekly["dollar_pnl"].round(2)
        weekly["cumulative"] = weekly["dollar_pnl"].cumsum().round(2)
        print(f"\n--- {coin} ---")
        print(weekly.to_string(index=False))
        print(f"{coin} TOTAL: {coin_trades['dollar_pnl'].sum():.2f} across {len(coin_trades)} trades")

    print("\n" + "=" * 70)
    print("COMBINED WEEKLY (all 3 coins together)")
    print("=" * 70)
    combined_weekly = combined.groupby("week").agg(
        trades=("dollar_pnl", "count"),
        dollar_pnl=("dollar_pnl", "sum"),
        win_rate=("dollar_pnl", lambda x: round((x > 0).mean() * 100, 1)),
    ).reset_index()
    combined_weekly["dollar_pnl"] = combined_weekly["dollar_pnl"].round(2)
    combined_weekly["cumulative"] = combined_weekly["dollar_pnl"].cumsum().round(2)
    print(combined_weekly.to_string(index=False))
    print(f"\nGRAND TOTAL: {combined['dollar_pnl'].sum():.2f} across {len(combined)} trades")

    print("\n" + "=" * 70)
    print("PROOF TABLE: baseline vs deployed config, SAME corrected engine")
    print("=" * 70)
    grand_baseline = 0.0
    grand_deployed = 0.0
    for coin in coins_to_run:
        coin_trades = combined[combined["coin"] == coin]
        deployed_total = coin_trades["dollar_pnl"].sum() if not coin_trades.empty else 0.0
        base_total = baseline_totals.get(coin, 0.0)
        delta = deployed_total - base_total
        pct = (delta / abs(base_total) * 100) if base_total != 0 else float("nan")
        verdict = "EDGE HOLDS" if delta > 0 else "EDGE DID NOT SURVIVE THE FIX" if base_total != 0 else "N/A"
        print(f"{coin:5} baseline=${base_total:8.2f}  deployed=${deployed_total:8.2f}  "
              f"delta=${delta:8.2f} ({pct:+.1f}%)  -> {verdict}")
        grand_baseline += base_total
        grand_deployed += deployed_total
    print(f"\nTOTAL   baseline=${grand_baseline:.2f}  deployed=${grand_deployed:.2f}  "
          f"delta=${grand_deployed - grand_baseline:.2f}")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    coin_tag = args.coin if args.coin else "ALL"
    out_path = f"results/weekly_pnl_live_config_{coin_tag}_{timestamp}.csv"
    combined_weekly.insert(0, "coin", "ALL")
    per_coin_weeklies = []
    for coin in coins_to_run:
        coin_trades = combined[combined["coin"] == coin]
        if coin_trades.empty:
            continue
        w = coin_trades.groupby("week").agg(
            trades=("dollar_pnl", "count"), dollar_pnl=("dollar_pnl", "sum"),
        ).reset_index()
        w.insert(0, "coin", coin)
        per_coin_weeklies.append(w)
    final_csv = pd.concat(per_coin_weeklies + [combined_weekly[["coin", "week", "trades", "dollar_pnl"]]], ignore_index=True)
    final_csv.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
