"""
Two direct answers, current DEPLOYED live config (SOL ATR-tighten still
included - not reverted, per explicit instruction), corrected zero-
lookahead engine:

  1. Overall win rate - to compare directly against real observed live
     experience (currently ~1 win in 6-7 trades, ~14-17%), not a
     reassuring aggregate from a much longer window.
  2. Trade-by-trade log for exits in the last 7 days from right now -
     so each real trade can be checked individually, not just summarized.

Reuses LIVE_CONFIG from weekly_pnl_live_config.py directly (no duplicate
table - the DRY mistake already found and fixed once this session isn't
getting repeated here).

Usage:
  python3 win_rate_and_recent_trades.py [--history-days 365]
"""
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from backtest_engine import run_backtest
from fee_model import apply_fees_and_interest, apply_dollar_pnl
from weekly_pnl_live_config import LIVE_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-days", type=int, default=365,
                         help="How far back to fetch for computing the OVERALL win rate. "
                              "The last-7-days trade log is always exactly the most recent 7 "
                              "days regardless of this value - this only controls how much "
                              "history backs the overall win-rate number.")
    parser.add_argument("--coin", type=str, default=None,
                         help="Run just this one coin (SOL/DOGE/ETH) instead of all 3. Used by "
                              "the parallel per-coin workflow - the sequential all-4-fetch "
                              "version already timed out once at 45 minutes, cancelled 30%% "
                              "through its 4th coin's fetch. Same fix as weekly_pnl_live_config.py.")
    args = parser.parse_args()

    coins_to_run = {args.coin: LIVE_CONFIG[args.coin]} if args.coin else LIVE_CONFIG

    now = datetime.now(timezone.utc)
    end_date = now.date()  # display/labeling only - NOT what gets fetched, see fetch_end_time below
    fetch_end_time = now.isoformat()  # the REAL current moment - str(end_date) alone parses as
                                       # midnight of today, silently dropping every hour since
                                       # midnight. For a 7-day window that's up to ~14% of the
                                       # data missing, and specifically the most recent day - the
                                       # one this whole report is supposed to be about.
    start_date = end_date - timedelta(days=args.history_days)
    seven_days_ago = now - timedelta(days=7)

    print(f"Deployed live config, corrected engine, {start_date} to {end_date} ({args.history_days}d)")
    print(f"'Last 7 days' cutoff: trades with exit_time >= {seven_days_ago} (exact, not rounded)\n")

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

    for coin, cfg in coins_to_run.items():
        cfg = dict(cfg)  # don't mutate the shared module-level table
        print(f"=== {coin}: fetching 1m data ===")
        candles_1m = fetch_coindcx_klines(coin, "1m", str(start_date), fetch_end_time)
        candles_5m = resample_candles(candles_1m, 5)
        print(f"  {coin}: {len(candles_1m)} 1m candles -> {len(candles_5m)} 5m candles")

        if coin == "ETH":
            candles_5m = candles_5m.join(
                btc_renamed[["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]], how="left"
            )

        max_hold_bars = round(cfg.pop("max_hold_hours") * 60 / 5)
        min_score = cfg.pop("min_score")

        trades, _ = run_backtest(
            coin, candles_5m, strategy="conservative",
            min_score=min_score, max_hold_bars=max_hold_bars,
            **cfg,
        )
        print(f"  {coin}: {len(trades)} total trades in the {args.history_days}-day window\n")

        if len(trades) == 0:
            continue

        trades = apply_fees_and_interest(trades, bar_minutes=5)
        trades = apply_dollar_pnl(trades)
        trades["coin"] = coin
        all_trades.append(trades)

    if not all_trades:
        print("No trades produced by any coin - nothing to report.")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    combined["entry_time"] = pd.to_datetime(combined["entry_time"], utc=True)
    combined["exit_time"] = pd.to_datetime(combined["exit_time"], utc=True)
    combined["outcome"] = combined["dollar_pnl"].apply(lambda x: "WIN" if x > 0 else "LOSS")

    # ---- Question 1: overall win rate ----
    print("=" * 70)
    print("QUESTION 1: OVERALL WIN RATE (deployed config, corrected engine)")
    print("=" * 70)
    for coin in coins_to_run:
        coin_trades = combined[combined["coin"] == coin]
        if coin_trades.empty:
            continue
        wins = (coin_trades["dollar_pnl"] > 0).sum()
        total = len(coin_trades)
        print(f"{coin:5} win rate: {wins}/{total} = {wins/total*100:.1f}%")
    overall_wins = (combined["dollar_pnl"] > 0).sum()
    overall_total = len(combined)
    print(f"\nCOMBINED (all 3 coins): {overall_wins}/{overall_total} = {overall_wins/overall_total*100:.1f}% win rate")
    print(f"\nFor direct comparison: your observed recent live experience was reported as "
          f"roughly 1 win in 6-7 trades (~14-17%).")

    # ---- Question 2: exact last-7-days trade log ----
    recent = combined[combined["exit_time"] >= seven_days_ago].sort_values("exit_time")
    print("\n" + "=" * 70)
    print(f"QUESTION 2: TRADE-BY-TRADE LOG, LAST 7 DAYS (since {seven_days_ago})")
    print("=" * 70)
    if recent.empty:
        print("No trades exited in the last 7 days for any coin in this backtest.")
    else:
        display_cols = ["coin", "direction", "entry_time", "exit_time", "entry_price",
                         "exit_price", "exit_reason", "dollar_pnl", "outcome"]
        display_cols = [c for c in display_cols if c in recent.columns]
        print(recent[display_cols].to_string(index=False))
        recent_wins = (recent["dollar_pnl"] > 0).sum()
        print(f"\nLast-7-days win rate: {recent_wins}/{len(recent)} = {recent_wins/len(recent)*100:.1f}%")
        print(f"Last-7-days total $ P&L: {recent['dollar_pnl'].sum():.2f}")


if __name__ == "__main__":
    main()
