"""
Cloud backtest runner. Replaces the Colab workflow entirely - no
Drive mount, no session disconnects, no compute credits. Runs on
GitHub Actions via workflow_dispatch (see .github/workflows/backtest.yml).

Fetches real CoinDCX 1m data fresh each run (GitHub Actions runners are
ephemeral - no persistent Drive-style cache), resamples to 5m, and runs
every requested coin x strategy combination through the fee-adjusted
backtest engine. Writes results to results/backtest_<timestamp>.csv and
commits it (handled by the workflow, not this script) so results
persist in the repo's git history the same way advisories.json etc. do.

Configurable via environment variables (all optional, sensible defaults):
  BT_COINS       comma-separated, default "BTC,ETH,SOL,XRP,DOGE"
  BT_STRATEGIES  comma-separated, default "ultra-short,aggressive,balanced,conservative,swing-trend"
  BT_DAYS        integer, default 365 (the 1-year cap requested to respect
                 compute/credit constraints - this used to be Colab
                 compute credits, now it's just GitHub Actions minutes,
                 but the same instinct to keep it bounded still applies)
  BT_MAX_HOLD_HOURS  default 36 (matches the live bot's own deployed cap)
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  optional - if both set, sends a
                 compact summary on completion (same env var names as
                 the live bot, for consistency)
"""
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from backtest_engine import run_backtest, summarize_results
from fee_model import apply_fees_and_interest

DEFAULT_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
DEFAULT_STRATEGIES = ["ultra-short", "aggressive", "balanced", "conservative", "swing-trend"]


def _env_list(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("(Telegram not configured - skipping notification)")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        if not resp.ok:
            print(f"Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Telegram send error: {e}")


def main():
    coins = _env_list("BT_COINS", DEFAULT_COINS)
    strategies = _env_list("BT_STRATEGIES", DEFAULT_STRATEGIES)
    days = int(os.environ.get("BT_DAYS", "365"))
    max_hold_hours = float(os.environ.get("BT_MAX_HOLD_HOURS", "36"))
    max_hold_bars = round(max_hold_hours * 60 / 5)  # 5m bars

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    print(f"Backtest window: {start_date} to {end_date} ({days} days)")
    print(f"Coins: {coins}")
    print(f"Strategies: {strategies}")
    print(f"Max hold: {max_hold_hours}h ({max_hold_bars} x 5m bars)")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    coins_tag = "-".join(coins)
    results_path = f"results/backtest_{coins_tag}_{timestamp}.csv"
    trades_path = f"results/trades_{coins_tag}_{timestamp}.csv"

    all_rows = []
    all_trades = []
    errors = []

    for coin in coins:
        print(f"\n=== {coin}: fetching 1m data ===")
        t0 = time.time()
        try:
            candles_1m = fetch_coindcx_klines(coin, "1m", str(start_date), str(end_date))
        except Exception as e:
            print(f"  FAILED to fetch {coin}: {e}")
            errors.append(f"{coin} fetch: {e}")
            continue
        candles_5m = resample_candles(candles_1m, "5m")
        print(f"  {coin}: {len(candles_1m)} 1m candles -> {len(candles_5m)} 5m candles in {time.time()-t0:.0f}s")

        for strategy in strategies:
            print(f"  --- {coin} / {strategy} ---")
            t1 = time.time()
            try:
                trades, equity = run_backtest(coin, candles_5m, strategy=strategy, max_hold_bars=max_hold_bars)
            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                errors.append(f"{coin}/{strategy} backtest: {e}")
                continue

            if len(trades) == 0:
                print(f"    0 trades ({time.time()-t1:.0f}s)")
                all_rows.append({"coin": coin, "strategy": strategy, "total_trades": 0})
                continue

            trades = apply_fees_and_interest(trades)
            trades["coin"] = coin  # already has 'strategy' from the engine itself
            all_trades.append(trades)

            summary = summarize_results(trades)
            row = {"coin": coin, "strategy": strategy, "days": days, **summary}
            row.pop("exit_reason_breakdown", None)  # dict - not CSV-friendly, kept in trades log instead
            all_rows.append(row)
            print(f"    {len(trades)} trades in {time.time()-t1:.0f}s | "
                  f"gross avgR {summary.get('avg_r_gross')} -> net avgR {summary.get('avg_r_net')} "
                  f"(fee/interest cost avg {summary.get('avg_fee_interest_r_cost')})")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}")

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(trades_path, index=False)
        print(f"Wrote {trades_path}")

    # ---- Telegram summary ----
    lines = [f"📊 *Cloud backtest complete* ({start_date} to {end_date}, {days}d)"]
    if not results_df.empty:
        for _, r in results_df.iterrows():
            if r.get("total_trades", 0) == 0:
                lines.append(f"{r['coin']}/{r['strategy']}: 0 trades")
            else:
                lines.append(
                    f"{r['coin']}/{r['strategy']}: {int(r['total_trades'])} trades | "
                    f"net avgR {r['avg_r_net']:+.3f} (gross {r['avg_r_gross']:+.3f}) | "
                    f"win {r['win_rate_pct_net']:.0f}%"
                )
    if errors:
        lines.append(f"⚠️ {len(errors)} error(s): " + " | ".join(errors[:3]))
    send_telegram("\n".join(lines))

    print("\nDone.")
    if errors:
        print(f"Completed with {len(errors)} error(s) - see above.")
        sys.exit(1 if not all_rows else 0)  # only hard-fail if NOTHING succeeded


if __name__ == "__main__":
    main()
