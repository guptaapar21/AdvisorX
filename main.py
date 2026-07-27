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
  BT_ATR_OVERRIDE    optional float - overrides a preset's default ATR stop multiplier
  BT_MIN_SCORE_OVERRIDE  optional float - overrides a preset's default min_score threshold
                 (added for the "find the sweet spot" sweep - loosening
                 conservative's bar specifically on SOL/DOGE, the two
                 coins already proven to work well with it)
  BT_FULL_CLOSE_AT_1R  "true"/"false" - close 100% at the first R target
                 instead of the normal 3-stage 1R/2R/3R exit
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  optional - if both set, sends a
                 compact summary on completion (same env var names as
                 the live bot, for consistency)

Every result is now reported in BOTH R-multiples and real dollars, using
a standing $500 capital / 5% fixed risk-per-trade convention (see
fee_model.py's DEFAULT_CAPITAL/DEFAULT_RISK_PCT) - added after a request
to see actual dollar terms rather than only abstract R-multiples.
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
from fee_model import apply_fees_and_interest, apply_dollar_pnl

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

    atr_override_raw = os.environ.get("BT_ATR_OVERRIDE", "").strip()
    atr_override = float(atr_override_raw) if atr_override_raw else None
    min_score_raw = os.environ.get("BT_MIN_SCORE_OVERRIDE", "").strip()
    min_score_override = float(min_score_raw) if min_score_raw else None
    reversal_threshold_raw = os.environ.get("BT_REVERSAL_THRESHOLD", "").strip()
    reversal_threshold = float(reversal_threshold_raw) if reversal_threshold_raw else 70
    raw_reversal_raw = os.environ.get("BT_RAW_REVERSAL_THRESHOLD", "").strip()
    raw_reversal_threshold = float(raw_reversal_raw) if raw_reversal_raw else None
    full_close_at_1r = os.environ.get("BT_FULL_CLOSE_AT_1R", "").strip().lower() in ("1", "true", "yes")

    print(f"Backtest window: {start_date} to {end_date} ({days} days)")
    print(f"Coins: {coins}")
    print(f"Strategies: {strategies}")
    print(f"Max hold: {max_hold_hours}h ({max_hold_bars} x 5m bars)")
    print(f"ATR multiplier override: {atr_override if atr_override is not None else '(preset default)'}")
    print(f"Min score override: {min_score_override if min_score_override is not None else '(preset default)'}")
    print(f"Reversal exit threshold: {reversal_threshold} (live default is 70)")
    print(f"Raw reversal threshold: {raw_reversal_threshold if raw_reversal_threshold is not None else '(disabled)'}")
    print(f"Full close at stage-1 R: {full_close_at_1r}")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    coins_tag = "-".join(coins)
    variant_tag = ""
    if atr_override is not None:
        variant_tag += f"_atr{atr_override}"
    if min_score_override is not None:
        variant_tag += f"_score{min_score_override}"
    if full_close_at_1r:
        variant_tag += "_fullclose1R"
    if reversal_threshold != 70:
        variant_tag += f"_rev{reversal_threshold:.0f}"
    if raw_reversal_threshold is not None:
        variant_tag += f"_rawrev{raw_reversal_threshold:.0f}"
    results_path = f"results/backtest_{coins_tag}{variant_tag}_{timestamp}.csv"
    trades_path = f"results/trades_{coins_tag}{variant_tag}_{timestamp}.csv"

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
                trades, equity = run_backtest(
                    coin, candles_5m, strategy=strategy, max_hold_bars=max_hold_bars,
                    atr_multiplier_override=atr_override, full_close_at_stage1=full_close_at_1r,
                    min_score=min_score_override, reversal_exit_threshold=reversal_threshold,
                    raw_reversal_threshold=raw_reversal_threshold,
                )
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
            trades = apply_dollar_pnl(trades)  # standing $500/5%-fixed convention, see fee_model.py
            trades["coin"] = coin  # already has 'strategy' from the engine itself
            all_trades.append(trades)

            summary = summarize_results(trades)
            row = {"coin": coin, "strategy": strategy, "days": days,
                   "atr_override": atr_override if atr_override is not None else "",
                   "min_score_override": min_score_override if min_score_override is not None else "",
                   "full_close_at_1r": full_close_at_1r, "reversal_exit_threshold": reversal_threshold, **summary}
            row.pop("exit_reason_breakdown", None)  # dict - not CSV-friendly, kept in trades log instead
            all_rows.append(row)
            print(f"    {len(trades)} trades in {time.time()-t1:.0f}s | "
                  f"gross avgR {summary.get('avg_r_gross')} -> net avgR {summary.get('avg_r_net')} "
                  f"(fee/interest cost avg {summary.get('avg_fee_interest_r_cost')}) | "
                  f"${summary.get('total_dollar_pnl')} total on $500 cap/5% risk")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}")

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(trades_path, index=False)
        print(f"Wrote {trades_path}")

    # ---- Telegram summary ----
    variant_desc = []
    if atr_override is not None:
        variant_desc.append(f"ATR {atr_override}x")
    if min_score_override is not None:
        variant_desc.append(f"min_score {min_score_override:.0f}")
    if full_close_at_1r:
        variant_desc.append("full-close@1R")
    if reversal_threshold != 70:
        variant_desc.append(f"reversal@{reversal_threshold:.0f}")
    if raw_reversal_threshold is not None:
        variant_desc.append(f"rawreversal@{raw_reversal_threshold:.0f}")
    variant_str = f" [{', '.join(variant_desc)}]" if variant_desc else ""
    lines = [f"📊 *Cloud backtest complete{variant_str}* ({start_date} to {end_date}, {days}d)"]
    if not results_df.empty:
        for _, r in results_df.iterrows():
            if r.get("total_trades", 0) == 0:
                lines.append(f"{r['coin']}/{r['strategy']}: 0 trades")
            else:
                beat_stop_note = ""
                if "raw_reversal_beat_stop_rate_pct" in r.index and pd.notna(r["raw_reversal_beat_stop_rate_pct"]):
                    beat_stop_note = f" | raw-reversal beat stop {r['raw_reversal_beat_stop_rate_pct']:.0f}% of the time it fired"
                lines.append(
                    f"{r['coin']}/{r['strategy']}: {int(r['total_trades'])} trades | "
                    f"net avgR {r['avg_r_net']:+.3f} (gross {r['avg_r_gross']:+.3f}) | "
                    f"win {r['win_rate_pct_net']:.0f}% | "
                    f"${r['total_dollar_pnl']:+.0f} on $500 cap/5% risk{beat_stop_note}"
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
