"""
Idea #1 runner: Portfolio Basket Circuit Breaker.

Fetches SOL/DOGE/ETH at their real deployed settings, runs each coin's
OWN independent backtest (unchanged engine), then applies the basket
circuit breaker overlay (basket_circuit_breaker.py) and reports BEFORE
vs AFTER for each coin side by side, plus every time the breaker fired.

This is a single, non-matrix job (unlike ideas #2-#5) because the
breaker needs all 3 coins' trades AND candles in the same process at
once - see basket_circuit_breaker.py's docstring for the approximation
this makes.

Env vars (all optional):
  BT_DAYS                        default 365
  BT_MIN_CORRELATED_POSITIONS    default 2
  BT_BASKET_NEGATIVE_MINUTES     default 45
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
"""
import os
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from backtest_engine import run_backtest, summarize_results
from fee_model import apply_fees_and_interest, apply_dollar_pnl
from basket_circuit_breaker import apply_basket_circuit_breaker

# Real deployed settings per coin - same as backtest.yml's coin_configs default.
COIN_SETTINGS = {
    "SOL": {"min_score": 80, "max_hold_hours": 18},
    "DOGE": {"min_score": 79, "max_hold_hours": 48},
    "ETH": {"min_score": 81, "max_hold_hours": 60},
}


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("(Telegram not configured - skipping notification)")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        if not resp.ok:
            print(f"Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Telegram send error: {e}")


def main():
    days = int(os.environ.get("BT_DAYS", "365"))
    min_correlated = int(os.environ.get("BT_MIN_CORRELATED_POSITIONS", "2"))
    basket_minutes = float(os.environ.get("BT_BASKET_NEGATIVE_MINUTES", "45"))

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    print(f"Idea #1: Basket Circuit Breaker | window {start_date} to {end_date} ({days}d)")
    print(f"min_correlated_positions={min_correlated}, basket_negative_minutes={basket_minutes}")

    per_coin_trades_before = {}
    per_coin_candles = {}
    errors = []

    for coin, cfg in COIN_SETTINGS.items():
        print(f"\n--- {coin} ---")
        try:
            candles_1m = fetch_coindcx_klines(coin, "1m", str(start_date), str(end_date))
            candles_5m = resample_candles(candles_1m, 5)
            print(f"  {len(candles_1m)} 1m candles -> {len(candles_5m)} 5m candles")
        except Exception as e:
            print(f"  FETCH FAILED: {e}")
            traceback.print_exc()
            errors.append(f"{coin} fetch: {e}")
            continue

        max_hold_bars = round(cfg["max_hold_hours"] * 60 / 5)
        try:
            trades, _ = run_backtest(
                coin, candles_5m, strategy="conservative",
                min_score=cfg["min_score"], max_hold_bars=max_hold_bars,
            )
        except Exception as e:
            print(f"  BACKTEST FAILED: {e}")
            traceback.print_exc()
            errors.append(f"{coin} backtest: {e}")
            continue

        if len(trades) == 0:
            print("  0 trades")
            continue

        per_coin_trades_before[coin] = trades
        per_coin_candles[coin] = candles_5m
        print(f"  {len(trades)} baseline trades")

    if len(per_coin_trades_before) < 2:
        msg = f"Idea #1 basket breaker: fewer than 2 coins produced trades - can't test correlated baskets. Errors: {errors}"
        print(msg)
        send_telegram(f"⚠️ {msg}")
        return

    adjusted_trades, breaker_events = apply_basket_circuit_breaker(
        per_coin_trades_before, per_coin_candles,
        min_correlated_positions=min_correlated, basket_negative_minutes=basket_minutes,
    )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    lines = [f"📊 <b>Idea #1: Basket Circuit Breaker</b> ({start_date} to {end_date}, {days}d)"]
    lines.append(f"min_correlated={min_correlated}, basket_negative_minutes={basket_minutes}")
    real_events = [e for e in breaker_events if e["real_early_close"]]
    lines.append(f"Breaker fired {len(breaker_events)} times ({len(real_events)} were real early closes, "
                  f"{len(breaker_events) - len(real_events)} coincided with the trade's own natural exit)")

    for coin in COIN_SETTINGS:
        if coin not in per_coin_trades_before:
            continue
        before = per_coin_trades_before[coin].copy()
        before = apply_fees_and_interest(before, bar_minutes=5)
        before = apply_dollar_pnl(before)
        before_summary = summarize_results(before)

        after = adjusted_trades[coin].copy()
        after = apply_fees_and_interest(after, bar_minutes=5)
        after = apply_dollar_pnl(after)
        after_summary = summarize_results(after)

        n_forced = (after["exit_reason"] == "basket_circuit_breaker").sum() if len(after) else 0

        before.to_csv(f"results/basket_before_{coin}_{timestamp}.csv", index=False)
        after.to_csv(f"results/basket_after_{coin}_{timestamp}.csv", index=False)

        lines.append(
            f"\n{coin}: BEFORE {before_summary.get('total_trades')} trades, "
            f"net avgR {before_summary.get('avg_r_net')}, ${before_summary.get('total_dollar_pnl')} | "
            f"AFTER {after_summary.get('total_trades')} trades, "
            f"net avgR {after_summary.get('avg_r_net')}, ${after_summary.get('total_dollar_pnl')} "
            f"({n_forced} force-closed by breaker)"
        )
        print(lines[-1])

    events_df = pd.DataFrame(breaker_events)
    events_df.to_csv(f"results/basket_breaker_events_{timestamp}.csv", index=False)

    if errors:
        lines.append(f"\nErrors: {errors}")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
