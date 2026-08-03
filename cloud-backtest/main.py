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
from binance_taker_volume_fetcher import fetch_binance_taker_volume
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
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
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
    primary_minutes = int(os.environ.get("BT_PRIMARY_MINUTES", "5").strip() or "5")
    max_hold_bars = round(max_hold_hours * 60 / primary_minutes)

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
    skip_filter_timeframe = os.environ.get("BT_SKIP_FILTER_TIMEFRAME", "").strip().lower() in ("1", "true", "yes")
    confirm_minutes = int(os.environ.get("BT_CONFIRM_MINUTES", "15").strip() or "15")
    filter_minutes = int(os.environ.get("BT_FILTER_MINUTES", "60").strip() or "60")
    full_close_at_1r = os.environ.get("BT_FULL_CLOSE_AT_1R", "").strip().lower() in ("1", "true", "yes")
    asymmetric_free_ride = os.environ.get("BT_ASYMMETRIC_FREE_RIDE", "").strip().lower() in ("1", "true", "yes")
    stage1_close_fraction = float(os.environ.get("BT_STAGE1_CLOSE_FRACTION", "0.65"))
    trailing_atr_multiplier = float(os.environ.get("BT_TRAILING_ATR_MULTIPLIER", "1.5"))
    use_adverse_drift = os.environ.get("BT_USE_ADVERSE_DRIFT", "").strip().lower() in ("1", "true", "yes")
    drift_net_threshold = float(os.environ.get("BT_DRIFT_NET_THRESHOLD", "10"))
    # --- Idea #2: time-decay dynamic reversal threshold ---
    dynamic_threshold_enabled = os.environ.get("BT_DYNAMIC_THRESHOLD_ENABLED", "").strip().lower() in ("1", "true", "yes")
    dynamic_threshold_after_minutes = float(os.environ.get("BT_DYNAMIC_THRESHOLD_AFTER_MINUTES", "45"))
    dynamic_threshold_drawdown_r = float(os.environ.get("BT_DYNAMIC_THRESHOLD_DRAWDOWN_R", "-0.4"))
    dynamic_threshold_tightened = float(os.environ.get("BT_DYNAMIC_THRESHOLD_TIGHTENED", "35"))
    # --- Idea #3: ATR stop compression on adverse drift ---
    drift_stop_tighten_enabled = os.environ.get("BT_DRIFT_STOP_TIGHTEN_ENABLED", "").strip().lower() in ("1", "true", "yes")
    drift_stop_tighten_atr_multiplier = float(os.environ.get("BT_DRIFT_STOP_TIGHTEN_ATR_MULTIPLIER", "1.2"))
    # --- Idea #4: OBV/price-divergence confirmation bonus (proxy for CVD - see indicators.py) ---
    obv_confirmation_bonus = float(os.environ.get("BT_OBV_CONFIRMATION_BONUS", "0"))
    obv_lookback_bars = int(os.environ.get("BT_OBV_LOOKBACK_BARS", "10"))
    obv_slope_threshold = float(os.environ.get("BT_OBV_SLOPE_THRESHOLD", "0.3"))
    use_real_cvd = os.environ.get("BT_USE_REAL_CVD", "").strip().lower() in ("1", "true", "yes")
    # --- Idea #6: BTC trend confirmation bonus + BTC stop-loss floor ---
    use_btc_trend_bonus = os.environ.get("BT_USE_BTC_TREND_BONUS", "").strip().lower() in ("1", "true", "yes")
    btc_trend_bonus = float(os.environ.get("BT_BTC_TREND_BONUS", "0"))
    btc_min_score_magnitude = float(os.environ.get("BT_BTC_MIN_SCORE_MAGNITUDE", "25"))
    use_btc_stop_floor = os.environ.get("BT_USE_BTC_STOP_FLOOR", "").strip().lower() in ("1", "true", "yes")
    btc_stop_beta = float(os.environ.get("BT_BTC_STOP_BETA", "1.0"))
    # --- Idea #7: volatility expansion entry gate (DOGE hypothesis) ---
    use_volatility_expansion_gate = os.environ.get("BT_USE_VOLATILITY_EXPANSION_GATE", "").strip().lower() in ("1", "true", "yes")
    min_atr_ratio_for_entry = float(os.environ.get("BT_MIN_ATR_RATIO_FOR_ENTRY", "1.0"))
    # --- Idea #8: asymmetric take-profit stage weighting (DOGE hypothesis) ---
    stage_fractions_raw = os.environ.get("BT_STAGE_FRACTIONS", "0.3333,0.3333,0.3334")
    stage_fractions = tuple(float(x) for x in stage_fractions_raw.split(","))
    # --- Idea #9: BTC lag-confirmation (DOGE hypothesis) ---
    use_btc_lag_bonus = os.environ.get("BT_USE_BTC_LAG_BONUS", "").strip().lower() in ("1", "true", "yes")
    btc_lag_bonus = float(os.environ.get("BT_BTC_LAG_BONUS", "0"))
    btc_lag_bars = int(os.environ.get("BT_BTC_LAG_BARS", "6"))
    btc_lag_min_score_magnitude = float(os.environ.get("BT_BTC_LAG_MIN_SCORE_MAGNITUDE", "25"))
    # --- Idea #5: mechanical soft-exit proxy (TRIM/TIGHTEN/FREEZE) - NOT real Gemini judgment ---
    soft_exit_enabled = os.environ.get("BT_SOFT_EXIT_ENABLED", "").strip().lower() in ("1", "true", "yes")
    soft_exit_trim_threshold = float(os.environ.get("BT_SOFT_EXIT_TRIM_THRESHOLD", "45"))
    soft_exit_trim_fraction = float(os.environ.get("BT_SOFT_EXIT_TRIM_FRACTION", "0.5"))
    soft_exit_tighten_threshold = float(os.environ.get("BT_SOFT_EXIT_TIGHTEN_THRESHOLD", "35"))
    soft_exit_tighten_atr_multiplier = float(os.environ.get("BT_SOFT_EXIT_TIGHTEN_ATR_MULTIPLIER", "1.5"))
    soft_exit_freeze_minutes = float(os.environ.get("BT_SOFT_EXIT_FREEZE_MINUTES", "60"))

    print(f"Backtest window: {start_date} to {end_date} ({days} days)")
    print(f"Coins: {coins}")
    print(f"Strategies: {strategies}")
    print(f"Max hold: {max_hold_hours}h ({max_hold_bars} x {primary_minutes}m bars)")
    print(f"ATR multiplier override: {atr_override if atr_override is not None else '(preset default)'}")
    print(f"Min score override: {min_score_override if min_score_override is not None else '(preset default)'}")
    print(f"Reversal exit threshold: {reversal_threshold} (live default is 70)")
    print(f"Raw reversal threshold: {raw_reversal_threshold if raw_reversal_threshold is not None else '(disabled)'}")
    print(f"Skip filter (1h) timeframe: {skip_filter_timeframe}")
    print(f"Timeframe combination: primary {primary_minutes}m / confirm {confirm_minutes}m / filter {filter_minutes}m (live default: 5m/15m/60m)")
    print(f"Full close at stage-1 R: {full_close_at_1r}")
    print(f"Asymmetric free-ride: {asymmetric_free_ride} (stage1 close fraction {stage1_close_fraction}, trailing ATR x{trailing_atr_multiplier})")
    print(f"Adverse drift detector (Aug 2 fix): {use_adverse_drift} (net threshold {drift_net_threshold})")

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
    if asymmetric_free_ride:
        variant_tag += f"_asymfr{stage1_close_fraction*100:.0f}pct_trail{trailing_atr_multiplier:.1f}x"
    if reversal_threshold != 70:
        variant_tag += f"_rev{reversal_threshold:.0f}"
    if raw_reversal_threshold is not None:
        variant_tag += f"_rawrev{raw_reversal_threshold:.0f}"
    if use_adverse_drift:
        variant_tag += f"_adversedrift{drift_net_threshold:.0f}"
    if skip_filter_timeframe:
        variant_tag += "_2tf"
    if (primary_minutes, confirm_minutes, filter_minutes) != (5, 15, 60):
        variant_tag += f"_tf{primary_minutes}-{confirm_minutes}-{filter_minutes}"
    if max_hold_hours != 36:
        variant_tag += f"_hold{max_hold_hours:.0f}h"
    results_path = f"results/backtest_{coins_tag}{variant_tag}_{timestamp}.csv"
    trades_path = f"results/trades_{coins_tag}{variant_tag}_{timestamp}.csv"

    all_rows = []
    all_trades = []
    errors = []

    btc_candles_5m = None
    if use_btc_trend_bonus or use_btc_stop_floor:
        print(f"\n=== BTC: fetching 1m data (shared across all coins this run) ===")
        try:
            btc_1m = fetch_coindcx_klines("BTC", "1m", str(start_date), str(end_date))
            btc_candles_5m = resample_candles(btc_1m, primary_minutes)
            print(f"  BTC: {len(btc_1m)} 1m candles -> {len(btc_candles_5m)} {primary_minutes}m candles")
        except Exception as e:
            print(f"  FAILED to fetch BTC: {e} - BTC trend bonus / stop floor will be inert this run "
                  f"(both checks skip silently when the btc_close column isn't present).")
            errors.append(f"BTC fetch: {e}")

    for coin in coins:
        print(f"\n=== {coin}: fetching 1m data ===")
        t0 = time.time()
        try:
            candles_1m = fetch_coindcx_klines(coin, "1m", str(start_date), str(end_date))
        except Exception as e:
            print(f"  FAILED to fetch {coin}: {e}")
            errors.append(f"{coin} fetch: {e}")
            continue
        candles_5m = resample_candles(candles_1m, primary_minutes)
        print(f"  {coin}: {len(candles_1m)} 1m candles -> {len(candles_5m)} {primary_minutes}m candles in {time.time()-t0:.0f}s")

        if btc_candles_5m is not None and coin != "BTC":
            btc_renamed = btc_candles_5m.rename(columns={
                "open": "btc_open", "high": "btc_high", "low": "btc_low",
                "close": "btc_close", "volume": "btc_volume",
            })
            candles_5m = candles_5m.join(btc_renamed[["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]], how="left")
            n_matched = candles_5m["btc_close"].notna().sum()
            print(f"  {coin}: BTC data merged - {n_matched}/{len(candles_5m)} bars matched "
                  f"({'full coverage' if n_matched == len(candles_5m) else 'PARTIAL - gaps between coin and BTC candle timestamps'})")

        if use_real_cvd:
            taker_df = fetch_binance_taker_volume(coin, f"{primary_minutes}m", str(start_date), str(end_date))
            if taker_df is not None:
                candles_5m = candles_5m.join(taker_df[["taker_buy_volume", "taker_sell_volume"]], how="left")
                n_matched = candles_5m["taker_buy_volume"].notna().sum()
                print(f"  {coin}: real CVD merged - {n_matched}/{len(candles_5m)} bars matched "
                      f"({'full coverage' if n_matched == len(candles_5m) else 'PARTIAL - timestamp gaps between CoinDCX and Binance candles'})")
            else:
                print(f"  {coin}: real CVD unavailable this run (see binance_taker_volume_fetcher.py log above) - "
                      f"will fall back to the OBV proxy for obv_confirmation_bonus.")

        for strategy in strategies:
            print(f"  --- {coin} / {strategy} ---")
            t1 = time.time()
            try:
                trades, equity = run_backtest(
                    coin, candles_5m, strategy=strategy, max_hold_bars=max_hold_bars,
                    atr_multiplier_override=atr_override, full_close_at_stage1=full_close_at_1r,
                    asymmetric_free_ride=asymmetric_free_ride, stage1_close_fraction=stage1_close_fraction,
                    trailing_atr_multiplier=trailing_atr_multiplier,
                    min_score=min_score_override, reversal_exit_threshold=reversal_threshold,
                    raw_reversal_threshold=raw_reversal_threshold,
                    skip_filter_timeframe=skip_filter_timeframe,
                    confirm_minutes=confirm_minutes, filter_minutes=filter_minutes,
                    primary_minutes=primary_minutes,
                    use_adverse_drift=use_adverse_drift, drift_net_threshold=drift_net_threshold,
                    dynamic_threshold_enabled=dynamic_threshold_enabled,
                    dynamic_threshold_after_minutes=dynamic_threshold_after_minutes,
                    dynamic_threshold_drawdown_r=dynamic_threshold_drawdown_r,
                    dynamic_threshold_tightened=dynamic_threshold_tightened,
                    drift_stop_tighten_enabled=drift_stop_tighten_enabled,
                    drift_stop_tighten_atr_multiplier=drift_stop_tighten_atr_multiplier,
                    obv_confirmation_bonus=obv_confirmation_bonus, obv_lookback_bars=obv_lookback_bars,
                    obv_slope_threshold=obv_slope_threshold,
                    use_btc_trend_bonus=use_btc_trend_bonus, btc_trend_bonus=btc_trend_bonus,
                    btc_min_score_magnitude=btc_min_score_magnitude,
                    use_btc_stop_floor=use_btc_stop_floor, btc_stop_beta=btc_stop_beta,
                    use_volatility_expansion_gate=use_volatility_expansion_gate,
                    min_atr_ratio_for_entry=min_atr_ratio_for_entry,
                    stage_fractions=stage_fractions,
                    use_btc_lag_bonus=use_btc_lag_bonus, btc_lag_bonus=btc_lag_bonus,
                    btc_lag_bars=btc_lag_bars, btc_lag_min_score_magnitude=btc_lag_min_score_magnitude,
                    soft_exit_enabled=soft_exit_enabled,
                    soft_exit_trim_threshold=soft_exit_trim_threshold,
                    soft_exit_trim_fraction=soft_exit_trim_fraction,
                    soft_exit_tighten_threshold=soft_exit_tighten_threshold,
                    soft_exit_tighten_atr_multiplier=soft_exit_tighten_atr_multiplier,
                    soft_exit_freeze_minutes=soft_exit_freeze_minutes,
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

            trades = apply_fees_and_interest(trades, bar_minutes=primary_minutes)
            trades = apply_dollar_pnl(trades)  # standing $500/5%-fixed convention, see fee_model.py
            trades["coin"] = coin  # already has 'strategy' from the engine itself
            trades["use_adverse_drift"] = use_adverse_drift
            trades["dynamic_threshold_enabled"] = dynamic_threshold_enabled
            trades["drift_stop_tighten_enabled"] = drift_stop_tighten_enabled
            trades["obv_confirmation_bonus"] = obv_confirmation_bonus
            trades["soft_exit_enabled"] = soft_exit_enabled
            all_trades.append(trades)

            summary = summarize_results(trades)
            if "avg_bars_held" in summary:
                # Bug 5 fix: avg_bars_held alone is silently misleading
                # once different timeframe combos are being compared -
                # "12 bars" means 12 minutes at primary=1m but 60 minutes
                # at primary=5m. Adding the real-time-converted version
                # right alongside it, computed here where primary_minutes
                # is actually known.
                summary["avg_hours_held"] = round(summary["avg_bars_held"] * primary_minutes / 60.0, 2)
            row = {"coin": coin, "strategy": strategy, "days": days,
                   "atr_override": atr_override if atr_override is not None else "",
                   "min_score_override": min_score_override if min_score_override is not None else "",
                   "full_close_at_1r": full_close_at_1r, "asymmetric_free_ride": asymmetric_free_ride,
                   "stage1_close_fraction": stage1_close_fraction if asymmetric_free_ride else "",
                   "trailing_atr_multiplier": trailing_atr_multiplier if asymmetric_free_ride else "",
                   "reversal_exit_threshold": reversal_threshold,
                   "use_adverse_drift": use_adverse_drift,
                   "drift_net_threshold": drift_net_threshold if use_adverse_drift else "",
                   "dynamic_threshold_enabled": dynamic_threshold_enabled,
                   "drift_stop_tighten_enabled": drift_stop_tighten_enabled,
                   "obv_confirmation_bonus": obv_confirmation_bonus,
                   "soft_exit_enabled": soft_exit_enabled,
                   # Bug 7 fix: previously only present in the filename tag,
                   # not as real columns - meaning concatenating multiple
                   # results CSVs together (which is how every sweep in
                   # this project has actually been analyzed) would lose
                   # track of which row came from which timeframe combo.
                   "primary_minutes": primary_minutes, "confirm_minutes": confirm_minutes,
                   "filter_minutes": filter_minutes, "skip_filter_timeframe": skip_filter_timeframe,
                   "max_hold_hours": max_hold_hours,
                   **summary}
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
    if asymmetric_free_ride:
        variant_desc.append(f"asym-free-ride({stage1_close_fraction*100:.0f}%@1R, trail{trailing_atr_multiplier:.1f}x)")
    if reversal_threshold != 70:
        variant_desc.append(f"reversal@{reversal_threshold:.0f}")
    if raw_reversal_threshold is not None:
        variant_desc.append(f"rawreversal@{raw_reversal_threshold:.0f}")
    if skip_filter_timeframe:
        variant_desc.append("2tf-no-filter")
    if (primary_minutes, confirm_minutes, filter_minutes) != (5, 15, 60):
        variant_desc.append(f"tf{primary_minutes}m-{confirm_minutes}m-{filter_minutes}m")
    if max_hold_hours != 36:
        variant_desc.append(f"hold{max_hold_hours:.0f}h")
    if use_adverse_drift:
        variant_desc.append(f"drift-ON(net{drift_net_threshold:.0f})")
    if dynamic_threshold_enabled:
        variant_desc.append(f"dynthresh-ON({dynamic_threshold_after_minutes:.0f}m,{dynamic_threshold_drawdown_r}R->{dynamic_threshold_tightened:.0f})")
    if drift_stop_tighten_enabled:
        variant_desc.append(f"stoptighten-ON({drift_stop_tighten_atr_multiplier}x)")
    if obv_confirmation_bonus:
        tag = f"obv-bonus{obv_confirmation_bonus:.0f}"
        if obv_lookback_bars != 10:
            tag += f",lookback{obv_lookback_bars}"
        if obv_slope_threshold != 0.3:
            tag += f",slopethresh{obv_slope_threshold}"
        variant_desc.append(tag + ("+realCVD" if use_real_cvd else ""))
    if soft_exit_enabled:
        variant_desc.append(f"softexit-ON(trim@{soft_exit_trim_threshold:.0f},tighten@{soft_exit_tighten_threshold:.0f})")
    if use_btc_trend_bonus and btc_trend_bonus:
        variant_desc.append(f"btc-bonus{btc_trend_bonus:.0f}(mag{btc_min_score_magnitude:.0f})")
    if use_btc_stop_floor:
        variant_desc.append(f"btc-stopfloor(beta{btc_stop_beta})")
    if use_volatility_expansion_gate:
        variant_desc.append(f"volgate(minratio{min_atr_ratio_for_entry})")
    if stage_fractions != (0.3333, 0.3333, 0.3334):
        variant_desc.append(f"stagefrac{stage_fractions_raw}")
    if use_btc_lag_bonus and btc_lag_bonus:
        variant_desc.append(f"btclag{btc_lag_bars}bars-bonus{btc_lag_bonus:.0f}(mag{btc_lag_min_score_magnitude:.0f})")
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
