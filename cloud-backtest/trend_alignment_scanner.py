"""
Trend-alignment + reversal-candle RVOL scanner (separate from Idea
#18F). This is a FOCUSED EXPERIMENT, not a complete trading strategy:
it only measures whether RVOL on a reversal candle carries useful
information - it does not enter, does not set a stop/target, does not
require a breakout, and does not track or act on anything after the
reversal candle.

STAGE 1 - trend qualification (checked every run, for every coin):
  Long requires ALL of:
    - 3m EMA9 > EMA21, and EMA9 rising (vs 3 candles back)
    - 3m candle closes above VWAP
    - 3m ADX14 >= 25, and rising
    - 15m EMA9 > EMA21, and EMA9 rising (vs 3 candles back on 15m)
  Short is the exact mirror.

STAGE 2 - pullback run (only while qualified):
  Watch for ONE OR MORE consecutive red candles (long case; green for
  short) - all of them together are the pullback/falling sequence. No
  requirement that they touch EMA9/EMA21.

STAGE 3 - reversal candle + RVOL:
  The FIRST candle that breaks the pullback run by closing the
  opposite color (green, for long) is the reversal candle. Its RVOL
  (this candle's volume / the average of the PRIOR 20 candles) is
  computed and reported, labeled per the standard scale (<1.0 weak,
  1.0-1.5 normal, 1.5-2.0 strong, >2.0 very strong/possible exhaustion)
  - informational only, never a filter, never an entry trigger.

DELIBERATELY NOT IMPLEMENTED, per explicit instruction: no EMA9/21
touch requirement on the pullback or reversal candles, no requirement
that the reversal candle break the pullback run's high/low, no
breakout tracking of any kind, no entry price, no stop-loss, no
target, no RVOL threshold, and nothing at all about the candle AFTER
the reversal candle - that one is for manual observation only.

All checks run every 1-minute poll, but the Telegram message itself is
only sent every 3 minutes, batching everything accumulated since the
last send. State persists across runs via the same git-committed JSON
pattern as live_signal_monitor_18f.py.

Usage:
  python3 trend_alignment_scanner.py --coins BTC,ETH,SOL,...
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import requests

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from gemini_advisor import get_trade_suggestions_batch

STATE_FILE = "trend_scanner_state.json"
RVOL_PERCENTILE_FILE = "rvol_percentiles.json"
DAILY_CANDLES_FILE = "daily_candles_30d.json"
# Trimmed from 150 to 100 15m candles after measuring the real fetch
# cost at 18 coins - 150 candles needed 3 API requests/coin (2250 min
# of 1m data), 100 needs only 2 (1500 min), saving ~14.4s across all 18
# coins. Still 10x the bare minimum (10) actually checked in the code
# below, real margin for EMA21/ADX14 to be stable, not just non-error.
# Trimmed to 65 15m candles (975 min) - fits in exactly 1 API request
# per coin instead of 2, still 4.6x ADX14's min_periods(14) warmup
# requirement and 6.5x the hard minimum (10) checked in code below.
# NOTE: the 24h multi-timeframe Gemini context does NOT need this
# widened - it's fetched separately, only for reversal-stage coins,
# in main()'s enrichment block below, keeping this routine per-cycle
# fetch (which runs for all coins, every cycle) unaffected.
FETCH_MINUTES_BACK = 15 * 65
MESSAGE_INTERVAL_MINUTES = 1


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"coins": {}, "last_sent_at": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def send_telegram(text, reply_markup=None, parse_mode="HTML"):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


def drop_still_forming_bucket(candles, now, bar_minutes):
    if len(candles) == 0:
        return candles
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    last_bucket_start = candles.index[-1]
    if last_bucket_start + pd.Timedelta(minutes=bar_minutes) > now:
        return candles.iloc[:-1]
    return candles


def compute_raw_stats(candles):
    """Deliberately NOT the old compute_indicators - no ADX, no EMA
    cross, no VWAP position. Those ARE the strategy-specific trend
    logic this scanner used to pre-filter with, which is exactly what
    the person asked to stop feeding Gemini - it should decide
    everything itself from raw, generic, strategy-agnostic numbers.
    Keeps ATR (a standard volatility measure, same formula as before -
    it's a generic stat, not a directional verdict) and RVOL (volume
    relative to recent average, also generic). Adds plain momentum:
    raw % price change over a few lookback windows in candle counts,
    not an indicator, just arithmetic."""
    candles = candles.copy()

    high, low, close = candles["high"], candles["low"], candles["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    candles["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    candles["avg_volume_20"] = candles["volume"].rolling(20).mean().shift(1)
    candles["rvol"] = candles["volume"] / candles["avg_volume_20"].replace(0, float("nan"))

    for n in (5, 20, 60):
        candles[f"momentum_pct_{n}"] = (candles["close"] / candles["close"].shift(n) - 1) * 100

    return candles


def rvol_label(rvol):
    if rvol is None or pd.isna(rvol):
        return "unknown"
    if rvol < 1.0:
        return "weak"
    if rvol < 1.5:
        return "normal"
    if rvol < 2.0:
        return "strong"
    return "very strong (possible exhaustion)"


def load_rvol_percentiles():
    """Reads the compact per-coin percentile file the separate
    rvol_percentile_refresh.py script produces. Read-only here - the
    live scanner never writes this file, only the daily refresh job
    does, keeping every 1-minute run's git commit small regardless of
    how much history backs the percentiles."""
    if os.path.exists(RVOL_PERCENTILE_FILE):
        try:
            with open(RVOL_PERCENTILE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


LEDGER_MAX_AGE_HOURS = 4  # entries older than this are pruned regardless of status


def resolve_ledger(ledger, fetched, now):
    """Checks every still-pending ledger entry against real price
    action since it was flagged - did target or stop actually get hit
    first, chronologically? Uses candle data already fetched this
    cycle (no extra API calls). An entry with neither touched yet
    stays pending. Mutates and returns the ledger, pruning anything
    older than LEDGER_MAX_AGE_HOURS regardless of status so it doesn't
    grow unbounded."""
    now_naive = now.replace(tzinfo=None) if now.tzinfo is not None else now
    for entry in ledger:
        if entry["status"] != "pending":
            continue
        # Validate BEFORE any arithmetic - confirmed directly as a
        # real, serious risk: entry_price<=0 silently produces
        # nonsensical P&L (a fake $0 "win", or a catastrophic
        # -1,090,000 "loss" labeled as a win with a negative price),
        # and a missing trade_amount_inr crashes this function with an
        # uncaught TypeError - which, since this runs on the
        # PERSISTED ledger every cycle, means one bad entry
        # permanently breaks every future run until manually fixed.
        # Marked "invalid" here instead - visible, excluded from the
        # scorecard, and never retried since it can never resolve.
        entry_price = entry.get("entry_price")
        trade_amount = entry.get("trade_amount_inr")
        if not entry_price or entry_price <= 0 or not trade_amount or trade_amount <= 0:
            print(f"  Ledger WARNING: {entry.get('coin')} entry marked invalid "
                  f"(entry_price={entry_price}, trade_amount_inr={trade_amount}) - excluding from resolution")
            entry["status"] = "invalid"
            continue

        coin = entry["coin"]
        if coin not in fetched:
            continue
        candles_3m, _, _ = fetched[coin]
        call_time = datetime.fromisoformat(entry["time"]).replace(tzinfo=None)
        since = candles_3m[candles_3m.index > call_time]
        if since.empty:
            continue
        direction = entry["direction"]
        target, stop = entry["target_price"], entry["stop_loss"]
        quantity = trade_amount / entry_price
        for _, row in since.iterrows():
            if direction == "long":
                target_hit = row["high"] >= target
                stop_hit = row["low"] <= stop
            else:
                target_hit = row["low"] <= target
                stop_hit = row["high"] >= stop
            if target_hit and stop_hit:
                # Both touched in the same candle - can't know which
                # came first from OHLC alone. Treat conservatively as
                # the stop, not the target - understating P&L is the
                # safer direction to be wrong in here.
                entry["status"] = "stop_hit"
                entry["resolved_pnl"] = -entry["max_loss_this_trade_inr"]
                entry["resolved_time"] = str(row.name)
                break
            if target_hit:
                entry["status"] = "target_hit"
                entry["resolved_pnl"] = round(quantity * abs(target - entry_price), 2)
                entry["resolved_time"] = str(row.name)
                break
            if stop_hit:
                entry["status"] = "stop_hit"
                entry["resolved_pnl"] = -entry["max_loss_this_trade_inr"]
                entry["resolved_time"] = str(row.name)
                break

    cutoff = now_naive - timedelta(hours=LEDGER_MAX_AGE_HOURS)
    return [e for e in ledger if datetime.fromisoformat(e["time"]).replace(tzinfo=None) > cutoff]


def compute_scorecard(ledger, now, window_hours=1):
    """Real trades-in / target-hit / stop-hit / pending / net P&L over
    the trailing window - computed from the ledger's actual resolved
    outcomes, not asked of Gemini (which has no memory and no way to
    verify this itself)."""
    now_naive = now.replace(tzinfo=None) if now.tzinfo is not None else now
    cutoff = now_naive - timedelta(hours=window_hours)
    window = [e for e in ledger if datetime.fromisoformat(e["time"]).replace(tzinfo=None) > cutoff]
    target_hit = [e for e in window if e["status"] == "target_hit"]
    stop_hit = [e for e in window if e["status"] == "stop_hit"]
    pending = [e for e in window if e["status"] == "pending"]
    net_pnl = round(sum(e.get("resolved_pnl", 0) for e in target_hit + stop_hit), 2)
    return {
        "total": len(window), "target_hit": len(target_hit), "stop_hit": len(stop_hit),
        "pending": len(pending), "net_pnl_inr": net_pnl,
    }


def find_pending_same_direction(ledger, coin, direction, now):
    """Is there already an unresolved call on this exact coin+direction?
    No time cutoff - "pending" status already means it hasn't hit
    target or stop yet, regardless of how long ago it was flagged.
    Confirmed directly as a real gap: an earlier version capped this
    at 60 minutes, which meant a genuinely still-open position from 90
    minutes ago was missed entirely - defeating the whole point of
    this check for exactly the longest-running open positions, where
    it matters most. (Ledger entries are pruned after
    LEDGER_MAX_AGE_HOURS regardless of status, so this is naturally
    bounded without needing its own separate cutoff.)"""
    now_naive = now.replace(tzinfo=None) if now.tzinfo is not None else now
    for entry in ledger:
        if entry["coin"] == coin and entry["direction"] == direction and entry["status"] == "pending":
            entry_time = datetime.fromisoformat(entry["time"]).replace(tzinfo=None)
            minutes_ago = (now_naive - entry_time).total_seconds() / 60
            return round(minutes_ago, 1)
    return None


def candles_to_compact(candles):
    """Same compact OHLCV dict format used throughout - one place so
    every tier (3m/15m/1h/1d) serializes identically for the Gemini
    payload."""
    return [
        {"t": str(idx), "o": round(float(r["open"]), 8), "h": round(float(r["high"]), 8),
         "l": round(float(r["low"]), 8), "c": round(float(r["close"]), 8), "v": round(float(r["volume"]), 2)}
        for idx, r in candles.iterrows()
    ]


def load_daily_candles_30d():
    """Reads the compact per-coin 30-day daily-candle file the daily
    refresh job produces (same script/schedule as RVOL percentiles,
    extended to also fetch this). Read-only here, same reasoning as
    load_rvol_percentiles - a 30-day historical pull is far too heavy
    for the 1-minute live loop to do itself."""
    if os.path.exists(DAILY_CANDLES_FILE):
        try:
            with open(DAILY_CANDLES_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def rvol_percentile_rank(rvol, coin_percentiles):
    """Estimates which percentile of a coin's OWN historical RVOL
    distribution the current value falls in, via linear interpolation
    against the stored breakpoint grid. Returns None if this coin has
    no stored history yet (backfill hasn't run for it) - caller should
    fall back to the fixed-scale rvol_label in that case, not error."""
    if coin_percentiles is None or rvol is None or pd.isna(rvol):
        return None
    grid = coin_percentiles.get("grid")
    breakpoints = coin_percentiles.get("breakpoints")
    if not grid or not breakpoints:
        return None
    # np.interp clamps at the edges: a value below the historical
    # minimum reads as 0th percentile, above the historical maximum
    # reads as 100th - not extrapolated beyond the observed range,
    # which is the honest behavior for a value genuinely outside
    # anything seen in the backfill window.
    return round(float(np.interp(rvol, breakpoints, grid)), 1)


def build_coin_snapshot(coin, candles_3m, candles_15m, rvol_percentiles=None):
    """Replaces the old staged process_coin entirely. No qualification,
    no direction, no pass/fail, no pullback/reversal state machine -
    just a flat snapshot of raw, descriptive data for this coin, every
    single cycle, for every coin. All judgment - including whether
    this coin is worth mentioning at all, and which direction it might
    favor - is left to Gemini. No per-coin state is needed at all -
    the candle-time dedup that decides whether to run this cycle at
    all happens once, globally, in main(), before any of this runs."""
    row = candles_3m.iloc[-1]
    candle_time = str(candles_3m.index[-1])

    rvol = row.get("rvol")
    coin_percentiles = (rvol_percentiles or {}).get(coin)
    snapshot = {
        "coin": coin,
        "candle_time": candle_time,
        "close": round(float(row["close"]), 8),
        "atr14_3m": round(float(row["atr14"]), 8) if not pd.isna(row.get("atr14", float("nan"))) else None,
        "rvol": round(float(rvol), 2) if rvol is not None and not pd.isna(rvol) else None,
        "rvol_label": rvol_label(rvol),
        "rvol_percentile": rvol_percentile_rank(rvol, coin_percentiles),
        "momentum_pct_5_3m": round(float(row["momentum_pct_5"]), 3) if not pd.isna(row.get("momentum_pct_5", float("nan"))) else None,
        "momentum_pct_20_3m": round(float(row["momentum_pct_20"]), 3) if not pd.isna(row.get("momentum_pct_20", float("nan"))) else None,
        "momentum_pct_60_3m": round(float(row["momentum_pct_60"]), 3) if not pd.isna(row.get("momentum_pct_60", float("nan"))) else None,
    }
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", type=str, required=True)
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(",")]
    now = datetime.now(timezone.utc)
    state = load_state()
    if state.pop("recent_flags", None) is not None:  # orphaned key from before the rename to call_history
        save_state(state)  # persist the cleanup now, not just in memory - this is the common early-exit path below

    # Compute the most recently CLOSED 3m candle's boundary purely
    # from the clock - no network call needed, since 3m boundaries are
    # deterministic (:00, :03, :06, ...). Skip the entire expensive
    # fetch+Gemini flow if this candle was already processed last run.
    # CONFIRMED REAL BUG this fixes: without this check, the full 24h
    # fetch (23 coins x ~2 API requests, threaded) was running on
    # EVERY 1-minute cycle regardless of whether a new candle actually
    # existed - since a 3m candle only changes every 3 minutes, that
    # meant roughly 3x more fetching than necessary, every single day.
    forming_start = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 3)
    expected_last_closed = forming_start - timedelta(minutes=3)
    expected_candle_key = str(expected_last_closed)
    if expected_candle_key == state.get("last_sent_candle_time"):
        print(f"No new 3m candle yet (still {expected_candle_key}) - skipping fetch entirely")
        return

    # Single fetch window covers 24h - feeds ALL FOUR candle tiers
    # (3m/15m/1h derived locally via resample, 1d from the daily
    # refresh file) directly from one pull per coin, instead of the
    # old design's two separate fetches (a trimmed ~16h main window
    # plus a second dedicated 24h "enrichment" fetch). Every coin now
    # needs the full tiered context every cycle - there's no more
    # staged pre-filter deciding which few coins "deserve" it - so
    # merging into one fetch avoids doubling the request count.
    start = now - timedelta(hours=24)
    rvol_percentiles = load_rvol_percentiles()
    daily_candles = load_daily_candles_30d()

    print(f"Raw-data scanner (no pre-filtering - Gemini decides) | coins={coins} | {now.isoformat()}")

    def fetch_one(coin):
        """One 24h pull per coin, all tiers derived locally from it -
        read-only, safe to run concurrently."""
        candles_1m = fetch_coindcx_klines(coin, "1m", start.isoformat(), now.isoformat(), stagger_delay=False)
        candles_3m = drop_still_forming_bucket(resample_candles(candles_1m, 3), now, 3)
        candles_15m = drop_still_forming_bucket(resample_candles(candles_1m, 15), now, 15)
        candles_1h = drop_still_forming_bucket(resample_candles(candles_1m, 60), now, 60)
        return coin, candles_3m, candles_15m, candles_1h

    fetched = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_one, coin): coin for coin in coins}
        for future in as_completed(futures):
            coin = futures[future]
            try:
                _, candles_3m, candles_15m, candles_1h = future.result()
                fetched[coin] = (candles_3m, candles_15m, candles_1h)
            except Exception as e:
                print(f"  {coin}: fetch failed ({e}), skipping")

    # Resolve every pending ledger entry against the candle data just
    # fetched - no extra API calls needed, this reuses what's already
    # in memory. Powers both the visible scorecard and the
    # repeat-flag detection below.
    ledger = resolve_ledger(state.get("ledger", []), fetched, now)
    state["ledger"] = ledger
    scorecard = compute_scorecard(ledger, now, window_hours=1)

    snapshots = []
    for coin in coins:
        try:
            if coin not in fetched:
                continue
            candles_3m, candles_15m, candles_1h = fetched[coin]
            if len(candles_3m) < 65:
                # 65, not 30 - confirmed directly that 30 let candles
                # through with too little history for momentum_pct_60
                # (needs 61+ candles), silently producing None for
                # that field instead of either skipping or ensuring
                # complete data. 65 gives real margin above 61.
                print(f"  {coin}: not enough closed candles yet, skipping")
                continue

            candles_3m = compute_raw_stats(candles_3m)
            snapshot = build_coin_snapshot(coin, candles_3m, candles_15m, rvol_percentiles)

            # Prior-call context: Gemini's API has no memory between
            # calls, so this is how it actually gets to see its own
            # recent track record - not just "don't contradict
            # yourself" as an instruction, but the real prior call and
            # what price has genuinely done since, so it can judge
            # whether that thesis is playing out or failing.
            call_history = state.get("call_history", {}).get(coin, [])
            if call_history:
                last_call = call_history[-1]
                minutes_since = (now - datetime.fromisoformat(last_call["time"])).total_seconds() / 60
                prior_entry = last_call.get("entry_price")
                if minutes_since <= 180 and prior_entry:  # only recent calls with a real entry price are usable context
                    price_change_pct = round((snapshot["close"] - prior_entry) / prior_entry * 100, 3)
                    # Explicit three-way check, not an equality trick -
                    # confirmed directly that (pct>0)==(direction=="long")
                    # incorrectly reported "favorable" for a SHORT call
                    # at EXACTLY 0% change (no movement at all), while
                    # correctly reporting "not favorable" for LONG at
                    # the same 0% - an inconsistent, asymmetric bug.
                    if last_call["direction"] == "long":
                        favorable = price_change_pct > 0
                    else:
                        favorable = price_change_pct < 0
                    snapshot["prior_call"] = {
                        "direction": last_call["direction"],
                        "take_trade": last_call["take_trade"],
                        "minutes_ago": round(minutes_since, 1),
                        "price_change_pct_since": price_change_pct,
                        "moved_favorably": favorable,
                    }

            # Tiered candle context, same compact format throughout -
            # 3m recent detail, 15m/1h progressively coarser, 1d from
            # the daily-refreshed file (no live fetch cost for that
            # tier at all).
            snapshot["ctx_3m"] = candles_to_compact(candles_3m.tail(20))
            snapshot["ctx_15m"] = candles_to_compact(candles_15m.tail(28))
            snapshot["ctx_1h"] = candles_to_compact(candles_1h.tail(16))
            snapshot["ctx_daily_30d"] = daily_candles.get(coin, [])

            snapshots.append(snapshot)
            print(f"  {coin}: close={snapshot['close']} rvol={snapshot['rvol']} "
                  f"mom5={snapshot['momentum_pct_5_3m']}")
        except Exception as e:
            print(f"  {coin}: ERROR - {e}")

    # Gate on a genuinely NEW 3m candle, same pattern as before - but
    # now covers ALL coins every cycle, not just ones a pre-filter
    # decided were worth tracking. No pre-filtering means every coin's
    # raw data goes to Gemini every new candle; Gemini alone decides
    # whether anything is worth flagging.
    candle_times_utc = [pd.Timestamp(s["candle_time"]) for s in snapshots if s.get("candle_time")]
    current_candle_time = str(max(candle_times_utc)) if candle_times_utc else None
    should_call_gemini = bool(snapshots) and current_candle_time != state.get("last_sent_candle_time")

    if not should_call_gemini:
        print(f"\nNot calling Gemini (has_snapshots={bool(snapshots)}, "
              f"same_candle_already_processed={current_candle_time == state.get('last_sent_candle_time')})")
        save_state(state)
        return

    # ONE Gemini call per cycle, covering ALL coins together - not
    # staged, not one call per coin. Gemini may return an empty array
    # if nothing across all coins looks worth mentioning - in which
    # case nothing gets sent, matching the "don't spam" preference.
    flagged = get_trade_suggestions_batch(snapshots, scorecard)

    if not flagged:
        print("\nGemini flagged nothing this cycle - not sending")
        state["last_sent_candle_time"] = current_candle_time
        save_state(state)
        return

    # Anti-whipsaw backstop: even with prior-call context now fed to
    # Gemini (see the snapshot-building loop above), it could still
    # reverse - the difference is it now does so WITH awareness of its
    # own track record, not blindly. This stays as a visible check
    # rather than being removed, since it costs nothing and catches
    # anything that slips through regardless of context.
    WHIPSAW_WINDOW_MINUTES = 15
    call_history = state.get("call_history", {})
    for coin, g in flagged.items():
        history = call_history.get(coin, [])
        if history:
            prior = history[-1]
            prior_time = datetime.fromisoformat(prior["time"])
            minutes_since = (now - prior_time).total_seconds() / 60
            if minutes_since <= WHIPSAW_WINDOW_MINUTES and prior["direction"] != g.get("direction"):
                g["whipsaw_warning"] = (f"reverses {prior['direction'].upper()} call from "
                                         f"{int(minutes_since)} min ago")
        history.append({
            "direction": g.get("direction"),
            "entry_price": g.get("entry_price"),
            "take_trade": g.get("take_trade"),
            "time": now.isoformat(),
        })
        call_history[coin] = history[-5:]  # capped - last 5 calls per coin, not unbounded growth

        # Repeat-flag detection: is this coin+direction already an
        # unresolved, pending call from earlier? Confirmed as a real
        # pattern (HEI LONG flagged 5 times in 15 minutes with no
        # acknowledgment any prior one was still open) - flagged
        # visibly, not silently suppressed, same principle as the
        # whipsaw warning above.
        if g.get("take_trade"):
            pending_minutes = find_pending_same_direction(ledger, coin, g.get("direction"), now)
            if pending_minutes is not None:
                g["repeat_warning"] = f"already have a pending {g.get('direction', '').upper()} call on {coin} from {pending_minutes:.0f} min ago"
            # Recorded regardless of repeat status - a genuine
            # continuation call is still real information, just
            # flagged so the person can judge whether it's fresh
            # conviction or the same extended move re-described.
            ledger.append({
                "coin": coin, "direction": g.get("direction"),
                "entry_price": g.get("entry_price"), "stop_loss": g.get("stop_loss"),
                "target_price": g.get("target_price"), "trade_amount_inr": g.get("trade_amount_inr"),
                "max_loss_this_trade_inr": g.get("max_loss_this_trade_inr"),
                "conviction": g.get("conviction"), "status": "pending", "time": now.isoformat(),
            })
    state["call_history"] = call_history
    state["ledger"] = ledger

    candle_start_utc = max(candle_times_utc)
    candle_start_ist = candle_start_utc + timedelta(hours=5, minutes=30)
    candle_close_ist = candle_start_ist + timedelta(minutes=3)
    detected_ist = now + timedelta(hours=5, minutes=30)

    import html
    net_pnl = scorecard["net_pnl_inr"]
    pnl_str = f"+\u20b9{net_pnl}" if net_pnl >= 0 else f"-\u20b9{abs(net_pnl)}"
    lines = [f"\U0001F4CA {candle_start_ist.strftime('%H:%M')}\u2192{candle_close_ist.strftime('%H:%M')} IST "
             f"(detected {detected_ist.strftime('%H:%M:%S')})"]
    lines.append(f"Last 1h: {scorecard['total']} calls \u2014 {scorecard['target_hit']} hit target, "
                  f"{scorecard['stop_hit']} hit stop, {scorecard['pending']} pending \u2014 net {pnl_str}")
    for coin, g in flagged.items():
        verdict = "TAKE" if g.get("take_trade") else "SKIP"
        direction = (g.get("direction") or "?").upper()
        conviction = g.get("conviction")
        verdict_emoji = "\u2705" if verdict == "TAKE" else "\u26aa"
        lines.append(f"\n{verdict_emoji} <b>{html.escape(coin)} {direction} \u2014 {verdict}</b> (conviction {conviction}/10)")
        lines.append(html.escape(g.get("reasoning", "-")))
        if g.get("whipsaw_warning"):
            lines.append(f"\u26a0\ufe0f Contradicts prior call: {html.escape(g['whipsaw_warning'])}")
        if g.get("repeat_warning"):
            lines.append(f"\U0001F501 {html.escape(g['repeat_warning'])} - still unresolved, not a fresh signal")
        lines.append(f"Entry {g.get('entry_price')} | SL {g.get('stop_loss')} | Target {g.get('target_price')}")
        lines.append(f"Amount \u20b9{g.get('trade_amount_inr')} (risk \u20b9{g.get('max_loss_this_trade_inr')})")
        # Fee math only shown when actually noteworthy - a healthy
        # trade's fee breakdown is backup detail, not something that
        # needs displaying every single time. Still ALWAYS computed
        # and enforced (the fee_override check above runs regardless
        # of whether this line is shown), just not always surfaced.
        if g.get("fee_override"):
            lines.append(f"\u26a0 Downgraded to SKIP \u2014 fees (\u20b9{g.get('estimated_fee_inr')}) would eat "
                          f"{g.get('fee_drag_pct')}% of the \u20b9{g.get('gross_profit_at_target_inr')} target profit")
        elif g.get("fee_drag_pct", 0) and g["fee_drag_pct"] > 15:
            lines.append(f"\u26a0 Fees \u2248{g['fee_drag_pct']}% of target profit (net \u20b9{g.get('net_profit_at_target_inr')})")
    message = "\n".join(lines)

    reply_markup = {"inline_keyboard": [
        [{"text": f"\U0001F4C8 {coin} chart", "url": f"https://coindcx.com/futures/B-{coin}_USDT"}]
        for coin in flagged
    ]}

    # Persist call_history NOW, separately from the send-success state
    # below - confirmed directly that send_telegram can raise with no
    # try/except above it, and previously save_state was only called
    # after the send, meaning a transient failure would crash the
    # script and silently discard the call_history just computed -
    # losing this cycle's self-tracking data exactly when a real
    # failure occurs. Deliberately NOT marking last_sent_candle_time
    # here though - that only gets set on CONFIRMED send success below,
    # so a failed send still gets retried next cycle instead of the
    # message being silently dropped forever.
    save_state(state)

    print(f"\nSending Telegram message:\n{message}")
    try:
        send_telegram(message, reply_markup)
        state["last_sent_candle_time"] = current_candle_time
        save_state(state)
    except Exception as e:
        print(f"Telegram send failed ({e}) - call_history was already saved, "
              f"this candle will be retried next cycle since last_sent_candle_time was not updated")


if __name__ == "__main__":
    main()
