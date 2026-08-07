"""
Idea #18: VWAP + 9/21 EMA pullback-rejection entries (from the
web-search trading guide, adapted for crypto), tested on BTC as
requested.

TIMING, DELIBERATELY DIFFERENT FROM EVERY PRIOR IDEA IN THIS PROJECT:
every prior idea decided AND entered using the same bar's own close
("decide at bar i's close, enter at bar i's close" - the established
zero-lookahead convention throughout this project). This strategy is
different by design: the guide specifies "enter on the OPEN of the
NEXT candle after a rejection candle forms" - a genuine one-candle
delay between signal and entry.

ZERO LOOKAHEAD, PRESERVED ACROSS THIS DELAY: the signal is detected
using ONLY bar i's own close/high/low/open (all known once bar i
closes). Entry then uses bar i+1's OWN OPEN PRICE ONLY - never bar
i+1's close, high, or low, which aren't known until bar i+1 itself
closes. Using a bar's own open the moment that bar begins is standard,
legitimate "enter at next candle's open" execution, not lookahead.

Usage:
  python3 run_ema_pullback_backtest.py --coin BTC --days 365 --adx-sweep
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from coindcx_fetcher import fetch_coindcx_klines, resample_candles
from momentum_scalp import detect_ema_pullback_rejection
from fee_model import apply_fees_and_interest, apply_dollar_pnl

MIN_CANDLES_NEEDED = 40
LOOKBACK_WINDOW_BARS = 100


def run_backtest(candles, r_target=1.5, max_hold_bars=60, trend_lookback=3,
                  momentum_checkpoint_bars=None, momentum_mfe_r_threshold=0.0,
                  min_stop_distance_pct=None, adx_min=None):
    """momentum_checkpoint_bars=None -> momentum-failure exit is fully
    disabled (this is the control / original strategy, unchanged).
    When set, a position still open at that many bars after entry is
    checked exactly ONCE: if its max favorable excursion so far (in R,
    the same R used for the 1.5R/2R target) never exceeded
    momentum_mfe_r_threshold (strictly - so a 0R threshold genuinely
    requires having moved into profit at all, not merely "not negative"),
    it's exited at that checkpoint's close.
    If it DID reach the threshold, nothing else changes - the trade
    keeps running under the existing SL / target / 3h max hold exactly
    as before. SL/TP hits always take precedence over the momentum
    check (checked first, below) - a trade that already stopped or
    hit target this bar never reaches the momentum check at all.

    min_stop_distance_pct=None -> no economic-viability filter (control,
    original strategy, unchanged). When set (e.g. 0.20 for 0.20%), a
    signal whose stop distance (as % of entry price) falls BELOW this
    floor is simply never taken: no trade is opened, no row is added to
    `trades`, and the engine immediately resumes scanning for the next
    signal on the very next bar - exactly as if the rejection candle had
    never fired. This is applied AT ENTRY, not as a post-hoc filter on
    the resulting trades list, so trade sequencing (one position at a
    time) is preserved: skipping a micro-stop trade can free the engine
    to catch a later, wider-stop signal it would otherwise have missed
    while "stuck" in the skipped trade. Entry signal detection itself
    (detect_ema_pullback_rejection) is completely untouched - this only
    gates whether a detected signal is actually acted on."""
    # ADX14 is computed once from completed 3m OHLC bars. At signal bar i we
    # only read ADX[i], which is known when that rejection candle closes;
    # entry remains at bar i+1 open, so there is no lookahead.
    # Wilder-style smoothing is approximated with EWM(alpha=1/14).
    candles = candles.copy()
    high = candles["high"]
    low = candles["low"]
    close = candles["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_dm14 = plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm14 = minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    plus_di14 = 100.0 * plus_dm14 / atr14.replace(0, float("nan"))
    minus_di14 = 100.0 * minus_dm14 / atr14.replace(0, float("nan"))
    di_sum = (plus_di14 + minus_di14).replace(0, float("nan"))
    dx = 100.0 * (plus_di14 - minus_di14).abs() / di_sum
    candles["plus_di14"] = pdi
    candles["minus_di14"] = mdi
    candles["adx14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    trades = []
    open_position = None
    pending_entry = None
    n = len(candles)

    for i in range(MIN_CANDLES_NEEDED, n):
        window = candles.iloc[max(0, i - LOOKBACK_WINDOW_BARS):i + 1]
        t = candles.index[i]
        bar_open = window["open"].iloc[-1]
        bar_close = window["close"].iloc[-1]

        if pending_entry is not None and open_position is None:
            direction = pending_entry["direction"]
            entry_price = bar_open
            stop_price = pending_entry["stop_price"]
            stop_distance = abs(entry_price - stop_price)
            stop_distance_pct_check = (stop_distance / entry_price * 100) if entry_price != 0 else 0.0
            # ECONOMIC-VIABILITY FILTER (Idea #18B): if the stop is tighter
            # than the floor, skip this trade entirely - no position opened,
            # no trade row recorded. stop_distance>0 guard is still applied
            # first (unchanged pre-existing behavior for a degenerate 0-width
            # stop), then the min-stop floor on top of it.
            passes_min_stop = (min_stop_distance_pct is None
                                or stop_distance_pct_check >= min_stop_distance_pct)
            if stop_distance > 0 and passes_min_stop:
                target_price = (entry_price + stop_distance * r_target if direction == "long"
                                 else entry_price - stop_distance * r_target)
                open_position = {
                    "direction": direction, "entry": entry_price, "entry_time": t, "entry_index": i,
                    "stop": stop_price, "target": target_price,
                    "mfe_r": 0.0, "momentum_checked": False,
                }
            pending_entry = None
            # NO continue here on purpose: the position was just opened
            # at this candle's OWN open, so this same candle's high/low
            # (after the open) can still hit the fresh stop or target.
            # Falling through to the open_position block below tests
            # this entry candle too, instead of skipping straight to the
            # next candle and missing an intrabar SL/TP hit on entry day.

        if open_position is not None:
            pos = open_position
            direction = pos["direction"]
            bar_high = window["high"].iloc[-1]
            bar_low = window["low"].iloc[-1]
            risk = abs(pos["entry"] - pos["stop"])
            # INTRABAR touch, not close-only: a 3m candle can pierce the
            # SL or TP and still close back inside, which the old
            # close-only check would have missed entirely.
            if direction == "long":
                hit_stop = bar_low <= pos["stop"]
                hit_target = bar_high >= pos["target"]
                bar_mfe = bar_high - pos["entry"]
            else:
                hit_stop = bar_high >= pos["stop"]
                hit_target = bar_low <= pos["target"]
                bar_mfe = pos["entry"] - bar_low
            pos["mfe_r"] = max(pos["mfe_r"], bar_mfe / risk if risk != 0 else 0.0)

            bars_held = i - pos["entry_index"]
            hit_max_hold = bars_held >= max_hold_bars

            # MOMENTUM-FAILURE CHECKPOINT: only evaluated once, only on
            # a position that survived SL/TP this bar, only once bars_held
            # reaches the checkpoint. SL/TP precedence is enforced by the
            # "not hit_stop and not hit_target" guard below.
            hit_momentum_failure = False
            if (momentum_checkpoint_bars is not None
                    and not pos["momentum_checked"]
                    and bars_held >= momentum_checkpoint_bars
                    and not hit_stop and not hit_target):
                pos["momentum_checked"] = True
                if pos["mfe_r"] <= momentum_mfe_r_threshold:
                    # <=, not <: mfe_r starts at 0.0 and (via the max()
                    # above) can never go negative, so a strict "<" against
                    # a 0.0 threshold could never be true - that variant
                    # would have been silently identical to the control.
                    # "<=" makes the 0R case a genuine "must have moved into
                    # profit at all" test, and keeps every other threshold's
                    # pass condition consistently "strictly greater than".
                    hit_momentum_failure = True

            if hit_stop or hit_target or hit_max_hold or hit_momentum_failure:
                # CONSERVATIVE TIEBREAK: 3m OHLC alone can't tell us
                # which level was touched first if both were hit inside
                # the same candle - assume the worse outcome (stop) hit
                # first. hit_stop is therefore checked ahead of hit_target.
                # Fill price is the actual stop/target level (not the
                # candle's close), matching how a real SL/TP order fills.
                if hit_stop:
                    exit_price = pos["stop"]
                    exit_reason = "stop"
                elif hit_target:
                    exit_price = pos["target"]
                    exit_reason = "target"
                elif hit_momentum_failure:
                    exit_price = bar_close
                    exit_reason = "momentum_failure"
                else:
                    exit_price = bar_close
                    exit_reason = "max_hold_time"
                profit = (exit_price - pos["entry"]) if direction == "long" else (pos["entry"] - exit_price)
                r_achieved = profit / risk if risk != 0 else 0
                trades.append({
                    "direction": direction, "entry_time": pos["entry_time"], "exit_time": t,
                    "entry_price": pos["entry"], "exit_price": exit_price,
                    "r_achieved": r_achieved, "exit_reason": exit_reason,
                    "stages_done": 0, "leverage": 5,
                    "stop_distance_pct": abs(pos["entry"] - pos["stop"]) / pos["entry"],
                    "bars_held": bars_held,
                })
                open_position = None
            continue

        for direction in ("long", "short"):
            if detect_ema_pullback_rejection(window, direction, trend_lookback=trend_lookback):
                signal_adx = float(window["adx14"].iloc[-1])
                # ADX measures trend strength, not direction. EMA9/21 + VWAP
                # still determine long/short direction; ADX only gates strength.
                if adx_min is not None:
                    if pd.isna(signal_adx) or signal_adx < adx_min:
                        continue
                stop_price = window["low"].iloc[-1] if direction == "long" else window["high"].iloc[-1]
                pending_entry = {"direction": direction, "stop_price": stop_price, "signal_adx14": signal_adx}
                break

    return pd.DataFrame(trades)


def run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback,
                   momentum_checkpoint_bars=None, momentum_mfe_r_threshold=0.0,
                   min_stop_distance_pct=None, adx_min=None):
    trades = run_backtest(candles, r_target=r_target, max_hold_bars=max_hold_bars, trend_lookback=trend_lookback,
                           momentum_checkpoint_bars=momentum_checkpoint_bars,
                           momentum_mfe_r_threshold=momentum_mfe_r_threshold,
                           min_stop_distance_pct=min_stop_distance_pct,
                           adx_min=adx_min)
    if len(trades) == 0:
        return {"coin": coin, "r_target": r_target, "trend_lookback": trend_lookback,
                "trades": 0, "win_rate": None, "gross_expected_r": None, "net_expected_r": None,
                "avg_cost_r": None, "profit_factor": None,
                "total_pnl": None, "exit_breakdown": {}}
    trades = trades.copy()
    trades["symbol"] = coin
    trades["strategy"] = "ema_pullback_vwap"
    trades = apply_fees_and_interest(trades, bar_minutes=3)
    trades = apply_dollar_pnl(trades)

    # STOP-DISTANCE DIAGNOSTICS: added to check, before any momentum/ADX
    # sweep, whether the ~1.1-1.2R fee cost is driven by structurally tiny
    # rejection-candle stops (CER-17-style issue) rather than a fee-model
    # bug. stop_distance_pct is already captured per-trade in the engine.
    sd = trades["stop_distance_pct"] * 100  # as a percentage, e.g. 0.10 -> 0.10%
    stop_distance_stats = {
        "mean_pct": round(sd.mean(), 4),
        "median_pct": round(sd.median(), 4),
        "p10_pct": round(sd.quantile(0.10), 4),
        "p25_pct": round(sd.quantile(0.25), 4),
        "p50_pct": round(sd.quantile(0.50), 4),
        "p75_pct": round(sd.quantile(0.75), 4),
        "p90_pct": round(sd.quantile(0.90), 4),
        "pct_under_0.10": round((sd < 0.10).mean() * 100, 1),
        "pct_under_0.20": round((sd < 0.20).mean() * 100, 1),
        "pct_under_0.30": round((sd < 0.30).mean() * 100, 1),
        "pct_over_0.50": round((sd > 0.50).mean() * 100, 1),
    }

    # NET R BY EXIT REASON: separates "the edge is fine but stops kill it"
    # from "even target-hitting winners don't clear costs" - the latter
    # would mean tightening stops further (to improve win rate) can't fix
    # this, since it would only shrink R further and raise fee_r_cost more.
    net_r_by_exit = {
        reason: round(group["net_r"].mean(), 4)
        for reason, group in trades.groupby("exit_reason")
    }

    # PROFIT FACTOR: gross $ won on winning trades / gross $ lost on losing
    # trades, computed on NET (post-fee) dollar P&L since that's what
    # actually determines viability. Trades with exactly 0 net P&L count
    # toward neither side (matches standard PF convention). Undefined
    # (None) if there are no losing trades, to avoid a divide-by-zero.
    gross_win = trades.loc[trades["dollar_pnl"] > 0, "dollar_pnl"].sum()
    gross_loss = trades.loc[trades["dollar_pnl"] < 0, "dollar_pnl"].sum()
    profit_factor = round(gross_win / abs(gross_loss), 3) if gross_loss < 0 else None

    return {
        "coin": coin, "r_target": r_target, "trend_lookback": trend_lookback,
        "trades": len(trades), "win_rate": round((trades["dollar_pnl"] > 0).mean() * 100, 1),
        "gross_expected_r": round(trades["r_achieved"].mean(), 4),
        # net_r already computed by apply_fees_and_interest (r_achieved
        # minus fee_interest_r_cost) - this is the number that actually
        # answers "does the edge survive costs", not just gross R.
        "net_expected_r": round(trades["net_r"].mean(), 4),
        # avg_cost_r: mean fee+interest R-cost per trade, i.e. exactly the
        # gap between gross_expected_r and net_expected_r above - reported
        # separately (rather than making the reader subtract) since Idea
        # #18B is specifically about tracking how this shrinks as the
        # min-stop floor rises.
        "avg_cost_r": round(trades["fee_interest_r_cost"].mean(), 4),
        "profit_factor": profit_factor,
        "total_pnl": round(trades["dollar_pnl"].sum(), 2),
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
        "stop_distance_stats": stop_distance_stats,
        "net_r_by_exit_reason": net_r_by_exit,
    }


def _run_hold_minutes_task(args):
    """Top-level (picklable) worker for the parallel exit-mode sweep.
    r_target and trend_lookback are held FIXED here - the only thing
    varying across parallel workers is max_hold_minutes, so any P&L
    difference between runs is attributable to the hold time alone,
    not tangled up with other parameters or (not touched at all here)
    an RSI filter."""
    candles, coin, r_target, hold_minutes, trend_lookback = args
    # -1, matching the same off-by-one fix already applied to the momentum
    # checkpoint: the entry candle itself (bars_held=0) already closes 3
    # real minutes after entry, since entry happens at THAT candle's own
    # open. Without the -1, hit_max_hold (bars_held >= max_hold_bars) fires
    # one full bar later than the label says - a "3m" hold was actually
    # exiting ~6m after entry, "6m" at ~9m, "9m" at ~12m. This makes the
    # labels match actual elapsed time from entry.
    max_hold_bars = max(0, round(hold_minutes / 3) - 1)  # 3-minute candles
    result = run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback)
    result["max_hold_minutes"] = hold_minutes
    return result


def _run_momentum_task(args):
    """Top-level (picklable) worker for the parallel momentum-failure
    ablation. r_target, trend_lookback, and max_hold_bars are held FIXED
    at the CLI defaults - the only things varying across workers are the
    checkpoint (in minutes) and the required MFE-in-R threshold. A
    checkpoint of None with threshold None is the CONTROL: momentum-
    failure exit fully disabled, i.e. the original strategy untouched."""
    candles, coin, r_target, max_hold_bars, trend_lookback, checkpoint_minutes, mfe_r_threshold = args
    checkpoint_bars = max(0, round(checkpoint_minutes / 3) - 1) if checkpoint_minutes is not None else None
    # -1 above corrects an off-by-one: the entry candle itself (bars_held=0)
    # already closes 3 minutes after entry, since entry happened at THAT
    # candle's own open. So bars_held=0 -> 3m elapsed, bars_held=1 -> 6m
    # elapsed, etc. Without the -1, a "3m" checkpoint was actually firing
    # at bars_held>=1 (6m elapsed), and "6m" was firing at 9m elapsed.
    result = run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback,
                            momentum_checkpoint_bars=checkpoint_bars,
                            momentum_mfe_r_threshold=mfe_r_threshold if mfe_r_threshold is not None else 0.0)
    result["checkpoint_minutes"] = checkpoint_minutes
    result["mfe_r_threshold"] = mfe_r_threshold
    return result


def _run_min_stop_task(args):
    """Top-level (picklable) worker for the parallel min-stop-distance /
    cost-to-R economic-viability ablation (Idea #18B). r_target,
    trend_lookback, and max_hold_bars are all held FIXED at the CLI
    values - the ONLY thing varying across workers is the minimum stop
    distance (as % of entry price) required to take a trade at all.
    min_stop_pct=None is the CONTROL: no floor, i.e. the original
    strategy completely unmodified."""
    candles, coin, r_target, max_hold_bars, trend_lookback, min_stop_pct = args
    result = run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback,
                            min_stop_distance_pct=min_stop_pct)
    result["min_stop_pct"] = min_stop_pct
    # cost_pct_of_1r: avg_cost_r expressed as "cost consumes X% of a full
    # 1R move" - the portable, coin/fee-tier-agnostic framing from the
    # analysis (e.g. avg_cost_r=0.5 -> costs eat 50% of 1R on average).
    # None when there were no trades to avoid a spurious 0.0.
    result["cost_pct_of_1r"] = (round(result["avg_cost_r"] * 100, 1)
                                 if result.get("avg_cost_r") is not None else None)
    return result


def _run_adx_task(args):
    """Idea #18C ADX14 ablation on the economically viable subset.
    The minimum stop floor is FIXED at 0.25% for every variant; only the
    ADX minimum threshold changes. ADX is a strength filter only; EMA/VWAP
    continue to determine trade direction. adx_min=None is the control.
    """
    candles, coin, r_target, max_hold_bars, trend_lookback, min_stop_pct, label, adx_min = args
    result = run_one_combo(candles, coin, r_target, max_hold_bars, trend_lookback,
                           min_stop_distance_pct=min_stop_pct, adx_min=adx_min)
    result["adx_variant"] = label
    result["adx_min"] = adx_min
    result["min_stop_pct"] = min_stop_pct
    result["cost_pct_of_1r"] = (round(result["avg_cost_r"] * 100, 1)
                                 if result.get("avg_cost_r") is not None else None)
    return result


def _prior_structure_target(candles, signal_i, direction, lookback):
    """Target from PRIOR completed bars only; excludes the rejection/signal bar."""
    start = max(0, signal_i - lookback)
    prior = candles.iloc[start:signal_i]
    if prior.empty:
        return None
    if direction == "long":
        return float(prior["high"].max())
    return float(prior["low"].min())


def _pullback_structure_stop(candles, signal_i, direction, stop_lookback):
    """Local swing stop using the signal bar plus preceding completed bars."""
    start = max(0, signal_i - stop_lookback + 1)
    w = candles.iloc[start:signal_i + 1]
    if direction == "long":
        return float(w["low"].min())
    return float(w["high"].max())


def run_structure_backtest(candles, coin, r_target_unused=1.5, trend_lookback=3,
                           adx_min=25.0, min_stop_distance_pct=0.25,
                           stop_lookback=3, target_lookback=10,
                           min_structural_rr=1.0, checkpoint_minutes=9,
                           checkpoint_mode="hard", checkpoint_profit_buffer_cost_mult=1.0,
                           final_max_hold_minutes=180, adx_filter_mode="level_only",
                           fee_rate=0.00059):
    """
    Idea #18D: structure-to-structure exit model.

    Direction/setup remains EMA9/21 + VWAP + rejection, with ADX14 strength gate.
    Entry is next 3m candle open.
    SL = local structure extreme over signal bar + previous stop_lookback-1 bars.
    TP = prior structure extreme over target_lookback bars, EXCLUDING signal bar.
    Entry is rejected if target is not beyond entry or structural RR is below threshold.

    checkpoint_mode:
      hard     -> at checkpoint, exit regardless of P&L if SL/TP not already hit.
      momentum -> at checkpoint:
                  * if mark-to-market profit covers estimated round-trip fee buffer,
                    keep running toward structural TP/SL until final max hold;
                  * otherwise exit at checkpoint.
    """
    # Reuse indicator preparation by mirroring the existing engine's calculations.
    candles = candles.copy()

    # EMA/VWAP
    candles["ema9"] = candles["close"].ewm(span=9, adjust=False).mean()
    candles["ema21"] = candles["close"].ewm(span=21, adjust=False).mean()
    typical = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    # Crypto is 24/7; preserve the existing session-like UTC-day VWAP convention.
    day = candles.index.floor("D") if isinstance(candles.index, pd.DatetimeIndex) else pd.to_datetime(candles["timestamp"]).dt.floor("D")
    pv = typical * candles["volume"]
    candles["vwap"] = pv.groupby(day).cumsum() / candles["volume"].groupby(day).cumsum().replace(0, float("nan"))

    # ADX14 (same Wilder/EWM formulation as Idea 18C)
    high, low, close = candles["high"], candles["low"], candles["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr14.replace(0, float("nan"))
    mdi = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr14.replace(0, float("nan"))
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, float("nan"))
    candles["adx14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # Per-side execution fee. Default preserves the old crypto assumption.
    # For CoinDCX low-fee commodity futures pass --fee-rate 0.0001 (0.01% per side).
    round_trip_fee_rate = 2 * fee_rate
    trades = []
    pending = None
    position = None

    checkpoint_bars = max(1, round(checkpoint_minutes / 3))
    final_max_bars = max(checkpoint_bars, round(final_max_hold_minutes / 3))

    # Enough warmup for EMA/ADX + structure windows.
    warmup = max(50, target_lookback + 2, stop_lookback + 2, trend_lookback + 2)

    for i in range(warmup, len(candles)):
        row = candles.iloc[i]

        # Enter pending signal at CURRENT bar open (signal was completed at i-1).
        if pending is not None and position is None:
            entry = float(row["open"])
            stop = pending["stop"]
            target = pending["target"]
            direction = pending["direction"]
            risk = (entry-stop) if direction == "long" else (stop-entry)
            reward = (target-entry) if direction == "long" else (entry-target)
            stop_pct = risk / entry * 100 if entry else 0
            rr = reward / risk if risk > 0 else -1

            if risk > 0 and reward > 0 and stop_pct >= min_stop_distance_pct and rr >= min_structural_rr:
                position = {
                    "direction": direction, "entry": entry, "stop": stop, "target": target,
                    "risk": risk, "rr": rr, "entry_i": i, "checkpoint_done": False,
                    "signal_adx14": pending["adx14"], "stop_pct": stop_pct
                }
            pending = None

        # Manage open position using this completed 3m bar.
        if position is not None:
            p = position
            d = p["direction"]
            bars_elapsed = i - p["entry_i"] + 1

            stop_hit = (float(row["low"]) <= p["stop"]) if d == "long" else (float(row["high"]) >= p["stop"])
            target_hit = (float(row["high"]) >= p["target"]) if d == "long" else (float(row["low"]) <= p["target"])

            # Conservative same-bar ambiguity: stop wins if both touched.
            exit_price = None
            exit_reason = None
            if stop_hit:
                exit_price, exit_reason = p["stop"], "stop"
            elif target_hit:
                exit_price, exit_reason = p["target"], "structure_target"
            elif (not p["checkpoint_done"]) and bars_elapsed >= checkpoint_bars:
                if checkpoint_mode == "hard":
                    exit_price, exit_reason = float(row["close"]), "checkpoint_hard"
                else:
                    mtm = (float(row["close"])-p["entry"]) if d == "long" else (p["entry"]-float(row["close"]))
                    # Profit must cover estimated round-trip fees times requested buffer.
                    required_profit = p["entry"] * round_trip_fee_rate * checkpoint_profit_buffer_cost_mult
                    if mtm < required_profit:
                        exit_price, exit_reason = float(row["close"]), "checkpoint_failed"
                    else:
                        p["checkpoint_done"] = True
            elif bars_elapsed >= final_max_bars:
                exit_price, exit_reason = float(row["close"]), "final_max_hold"

            if exit_price is not None:
                gross_pnl = (exit_price-p["entry"]) if d == "long" else (p["entry"]-exit_price)
                gross_r = gross_pnl / p["risk"]
                # Fee cost in R based on entry + exit notionals.
                cost_r = fee_rate * (p["entry"] + exit_price) / p["risk"]
                net_r = gross_r - cost_r
                trades.append({
                    "gross_r": gross_r, "cost_r": cost_r, "net_r": net_r,
                    "exit_reason": exit_reason, "structural_rr": p["rr"],
                    "stop_pct": p["stop_pct"], "adx14": p["signal_adx14"]
                })
                position = None

        # Generate a fresh signal only when flat and no pending entry.
        if position is None and pending is None and i < len(candles)-1:
            # Completed-bar rejection logic aligned with existing Idea 18:
            # trend direction from EMA/VWAP; rejection candle touches EMA zone
            # and closes back in trend direction.
            w = candles.iloc[max(0, i-trend_lookback+1):i+1]
            ema9 = float(row["ema9"]); ema21 = float(row["ema21"]); vwap = float(row["vwap"])
            adx = float(row["adx14"]) if not pd.isna(row["adx14"]) else float("nan")
            if pd.isna(adx) or adx < adx_min:
                continue

            plus_di = float(row["plus_di14"]) if not pd.isna(row["plus_di14"]) else float("nan")
            minus_di = float(row["minus_di14"]) if not pd.isna(row["minus_di14"]) else float("nan")
            adx_prev1 = float(candles["adx14"].iloc[i-1])
            adx_prev3 = float(candles["adx14"].iloc[i-3])

            adx_rising_1 = adx > adx_prev1
            adx_rising_3 = adx > adx_prev3

            ema9_prev = float(w["ema9"].iloc[0])
            ema21_prev = float(w["ema21"].iloc[0])
            long_trend = ema9 > ema21 and ema9 >= ema9_prev and ema21 >= ema21_prev and float(row["close"]) > vwap
            short_trend = ema9 < ema21 and ema9 <= ema9_prev and ema21 <= ema21_prev and float(row["close"]) < vwap

            # Touch either EMA9/EMA21 and reject back in trend direction.
            zone_lo, zone_hi = min(ema9, ema21), max(ema9, ema21)
            touched_zone = float(row["low"]) <= zone_hi and float(row["high"]) >= zone_lo
            bullish_rejection = touched_zone and float(row["close"]) > float(row["open"]) and float(row["close"]) >= ema9
            bearish_rejection = touched_zone and float(row["close"]) < float(row["open"]) and float(row["close"]) <= ema9

            direction = "long" if (long_trend and bullish_rejection) else ("short" if (short_trend and bearish_rejection) else None)

            if direction:
                di_ok = (plus_di > minus_di) if direction == "long" else (minus_di > plus_di)

                if adx_filter_mode == "level_only":
                    adx_confirmation_ok = True
                elif adx_filter_mode == "di":
                    adx_confirmation_ok = di_ok
                elif adx_filter_mode == "slope1":
                    adx_confirmation_ok = adx_rising_1
                elif adx_filter_mode == "slope3":
                    adx_confirmation_ok = adx_rising_3
                elif adx_filter_mode == "full_slope1":
                    adx_confirmation_ok = di_ok and adx_rising_1
                elif adx_filter_mode == "full_slope3":
                    adx_confirmation_ok = di_ok and adx_rising_3
                else:
                    raise ValueError(f"Unknown adx_filter_mode: {adx_filter_mode}")

                if not adx_confirmation_ok:
                    continue
                stop = _pullback_structure_stop(candles, i, direction, stop_lookback)
                target = _prior_structure_target(candles, i, direction, target_lookback)
                if target is not None:
                    pending = {"direction": direction, "stop": stop, "target": target, "adx14": adx}

    if not trades:
        return {"trades": 0, "win_rate": None, "gross_expected_r": None, "avg_cost_r": None,
                "net_expected_r": None, "profit_factor": None, "total_pnl": None,
                "exit_breakdown": {}}

    t = pd.DataFrame(trades)
    wins = t["net_r"] > 0
    gross_profit = t.loc[t["net_r"] > 0, "net_r"].sum()
    gross_loss = -t.loc[t["net_r"] < 0, "net_r"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Keep total_pnl comparable to the existing engine's convention of $25 risk/trade.
    total_pnl = float((t["net_r"] * 25.0).sum())
    return {
        "trades": len(t),
        "win_rate": round(float(wins.mean()*100), 1),
        "gross_expected_r": round(float(t["gross_r"].mean()), 4),
        "avg_cost_r": round(float(t["cost_r"].mean()), 4),
        "net_expected_r": round(float(t["net_r"].mean()), 4),
        "profit_factor": round(float(pf), 3),
        "total_pnl": round(total_pnl, 2),
        "avg_structural_rr": round(float(t["structural_rr"].mean()), 3),
        "avg_stop_pct": round(float(t["stop_pct"].mean()), 3),
        "exit_breakdown": t["exit_reason"].value_counts().to_dict(),
    }


def _run_structure_task(args):
    candles, coin, adx_min, min_stop, stop_lb, target_lb, min_rr, checkpoint, mode = args
    r = run_structure_backtest(
        candles, coin, trend_lookback=3, adx_min=adx_min,
        min_stop_distance_pct=min_stop, stop_lookback=stop_lb,
        target_lookback=target_lb, min_structural_rr=min_rr,
        checkpoint_minutes=checkpoint, checkpoint_mode=mode
    )
    r.update({
        "adx_min": adx_min, "min_stop_pct": min_stop, "stop_lookback": stop_lb,
        "target_lookback": target_lb, "min_structural_rr": min_rr,
        "checkpoint_minutes": checkpoint, "checkpoint_mode": mode
    })
    return r


def ensure_adx_di_columns(candles, period=14):
    """Guarantee ADX14, +DI14 and -DI14 on the exact dataframe used for the sweep."""
    required = {"adx14", "plus_di14", "minus_di14"}
    if required.issubset(candles.columns):
        return candles

    candles = candles.copy()
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    close = candles["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_sm = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_sm = minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_sm / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_sm / atr.replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom

    candles["plus_di14"] = plus_di
    candles["minus_di14"] = minus_di
    candles["adx14"] = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    return candles

def _run_adx_quality_task(args):
    candles, coin, adx_min, adx_filter_mode, fee_rate = args
    candles = ensure_adx_di_columns(candles)
    # Freeze the best Idea 18D structure:
    # target lookback=10, structural RR>=1.0, 3-bar swing SL,
    # min stop=0.25%, 6m momentum checkpoint.
    r = run_structure_backtest(
        candles, coin,
        trend_lookback=3,
        adx_min=adx_min,
        min_stop_distance_pct=0.25,
        stop_lookback=3,
        target_lookback=10,
        min_structural_rr=1.0,
        checkpoint_minutes=6,
        checkpoint_mode="momentum",
        adx_filter_mode=adx_filter_mode,
        fee_rate=fee_rate,
    )
    r.update({"adx_min": adx_min, "adx_filter_mode": adx_filter_mode, "fee_rate": fee_rate})
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="BTC")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--fee-rate", type=float, default=0.00059,
                        help="Per-side execution fee as decimal. Example: 0.0001 = 0.01%% per side.")
    parser.add_argument("--max-hold-hours", type=float, default=3.0)
    parser.add_argument("--r-target", type=float, default=1.5)
    parser.add_argument("--trend-lookback", type=int, default=3)
    parser.add_argument("--sweep-all", action="store_true")
    parser.add_argument("--momentum-sweep", action="store_true",
                         help="Momentum-failure-exit ablation: control (no momentum check) vs "
                              "checkpoint in {3,6} minutes x required MFE in {>0R,0.1R,0.25R,0.5R}. "
                              "r_target/trend_lookback/max_hold_hours held fixed at their CLI values.")
    parser.add_argument("--adx-sweep", action="store_true",
                         help="Idea #18C: ADX14 trend-strength ablation with min-stop fixed at 0.25%%. "
                              "Tests control and ADX >= {15,20,25,30,35,40}. "
                              "EMA9/21 + VWAP continue to determine direction.")
    parser.add_argument("--min-stop-sweep", action="store_true",
                         help="Idea #18B: economic-viability / minimum stop-distance ablation. "
                              "Control (no floor) vs stop-distance floors in "
                              "{0.10,0.15,0.20,0.25,0.30,0.40,0.50}%% of entry price. Same entries, "
                              "same r_target/trend_lookback/max_hold_hours - only whether a detected "
                              "signal is skipped for being economically too tight varies.")
    parser.add_argument("--structure-sweep", action="store_true",
                         help="Idea #18D: structural SL/TP + 6/9m checkpoint sweep. "
                              "ADX>=25 and min-stop>=0.25%% fixed.")
    parser.add_argument(
        "--adx-quality-sweep", action="store_true",
        help="Idea #18E: sweep ADX thresholds plus DI and ADX-slope confirmation."
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_date = now.date() - timedelta(days=args.days)
    fetch_end_time = now.isoformat()

    mode = ("adx-quality-sweep" if args.adx_quality_sweep
            else "structure-sweep" if args.structure_sweep
            else "adx-sweep" if args.adx_sweep
            else "min-stop-sweep" if args.min_stop_sweep
            else "momentum-sweep" if args.momentum_sweep
            else "hold-sweep" if args.sweep_all
            else "single")
    print(f"Idea #18 [{mode}]: EMA Pullback + VWAP | {args.coin} | 3m | "
          f"{start_date} to {now.date()} ({args.days}d)")

    candles_1m = fetch_coindcx_klines(args.coin, "1m", str(start_date), fetch_end_time)
    candles_3m = resample_candles(candles_1m, 3)
    print(f"{args.coin}: {len(candles_1m)} 1m candles -> {len(candles_3m)} 3m candles"
          + (" (ONE fetch, reused for every combo below)" if mode != "single" else ""))

    if mode == "single":
        # -1: same off-by-one correction applied everywhere else in this
        # file - the entry candle itself (bars_held=0) already represents
        # the first 3 elapsed minutes, since entry happens at that candle's
        # own open. Without the -1, a "3h" max hold would actually fire one
        # 3m candle (one full bar) late.
        max_hold_bars = max(0, round(args.max_hold_hours * 60 / 3) - 1)
        result = run_one_combo(candles_3m, args.coin, args.r_target, max_hold_bars, args.trend_lookback)
        print(f"\n=== RESULTS: {args.coin} ===")
        print(f"Trades: {result['trades']} | Win rate: {result['win_rate']} | "
              f"Gross expected R: {result['gross_expected_r']} | Net expected R (after fees): {result['net_expected_r']} | "
              f"Total $ P&L: {result['total_pnl']}")
        print(f"Exit reason breakdown: {result['exit_breakdown']}")
        print(f"Net R by exit reason: {result['net_r_by_exit_reason']}")
        print(f"Stop distance %% stats: {result['stop_distance_stats']}")
        return

    if mode == "momentum-sweep":
        # MOMENTUM-FAILURE ABLATION: r_target, trend_lookback, and the 3h
        # max hold are all held FIXED at their CLI values - nothing about
        # the entry, target sizing, or final backstop hold changes. The
        # ONLY thing varying is whether/when/how strictly a still-open
        # position gets killed for failing to show favorable movement.
        # SL/TP always take precedence over this check (enforced inside
        # run_backtest, not here). Checkpoint restricted to {3,6} minutes
        # since 3m candles can't resolve a precise 5-minute mark. Control
        # (checkpoint=None) is the original strategy, completely unmodified -
        # included so every other variant is compared against it directly.
        # 0.25R is the primary hypothesis per the spec; the surrounding
        # thresholds (>0R, 0.1R, 0.5R) are there to see if there's a broad
        # relationship rather than cherry-picking whichever number wins.
        # -1: same off-by-one correction as the 3/6/9m hold-sweep and
        # single mode - bars_held=0 already represents the first 3 elapsed
        # minutes (entry happens at that candle's own open), so without
        # the -1 the nominal max_hold_hours fires one 3m candle late.
        max_hold_bars = max(0, round(args.max_hold_hours * 60 / 3) - 1)
        checkpoints_minutes = [3, 6]
        mfe_r_thresholds = [0.0, 0.1, 0.25, 0.5]

        variants = [(None, None)]  # control: momentum-failure exit disabled
        variants += [(cp, thr) for cp in checkpoints_minutes for thr in mfe_r_thresholds]

        tasks = [
            (candles_3m, args.coin, args.r_target, max_hold_bars, args.trend_lookback, cp, thr)
            for cp, thr in variants
        ]

        results = []
        max_workers = min(len(tasks), os.cpu_count() or 1)
        print(f"Running {len(tasks)} momentum-failure variants in parallel ({max_workers} workers)...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_momentum_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                label = "CONTROL (no momentum check)" if result["checkpoint_minutes"] is None \
                    else f"checkpoint={result['checkpoint_minutes']}m, MFE>{result['mfe_r_threshold']}R"
                print(f"  {label} -> {result['trades']:5} trades | win_rate={result['win_rate']} | "
                      f"gross_R={result['gross_expected_r']} | net_R={result['net_expected_r']} | "
                      f"total_pnl={result['total_pnl']} | exits={result['exit_breakdown']}")

        results_df = pd.DataFrame(results).sort_values(
            by=["checkpoint_minutes", "mfe_r_threshold"], na_position="first")
        print(f"\n=== MOMENTUM-FAILURE ABLATION: {args.coin} "
              f"(r_target={args.r_target}, trend_lookback={args.trend_lookback}, "
              f"max_hold={args.max_hold_hours}h all fixed; {len(tasks)} variants incl. control, 1 fetch) ===")
        print(results_df.to_string(index=False))

        valid = results_df.dropna(subset=["total_pnl"])
        if len(valid):
            best = valid.loc[valid["total_pnl"].idxmax()]
            best_label = "CONTROL" if pd.isna(best["checkpoint_minutes"]) \
                else f"checkpoint={best['checkpoint_minutes']}m, MFE>{best['mfe_r_threshold']}R"
            print(f"\nBest variant for {args.coin}: {best_label} -> ${best['total_pnl']:.2f} "
                  f"({best['trades']} trades, {best['win_rate']}% win rate, "
                  f"gross R={best['gross_expected_r']}, net R={best['net_expected_r']})")
        return

    if mode == "adx-quality-sweep":
        candles_3m = ensure_adx_di_columns(candles_3m)
        required = ["adx14", "plus_di14", "minus_di14"]
        missing = [c for c in required if c not in candles_3m.columns]
        if missing:
            raise RuntimeError(f"Missing ADX/DI columns: {missing}")
        print("ADX/DI columns ready:", required)
        thresholds = (20.0, 25.0, 30.0, 35.0, 40.0)
        modes = ("level_only", "di", "slope1", "slope3", "full_slope1", "full_slope3")
        tasks = [(candles_3m, args.coin, threshold, filt, args.fee_rate)
                 for threshold in thresholds for filt in modes]

        print(f"Running Idea #18E ADX quality sweep: {len(tasks)} variants")
        print(f"Fee assumption: {args.fee_rate*100:.4f}% per side ({2*args.fee_rate*100:.4f}% round trip before other costs)")
        print("FIXED from best 18D: targetLB=10, structural RR>=1.0, "
              "3-bar swing SL, min-stop>=0.25%, 6m momentum checkpoint")
        print("ADX thresholds: 20, 25, 30, 35, 40")
        print("Modes: level_only, DI direction, 1-bar ADX slope, 3-bar ADX slope, "
              "DI+1-bar slope, DI+3-bar slope")

        results = []
        max_workers = min(len(tasks), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_adx_quality_task, task): task for task in tasks}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                print(f"  ADX>={r['adx_min']:>4.0f} {r['adx_filter_mode']:<12} -> "
                      f"n={r['trades']:4} win={r['win_rate']}% "
                      f"grossR={r['gross_expected_r']} costR={r['avg_cost_r']} "
                      f"netR={r['net_expected_r']} PF={r['profit_factor']} "
                      f"PnL={r['total_pnl']}")

        df = pd.DataFrame(results).sort_values(["adx_min", "adx_filter_mode"])
        print("\n=== IDEA #18E: ADX THRESHOLD + DIRECTION + SLOPE SWEEP ===")
        cols = ["adx_min","adx_filter_mode","trades","win_rate","gross_expected_r",
                "avg_cost_r","net_expected_r","profit_factor","total_pnl",
                "avg_structural_rr","avg_stop_pct","exit_breakdown"]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))

        valid = df.dropna(subset=["net_expected_r"])
        if len(valid):
            best = valid.loc[valid["net_expected_r"].idxmax()]
            print(f"\nBest net-R variant: ADX>={best['adx_min']:.0f}, "
                  f"mode={best['adx_filter_mode']} -> n={best['trades']}, "
                  f"win={best['win_rate']}%, grossR={best['gross_expected_r']}, "
                  f"costR={best['avg_cost_r']}, netR={best['net_expected_r']}, "
                  f"PF={best['profit_factor']}, PnL={best['total_pnl']}")
        return

    if mode == "structure-sweep":
        # Idea #18D. Freeze ADX>=25 and economic min-stop>=0.25%.
        # Sweep only the user-observed structural choices:
        # stop lookback 3 bars; target lookback 5/10 bars;
        # minimum structural RR 1.0/1.25/1.5;
        # checkpoint 6/9m; hard exit vs momentum checkpoint.
        adx_min = 25.0
        min_stop = 0.25
        tasks = []
        for target_lb in (5, 10):
            for min_rr in (1.0, 1.25, 1.5):
                for checkpoint in (6, 9):
                    for checkpoint_mode in ("hard", "momentum"):
                        tasks.append((candles_3m, args.coin, adx_min, min_stop, 3,
                                      target_lb, min_rr, checkpoint, checkpoint_mode))

        results = []
        max_workers = min(len(tasks), os.cpu_count() or 1)
        print(f"Running Idea #18D structural sweep: {len(tasks)} variants, {max_workers} workers")
        print("FIXED: ADX14>=25, min_stop>=0.25%, local swing SL=3 bars")
        print("SWEEP: target lookback={5,10}, min structural RR={1.0,1.25,1.5}, "
              "checkpoint={6m,9m}, mode={hard,momentum}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_structure_task, task): task for task in tasks}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                print(f"  targetLB={r['target_lookback']:2} RR>={r['min_structural_rr']:<4} "
                      f"cp={r['checkpoint_minutes']}m {r['checkpoint_mode']:<8} -> "
                      f"n={r['trades']:4} grossR={r['gross_expected_r']} "
                      f"costR={r['avg_cost_r']} netR={r['net_expected_r']} "
                      f"PF={r['profit_factor']} pnl={r['total_pnl']}")

        df = pd.DataFrame(results).sort_values(
            ["target_lookback","min_structural_rr","checkpoint_minutes","checkpoint_mode"]
        )
        print("\n=== IDEA #18D: STRUCTURE-TO-STRUCTURE RESULTS ===")
        cols = ["target_lookback","min_structural_rr","checkpoint_minutes","checkpoint_mode",
                "trades","win_rate","gross_expected_r","avg_cost_r","net_expected_r",
                "profit_factor","total_pnl","avg_structural_rr","avg_stop_pct","exit_breakdown"]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        valid = df.dropna(subset=["net_expected_r"])
        if len(valid):
            best = valid.loc[valid["net_expected_r"].idxmax()]
            print(f"\nBest net-R variant: targetLB={best['target_lookback']}, "
                  f"RR>={best['min_structural_rr']}, cp={best['checkpoint_minutes']}m "
                  f"{best['checkpoint_mode']} -> n={best['trades']}, "
                  f"grossR={best['gross_expected_r']}, costR={best['avg_cost_r']}, "
                  f"netR={best['net_expected_r']}, PF={best['profit_factor']}")
        return

    if mode == "adx-sweep":
        # IDEA #18C: clean ADX14 trend-strength ablation. Based on #18B,
        # min-stop is held fixed at >=0.25% for every row. The ONLY changing
        # variable is the minimum ADX14 required on the completed signal bar.
        max_hold_bars = max(0, round(args.max_hold_hours * 60 / 3) - 1)
        fixed_min_stop_pct = 0.25
        variants = [
            ("CONTROL_no_ADX", None),
            ("ADX_ge_15", 15.0),
            ("ADX_ge_20", 20.0),
            ("ADX_ge_25", 25.0),
            ("ADX_ge_30", 30.0),
            ("ADX_ge_35", 35.0),
            ("ADX_ge_40", 40.0),
        ]
        tasks = [
            (candles_3m, args.coin, args.r_target, max_hold_bars, args.trend_lookback,
             fixed_min_stop_pct, label, threshold)
            for label, threshold in variants
        ]
        results = []
        max_workers = min(len(tasks), os.cpu_count() or 1)
        print(f"Running {len(tasks)} ADX14 variants in parallel ({max_workers} workers); "
              f"min_stop fixed at >={fixed_min_stop_pct}%...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_adx_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"  {result['adx_variant']} -> {result['trades']:5} trades | "
                      f"win_rate={result['win_rate']} | gross_R={result['gross_expected_r']} | "
                      f"avg_cost_R={result['avg_cost_r']} | net_R={result['net_expected_r']} | "
                      f"PF={result['profit_factor']} | total_pnl={result['total_pnl']}")

        order = {label: i for i, (label, _) in enumerate(variants)}
        results_df = pd.DataFrame(results)
        results_df["_order"] = results_df["adx_variant"].map(order)
        results_df = results_df.sort_values("_order").drop(columns="_order")
        print_cols = [c for c in results_df.columns if c not in
                      ("stop_distance_stats", "net_r_by_exit_reason", "exit_breakdown")]
        print(f"\n=== ADX14 ABLATION (Idea #18C): {args.coin} "
              f"(min_stop>={fixed_min_stop_pct}% FIXED, r_target={args.r_target}, "
              f"trend_lookback={args.trend_lookback}, max_hold={args.max_hold_hours}h; "
              f"{len(tasks)} variants incl. control, 1 fetch) ===")
        print(results_df[print_cols].to_string(index=False))

        print("\nGross/net R by ADX threshold:")
        for _, row in results_df.iterrows():
            print(f"  {row['adx_variant']:>14}: gross_R={row['gross_expected_r']:>8} | "
                  f"cost_R={row['avg_cost_r']:>8} | net_R={row['net_expected_r']:>8} | "
                  f"n={int(row['trades'])}")

        valid = results_df.dropna(subset=["total_pnl"])
        if len(valid):
            best = valid.loc[valid["total_pnl"].idxmax()]
            print(f"\nBest ADX variant for {args.coin}: {best['adx_variant']} -> "
                  f"${best['total_pnl']:.2f} ({best['trades']} trades, "
                  f"{best['win_rate']}% win rate, gross R={best['gross_expected_r']}, "
                  f"net R={best['net_expected_r']}, PF={best['profit_factor']})")
        return

    if mode == "min-stop-sweep":
        # ECONOMIC-VIABILITY / MIN-STOP-DISTANCE ABLATION (Idea #18B): a
        # separate, simpler experiment from the momentum-failure ablation
        # above. r_target, trend_lookback, and max_hold are all held FIXED
        # at their CLI values - the ONLY thing varying is the minimum stop
        # distance (as % of entry price) required to actually take a
        # detected signal. This directly tests the diagnosis from the
        # stop-distance-vs-fee analysis: does removing the tightest-stop
        # (highest cost/R) trades let net expectancy recover, or does gross
        # R stay near zero even after they're gone (which would mean the
        # entry signal itself lacks edge, independent of stop sizing)?
        # Thresholds intentionally broad, not tuned to "the best number" -
        # per the analysis, the goal is the shape of the relationship.
        # -1: same off-by-one correction as everywhere else - bars_held=0
        # already represents the first 3 elapsed minutes since entry
        # happens at that candle's own open, so without the -1 the nominal
        # max_hold_hours fires one 3m candle late.
        max_hold_bars = max(0, round(args.max_hold_hours * 60 / 3) - 1)
        min_stop_thresholds = [None, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

        tasks = [
            (candles_3m, args.coin, args.r_target, max_hold_bars, args.trend_lookback, thr)
            for thr in min_stop_thresholds
        ]

        results = []
        max_workers = min(len(tasks), os.cpu_count() or 1)
        print(f"Running {len(tasks)} min-stop-distance variants in parallel ({max_workers} workers)...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_min_stop_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                label = "CONTROL (no min stop)" if result["min_stop_pct"] is None \
                    else f"min_stop>={result['min_stop_pct']}%"
                print(f"  {label} -> {result['trades']:5} trades | win_rate={result['win_rate']} | "
                      f"gross_R={result['gross_expected_r']} | avg_cost_R={result['avg_cost_r']} "
                      f"({result['cost_pct_of_1r']}% of 1R) | net_R={result['net_expected_r']} | "
                      f"PF={result['profit_factor']} | total_pnl={result['total_pnl']}")

        results_df = pd.DataFrame(results).sort_values(by="min_stop_pct", na_position="first")
        print_cols = [c for c in results_df.columns if c not in ("stop_distance_stats", "net_r_by_exit_reason", "exit_breakdown")]
        print(f"\n=== MIN-STOP-DISTANCE ABLATION (Idea #18B): {args.coin} "
              f"(r_target={args.r_target}, trend_lookback={args.trend_lookback}, "
              f"max_hold={args.max_hold_hours}h all fixed; {len(tasks)} variants incl. control, 1 fetch) ===")
        print(results_df[print_cols].to_string(index=False))

        valid = results_df.dropna(subset=["gross_expected_r"])
        if len(valid):
            print("\nGross R by threshold (does the signal have edge once micro-stop trades are removed?):")
            for _, row in valid.iterrows():
                label = "CONTROL" if pd.isna(row["min_stop_pct"]) else f">={row['min_stop_pct']}%"
                print(f"  {label:>10}: gross_R={row['gross_expected_r']:>8} | net_R={row['net_expected_r']:>8} | "
                      f"n={row['trades']}")

        valid_pnl = results_df.dropna(subset=["total_pnl"])
        if len(valid_pnl):
            best = valid_pnl.loc[valid_pnl["total_pnl"].idxmax()]
            best_label = "CONTROL" if pd.isna(best["min_stop_pct"]) else f"min_stop>={best['min_stop_pct']}%"
            print(f"\nBest variant for {args.coin}: {best_label} -> ${best['total_pnl']:.2f} "
                  f"({best['trades']} trades, {best['win_rate']}% win rate, "
                  f"gross R={best['gross_expected_r']}, net R={best['net_expected_r']}, "
                  f"PF={best['profit_factor']})")
        return

    # HOLD-TIME SWEEP (--sweep-all): a separate, simpler experiment from
    # the momentum-failure ablation above - this tests a hard timeout
    # ("exit everything still open after N minutes, regardless of
    # profitability"), not the momentum-failure rule. r_target and
    # trend_lookback held fixed; NOT mixed with ADX or any other change.
    hold_minutes_options = [3, 6, 9]

    tasks = [
        (candles_3m, args.coin, args.r_target, hold_minutes, args.trend_lookback)
        for hold_minutes in hold_minutes_options
    ]

    results = []
    max_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"Running {len(tasks)} hold-time variants in parallel ({max_workers} workers)...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_hold_minutes_task, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  hold={result['max_hold_minutes']}m, r_target={result['r_target']}, "
                  f"trend_lookback={result['trend_lookback']} -> "
                  f"{result['trades']:5} trades | win_rate={result['win_rate']} | "
                  f"gross_R={result['gross_expected_r']} | net_R={result['net_expected_r']} | "
                  f"total_pnl={result['total_pnl']}")

    results_df = pd.DataFrame(results).sort_values("max_hold_minutes")
    print_cols = [c for c in results_df.columns if c not in ("stop_distance_stats", "net_r_by_exit_reason")]
    print(f"\n=== EXIT-MODE SWEEP RESULTS: {args.coin} "
          f"(r_target={args.r_target}, trend_lookback={args.trend_lookback} fixed; "
          f"{len(tasks)} hold-time variants, 1 fetch) ===")
    print(results_df[print_cols].to_string(index=False))

    # Stop distance is a function of the entry signal only, not the hold
    # time, so it's ~identical across the 3/6/9m variants (same entries,
    # only the exit rule differs) - print it once rather than 3x.
    print(f"\nStop distance %% stats ({args.coin}, same across hold variants - entries are unchanged): "
          f"{results[0]['stop_distance_stats']}")
    for r in sorted(results, key=lambda x: x["max_hold_minutes"]):
        print(f"  hold={r['max_hold_minutes']}m net R by exit reason: {r['net_r_by_exit_reason']}")

    valid = results_df.dropna(subset=["total_pnl"])
    if len(valid):
        best = valid.loc[valid["total_pnl"].idxmax()]
        print(f"\nBest hold time for {args.coin}: {best['max_hold_minutes']}m -> "
              f"${best['total_pnl']:.2f} ({best['trades']} trades, "
              f"{best['win_rate']}% win rate, gross R={best['gross_expected_r']}, "
              f"net R={best['net_expected_r']})")


if __name__ == "__main__":
    main()
