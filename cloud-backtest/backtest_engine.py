"""
The backtest engine. Walks forward through 5m candles (the "primary"
timeframe), building 15m ("confirm") and 1h ("filter") context at each
step from ONLY fully-closed higher-timeframe candles as of that moment -
avoids look-ahead bias.

REBUILT to be preset-aware: takes a `strategy` name (one of ultra-short/
aggressive/balanced/conservative/swing-trend) and resolves that preset's
real min_score, ATR stop-multiplier/distance bounds, and leverage bounds
from scoring_stoploss.py's verified tables - instead of always using
balanced's numbers regardless of what preset was requested.

Also now captures per-trade: stages_done, leverage, and stop_distance_pct
- needed by fee_model.py to compute real fee/interest costs after the
fact, as a separate, clearly-labeled step rather than baked silently into
this core simulation loop.
"""
import pandas as pd
import numpy as np

from indicators import ema, rsi, macd, macd_histogram_turn, atr_ratio, atr_wilder, detect_obv_price_divergence, detect_real_cvd_divergence
from market_state_analyzer import (
    build_timeframe_indicators, determine_trend_strength, determine_momentum_state,
    determine_market_state, calculate_triple_timeframe_consistency, calculate_trend_score,
    calculate_reversal_score, detect_btc_trend_adverse, calculate_stop_loss_with_btc_floor,
)
from strategy_logic import route_strategy
from scoring_stoploss import (
    score_opportunity, should_open_position, analyze_market_volatility,
    calculate_r_multiple, calculate_target_price,
    STRATEGY_SCORE_WEIGHTS, STRATEGY_STOP_LOSS, STRATEGY_LEVERAGE_BOUNDS,
)
from coindcx_fetcher import resample_candles

MIN_CANDLES_NEEDED = 55  # matches the live bot's own minimum


def _closed_bucket_candles(resampled, as_of_time, rule_minutes, window=200, primary_step_minutes=5):
    """Returns the last `window` fully-closed buckets as of `as_of_time` -
    bounded rolling window (fixes the O(n^2) unbounded-slice bug that
    likely caused Colab hangs on full-year runs).

    primary_step_minutes: how far the outer loop actually steps forward
    each iteration. Previously hardcoded to a bare 5-minute tolerance,
    silently assuming the primary timeframe was always 5m - now that it's
    configurable, using the wrong tolerance here means treating a candle
    as "closed" earlier or later than it really is, which is a real
    lookahead-bias risk specifically on faster combos like 1m primary."""
    sliced = resampled.loc[:as_of_time].tail(window)
    if len(sliced) == 0:
        return sliced
    last_bucket_start = sliced.index[-1]
    bucket_end = last_bucket_start + pd.Timedelta(minutes=rule_minutes)
    if bucket_end > as_of_time + pd.Timedelta(minutes=primary_step_minutes):
        return sliced.iloc[:-1]
    return sliced


def run_backtest(symbol, candles_5m, strategy="balanced", min_score=None, max_positions=1,
                  max_hold_bars=None, stage_multipliers=(1, 2, 3),
                  direction_filter=None, setup_filter=None, verbose=False,
                  full_close_at_stage1=False, atr_multiplier_override=None,
                  reversal_exit_threshold=70, raw_reversal_threshold=None,
                  skip_filter_timeframe=False, confirm_minutes=15, filter_minutes=60,
                  primary_minutes=5, asymmetric_free_ride=False,
                  stage1_close_fraction=0.65, trailing_atr_multiplier=1.5,
                  use_adverse_drift=False, drift_net_threshold=10,
                  # --- Idea #2: time-decay dynamic reversal threshold ---
                  dynamic_threshold_enabled=False, dynamic_threshold_after_minutes=45,
                  dynamic_threshold_drawdown_r=-0.4, dynamic_threshold_tightened=35,
                  # --- Idea #3: ATR stop compression on adverse drift ---
                  drift_stop_tighten_enabled=False, drift_stop_tighten_atr_multiplier=1.2,
                  # --- Idea #4: OBV/price-divergence confirmation bonus (proxy for CVD, see indicators.py) ---
                  obv_confirmation_bonus=0, obv_lookback_bars=10, obv_slope_threshold=0.3,
                  use_btc_trend_bonus=False, btc_trend_bonus=0, btc_min_score_magnitude=25,
                  use_btc_stop_floor=False, btc_stop_beta=1.0,
                  # --- Idea #5: mechanical soft-exit proxy (TRIM/TIGHTEN/FREEZE) - NOT real Gemini judgment ---
                  soft_exit_enabled=False, soft_exit_trim_threshold=45, soft_exit_trim_fraction=0.5,
                  soft_exit_tighten_threshold=35, soft_exit_tighten_atr_multiplier=1.5,
                  soft_exit_freeze_minutes=60):
    """
    strategy: one of "ultra-short"/"aggressive"/"balanced"/"conservative"/
    "swing-trend". Resolves that preset's real min_score, ATR stop
    multiplier/distance bounds, and leverage from the verified tables in
    scoring_stoploss.py.

    full_close_at_stage1: if True, closes 100% of the position the moment
    the (volatility-adjusted) 1R target is hit, instead of the normal
    33.33%/33.33%/0% staged exit across 1R/2R/3R. A genuinely different
    exit policy, not a variant of "fewer fee legs" - it changes which R
    the position books, not just how many fills it takes to book it.

    asymmetric_free_ride: a THIRD, distinct exit policy (mutually
    exclusive with full_close_at_stage1) - closes stage1_close_fraction
    (default 65%) of the position at the volatility-adjusted 1R target
    and moves the stop to breakeven (same as the normal staged exit's
    stage 1), but the REMAINING fraction has no fixed 2R/3R target at
    all - instead it rides on a trailing stop, recalculated from the
    CURRENT price each bar using trailing_atr_multiplier x the CURRENT
    ATR (not the entry-time ATR), only ever moving in the favorable
    direction. Tests whether "secure most of the win immediately, let a
    small remainder ride with zero fixed ceiling" beats the existing
    proven staged exit, rather than assuming it does.

    atr_multiplier_override: if set, overrides the preset's own default
    ATR stop-loss multiplier (e.g. testing conservative/swing-trend at a
    tighter 2.0x/2.25x instead of their default 2.5x, while keeping every
    other preset parameter - min_score, leverage, position size -
    unchanged). None = use the preset's real deployed value.

    reversal_exit_threshold: the BUCKETED reversal score that triggers an
    exit (live default 70, "close immediately"). This score only ever
    takes fixed stacked values (12/20/25/40/52/57/65/77/...), so testing
    55 vs 60 vs 65 always gives identical results if no trade's real
    reading ever fell between them - confirmed empirically already.

    raw_reversal_threshold: NEW - bypasses the bucket entirely and checks
    the underlying continuous primary-timeframe trend number directly
    (e.g. -8, -23, -41, not a fixed step). None = disabled, only the
    bucketed check above runs (matches current live behavior exactly).
    When set, adds a genuinely continuous exit check alongside the
    bucketed one - whichever fires first still wins. Each trade's
    exit_reason will show "raw_reversal_beat_stop" (this genuinely
    triggered before the stop would have on the same bar - a real save)
    vs "raw_reversal_same_bar_as_stop" (fired on the same bar the stop
    also would have - no actual benefit over just waiting for the stop),
    so this data point answers "does this actually save anything" directly,
    not just "does it exit differently."

    skip_filter_timeframe: NEW - when True, substitutes tf_confirm (15m)
    everywhere tf_filter (1h) would normally be used (market_state's
    atr_ratio, route_strategy, score_opportunity, timeframe-alignment
    scoring), and never builds the 1h slice or its indicators at all.
    Directly tests whether the filter timeframe earns its own
    computational cost, or whether entries are just as good using only
    2 timeframes (primary+confirm) instead of 3. Real speed test, not a
    approximation - the 1h resample/indicator work is skipped entirely,
    not just cached.

    confirm_minutes / filter_minutes: NEW - the actual confirm/filter
    timeframe intervals, in minutes. Defaults (15/60) match the live
    bot's real deployed 15m/1h setup exactly. Override to test genuinely
    faster combinations (e.g. confirm_minutes=5, filter_minutes=15 for a
    1m/5m/15m setup, alongside a primary_minutes override in main.py) -
    tests whether a quicker-reacting timeframe combination still
    produces good entries, given real evidence that scores can shift a
    lot within a single 15-minute window.

    Returns (trades_df, equity_curve_series). trades_df includes
    stages_done/leverage/stop_distance_pct for fee_model.py to consume.
    """
    weights = STRATEGY_SCORE_WEIGHTS[strategy]
    stop_cfg = STRATEGY_STOP_LOSS[strategy]
    atr_multiplier = atr_multiplier_override if atr_multiplier_override is not None else stop_cfg["atr_multiplier"]

    if full_close_at_stage1 and asymmetric_free_ride:
        raise ValueError("full_close_at_stage1 and asymmetric_free_ride are mutually exclusive exit policies - only one can be active per run.")
    lev_bounds = STRATEGY_LEVERAGE_BOUNDS[strategy]
    resolved_leverage = (lev_bounds["min"] + lev_bounds["max"]) / 2
    effective_min_score = min_score if min_score is not None else weights["min_score"]

    candles_confirm_full = resample_candles(candles_5m, confirm_minutes)
    candles_filter_full = None if skip_filter_timeframe else resample_candles(candles_5m, filter_minutes)

    # Bug 3 fix: previously nothing stopped a nonsensical timeframe
    # ordering (e.g. primary >= confirm) from silently running a
    # logically broken backtest instead of failing clearly.
    if primary_minutes >= confirm_minutes:
        raise ValueError(f"primary_minutes ({primary_minutes}) must be smaller than confirm_minutes ({confirm_minutes})")
    if not skip_filter_timeframe and confirm_minutes >= filter_minutes:
        raise ValueError(f"confirm_minutes ({confirm_minutes}) must be smaller than filter_minutes ({filter_minutes})")

    # Bug 2 fix: preserves the ORIGINAL real-world lookback span (not
    # just a fixed bar count) regardless of which interval combination is
    # used - previously a fixed 200-bar window meant faster combos (e.g.
    # 1m primary) silently got 5x LESS real lookback history for the
    # exact same indicators (EMA20/50 etc.), confounding any comparison
    # between combos. These minute-targets match what 200 bars
    # represented at the combo this was originally tuned around
    # (5m/15m/60m).
    primary_window_bars = max(55, round(1000 / primary_minutes))
    confirm_window_bars = max(55, round(3000 / confirm_minutes))
    filter_window_bars = max(55, round(12000 / filter_minutes))

    trades = []
    open_position = None
    trend_history = {"primary": [], "confirm": [], "filter": []}
    equity = 1.0
    equity_curve = []
    # Idea #5 (FREEZE_REENTRY proxy): index (bar count, not wall time) up
    # to which new entries in a given direction are blocked, set after a
    # basket/drift-related early exit. -1 = no freeze active. Keyed by
    # direction since a stopped-out short shouldn't necessarily block a
    # fresh long signal.
    freeze_until_index = {"long": -1, "short": -1}

    n = len(candles_5m)
    # Confirm(15m)/filter(1h) only actually change once every 3 / 12
    # primary(5m) iterations respectively - but were being fully
    # recomputed from scratch every single iteration regardless. Caching
    # by the slice's own last timestamp - a cheap, correctness-preserving
    # check - and only recomputing when it's genuinely a new closed
    # candle cuts ~2/3 of confirm computation and ~11/12 of filter
    # computation, since indicator math is otherwise identical to the
    # last time this exact slice was seen.
    cached_confirm_end = None
    cached_tf_confirm = None
    cached_filter_end = None
    cached_tf_filter = None

    for i in range(MIN_CANDLES_NEEDED, n):
        t = candles_5m.index[i]
        primary_slice = candles_5m.iloc[max(0, i - primary_window_bars):i + 1]
        confirm_slice = _closed_bucket_candles(candles_confirm_full, t, confirm_minutes, window=confirm_window_bars, primary_step_minutes=primary_minutes)
        filter_slice = None if skip_filter_timeframe else _closed_bucket_candles(candles_filter_full, t, filter_minutes, window=filter_window_bars, primary_step_minutes=primary_minutes)

        filter_len_ok = True if skip_filter_timeframe else len(filter_slice) >= MIN_CANDLES_NEEDED
        if len(primary_slice) < MIN_CANDLES_NEEDED or len(confirm_slice) < MIN_CANDLES_NEEDED or not filter_len_ok:
            equity_curve.append((t, equity))
            continue

        current_price = primary_slice["close"].iloc[-1]

        tf_primary = build_timeframe_indicators(primary_slice)

        confirm_end = confirm_slice.index[-1]
        if confirm_end != cached_confirm_end:
            cached_tf_confirm = build_timeframe_indicators(confirm_slice)
            cached_confirm_end = confirm_end
        tf_confirm = cached_tf_confirm

        if skip_filter_timeframe:
            # Real speed test, not an approximation - the 1h slice/
            # indicators are never built at all in this mode, not just
            # cached. tf_confirm stands in wherever tf_filter would be
            # used downstream.
            tf_filter = tf_confirm
        else:
            filter_end = filter_slice.index[-1]
            if filter_end != cached_filter_end:
                cached_tf_filter = build_timeframe_indicators(filter_slice)
                cached_filter_end = filter_end
            tf_filter = cached_tf_filter

        if open_position is not None:
            pos = open_position
            direction = pos["direction"]

            reversal = calculate_reversal_score(tf_primary, tf_confirm, tf_filter, direction, trend_history,
                                                 use_adverse_drift=use_adverse_drift, drift_net_threshold=drift_net_threshold)

            # Idea #4: OBV/CVD confirmation bonus. Prefers REAL CVD
            # (genuine taker buy/sell split, e.g. from Binance via
            # binance_taker_volume_fetcher.py, merged onto candles as
            # taker_buy_volume/taker_sell_volume columns in main.py) when
            # those columns are present on primary_slice. Falls back to
            # the OBV proxy (see indicators.py for the honest caveat on
            # what OBV can't distinguish) when real data isn't available -
            # e.g. Binance blocked this run, or real-CVD wasn't requested.
            # obv_confirmation_bonus=0 (default) means this never changes
            # anything vs the existing score either way.
            if obv_confirmation_bonus:
                cvd_signal = detect_real_cvd_divergence(primary_slice, direction, lookback=obv_lookback_bars, slope_threshold=obv_slope_threshold)
                if cvd_signal is not None:
                    cvd_source_used = "real_cvd"
                    pos["used_real_cvd"] = True
                    signal_fired = cvd_signal["cvd_slope_adverse"]
                    slope_norm = cvd_signal["cvd_slope_norm"]
                else:
                    cvd_source_used = "obv_proxy"
                    obv_signal = detect_obv_price_divergence(primary_slice, direction, lookback=obv_lookback_bars, slope_threshold=obv_slope_threshold)
                    signal_fired = obv_signal["obv_slope_adverse"]
                    slope_norm = obv_signal["obv_slope_norm"]
                if signal_fired:
                    reversal["reversal_score"] += obv_confirmation_bonus
                    reversal["details"].append(f"{cvd_source_used} confirms adverse flow (slope_norm {slope_norm})")

            # Idea #6: BTC trend confirmation bonus. Uses BTC's own OHLCV,
            # merged onto candles_5m as btc_open/btc_high/btc_low/btc_close/
            # btc_volume columns by main.py (same join pattern already
            # proven for real CVD) - reconstructs a candles-shaped slice on
            # the fly here rather than threading a second DataFrame through
            # every function signature.
            if use_btc_trend_bonus and btc_trend_bonus and "btc_close" in primary_slice.columns:
                btc_slice = primary_slice[["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]].rename(
                    columns={"btc_open": "open", "btc_high": "high", "btc_low": "low", "btc_close": "close", "btc_volume": "volume"}
                ).dropna()
                if len(btc_slice) >= 55:  # MIN_CANDLES_NEEDED
                    btc_signal = detect_btc_trend_adverse(btc_slice, direction, min_score_magnitude=btc_min_score_magnitude)
                    if btc_signal["btc_adverse"]:
                        reversal["reversal_score"] += btc_trend_bonus
                        reversal["details"].append(f"BTC trend confirms adverse move (btc_score={btc_signal['btc_trend_score']})")

            # Idea #3: ATR stop compression on adverse drift. Fires
            # independently of the reversal_score exit gate below - a
            # drift-only signal (max +36 points, never reaches the
            # default 70 exit threshold on its own, see the Aug 2
            # investigation) still tightens the stop instead of doing
            # nothing, WITHOUT forcing an exit. Only ever tightens
            # (moves the stop toward current price), never loosens -
            # same one-directional guarantee as the existing trailing-
            # stop logic elsewhere in this function.
            if drift_stop_tighten_enabled and use_adverse_drift:
                drift_fired = any("drift" in f for f in reversal["reversed_frames"])
                if drift_fired:
                    fresh_atr = atr_wilder(confirm_slice, 14)
                    tighten_distance = fresh_atr * drift_stop_tighten_atr_multiplier
                    dir_sign = 1 if direction == "long" else -1
                    candidate_stop = current_price - dir_sign * tighten_distance
                    if direction == "long":
                        new_stop = max(pos["stop"], candidate_stop)
                    else:
                        new_stop = min(pos["stop"], candidate_stop)
                    if new_stop != pos["stop"]:
                        pos["stop"] = new_stop
                        pos["drift_stop_tightened"] = True

            hit_stop = current_price <= pos["stop"] if direction == "long" else current_price >= pos["stop"]

            # Idea #2: time-decay dynamic reversal threshold. Keep the
            # proven default (70) for fresh trades - only tighten the
            # gate once a trade is BOTH old enough AND already
            # underwater, so normal entry noise on a fresh position never
            # gets cut. current_r_for_gate uses the position's CURRENT
            # stop (not initial_stop) deliberately - matches what the
            # trade is actually risking right now, same convention
            # calculate_r_multiple uses elsewhere in this function.
            effective_reversal_threshold = reversal_exit_threshold
            dynamic_threshold_active = False
            if dynamic_threshold_enabled:
                time_in_trade_minutes = (i - pos["entry_index"]) * primary_minutes
                current_r_for_gate = calculate_r_multiple(pos["entry"], current_price, pos["stop"], direction)
                if time_in_trade_minutes > dynamic_threshold_after_minutes and current_r_for_gate < dynamic_threshold_drawdown_r:
                    effective_reversal_threshold = dynamic_threshold_tightened
                    dynamic_threshold_active = True

            # Raw-continuous check: uses the underlying primary trend
            # number directly (e.g. -8, -23, -41) instead of the bucketed
            # reversal_score (12/20/25/40...), which jumps in fixed steps
            # and can't distinguish "just starting to turn" from "well
            # advanced" within the same bucket - exactly why 55/60/65
            # produced identical results earlier. target_sign matches
            # calculate_reversal_score's own convention (long positions
            # watch for the score turning negative, short positions watch
            # for it turning positive).
            raw_primary = reversal["trend_scores"]["primary"]
            target_sign = -1 if direction == "long" else 1
            raw_reversal_triggered = (
                raw_reversal_threshold is not None
                and np.sign(raw_primary) == target_sign
                and abs(raw_primary) >= raw_reversal_threshold
            )

            exit_reason = None
            exit_price = current_price

            # Idea #5: mechanical soft-exit proxy (TRIM_50%/TIGHTEN_STOP).
            # EXPLICITLY a rule-based STAND-IN for what Gemini's live
            # judgment might choose - not a simulation of the LLM itself
            # (that reasoning step doesn't exist in this backtest, same
            # limitation documented for the original drift fix). Trims a
            # fixed fraction the FIRST time score crosses
            # soft_exit_trim_threshold (below the hard exit gate), and
            # tightens the stop the first time it crosses
            # soft_exit_tighten_threshold - both independent of, and
            # evaluated BEFORE, the hard reversal/raw-reversal/stop exit
            # checks below, matching "give it a middle option before the
            # binary HOLD/CLOSE decision" from the idea.
            if soft_exit_enabled and not hit_stop:
                if (not pos.get("soft_tighten_done") and
                        reversal["reversal_score"] >= soft_exit_tighten_threshold and
                        reversal["reversal_score"] < effective_reversal_threshold):
                    fresh_atr = atr_wilder(confirm_slice, 14)
                    tighten_distance = fresh_atr * soft_exit_tighten_atr_multiplier
                    dir_sign = 1 if direction == "long" else -1
                    candidate_stop = current_price - dir_sign * tighten_distance
                    new_stop = max(pos["stop"], candidate_stop) if direction == "long" else min(pos["stop"], candidate_stop)
                    if new_stop != pos["stop"]:
                        pos["stop"] = new_stop
                    pos["soft_tighten_done"] = True

                if (not pos.get("soft_trim_done") and
                        reversal["reversal_score"] >= soft_exit_trim_threshold and
                        reversal["reversal_score"] < effective_reversal_threshold):
                    # Realize soft_exit_trim_fraction of the position NOW
                    # at current_price, exactly like the existing staged-
                    # exit bookkeeping (adds to realized_r, reduces what
                    # remaining_fraction will book at final exit).
                    trim_r = calculate_r_multiple(pos["entry"], current_price, pos["initial_stop"], direction)
                    pos["realized_r"] += trim_r * soft_exit_trim_fraction
                    pos["soft_trim_fraction_taken"] = pos.get("soft_trim_fraction_taken", 0.0) + soft_exit_trim_fraction
                    pos["soft_trim_done"] = True

            if reversal["reversal_score"] >= effective_reversal_threshold:
                exit_reason = "reversal_dynamic_threshold" if dynamic_threshold_active else "reversal"
            elif raw_reversal_triggered:
                # This is the exit we're actually testing: did it fire
                # BEFORE the stop would have, i.e. did it genuinely save
                # something, or is it just closing early on a move the
                # stop was about to catch anyway? hit_stop is already
                # computed above for this exact bar, so this is a direct,
                # honest same-bar comparison, not an estimate.
                exit_reason = "raw_reversal_beat_stop" if not hit_stop else "raw_reversal_same_bar_as_stop"
            elif hit_stop:
                exit_reason = "stop"
                exit_price = pos["stop"]
            elif max_hold_bars is not None and (i - pos["entry_index"]) >= max_hold_bars:
                exit_reason = "max_hold_time"
            else:
                vol = analyze_market_volatility(confirm_slice)
                current_r = calculate_r_multiple(pos["entry"], current_price, pos["stop"], direction)
                s1, s2, s3 = stage_multipliers
                adj_r = {"1": s1 * vol["adjustment_factor"], "2": s2 * vol["adjustment_factor"], "3": s3 * vol["adjustment_factor"]}
                if full_close_at_stage1 and current_r >= adj_r["1"] and pos["stages_done"] == 0:
                    # Close the ENTIRE position here - no partial, no
                    # trailing continuation. stages_done stays 0 so the
                    # remaining_fraction math below correctly treats this
                    # as a 100%-of-position exit at this R level.
                    exit_reason = "target_1r_full_close"
                elif asymmetric_free_ride:
                    if pos["stages_done"] == 0 and current_r >= adj_r["1"]:
                        # Realize the LARGER stage1 fraction (not the
                        # normal 33.33%), move stop to breakeven exactly
                        # like the standard staged exit's stage 1 does,
                        # and switch the rest into trailing mode - no
                        # fixed 2R/3R target for the remainder at all.
                        pos["stages_done"] = 1
                        pos["realized_r"] += adj_r["1"] * stage1_close_fraction
                        pos["last_staged_r"] = adj_r["1"]
                        pos["stop"] = pos["entry"]
                        pos["trailing_active"] = True
                    elif pos.get("trailing_active"):
                        # Recompute the trailing stop from the CURRENT
                        # price using the CURRENT ATR (not the entry-time
                        # ATR) - only ever moves in the favorable
                        # direction, exactly like a real trailing stop.
                        # The actual exit still happens through the same
                        # hit_stop check already run earlier this bar
                        # against pos["stop"] - no separate exit path
                        # needed, this just updates where that stop sits.
                        fresh_atr = atr_wilder(confirm_slice, 14)
                        trail_distance = fresh_atr * trailing_atr_multiplier
                        dir_sign = 1 if direction == "long" else -1
                        candidate_stop = current_price - dir_sign * trail_distance
                        if direction == "long":
                            pos["stop"] = max(pos["stop"], candidate_stop)
                        else:
                            pos["stop"] = min(pos["stop"], candidate_stop)
                elif current_r >= adj_r["3"] and pos["stages_done"] == 2:
                    pos["stages_done"] = 3
                    pos["stop"] = calculate_target_price(pos["entry"], pos["stop"], s2, direction)
                elif current_r >= adj_r["2"] and pos["stages_done"] == 1:
                    pos["stages_done"] = 2
                    pos["realized_r"] += adj_r["2"] * 0.3333
                    pos["last_staged_r"] = adj_r["2"]
                    pos["stop"] = calculate_target_price(pos["entry"], pos["stop"], s1, direction)
                elif current_r >= adj_r["1"] and pos["stages_done"] == 0:
                    pos["stages_done"] = 1
                    pos["realized_r"] += adj_r["1"] * 0.3333
                    pos["last_staged_r"] = adj_r["1"]
                    pos["stop"] = pos["entry"]

            if exit_reason:
                final_r = calculate_r_multiple(pos["entry"], exit_price, pos["initial_stop"], direction)
                if asymmetric_free_ride:
                    # stages_done is 0 (never reached 1R - full position
                    # still open at exit) or 1 (stage1 fraction already
                    # realized, this is the remaining runner closing).
                    remaining_fraction = 1.0 if pos["stages_done"] == 0 else (1.0 - stage1_close_fraction)
                else:
                    remaining_fraction = 1.0 - (0.3333 * pos["stages_done"] if pos["stages_done"] < 3 else 0.6667)
                # Idea #5: subtract whatever fraction the soft-exit TRIM
                # already realized early, so it isn't double-counted at
                # final exit (that fraction's R was already banked into
                # realized_r at trim time, using the entry-time
                # initial_stop convention - same convention every other
                # realized_r addition in this function uses).
                remaining_fraction = max(0.0, remaining_fraction - pos.get("soft_trim_fraction_taken", 0.0))
                total_r = pos["realized_r"] + final_r * remaining_fraction
                trades.append({
                    "symbol": symbol, "strategy": strategy, "direction": direction,
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "entry_price": pos["entry"], "exit_price": exit_price, "r_achieved": total_r,
                    "exit_reason": exit_reason, "setup_type": pos["setup_type"],
                    "is_breakout_extension": pos.get("is_breakout_extension", False),
                    "bars_held": i - pos["entry_index"],
                    # ---- new fields, needed by fee_model.py ----
                    "stages_done": pos["stages_done"],
                    "leverage": pos["leverage"],
                    "stop_distance_pct": pos["stop_distance_pct"],
                    # ---- idea #2/#3/#5 diagnostic fields, all default-inert ----
                    "dynamic_threshold_active": dynamic_threshold_active,
                    "drift_stop_tightened": pos.get("drift_stop_tightened", False),
                    "soft_trim_done": pos.get("soft_trim_done", False),
                    "soft_tighten_done": pos.get("soft_tighten_done", False),
                    "used_real_cvd": pos.get("used_real_cvd", False),
                })
                equity *= (1 + total_r * 0.01)  # 1% risk per trade convention (unchanged)
                # Idea #5 (FREEZE_REENTRY proxy): only freeze re-entry in
                # the SAME direction after a reversal-driven exit (a plain
                # stop-out or clean target hit isn't the "trend actually
                # turned against me" signal this is meant to guard
                # against).
                if soft_exit_enabled and exit_reason in ("reversal", "reversal_dynamic_threshold"):
                    freeze_bars = max(1, round(soft_exit_freeze_minutes / primary_minutes))
                    freeze_until_index[direction] = i + freeze_bars
                open_position = None

        elif open_position is None:
            trend_strength = determine_trend_strength(tf_primary)
            momentum_state = determine_momentum_state(tf_confirm)
            market_state = determine_market_state(trend_strength, momentum_state, tf_confirm)
            market_state["atr_ratio"] = tf_filter["atr_ratio"]

            alignment_score = calculate_triple_timeframe_consistency(tf_primary, tf_confirm, tf_filter)
            strategy_result = route_strategy(symbol, market_state, tf_confirm, tf_filter)
            opp = score_opportunity(strategy_result, market_state, alignment_score, tf_filter["atr_ratio"],
                                     symbol, strategy=strategy, leverage=resolved_leverage)

            if strategy_result["action"] != "wait" and opp["total_score"] >= effective_min_score:
                direction = strategy_result["action"]
                setup_type = strategy_result["strategy_type"]
                # Idea #5 (FREEZE_REENTRY proxy): block a fresh entry in a
                # direction still under cooldown from a recent reversal-
                # driven exit. soft_exit_enabled=False (default) means
                # freeze_until_index never gets set above 0, so this
                # never blocks anything unless explicitly opted in.
                if soft_exit_enabled and i < freeze_until_index.get(direction, -1):
                    pass
                elif direction_filter is not None and direction != direction_filter:
                    pass
                elif setup_filter is not None and setup_type != setup_filter:
                    pass
                else:
                    can_open, sl_result = should_open_position(
                        tf_filter["candles"], direction, current_price,
                        atr_multiplier=atr_multiplier,
                        min_stop_pct=stop_cfg["min_distance"], max_stop_pct=stop_cfg["max_distance"],
                    )
                    if can_open and use_btc_stop_floor and "btc_close" in candles_5m.columns:
                        # NOTE: sourced from candles_5m directly, NOT tf_filter["candles"] -
                        # resample_candles() (used to build tf_filter) only knows about the
                        # fixed open/high/low/close/volume columns and silently drops anything
                        # else, including btc_*. Confirmed via a real before/after test: the
                        # floor never engaged even with deliberately high BTC volatility until
                        # this was switched to read from candles_5m instead.
                        btc_window = candles_5m[["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]].iloc[max(0, i - 60):i + 1].rename(
                            columns={"btc_open": "open", "btc_high": "high", "btc_low": "low", "btc_close": "close", "btc_volume": "volume"}
                        ).dropna()
                        if len(btc_window) >= 15:  # atr_wilder needs 14+1
                            own_distance = abs(current_price - sl_result["stop_price"])
                            widened_distance = calculate_stop_loss_with_btc_floor(
                                own_distance, btc_window, current_price, beta=btc_stop_beta
                            )
                            if widened_distance > own_distance:
                                sl_result = dict(sl_result)  # don't mutate the original dict from should_open_position
                                sl_result["stop_price"] = (current_price - widened_distance if direction == "long"
                                                            else current_price + widened_distance)
                                sl_result["stop_distance_pct"] = (widened_distance / current_price) * 100
                                sl_result["btc_floor_applied"] = True
                    if can_open:
                        stop_distance_pct = abs(current_price - sl_result["stop_price"]) / current_price
                        open_position = {
                            "direction": direction, "entry": current_price, "entry_time": t, "entry_index": i,
                            "stop": sl_result["stop_price"], "initial_stop": sl_result["stop_price"],
                            "stages_done": 0, "realized_r": 0.0, "last_staged_r": 0.0,
                            "setup_type": strategy_result["strategy_type"],
                            "is_breakout_extension": strategy_result.get("is_breakout_extension", False),
                            "leverage": resolved_leverage,
                            "stop_distance_pct": stop_distance_pct,
                            # ---- idea #3/#5 per-position state, all inert unless opted in ----
                            "drift_stop_tightened": False, "soft_trim_done": False,
                            "used_real_cvd": False,
                            "soft_tighten_done": False, "soft_trim_fraction_taken": 0.0,
                        }


        for key, tf in (("primary", tf_primary), ("confirm", tf_confirm), ("filter", tf_filter)):
            trend_history[key].append(calculate_trend_score(tf))
            if len(trend_history[key]) > 5:
                trend_history[key].pop(0)

        equity_curve.append((t, equity))

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(dict(equity_curve))
    return trades_df, equity_series


def summarize_results(trades_df):
    """Reports gross (r_achieved, pre-fee), net (net_r, post-fee), AND
    real-dollar numbers side by side, once fee_model.apply_fees_and_interest()
    and apply_dollar_pnl() have been run on trades_df. Dollar figures use
    the standing $500 capital / 5% fixed risk per trade convention
    (fee_model.DEFAULT_CAPITAL / DEFAULT_RISK_PCT) - added because R-
    multiples alone were hard to reason about in practical terms."""
    if len(trades_df) == 0:
        return {"total_trades": 0}
    has_net = "net_r" in trades_df.columns
    has_dollar = "dollar_pnl" in trades_df.columns
    wins_gross = trades_df[trades_df["r_achieved"] > 0]
    losses_gross = trades_df[trades_df["r_achieved"] <= 0]

    summary = {
        "total_trades": len(trades_df),
        "win_rate_pct_gross": round(len(wins_gross) / len(trades_df) * 100, 1),
        "avg_r_gross": round(trades_df["r_achieved"].mean(), 3),
        "total_r_gross": round(trades_df["r_achieved"].sum(), 2),
        "avg_bars_held": round(trades_df["bars_held"].mean(), 1),
        # Bug 5 fix: avg_bars_held alone is silently misleading when
        # comparing across different timeframe combos, since a "bar" now
        # means a different real duration depending on primary_minutes.
        # This needs the actual interval to convert correctly, so it's
        # computed by the caller (main.py) after this function returns -
        # see avg_hours_held added to the row dict there.
        "exit_reason_breakdown": trades_df["exit_reason"].value_counts().to_dict(),
        "breakout_extension_trades": int(trades_df["is_breakout_extension"].sum()),
    }
    # The key number the raw-reversal-threshold feature exists to answer:
    # of the trades that exited via the raw check, what fraction actually
    # beat the stop (a genuine save) vs just coincided with it (no real
    # benefit over waiting)?
    beat_stop = int((trades_df["exit_reason"] == "raw_reversal_beat_stop").sum())
    same_bar = int((trades_df["exit_reason"] == "raw_reversal_same_bar_as_stop").sum())
    if beat_stop + same_bar > 0:
        summary["raw_reversal_beat_stop_rate_pct"] = round(beat_stop / (beat_stop + same_bar) * 100, 1)
        summary["raw_reversal_beat_stop_count"] = beat_stop
        summary["raw_reversal_same_bar_count"] = same_bar
    if has_net:
        wins_net = trades_df[trades_df["net_r"] > 0]
        summary.update({
            "win_rate_pct_net": round(len(wins_net) / len(trades_df) * 100, 1),
            "avg_r_net": round(trades_df["net_r"].mean(), 3),
            "total_r_net": round(trades_df["net_r"].sum(), 2),
            "avg_fee_interest_r_cost": round(trades_df["fee_interest_r_cost"].mean(), 3),
        })
    if has_dollar:
        wins_d = trades_df[trades_df["dollar_pnl"] > 0]
        losses_d = trades_df[trades_df["dollar_pnl"] <= 0]
        summary.update({
            "total_dollar_pnl": round(trades_df["dollar_pnl"].sum(), 2),
            "avg_dollar_win": round(wins_d["dollar_pnl"].mean(), 2) if len(wins_d) else 0,
            "avg_dollar_loss": round(losses_d["dollar_pnl"].mean(), 2) if len(losses_d) else 0,
            "biggest_dollar_win": round(trades_df["dollar_pnl"].max(), 2),
            "biggest_dollar_loss": round(trades_df["dollar_pnl"].min(), 2),
            "final_dollar_running_total": round(trades_df["dollar_running_total"].iloc[-1], 2),
        })
    return summary
