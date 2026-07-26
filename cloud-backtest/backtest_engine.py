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

from indicators import ema, rsi, macd, macd_histogram_turn, atr_ratio, atr_wilder
from market_state_analyzer import (
    build_timeframe_indicators, determine_trend_strength, determine_momentum_state,
    determine_market_state, calculate_triple_timeframe_consistency, calculate_trend_score,
    calculate_reversal_score,
)
from strategy_logic import route_strategy
from scoring_stoploss import (
    score_opportunity, should_open_position, analyze_market_volatility,
    calculate_r_multiple, calculate_target_price,
    STRATEGY_SCORE_WEIGHTS, STRATEGY_STOP_LOSS, STRATEGY_LEVERAGE_BOUNDS,
)
from coindcx_fetcher import resample_candles

MIN_CANDLES_NEEDED = 55  # matches the live bot's own minimum


def _closed_bucket_candles(resampled, as_of_time, rule_minutes, window=200):
    """Returns the last `window` fully-closed buckets as of `as_of_time` -
    bounded rolling window (fixes the O(n^2) unbounded-slice bug that
    likely caused Colab hangs on full-year runs)."""
    sliced = resampled.loc[:as_of_time].tail(window)
    if len(sliced) == 0:
        return sliced
    last_bucket_start = sliced.index[-1]
    bucket_end = last_bucket_start + pd.Timedelta(minutes=rule_minutes)
    if bucket_end > as_of_time + pd.Timedelta(minutes=5):
        return sliced.iloc[:-1]
    return sliced


def run_backtest(symbol, candles_5m, strategy="balanced", min_score=None, max_positions=1,
                  max_hold_bars=None, stage_multipliers=(1, 2, 3),
                  direction_filter=None, setup_filter=None, verbose=False,
                  full_close_at_stage1=False, atr_multiplier_override=None):
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

    atr_multiplier_override: if set, overrides the preset's own default
    ATR stop-loss multiplier (e.g. testing conservative/swing-trend at a
    tighter 2.0x/2.25x instead of their default 2.5x, while keeping every
    other preset parameter - min_score, leverage, position size -
    unchanged). None = use the preset's real deployed value.

    Returns (trades_df, equity_curve_series). trades_df includes
    stages_done/leverage/stop_distance_pct for fee_model.py to consume.
    """
    weights = STRATEGY_SCORE_WEIGHTS[strategy]
    stop_cfg = STRATEGY_STOP_LOSS[strategy]
    atr_multiplier = atr_multiplier_override if atr_multiplier_override is not None else stop_cfg["atr_multiplier"]
    lev_bounds = STRATEGY_LEVERAGE_BOUNDS[strategy]
    resolved_leverage = (lev_bounds["min"] + lev_bounds["max"]) / 2
    effective_min_score = min_score if min_score is not None else weights["min_score"]

    candles_15m_full = resample_candles(candles_5m, "15m")
    candles_1h_full = resample_candles(candles_5m, "1h")

    trades = []
    open_position = None
    trend_history = {"primary": [], "confirm": [], "filter": []}
    equity = 1.0
    equity_curve = []

    n = len(candles_5m)
    for i in range(MIN_CANDLES_NEEDED, n):
        t = candles_5m.index[i]
        primary_slice = candles_5m.iloc[max(0, i - 200):i + 1]
        confirm_slice = _closed_bucket_candles(candles_15m_full, t, 15)
        filter_slice = _closed_bucket_candles(candles_1h_full, t, 60)

        if len(primary_slice) < MIN_CANDLES_NEEDED or len(confirm_slice) < MIN_CANDLES_NEEDED or len(filter_slice) < MIN_CANDLES_NEEDED:
            equity_curve.append((t, equity))
            continue

        current_price = primary_slice["close"].iloc[-1]

        tf_primary = build_timeframe_indicators(primary_slice)
        tf_confirm = build_timeframe_indicators(confirm_slice)
        tf_filter = build_timeframe_indicators(filter_slice)

        if open_position is not None:
            pos = open_position
            direction = pos["direction"]

            reversal = calculate_reversal_score(tf_primary, tf_confirm, tf_filter, direction, trend_history)
            hit_stop = current_price <= pos["stop"] if direction == "long" else current_price >= pos["stop"]

            exit_reason = None
            exit_price = current_price
            if reversal["reversal_score"] >= 70:
                exit_reason = "reversal"
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
                remaining_fraction = 1.0 - (0.3333 * pos["stages_done"] if pos["stages_done"] < 3 else 0.6667)
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
                })
                equity *= (1 + total_r * 0.01)  # 1% risk per trade convention (unchanged)
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
                if direction_filter is not None and direction != direction_filter:
                    pass
                elif setup_filter is not None and setup_type != setup_filter:
                    pass
                else:
                    can_open, sl_result = should_open_position(
                        tf_filter["candles"], direction, current_price,
                        atr_multiplier=atr_multiplier,
                        min_stop_pct=stop_cfg["min_distance"], max_stop_pct=stop_cfg["max_distance"],
                    )
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
        "exit_reason_breakdown": trades_df["exit_reason"].value_counts().to_dict(),
        "breakout_extension_trades": int(trades_df["is_breakout_extension"].sum()),
    }
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
