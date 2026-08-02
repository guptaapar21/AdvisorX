"""
Idea #1: Portfolio-level "Basket Drift" Circuit Breaker.

HONEST SCOPE NOTE, read before using: this is a POST-HOC OVERLAY, not a
full joint re-simulation. run_backtest() (backtest_engine.py) simulates
one coin at a time and has no concept of "what's happening on the other
two coins right now." Rebuilding it into a single shared per-bar stepper
across coins would be the correct long-term fix, but is a much larger
refactor. This module instead:

  1. Runs each coin's OWN existing run_backtest() independently (unchanged).
  2. Replays each coin's own price candles bar-by-bar to reconstruct
     UNREALIZED R for every open position at every moment (using the
     trade's real entry/stop/direction - this part is exact, not
     approximated).
  3. Whenever >= min_correlated_positions coins are open in the SAME
     direction at the same moment, sums their unrealized R into a
     combined basket R and tracks how long it's been negative.
  4. Once combined R has been negative for >= basket_negative_minutes
     straight, force-closes the WEAKEST open position (worst unrealized R)
     at that bar's price - replacing that trade's real outcome with the
     early-exit outcome.

APPROXIMATION FLAGGED: step 4's early close ignores whatever staged
partial-exit fractions (1R/2R/3R) that trade had already realized before
the cutoff - it re-derives R from entry/stop/exit-price directly, same
convention as run_backtest's own final_r calculation, but does not carry
forward pos["realized_r"] bookkeeping mid-trade (that state lives inside
run_backtest's closed loop and isn't exposed after the fact). For trades
NOT hit by the circuit breaker, the original result is used unchanged.
Good enough to answer "does this rule help or hurt overall", not
precise enough to treat as the exact same accounting as the base engine.

Usage:
    from backtest_engine import run_backtest
    from basket_circuit_breaker import apply_basket_circuit_breaker

    per_coin_trades = {}
    per_coin_candles = {}
    for coin, candles in coin_candles.items():
        trades, _ = run_backtest(coin, candles, strategy="conservative", ...)
        per_coin_trades[coin] = trades
        per_coin_candles[coin] = candles

    adjusted_trades, breaker_events = apply_basket_circuit_breaker(
        per_coin_trades, per_coin_candles,
        min_correlated_positions=2, basket_negative_minutes=45,
    )
"""
import pandas as pd
import numpy as np

from scoring_stoploss import calculate_r_multiple


def _build_position_timeline(trades_df, candles, primary_minutes=5):
    """For one coin's trades_df, returns a per-bar DataFrame (indexed by
    the coin's own candle timestamps) with columns: direction (None if
    flat), unrealized_r (0 if flat), trade_idx (row index into trades_df,
    -1 if flat). Only covers the span trades_df actually spans - bars
    with no open position anywhere in the whole backtest are direction=None.
    """
    idx = candles.index
    direction_col = pd.Series(None, index=idx, dtype=object)
    unrealized_r_col = pd.Series(0.0, index=idx)
    trade_idx_col = pd.Series(-1, index=idx)

    for row_i, trade in trades_df.iterrows():
        mask = (idx >= trade["entry_time"]) & (idx <= trade["exit_time"])
        if not mask.any():
            continue
        entry = trade["entry_price"]
        stop = trade.get("initial_stop", None)
        if stop is None or pd.isna(stop):
            # initial_stop isn't always carried in trades_df depending on
            # caller - fall back to a synthetic stop 1 stop_distance_pct
            # away, which is exact for R purposes since calculate_r_multiple
            # only uses (entry - stop) as the risk unit.
            stop_dist = trade.get("stop_distance_pct", 0.01) * entry
            stop = entry - stop_dist if trade["direction"] == "long" else entry + stop_dist

        direction_col.loc[mask] = trade["direction"]
        trade_idx_col.loc[mask] = row_i
        closes_in_window = candles.loc[mask, "close"]
        unrealized_r_col.loc[mask] = closes_in_window.apply(
            lambda px: calculate_r_multiple(entry, px, stop, trade["direction"])
        )

    return pd.DataFrame({
        "direction": direction_col,
        "unrealized_r": unrealized_r_col,
        "trade_idx": trade_idx_col,
    })


def apply_basket_circuit_breaker(per_coin_trades, per_coin_candles, primary_minutes=5,
                                  min_correlated_positions=2, basket_negative_minutes=45,
                                  action="close_weakest"):
    """
    per_coin_trades: {coin: trades_df} - output of run_backtest per coin,
        must include entry_time/exit_time/entry_price/direction/
        stop_distance_pct (initial_stop optional, reconstructed if absent).
    per_coin_candles: {coin: candles_5m DataFrame} - the SAME raw candles
        passed into run_backtest for that coin (needed to replay price
        path for unrealized R).
    action: only "close_weakest" implemented - closes the single worst
        (most negative unrealized R) position in the triggering basket.

    Returns (adjusted_trades: {coin: trades_df}, breaker_events: list of
    dicts logging every time the breaker fired - timestamp, coins
    involved, direction, combined_r, which coin got closed).
    """
    if action != "close_weakest":
        raise NotImplementedError(f"action={action!r} not implemented - only 'close_weakest'.")

    coins = list(per_coin_trades.keys())
    timelines = {
        coin: _build_position_timeline(per_coin_trades[coin], per_coin_candles[coin], primary_minutes)
        for coin in coins
    }

    # Align all coins onto a shared timestamp index (union, forward-filled
    # is NOT used here deliberately - a bar with no timeline entry for a
    # coin just means that coin had no candle at that exact timestamp,
    # which reindex+left-join naturally treats as "not open" rather than
    # smearing a stale unrealized_r forward).
    common_index = sorted(set().union(*[set(t.index) for t in timelines.values()]))

    breaker_events = []
    # Tracks, per direction, how many consecutive bars the combined R for
    # the currently-open correlated basket has been negative, and which
    # coins are in that basket (baskets can change membership bar to bar
    # as positions open/close, so this resets whenever membership shrinks
    # below min_correlated_positions).
    negative_streak_bars = {"long": 0, "short": 0}
    forced_close_bar = {coin: None for coin in coins}  # coin -> timestamp once force-closed

    bars_needed = max(1, round(basket_negative_minutes / primary_minutes))

    for t in common_index:
        for direction in ("long", "short"):
            open_coins = []
            for coin in coins:
                if forced_close_bar[coin] is not None and t >= forced_close_bar[coin]:
                    continue  # already forced closed earlier - not part of any further basket
                tl = timelines[coin]
                if t in tl.index and tl.loc[t, "direction"] == direction:
                    open_coins.append(coin)

            if len(open_coins) < min_correlated_positions:
                negative_streak_bars[direction] = 0
                continue

            combined_r = sum(timelines[c].loc[t, "unrealized_r"] for c in open_coins)
            if combined_r < 0:
                negative_streak_bars[direction] += 1
            else:
                negative_streak_bars[direction] = 0

            if negative_streak_bars[direction] >= bars_needed:
                # Close the single WEAKEST position in this basket right now.
                weakest_coin = min(open_coins, key=lambda c: timelines[c].loc[t, "unrealized_r"])
                natural_exit_time = per_coin_trades[weakest_coin].loc[
                    timelines[weakest_coin].loc[t, "trade_idx"], "exit_time"
                ]
                # Same distinction the codebase already makes for
                # raw_reversal_beat_stop vs raw_reversal_same_bar_as_stop:
                # if the trade's own natural exit was going to happen on
                # this EXACT bar anyway, the breaker didn't actually save
                # or change anything - it's a no-op, not a real early close.
                is_real_early_close = t < natural_exit_time
                forced_close_bar[weakest_coin] = t
                breaker_events.append({
                    "timestamp": t, "direction": direction, "coins_in_basket": list(open_coins),
                    "combined_r": round(float(combined_r), 3), "closed_coin": weakest_coin,
                    "closed_unrealized_r": round(float(timelines[weakest_coin].loc[t, "unrealized_r"]), 3),
                    "real_early_close": bool(is_real_early_close),
                })
                negative_streak_bars[direction] = 0  # basket relieved, reset

    # Apply the forced closes: for every trade whose timeline shows a
    # forced_close_bar timestamp inside its own [entry_time, exit_time]
    # window, replace its exit with the earlier forced-close bar/price.
    adjusted_trades = {}
    for coin in coins:
        trades_df = per_coin_trades[coin].copy()
        if forced_close_bar[coin] is None or trades_df.empty:
            adjusted_trades[coin] = trades_df
            continue
        cutoff = forced_close_bar[coin]
        candles = per_coin_candles[coin]

        hit_mask = (trades_df["entry_time"] <= cutoff) & (trades_df["exit_time"] > cutoff)
        for row_i in trades_df[hit_mask].index:
            trade = trades_df.loc[row_i]
            entry = trade["entry_price"]
            direction = trade["direction"]
            stop_dist = trade.get("stop_distance_pct", 0.01) * entry
            stop = entry - stop_dist if direction == "long" else entry + stop_dist
            if cutoff not in candles.index:
                continue  # shouldn't happen given common_index construction, but stay safe
            exit_price = candles.loc[cutoff, "close"]
            new_r = calculate_r_multiple(entry, exit_price, stop, direction)
            trades_df.loc[row_i, "exit_time"] = cutoff
            trades_df.loc[row_i, "exit_price"] = exit_price
            trades_df.loc[row_i, "r_achieved"] = new_r  # APPROXIMATION: see module docstring
            trades_df.loc[row_i, "exit_reason"] = "basket_circuit_breaker"
        adjusted_trades[coin] = trades_df

    return adjusted_trades, breaker_events
