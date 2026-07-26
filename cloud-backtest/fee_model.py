"""
Fee + margin-interest model for the backtest, added because the engine
had NEVER modeled either before - every prior result (gross R, Return %)
assumed completely free trading. Confirmed assumptions (from the person
directly, not guessed): CoinDCX Regular tier, market orders on every leg
(entry, each partial take-profit close, final exit) - so taker fee applies
every time, never maker.

Rates, from CoinDCX's own published fee schedule (Futures USDT-M):
- Taker fee: 0.05%, +18% GST = 0.059% effective, per leg.
- Margin interest: 0.0028%/hour on the BORROWED portion of a leveraged
  position, first hour free.

WHY THIS IS EXPRESSED IN R-MULTIPLES, NOT DOLLARS:
The engine tracks P&L purely as R-multiples (a price-ratio, independent of
account size, leverage, or position size) and only converts to an equity
curve at the very end via a fixed "risk this % of equity per trade"
convention. Fees/interest are dollar costs proportional to NOTIONAL,
which this engine doesn't track directly - but the conversion works out
cleanly:

  1R (in dollars) = position_notional x stop_distance_fraction
  fee (in dollars) = position_notional x fee_rate
  => fee, expressed in R units = fee_rate / stop_distance_fraction

This is independent of account size, leverage, and the risk-per-trade
convention entirely - it only depends on the fee rate and how tight the
stop is. A tighter stop (e.g. ultra-short's 0.3-2.0% range) means the same
flat % fee eats a LARGER share of 1R, since 1R itself is a smaller dollar
amount for the same notional. This is exactly why fee modeling was
expected to hurt high-frequency, tight-stop presets hardest.

Margin interest follows similarly, additionally scaled by (leverage-1)/
leverage (only the borrowed portion accrues interest) and by hours held.
SIMPLIFICATION (clearly flagged, not hidden): interest is computed against
the FULL initial notional for the entire hold duration, even though real
notional shrinks after each partial close. This slightly overstates
interest cost - a conservative direction, and a small one, since interest
is a much smaller effect than fees for any trade under a few days.
"""

TAKER_FEE_RATE = 0.0005          # 0.05%, CoinDCX Regular tier, Futures USDT-M
GST_RATE = 0.18                  # 18% GST on top of the fee itself
EFFECTIVE_TAKER_FEE = TAKER_FEE_RATE * (1 + GST_RATE)   # 0.00059 (0.059%)

MARGIN_INTEREST_HOURLY = 0.000028   # 0.0028%/hour
FREE_HOURS = 1                       # first hour free, per CoinDCX's fee page


def fee_and_interest_r_cost(stages_done, leverage, stop_distance_pct, bars_held, bar_minutes=5):
    """Returns the total R-multiple cost (fees + margin interest) for one
    trade, given:
    - stages_done: 0-3, how many of the 3 take-profit stages were reached
      (each reached stage = one additional partial-close fee leg, since
      stages 1 and 2 both realize 33.33% of the position; stage 3 doesn't
      close anything additional per the real deployed logic).
    - leverage: the leverage used on this trade.
    - stop_distance_pct: the ORIGINAL stop distance as a fraction of entry
      price (e.g. 0.02 for a 2% stop) - this is the fixed conversion factor
      between dollar amounts and R-multiples for this specific trade,
      matching how the engine already normalizes r_achieved to the
      initial stop, not the trailing one.
    - bars_held: number of 5m bars the position was open.
    """
    if stop_distance_pct <= 0:
        return 0.0

    # Fee legs: 1 entry + up to 2 partial-closes (stage 1, stage 2) + 1 final exit.
    # Stage 3 never adds a NEW close leg (closePercent=0 in the real logic).
    fee_legs = 1 + min(stages_done, 2) + 1
    fee_r_cost = fee_legs * (EFFECTIVE_TAKER_FEE / stop_distance_pct)

    hours_held = bars_held * bar_minutes / 60.0
    billable_hours = max(0.0, hours_held - FREE_HOURS)
    borrowed_fraction = (leverage - 1) / leverage if leverage > 0 else 0.0
    interest_r_cost = borrowed_fraction * MARGIN_INTEREST_HOURLY * billable_hours / stop_distance_pct

    return fee_r_cost + interest_r_cost


def apply_fees_and_interest(trades_df):
    """Adds 'fee_interest_r_cost' and 'net_r' columns to a trades
    DataFrame produced by run_backtest(). Requires the trade dict to
    include 'stages_done', 'leverage', and 'stop_distance_pct' - all three
    are now captured by the updated backtest engine specifically so this
    can run as a separate, clearly-labeled post-processing step rather
    than being baked invisibly into the core simulation loop."""
    if len(trades_df) == 0:
        trades_df["fee_interest_r_cost"] = []
        trades_df["net_r"] = []
        return trades_df

    trades_df = trades_df.copy()
    trades_df["fee_interest_r_cost"] = trades_df.apply(
        lambda row: fee_and_interest_r_cost(
            row["stages_done"], row["leverage"], row["stop_distance_pct"], row["bars_held"]
        ),
        axis=1,
    )
    trades_df["net_r"] = trades_df["r_achieved"] - trades_df["fee_interest_r_cost"]
    return trades_df
