"""
Idea #10: "5m Momentum Scalp" - a fully SEPARATE entry path, not a
modification to the existing bot. The existing system's route_strategy()
generates entries from tf15m/tf1h ONLY (confirmed by reading the actual
code) - the 5m primary chart barely participates in entry decisions at
all. This module is the opposite: entry and stop/target sizing are both
computed purely from the 5m chart itself.

Three required confirmations (all must agree - deliberately strict):
  1. EMA(5) vs EMA(13) crossed in the trade direction AND the gap between
     them has been widening for the last 3 candles (filters out flat,
     barely-crossed chop).
  2. Price structure: last 3 closed 5m candles show higher-highs AND
     higher-lows (long) or lower-highs AND lower-lows (short) - a
     mechanical version of "a human would call this a visible trend."
  3. OBV sloping the same direction as price over the same lookback -
     same proxy already validated for ETH's OBV bonus (real aggressor-
     side data unavailable from CoinDCX, same caveat applies here).

MACD and RSI are OPTIONAL EXTRA confirmations, off by default - only
added if they demonstrably improve results, per the instruction to use
them "only if required." Tested as an ablation: core-3 alone vs core-3
plus each optional filter vs core-3 plus both.

ZERO LOOKAHEAD, BY CONSTRUCTION: every function here takes a candles
DataFrame that the CALLER has already sliced to end at the current bar
(candles.iloc[:i+1]) - never a full-length DataFrame, never anything
past "now". Kept in its own file specifically so this is trivial to
audit independently of the main engine (already audited once this
session) - a bug here can't hide inside a bigger, more complex function.
"""
import numpy as np

from indicators import ema, rsi, macd_series, atr_wilder, avg_volume, detect_volume_spike

def detect_consolidation_breakout(candles, lookback=10, range_atr_ratio=1.5, volume_spike_threshold=1.5):
    """Idea #11: genuinely different mechanism from idea #10's momentum
    scalp - that one required a trend already confirmed three lagging
    ways (EMA cross+widening, 3 higher-highs/lows, OBV slope), which
    testing showed fires only AFTER the move is essentially over
    (gross expected R near zero at every setting tested). This instead
    triggers AT THE MOMENT price escapes a tight range - the signal
    exists before the move develops, not after.

    Two conditions, both on data already available in the caller-bounded
    `candles` window:
      1. CONSOLIDATION: the `lookback` candles BEFORE the current one
         have a tight high-low range relative to volatility - range
         under `range_atr_ratio` x ATR(14). This is checked on the prior
         candles only, excluding the current (possibly-breaking) one.
      2. BREAKOUT: the CURRENT candle's close is outside that prior
         range, confirmed by a real volume spike (reusing the existing
         detect_volume_spike/avg_volume functions - previously flagged
         as missing entirely from breakout-style signals in this
         project's Python port).

    Returns {"action": "long"/"short"/"wait", "range_high": ..., "range_low": ...}
    - the range bounds are returned so the caller can size the stop
      structurally (opposite side of the range) instead of pure ATR,
      a genuinely different risk model from idea #10 too.
    """
    if len(candles) < lookback + 15:  # +15 margin for atr_wilder/avg_volume's own needs
        return {"action": "wait"}

    prior = candles.iloc[-(lookback + 1):-1]   # the lookback candles BEFORE the current one
    current = candles.iloc[-1]

    range_high = prior["high"].max()
    range_low = prior["low"].min()
    range_width = range_high - range_low

    atr14 = atr_wilder(candles.iloc[:-1], 14)  # ATR computed on prior candles only, not including
                                                 # the current (possibly-breaking) one - the whole
                                                 # point is measuring how tight things WERE, not
                                                 # letting today's breakout candle inflate its own
                                                 # volatility comparison.
    if atr14 == 0:
        return {"action": "wait"}

    is_consolidating = range_width < (range_atr_ratio * atr14)
    if not is_consolidating:
        return {"action": "wait"}

    avg_vol = avg_volume(prior, period=lookback)
    vol_signal = detect_volume_spike(current["volume"], avg_vol, threshold=volume_spike_threshold)
    if not vol_signal["is_spike"]:
        return {"action": "wait"}

    if current["close"] > range_high:
        return {"action": "long", "range_high": range_high, "range_low": range_low}
    elif current["close"] < range_low:
        return {"action": "short", "range_high": range_high, "range_low": range_low}
    return {"action": "wait"}


def calculate_breakout_stop_target(direction, entry_price, range_high, range_low, r_target=1.5):
    """Structural stop: the OPPOSITE side of the consolidation range that
    just broke - not an ATR multiple. If the breakout is real, price
    shouldn't come back through the range it just escaped; if it does,
    that's a clean, structurally-motivated invalidation, not an
    arbitrary volatility-based distance."""
    if direction == "long":
        stop_price = range_low
        stop_distance = entry_price - stop_price
        target_price = entry_price + stop_distance * r_target
    else:
        stop_price = range_high
        stop_distance = stop_price - entry_price
        target_price = entry_price - stop_distance * r_target
    return {"stop_price": stop_price, "target_price": target_price, "stop_distance": stop_distance}




def detect_ema_slope_widening(candles, direction, fast=5, slow=13, lookback=3):
    """EMA(fast) vs EMA(slow) crossed in `direction`, AND the gap between
    them is bigger NOW than it was `lookback` candles ago - a net
    comparison, not a strict "every single candle must widen" check.
    That stricter version was tested directly against a deliberate,
    steady linear uptrend and NEVER fired - the EMA gap converges to a
    constant value in steady-state (confirmed: gap stayed flat at 1.2
    for 6 straight candles on a perfectly linear trend), so requiring
    monotonic widening every step would make this condition almost
    impossible to satisfy on any real, sustained trend. A net comparison
    correctly captures "this trend has strengthened over the window"
    without demanding something even an ideal trend doesn't do."""
    closes = candles["close"].values
    if len(closes) < slow + lookback:
        return False
    ema_fast_series = ema(closes, fast)
    ema_slow_series = ema(closes, slow)
    offset = len(ema_fast_series) - len(ema_slow_series)
    ema_fast_aligned = ema_fast_series[offset:]
    if len(ema_fast_aligned) < lookback + 1:
        return False
    gap = ema_fast_aligned - ema_slow_series
    recent_gap = gap[-(lookback + 1):]

    if direction == "long":
        crossed = recent_gap[-1] > 0
        widening = recent_gap[-1] > recent_gap[0]
    else:
        crossed = recent_gap[-1] < 0
        widening = recent_gap[-1] < recent_gap[0]
    return bool(crossed and widening)


def detect_price_structure(candles, direction, lookback=3):
    """Last `lookback` closed 5m candles show higher-highs AND
    higher-lows (long) or lower-highs AND lower-lows (short). Uses only
    candles already in the passed-in (caller-bounded) DataFrame."""
    if len(candles) < lookback:
        return False
    recent = candles.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    if direction == "long":
        return bool(all(highs[i] < highs[i + 1] for i in range(len(highs) - 1)) and
                     all(lows[i] < lows[i + 1] for i in range(len(lows) - 1)))
    else:
        return bool(all(highs[i] > highs[i + 1] for i in range(len(highs) - 1)) and
                     all(lows[i] > lows[i + 1] for i in range(len(lows) - 1)))


def detect_obv_slope_confirmation(candles, direction, lookback=3):
    """OBV sloping the SAME direction as the trade (confirming
    participation, not divergence - opposite convention from the
    adverse-flow OBV bonus used elsewhere in this project, which looks
    for OBV moving AGAINST an open position). Same OBV formula/caveat as
    the rest of this project: signed by candle direction, not real
    taker buy/sell split (CoinDCX's public candles don't expose that)."""
    if len(candles) < lookback + 1:
        return False
    window = candles.tail(lookback + 1)
    closes = window["close"].values
    vols = window["volume"].values
    obv_dir = np.sign(np.diff(closes, prepend=closes[0]))
    obv_series = np.cumsum(obv_dir * vols)
    slope = obv_series[-1] - obv_series[0]
    target_sign = 1 if direction == "long" else -1
    return bool(np.sign(slope) == target_sign and slope != 0)


def detect_macd_confirmation(candles, direction):
    """Optional extra filter: MACD histogram positive (long) or negative
    (short) - i.e. MACD line above/below its signal line right now."""
    closes = candles["close"].values
    hist_series = macd_series(closes)
    if len(hist_series) == 0:
        return False
    hist = hist_series[-1]
    return bool((direction == "long" and hist > 0) or (direction == "short" and hist < 0))


def detect_rsi_confirmation(candles, direction, long_threshold=50, short_threshold=50):
    """Optional extra filter: RSI above 50 (long) or below 50 (short) -
    simple momentum-side confirmation, not overbought/oversold."""
    closes = candles["close"].values
    r = rsi(closes)
    if r is None:
        return False
    return bool((direction == "long" and r > long_threshold) or (direction == "short" and r < short_threshold))


def detect_momentum_scalp_signal(candles, use_macd=False, use_rsi=False, lookback=3):
    """Checks BOTH directions, returns the first that satisfies every
    required confirmation (core-3 always required; MACD/RSI only checked
    if their respective use_* flag is True). candles must already be
    bounded to "up to and including the current closed bar" by the
    caller - see module docstring."""
    for direction in ("long", "short"):
        if not detect_ema_slope_widening(candles, direction, lookback=lookback):
            continue
        if not detect_price_structure(candles, direction, lookback=lookback):
            continue
        if not detect_obv_slope_confirmation(candles, direction, lookback=lookback):
            continue
        if use_macd and not detect_macd_confirmation(candles, direction):
            continue
        if use_rsi and not detect_rsi_confirmation(candles, direction):
            continue
        return {"action": direction}
    return {"action": "wait"}


def calculate_scalp_stop_target(candles, direction, entry_price, atr_multiplier=1.2, r_target=1.5):
    """Stop and target sized to the 5m chart itself - a SEPARATE ATR
    calculation from the existing 1h-based stop-loss used everywhere
    else in this project. atr_multiplier is deliberately much smaller
    than the existing 2.5x (calibrated for 15m/1h swings) since 5m ATR
    is inherently tighter. r_target is a SINGLE fixed target, closed
    100% at once - not the existing staged 1R/2R/3R ladder, which is
    built for multi-hour trend-riding, not a scalp."""
    atr14 = atr_wilder(candles, 14)
    stop_distance = atr14 * atr_multiplier
    if direction == "long":
        stop_price = entry_price - stop_distance
        target_price = entry_price + stop_distance * r_target
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - stop_distance * r_target
    return {"stop_price": stop_price, "target_price": target_price, "stop_distance": stop_distance}
