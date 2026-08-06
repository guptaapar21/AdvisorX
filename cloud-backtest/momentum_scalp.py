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

from indicators import ema, rsi, macd, macd_series, atr_wilder, avg_volume, detect_volume_spike

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


def calculate_supertrend(candles, period=10, multiplier=3.0):
    """SuperTrend is RECURSIVE, unlike every other indicator in this
    file (EMA/RSI/MACD are memoryless - recomputed fresh from a window
    each call). SuperTrend's bands ratchet forward from the PREVIOUS
    bar's final bands, so it needs the recursive history replayed, not
    just the current bar's raw values. This function replays the full
    band/trend recursion across the given (already-bounded, zero-
    lookahead) candles window from its own start - it does NOT thread
    state across separate calls, avoiding any risk of that state
    accidentally holding information from before what the caller
    intended to bound. Window must be long enough for the ratchet to
    converge - same convergence property already validated for Wilder
    ATR (100 bars is far more than the ~40-50 needed)."""
    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values
    n = len(candles)
    if n < period + 2:
        return {"trend": None, "value": None, "flipped": False}

    trs = np.zeros(n)
    for i in range(1, n):
        trs[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr = np.zeros(n)
    atr[period] = trs[1:period + 1].mean()
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period

    hl2 = (highs + lows) / 2
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    trend = np.ones(n, dtype=int)

    start = period
    final_upper[start] = hl2[start] + multiplier * atr[start]
    final_lower[start] = hl2[start] - multiplier * atr[start]
    trend[start] = 1 if closes[start] > final_upper[start] else -1

    for i in range(start + 1, n):
        basic_upper = hl2[i] + multiplier * atr[i]
        basic_lower = hl2[i] - multiplier * atr[i]
        final_upper[i] = basic_upper if (basic_upper < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        final_lower[i] = basic_lower if (basic_lower > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if closes[i] < final_lower[i] else 1
        else:
            trend[i] = 1 if closes[i] > final_upper[i] else -1

    current_trend = "up" if trend[-1] == 1 else "down"
    flipped = bool(trend[-1] != trend[-2]) if n > start + 1 else False
    st_value = final_lower[-1] if trend[-1] == 1 else final_upper[-1]
    return {"trend": current_trend, "value": float(st_value), "flipped": flipped}


def classify_1m_confirmation(candles_1m, direction, fast=9, slow=21, strong_threshold_pct=0.0015):
    """Uses a FASTER timeframe (1m) to gauge conviction behind a 3m
    SuperTrend flip, returning one of three tiers used to dynamically
    size the exit - not a fixed stop/target regardless of context.

    Uses the EMA GAP MAGNITUDE (normalized by price), not a binary
    "widening over N bars" flag. The widening-flag version was tested
    directly against a genuinely ambiguous case (flat/choppy history,
    only barely tipping positive in the last 3 bars) and STILL returned
    "strong" - the gap grows monotonically almost any time the sign
    agrees at all, making a widening-based "weak" tier nearly
    unreachable in practice. A magnitude threshold is a real,
    continuously-varying signal instead.

      "contradicting" - 1m EMA(fast) on the WRONG side of EMA(slow)
                          relative to the 3m trade direction. Skip the
                          trade entirely - this also directly helps the
                          frequency/fee problem (idea #12's 3m result
                          showed real gross edge but too many trades).
      "strong"          - agrees with direction AND the gap is at least
                          `strong_threshold_pct` of price - real
                          separation, not just barely on the right side.
                          Use a WIDER target, let it run.
      "weak"            - agrees with direction but the gap is smaller
                          than the threshold - barely on the right
                          side. Use a TIGHTER target, lock in gains
                          fast rather than risk giving them back.

    candles_1m must already be bounded by the caller to end at the same
    moment the 3m bar itself closed - zero lookahead preserved."""
    closes = candles_1m["close"].values
    if len(closes) < slow:
        return "weak"  # not enough 1m history yet - default to the cautious tier, never to "strong"
    ema_fast_series = ema(closes, fast)
    ema_slow_series = ema(closes, slow)
    offset = len(ema_fast_series) - len(ema_slow_series)
    gap_now = ema_fast_series[offset:][-1] - ema_slow_series[-1]
    current_price = closes[-1]

    target_sign = 1 if direction == "long" else -1
    if np.sign(gap_now) != target_sign:
        return "contradicting"
    gap_pct = abs(gap_now) / current_price if current_price != 0 else 0
    return "strong" if gap_pct >= strong_threshold_pct else "weak"


def detect_macd_zero_cross(candles, direction):
    """Bullish: MACD line crosses ABOVE its signal line (histogram flips
    from <=0 to >0 between the last two bars) AND the MACD line itself
    is above the zero line at that moment. Bearish (short): mirrored -
    crosses below signal AND MACD line is below zero. Stricter than a
    plain crossover - specifically requires the cross to already be on
    the "confirming" side of zero, not a reversal crossover from deep
    negative/positive territory."""
    closes = candles["close"].values
    hist_series = macd_series(closes)
    if len(hist_series) < 2:
        return False
    m = macd(closes)
    if m is None:
        return False
    macd_line_value = m["macd"]

    if direction == "long":
        crossed_up = hist_series[-2] <= 0 and hist_series[-1] > 0
        return bool(crossed_up and macd_line_value > 0)
    else:
        crossed_down = hist_series[-2] >= 0 and hist_series[-1] < 0
        return bool(crossed_down and macd_line_value < 0)


def calculate_vwap(candles):
    """Volume-weighted average price over the given (already-bounded,
    zero-lookahead) window - a rolling VWAP over the same window used
    for every other indicator in this file, not a session-anchored
    reset (crypto trades 24/7, no natural session boundary, and a
    rolling window avoids introducing a new alignment/reset-timing risk
    alongside everything else already carefully bounded in this
    project). Uses typical price (high+low+close)/3, the standard VWAP
    convention, not close alone."""
    typical_price = (candles["high"] + candles["low"] + candles["close"]) / 3
    total_volume = candles["volume"].sum()
    if total_volume == 0:
        return None
    return float((typical_price * candles["volume"]).sum() / total_volume)


def detect_ema_pullback_rejection(candles, direction, fast=9, slow=21, trend_lookback=3):
    """Idea #18: VWAP directional bias + 9/21 EMA trend alignment + a
    genuine PULLBACK to the fast EMA + a REJECTION candle at that level.
    This is the signal on the CURRENT (last) candle in the window - the
    caller is responsible for entering on the NEXT candle's open, not
    this candle's close (see run_ema_pullback_backtest.py for why).

    Long requires ALL of:
      - price above VWAP (directional bias)
      - EMA9 > EMA21 (trend alignment)
      - a genuine prior trend leg: `trend_lookback` candles ago, price
        was clearly separated from EMA9 (confirms this is a real
        pullback, not just chop sitting on the EMA the whole time)
      - the current candle's LOW touches or dips to EMA9 (the pullback
        itself)
      - the current candle's CLOSE is back above EMA9 and above its own
        OPEN (a real bullish rejection candle, not just a touch)
    Short is the exact mirror.
    """
    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values
    opens = candles["open"].values
    n = len(candles)
    if n < slow + trend_lookback + 1:
        return False

    ema_fast_series = ema(closes, fast)
    ema_slow_series = ema(closes, slow)
    offset = len(ema_fast_series) - len(ema_slow_series)
    ema_fast_aligned = ema_fast_series[offset:]

    vwap = calculate_vwap(candles)
    if vwap is None:
        return False

    current_close = closes[-1]
    current_open = opens[-1]
    current_low = lows[-1]
    current_high = highs[-1]
    ema9_now = ema_fast_aligned[-1]
    ema21_now = ema_slow_series[-1]
    ema9_prior = ema_fast_aligned[-(trend_lookback + 1)]
    price_prior = closes[-(trend_lookback + 1)]
    # Meaningful separation threshold, not just "> 0" - a near-zero gap
    # is satisfied by ordinary noise even in flat/choppy data with no
    # real trend leg at all (confirmed directly: tested against
    # deliberately flat/choppy synthetic data and the naive ">0" check
    # incorrectly returned True). 0.1% of price is a real, non-trivial
    # separation.
    min_separation = current_close * 0.001

    if direction == "long":
        vwap_ok = current_close > vwap
        trend_ok = ema9_now > ema21_now
        had_real_leg = (price_prior - ema9_prior) > min_separation
        touched = current_low <= ema9_now
        rejected = current_close > ema9_now and current_close > current_open
        return bool(vwap_ok and trend_ok and had_real_leg and touched and rejected)
    else:
        vwap_ok = current_close < vwap
        trend_ok = ema9_now < ema21_now
        had_real_leg = (ema9_prior - price_prior) > min_separation
        touched = current_high >= ema9_now
        rejected = current_close < ema9_now and current_close < current_open
        return bool(vwap_ok and trend_ok and had_real_leg and touched and rejected)
