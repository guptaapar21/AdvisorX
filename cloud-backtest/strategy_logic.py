"""
Faithful port of strategyUtils.js / trendFollowingStrategy.js /
meanReversionStrategy.js / breakoutStrategy.js / strategyRouter.js.

Matches what's actually DEPLOYED (not the untouched original): the
macdHist bug is fixed, breakout is wired in as a fallback-only extension
tagged is_breakout_extension.
"""
import numpy as np


def calculate_signal_strength(rsi7, macd_val, macd_signal, ema_alignment, price_position, trend_consistency):
    score = 0
    if rsi7 < 25:
        score += 25 * (25 - rsi7) / 25
    elif rsi7 > 75:
        score += 25 * (rsi7 - 75) / 25
    elif 30 <= rsi7 <= 70:
        score += 15

    macd_diff = macd_val - macd_signal
    if abs(macd_diff) > 0:
        score += 20 * min(abs(macd_diff) / 100, 1)

    if ema_alignment:
        score += 25

    abs_dev = abs(price_position)
    if abs_dev < 3:
        score += 15 * (1 - abs_dev / 3)

    score += 15 * trend_consistency
    return min(score / 100, 1)


def check_mtf_alignment(tf15m, tf1h, direction):
    alignment = 0
    ema_15 = tf15m["ema20"] > tf15m["ema50"]
    ema_1h = tf1h["ema20"] > tf1h["ema50"]
    if direction == "long" and ema_15 and ema_1h:
        alignment += 30
    elif direction == "short" and not ema_15 and not ema_1h:
        alignment += 30
    elif direction == "long" and ema_1h:
        alignment += 15
    elif direction == "short" and not ema_1h:
        alignment += 15

    macd_15 = tf15m["macd"] > 0
    macd_1h = tf1h["macd"] > 0
    if direction == "long" and macd_1h:
        alignment += 25
        if macd_15:
            alignment += 10
    elif direction == "short" and not macd_1h:
        alignment += 25
        if not macd_15:
            alignment += 10

    if direction == "long":
        if tf1h["rsi14"] < 70:
            alignment += 15
        if tf15m["rsi7"] < 30:
            alignment += 10
    else:
        if tf1h["rsi14"] > 30:
            alignment += 15
        if tf15m["rsi7"] > 70:
            alignment += 10

    if direction == "long" and tf1h["current_price"] > tf1h["ema20"]:
        alignment += 10
    elif direction == "short" and tf1h["current_price"] < tf1h["ema20"]:
        alignment += 10

    final_score = alignment / 100
    return {"aligned": final_score >= 0.6, "score": final_score}


def calculate_volatility_adjustment(atr, atr_ma=1.0):
    ratio = atr / atr_ma if atr_ma != 0 else 1
    if ratio < 0.8:
        return {"leverage_multiplier": 1.0, "status": "low"}
    if ratio < 1.2:
        return {"leverage_multiplier": 1.0, "status": "normal"}
    if ratio < 1.8:
        return {"leverage_multiplier": 0.8, "status": "high"}
    return {"leverage_multiplier": 0.6, "status": "extreme"}


def detect_macd_histogram_reversal(current_hist, previous_hist, direction):
    if direction == "bullish":
        return current_hist > previous_hist and previous_hist < 0
    return current_hist < previous_hist and previous_hist > 0


def identify_key_levels(candles, lookback=20):
    recent = candles.tail(lookback)
    return {"resistance": recent["high"].max(), "support": recent["low"].min()}


# ---- Trend-following ----

def trend_following_signal(symbol, direction, tf15m, tf1h, market_state):
    warnings = []
    if direction == "long":
        trend_confirmed = tf1h["ema20"] > tf1h["ema50"]
    else:
        trend_confirmed = tf1h["ema20"] < tf1h["ema50"]

    if not trend_confirmed:
        return {"action": "wait", "signal_strength": 0, "strategy_type": "trend_following"}

    signal_strength = 0
    if market_state["state"] in ("uptrend_continuation", "downtrend_continuation"):
        rsi_ok = (45 <= tf15m["rsi7"] <= 65) if direction == "long" else (35 <= tf15m["rsi7"] <= 55)
        if rsi_ok:
            signal_strength = 0.5
        else:
            return {"action": "wait", "signal_strength": 0, "strategy_type": "trend_following"}
    else:
        pullback_ok = tf15m["rsi7"] < 40 if direction == "long" else tf15m["rsi7"] > 60
        if not pullback_ok:
            return {"action": "wait", "signal_strength": 0, "strategy_type": "trend_following"}
        alignment = check_mtf_alignment(tf15m, tf1h, direction)
        price_pos = (tf15m["current_price"] - tf15m["ema20"]) / tf15m["ema20"] * 100
        signal_strength = calculate_signal_strength(
            tf15m["rsi7"], tf1h["macd"], tf1h["macd_signal"], trend_confirmed, price_pos, alignment["score"])

    vol_adj = calculate_volatility_adjustment(market_state["atr_ratio"], 1.0)
    if vol_adj["status"] == "extreme":
        signal_strength *= 0.7
    elif vol_adj["status"] == "high":
        signal_strength *= 0.85

    return {"action": direction, "signal_strength": signal_strength, "strategy_type": "trend_following"}


# ---- Mean-reversion (macdHist bug FIXED, matches deployed version) ----

def mean_reversion_signal(symbol, direction, tf15m, tf1h, market_state):
    extreme = tf15m["rsi7"] < 35 if direction == "long" else tf15m["rsi7"] > 65
    if not extreme:
        return {"action": "wait", "signal_strength": 0, "strategy_type": "mean_reversion"}

    if direction == "long":
        strong_opposite = tf1h["ema20"] < tf1h["ema50"] and tf1h["macd"] < -50
    else:
        strong_opposite = tf1h["ema20"] > tf1h["ema50"] and tf1h["macd"] > 50
    if strong_opposite:
        return {"action": "wait", "signal_strength": 0, "strategy_type": "mean_reversion"}

    alignment = check_mtf_alignment(tf15m, tf1h, direction)
    price_pos = (tf15m["current_price"] - tf15m["ema20"]) / tf15m["ema20"] * 100
    ema_align = tf1h["ema20"] > tf1h["ema50"] if direction == "long" else tf1h["ema20"] < tf1h["ema50"]
    signal_strength = calculate_signal_strength(
        tf15m["rsi7"], tf15m["macd"], tf15m["macd_signal"], ema_align, price_pos, alignment["score"] * 0.7)

    if direction == "long" and tf15m["rsi7"] < 25:
        signal_strength = min(signal_strength * 1.2, 1.0)
    elif direction == "short" and tf15m["rsi7"] > 75:
        signal_strength = min(signal_strength * 1.2, 1.0)

    # Fixed bug: consistent field name + real prevMacdHistogram (this
    # backtest computes it directly, matching the deployed fix)
    macd_reversal = detect_macd_histogram_reversal(
        tf15m["macd_histogram"], tf15m.get("prev_macd_histogram", tf15m["macd_histogram"]),
        "bullish" if direction == "long" else "bearish")
    if macd_reversal:
        signal_strength = min(signal_strength * 1.15, 1.0)

    vol_adj = calculate_volatility_adjustment(market_state["atr_ratio"], 1.0)
    if vol_adj["status"] == "extreme":
        signal_strength *= 0.6
    elif vol_adj["status"] == "high":
        signal_strength *= 0.8

    return {"action": direction, "signal_strength": signal_strength, "strategy_type": "mean_reversion"}


# ---- Breakout (flagged extension, matches deployed version - the
# original never actually calls this) ----

def breakout_signal(symbol, direction, tf15m, tf1h, market_state):
    candles = tf15m["candles"]
    if len(candles) < 20:
        return {"action": "wait", "signal_strength": 0, "strategy_type": "breakout"}

    levels = identify_key_levels(candles, 20)
    price = tf15m["current_price"]

    if direction == "long":
        broke = price > levels["resistance"] * 0.998
    else:
        broke = price < levels["support"] * 1.002
    if not broke:
        return {"action": "wait", "signal_strength": 0, "strategy_type": "breakout"}

    if direction == "long":
        rsi_ok = 35 <= tf15m["rsi7"] <= 75
    else:
        rsi_ok = 25 <= tf15m["rsi7"] <= 65
    if not rsi_ok:
        return {"action": "wait", "signal_strength": 0, "strategy_type": "breakout"}

    alignment = check_mtf_alignment(tf15m, tf1h, direction)
    ref_level = levels["resistance"] if direction == "long" else levels["support"]
    price_pos = (price - ref_level) / ref_level * 100
    ema_align = tf1h["ema20"] > tf1h["ema50"] if direction == "long" else tf1h["ema20"] < tf1h["ema50"]
    signal_strength = calculate_signal_strength(
        tf15m["rsi7"], tf1h["macd"], tf1h["macd_signal"], ema_align, price_pos, alignment["score"])

    vol_adj = calculate_volatility_adjustment(market_state["atr_ratio"], 1.0)
    if vol_adj["status"] == "extreme":
        signal_strength *= 0.7
    elif vol_adj["status"] == "high":
        signal_strength *= 0.85

    return {"action": direction, "signal_strength": signal_strength, "strategy_type": "breakout",
            "is_breakout_extension": True}


# ---- Router (matches strategyRouter.js's real state->strategy mapping) ----

STATE_STRATEGY_MAP = {
    "uptrend_oversold": ("trend_following", "long"),
    "downtrend_overbought": ("trend_following", "short"),
    "downtrend_oversold": ("mean_reversion", "long"),
    "uptrend_overbought": ("mean_reversion", "short"),
    "uptrend_continuation": ("trend_following", "long"),
    "downtrend_continuation": ("trend_following", "short"),
    "ranging_oversold": ("mean_reversion", "long"),
    "ranging_overbought": ("mean_reversion", "short"),
}


def route_strategy(symbol, market_state, tf15m, tf1h):
    state = market_state["state"]
    mapping = STATE_STRATEGY_MAP.get(state)

    if mapping:
        strategy_type, direction = mapping
        if strategy_type == "trend_following":
            base_result = trend_following_signal(symbol, direction, tf15m, tf1h, market_state)
        else:
            base_result = mean_reversion_signal(symbol, direction, tf15m, tf1h, market_state)
    else:
        base_result = {"action": "wait", "signal_strength": 0, "strategy_type": "none"}

    if base_result["action"] == "wait":
        long_breakout = breakout_signal(symbol, "long", tf15m, tf1h, market_state)
        if long_breakout["action"] == "long":
            return long_breakout
        short_breakout = breakout_signal(symbol, "short", tf15m, tf1h, market_state)
        if short_breakout["action"] == "short":
            return short_breakout

    return base_result
