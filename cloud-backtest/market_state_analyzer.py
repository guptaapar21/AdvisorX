"""
Faithful port of marketStateAnalyzer.js. Every threshold/formula copied
directly from the deployed JS source.

SCOPE NOTE (flagged, not hidden): the reversal score's MACD/RSI divergence
components (10%+10% of the JS version's weighting) are NOT ported here -
reconstructing them for a fast vectorized backtest adds real complexity for
a relatively small slice of the score. The primary/confirm/filter
trend-weakening components (80% of the weighting) ARE ported in full. This
means reversal-triggered exits in this backtest are somewhat less sensitive
than the live bot - if this matters for your conclusions, ask and it can be
added.
"""
import numpy as np
from indicators import ema, rsi, macd, macd_histogram_turn, atr_ratio, avg_volume, detect_volume_spike

OVERSOLD_EXTREME = 20
OVERSOLD_MILD = 30
OVERBOUGHT_EXTREME = 80
OVERBOUGHT_MILD = 70


def build_timeframe_indicators(candles):
    closes = candles["close"].values
    current_price = closes[-1]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    e20 = ema20[-1] if len(ema20) else 0
    e50 = ema50[-1] if len(ema50) else 0
    m = macd(closes) or {"macd": 0, "signal": 0, "histogram": 0}
    deviation = ((current_price - e20) / e20 * 100) if e20 != 0 else 0
    return {
        "current_price": current_price,
        "ema20": e20,
        "ema50": e50,
        "macd": m["macd"],
        "macd_signal": m["signal"],
        "macd_histogram": m["histogram"],
        "macd_turn": macd_histogram_turn(closes),
        "rsi7": rsi(closes, 7) if rsi(closes, 7) is not None else 50,
        "rsi14": rsi(closes, 14) if rsi(closes, 14) is not None else 50,
        "atr_ratio": atr_ratio(candles, 14),
        "deviation_from_ema20": deviation,
        "volume": candles["volume"].iloc[-1],
        "avg_volume": avg_volume(candles, 20),
        "candles": candles,
    }


def determine_trend_strength(tf):
    if tf["ema20"] > tf["ema50"] and tf["macd"] > 0:
        return "trending_up"
    if tf["ema20"] < tf["ema50"] and tf["macd"] < 0:
        return "trending_down"
    return "ranging"


def determine_momentum_state(tf):
    if tf["rsi7"] < OVERSOLD_EXTREME:
        return "oversold_extreme"
    if tf["rsi7"] < OVERSOLD_MILD:
        return "oversold_mild"
    if tf["rsi7"] > OVERBOUGHT_EXTREME:
        return "overbought_extreme"
    if tf["rsi7"] > OVERBOUGHT_MILD:
        return "overbought_mild"
    return "neutral"


def determine_market_state(trend_strength, momentum_state, tf_confirm):
    state = "no_clear_signal"
    confidence = 0.3

    if trend_strength == "trending_up" and momentum_state == "oversold_extreme":
        state, confidence = "uptrend_oversold", 0.9
    elif trend_strength == "trending_down" and momentum_state == "overbought_extreme":
        state, confidence = "downtrend_overbought", 0.9
    elif trend_strength == "trending_down" and momentum_state == "oversold_extreme":
        state, confidence = "downtrend_oversold", 0.6
    elif trend_strength == "trending_up" and momentum_state == "overbought_extreme":
        state, confidence = "uptrend_overbought", 0.6
    elif trend_strength == "trending_up" and momentum_state in ("oversold_mild", "neutral"):
        state, confidence = "uptrend_continuation", 0.7
    elif trend_strength == "trending_down" and momentum_state in ("overbought_mild", "neutral"):
        state, confidence = "downtrend_continuation", 0.7
    elif trend_strength == "trending_down" and momentum_state == "oversold_mild":
        state, confidence = "downtrend_oversold", 0.5
    elif trend_strength == "trending_up" and momentum_state == "overbought_mild":
        state, confidence = "uptrend_overbought", 0.5
    elif trend_strength == "ranging" and momentum_state == "oversold_extreme":
        state, confidence = "ranging_oversold", 0.8
    elif trend_strength == "ranging" and momentum_state == "overbought_extreme":
        state, confidence = "ranging_overbought", 0.8
    elif trend_strength == "ranging" and momentum_state == "neutral":
        state, confidence = "ranging_neutral", 0.5

    if tf_confirm["macd_turn"] == 1 and state in ("uptrend_oversold", "ranging_oversold"):
        confidence = min(confidence + 0.1, 1.0)
    if tf_confirm["macd_turn"] == -1 and state in ("downtrend_overbought", "ranging_overbought"):
        confidence = min(confidence + 0.1, 1.0)

    return {"state": state, "confidence": confidence}


def calculate_trend_consistency(ema20a, ema50a, ema20b, ema50b, macd_a, macd_b):
    score = 0
    trend_a = 1 if ema20a > ema50a else -1
    momentum_a = 1 if macd_a > 0 else -1
    trend_b = 1 if ema20b > ema50b else -1
    momentum_b = 1 if macd_b > 0 else -1
    if trend_a == trend_b:
        score += 0.4
    if momentum_a == momentum_b:
        score += 0.3
    if trend_a == momentum_a:
        score += 0.15
    if trend_b == momentum_b:
        score += 0.15
    return max(0, min(1, score))


def calculate_triple_timeframe_consistency(tf_primary, tf_confirm, tf_filter):
    pc = calculate_trend_consistency(
        tf_primary["ema20"], tf_primary["ema50"], tf_confirm["ema20"], tf_confirm["ema50"],
        tf_primary["macd"], tf_confirm["macd"])
    cf = calculate_trend_consistency(
        tf_confirm["ema20"], tf_confirm["ema50"], tf_filter["ema20"], tf_filter["ema50"],
        tf_confirm["macd"], tf_filter["macd"])
    return pc * 0.6 + cf * 0.4


def calculate_trend_score(tf):
    """-100..100. Matches source exactly: EMA gap (40%) + MACD/price
    normalized (30%) + price deviation from EMA20 (20%) + RSI trend (10%)."""
    score = 0
    ema_gap = (tf["ema20"] - tf["ema50"]) / tf["ema50"] if tf["ema50"] != 0 else 0
    score += max(-40, min(40, ema_gap * 1000))
    macd_normalized = tf["macd"] / tf["current_price"] if tf["current_price"] != 0 else 0
    score += max(-30, min(30, macd_normalized * 10000))
    score += max(-20, min(20, tf["deviation_from_ema20"] * 2))
    rsi_trend = (tf["rsi7"] - 50) / 5
    score += max(-10, min(10, rsi_trend))
    return round(score)


def detect_trend_weakening(current_score, score_history):
    """Matches source exactly: 20%-relative-drop weakening, +/-20 crossing
    = reversing."""
    previous_score = score_history[-1] if len(score_history) > 0 else current_score
    is_weakening = abs(current_score) < abs(previous_score) * 0.8
    weakening_severity = (
        round((1 - abs(current_score) / abs(previous_score)) * 100) if is_weakening and previous_score != 0 else 0
    )
    return {"current_score": current_score, "previous_score": previous_score, "is_weakening": is_weakening,
            "weakening_severity": weakening_severity}


def detect_adverse_drift(current_score, score_history, target_sign, net_threshold=10):
    """Faithful port of the JS fix (marketStateAnalyzer.js, added Aug 2 after
    the ETH/DOGE reversal-blind-spot incident). detect_trend_weakening above
    only fires on a single-step MAGNITUDE DROP - it's direction-blind, so a
    score climbing steadily toward the adverse side (14 -> 20 -> 22 -> 27,
    each step real, none of them a "drop") never registers. This looks at
    the last few cycles instead and asks a direction-based question: has the
    score moved toward the target (adverse) side on nearly every recent
    step, covering real cumulative distance - regardless of whether any
    single step was a big enough drop to trip the old check.

    net_threshold=10 was tuned against the real ETH short that triggered
    this fix (primary score 14/20/22/27, net +13 over 3 cycles) - NOT
    independently validated against 2 years of history yet. Treat this
    default as a hypothesis to backtest, not a settled number - see
    compare_adverse_drift.py.
    """
    sequence = list(score_history) + [current_score]
    if len(sequence) < 3:
        return {"is_drifting": False, "net_movement": 0, "adverse_steps": 0, "total_steps": 0}

    window = sequence[-4:]
    steps = [window[i] - window[i - 1] for i in range(1, len(window))]
    adverse_steps = sum(1 for s in steps if np.sign(s) == target_sign)
    net_movement = (window[-1] - window[0]) * target_sign

    is_drifting = (
        len(steps) >= 2
        and adverse_steps >= len(steps) - 1
        and net_movement >= net_threshold
        and np.sign(current_score) == target_sign
    )
    return {"is_drifting": bool(is_drifting), "net_movement": round(net_movement),
            "adverse_steps": adverse_steps, "total_steps": len(steps)}


def calculate_reversal_score(tf_primary, tf_confirm, tf_filter, position_direction, history,
                              use_adverse_drift=False, drift_net_threshold=10):
    """Weighted 40/25/15 across primary/confirm/filter timeframes (the
    trend-weakening portion of the source's reversal score - MACD/RSI
    divergence NOT included, see module docstring).

    use_adverse_drift: NEW, default False so every existing sweep/backtest
    call reproduces its old results unchanged unless explicitly opted in.
    When True, adds the drift branch (see detect_adverse_drift) to each
    timeframe's scoring ladder - this is the thing compare_adverse_drift.py
    is for: run the SAME data with this False vs True and compare.
    """
    score = 0
    details = []
    reversed_frames = []
    target_sign = -1 if position_direction == "long" else 1

    score_primary = calculate_trend_score(tf_primary)
    score_confirm = calculate_trend_score(tf_confirm)
    score_filter = calculate_trend_score(tf_filter)

    primary_change = detect_trend_weakening(score_primary, history.get("primary", []))
    primary_drift = detect_adverse_drift(score_primary, history.get("primary", []), target_sign, drift_net_threshold) if use_adverse_drift else None
    if np.sign(score_primary) == target_sign and abs(score_primary) > 30:
        score += 40
        details.append(f"primary strongly reversed ({score_primary})")
        reversed_frames.append("primary")
    elif primary_change["is_weakening"] and primary_change["weakening_severity"] > 40:
        score += 20
        details.append(f"primary weakening ({primary_change['weakening_severity']}%)")
    elif use_adverse_drift and primary_drift["is_drifting"]:
        score += 18
        details.append(f"primary drifting adverse (net {primary_drift['net_movement']})")
        reversed_frames.append("primary_drift")
    elif abs(score_primary) < 20:
        score += 12

    confirm_change = detect_trend_weakening(score_confirm, history.get("confirm", []))
    confirm_drift = detect_adverse_drift(score_confirm, history.get("confirm", []), target_sign, drift_net_threshold) if use_adverse_drift else None
    if np.sign(score_confirm) == target_sign and abs(score_confirm) > 30:
        score += 25
        details.append(f"confirm strongly reversed ({score_confirm})")
        reversed_frames.append("confirm")
    elif confirm_change["is_weakening"] and confirm_change["weakening_severity"] > 40:
        score += 12
    elif use_adverse_drift and confirm_drift["is_drifting"]:
        score += 11
        details.append(f"confirm drifting adverse (net {confirm_drift['net_movement']})")
        reversed_frames.append("confirm_drift")

    filter_change = detect_trend_weakening(score_filter, history.get("filter", []))
    filter_drift = detect_adverse_drift(score_filter, history.get("filter", []), target_sign, drift_net_threshold) if use_adverse_drift else None
    if np.sign(score_filter) == target_sign and abs(score_filter) > 30:
        score += 15
        details.append(f"filter reversed ({score_filter})")
        reversed_frames.append("filter")
    elif use_adverse_drift and filter_drift["is_drifting"]:
        score += 7
        details.append(f"filter drifting adverse (net {filter_drift['net_movement']})")
        reversed_frames.append("filter_drift")

    return {
        "reversal_score": score,
        "reversed_frames": reversed_frames,
        "details": details,
        "trend_scores": {"primary": score_primary, "confirm": score_confirm, "filter": score_filter},
    }
