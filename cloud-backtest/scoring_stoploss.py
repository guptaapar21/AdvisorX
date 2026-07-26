"""
Faithful port of opportunityScorer.js, strategyParams.js's stop-loss/
leverage bounds, stopLossCalculator.js, and takeProfitManagement.js's REAL
enforced logic (confirmed via the deployed code: always 1R/2R/3R at
33.33/33.33/0%, volatility-adjusted).

REBUILT July 2026: the previous version of this file (and every prior
Colab session) only ever had BALANCED_WEIGHTS hardcoded - preset switching
for aggressive/conservative/swing-trend/ultra-short was done via an ad-hoc
edit made directly in a live Colab kernel that was never saved to any
notebook file, and that session is now gone/unrecoverable. Rather than
guess at what that lost code approximately did, every table below was
re-extracted directly from the real, currently-deployed
opportunityScorer.js and strategyParams.js in the AdvisorX-main repo -
this is a faithful reconstruction from source, not a reconstruction of a
lost approximation.
"""
import numpy as np
from indicators import atr_wilder

# ---- Opportunity scorer: real per-preset weight tables ----
# (exact match to STRATEGY_SCORE_WEIGHTS in opportunityScorer.js)

STRATEGY_SCORE_WEIGHTS = {
    "ultra-short":  {"signal_strength": 35, "trend_consistency": 20, "volatility_fit": 20, "risk_reward": 10, "liquidity": 15, "min_score": 65},
    "aggressive":   {"signal_strength": 30, "trend_consistency": 25, "volatility_fit": 20, "risk_reward": 12, "liquidity": 13, "min_score": 70},
    "balanced":     {"signal_strength": 30, "trend_consistency": 25, "volatility_fit": 20, "risk_reward": 15, "liquidity": 10, "min_score": 75},
    "conservative": {"signal_strength": 25, "trend_consistency": 30, "volatility_fit": 15, "risk_reward": 20, "liquidity": 10, "min_score": 80},
    "swing-trend":  {"signal_strength": 20, "trend_consistency": 35, "volatility_fit": 15, "risk_reward": 20, "liquidity": 10, "min_score": 78},
}

# (exact match to STRATEGY_VOLATILITY_PREFS in opportunityScorer.js)
STRATEGY_VOLATILITY_PREFS = {
    "ultra-short":  {"ideal_min": 1.0, "ideal_max": 1.5, "acceptable_min": 0.8, "acceptable_max": 2.0, "penalty_factor": 0.4},
    "aggressive":   {"ideal_min": 0.9, "ideal_max": 1.4, "acceptable_min": 0.7, "acceptable_max": 1.8, "penalty_factor": 0.5},
    "balanced":     {"ideal_min": 0.8, "ideal_max": 1.2, "acceptable_min": 0.6, "acceptable_max": 1.5, "penalty_factor": 0.5},
    "conservative": {"ideal_min": 0.6, "ideal_max": 1.0, "acceptable_min": 0.5, "acceptable_max": 1.3, "penalty_factor": 0.7},
    "swing-trend":  {"ideal_min": 0.7, "ideal_max": 1.1, "acceptable_min": 0.5, "acceptable_max": 1.4, "penalty_factor": 0.6},
}

# (exact match to getStrategyParams() in strategyParams.js, at the live
# bot's default maxLeverage=15 - leverageMin/Max below are already the
# resolved integers at that leverage, not the raw percentage formulas)
STRATEGY_LEVERAGE_BOUNDS = {
    "ultra-short":  {"min": 8, "max": 12},
    "aggressive":   {"min": 13, "max": 15},
    "balanced":     {"min": 9, "max": 13},
    "conservative": {"min": 5, "max": 9},
    "swing-trend":  {"min": 3, "max": 8},
}

# (exact match to scientificStopLoss in strategyParams.js)
STRATEGY_STOP_LOSS = {
    "ultra-short":  {"atr_multiplier": 1.5, "min_distance": 0.3, "max_distance": 2.0},
    "aggressive":   {"atr_multiplier": 1.5, "min_distance": 0.5, "max_distance": 5.0},
    "balanced":     {"atr_multiplier": 2.0, "min_distance": 0.5, "max_distance": 5.0},
    "conservative": {"atr_multiplier": 2.5, "min_distance": 1.0, "max_distance": 4.0},
    "swing-trend":  {"atr_multiplier": 2.5, "min_distance": 1.0, "max_distance": 6.0},
}

TIER1 = {"BTC", "ETH"}
TIER2 = {"BNB", "SOL", "XRP", "ADA"}
TIER3 = {"DOGE", "AVAX", "DOT", "MATIC", "LTC", "ARB", "OP"}


def calculate_volatility_fit_score(atr_ratio_val, strategy="balanced"):
    p = STRATEGY_VOLATILITY_PREFS[strategy]
    if p["ideal_min"] <= atr_ratio_val <= p["ideal_max"]:
        return 1.0
    if p["acceptable_min"] <= atr_ratio_val <= p["acceptable_max"]:
        if atr_ratio_val < p["ideal_min"]:
            distance = p["ideal_min"] - atr_ratio_val
            rng = p["ideal_min"] - p["acceptable_min"]
        else:
            distance = atr_ratio_val - p["ideal_max"]
            rng = p["acceptable_max"] - p["ideal_max"]
        return 1.0 - (distance / rng) * p["penalty_factor"]
    return 0.3


def calculate_risk_reward_score(market_state_name, leverage, strategy="balanced"):
    """Exact match to calculateRiskRewardScore in opportunityScorer.js,
    including the leverage-based scaling and the two strategy-specific
    adjustments (conservative penalty, ultra-short boost) - these were
    missing entirely from every prior Python port, which only had the
    base_rr table with no leverage/strategy adjustment at all."""
    base_rr = 0.5
    if market_state_name in ("uptrend_oversold", "downtrend_overbought"):
        base_rr = 0.9
    elif market_state_name in ("uptrend_continuation", "downtrend_continuation"):
        base_rr = 0.7
    elif market_state_name in ("ranging_oversold", "ranging_overbought"):
        base_rr = 0.8

    if leverage <= 2:
        base_rr *= 0.95
    elif leverage >= 5:
        base_rr *= 0.75

    if strategy == "conservative" and base_rr < 0.7:
        base_rr *= 0.8
    if strategy == "ultra-short":
        base_rr = min(1.0, base_rr + 0.1)

    return base_rr


def calculate_liquidity_score(symbol):
    if symbol in TIER1:
        return 1.0
    if symbol in TIER2:
        return 0.85
    if symbol in TIER3:
        return 0.7
    return 0.6


def score_opportunity(strategy_result, market_state, alignment_score, atr_ratio_val, symbol,
                       strategy="balanced", leverage=None):
    if strategy_result["action"] == "wait":
        return {"total_score": 0, "confidence": "low"}

    w = STRATEGY_SCORE_WEIGHTS[strategy]
    if leverage is None:
        leverage = STRATEGY_LEVERAGE_BOUNDS[strategy]["min"]  # conservative default if not resolved yet

    signal_score = strategy_result["signal_strength"] * w["signal_strength"]
    trend_score = alignment_score * w["trend_consistency"]
    vol_score = calculate_volatility_fit_score(atr_ratio_val, strategy) * w["volatility_fit"]
    rr_score = calculate_risk_reward_score(market_state["state"], leverage, strategy) * w["risk_reward"]
    liq_score = calculate_liquidity_score(symbol) * w["liquidity"]

    total = signal_score + trend_score + vol_score + rr_score + liq_score
    high_threshold = w["min_score"]
    medium_threshold = high_threshold - 15
    confidence = "high" if total >= high_threshold else ("medium" if total >= medium_threshold else "low")

    return {"total_score": round(total), "confidence": confidence}


# ---- Stop-loss calculator (hybrid ATR + support/resistance) ----

def find_support_level(candles, lookback=20):
    if len(candles) < lookback:
        return 0
    recent = candles.tail(lookback).reset_index(drop=True)
    lows = recent["low"].values
    local_lows = []
    for i in range(2, len(recent) - 2):
        if lows[i] < min(lows[i - 1], lows[i - 2]) and lows[i] < min(lows[i + 1], lows[i + 2]):
            local_lows.append(lows[i])
    return min(local_lows) if local_lows else lows.min()


def find_resistance_level(candles, lookback=20):
    if len(candles) < lookback:
        return 0
    recent = candles.tail(lookback).reset_index(drop=True)
    highs = recent["high"].values
    local_highs = []
    for i in range(2, len(recent) - 2):
        if highs[i] > max(highs[i - 1], highs[i - 2]) and highs[i] > max(highs[i + 1], highs[i + 2]):
            local_highs.append(highs[i])
    return max(local_highs) if local_highs else highs.max()


def calculate_scientific_stop_loss(candles, side, entry_price, atr_multiplier=2.0,
                                    min_stop_pct=0.5, max_stop_pct=5.0):
    current_price = candles["close"].iloc[-1]
    atr = atr_wilder(candles, 14)
    atr_pct = (atr / current_price) * 100 if current_price != 0 else 0
    atr_distance = atr * atr_multiplier
    atr_stop = entry_price - atr_distance if side == "long" else entry_price + atr_distance

    sr_stop = None
    if side == "long":
        support = find_support_level(candles, 20)
        buf = support * 0.001
        candidate = support - buf
        if candidate < entry_price:
            sr_stop = candidate
    else:
        resistance = find_resistance_level(candles, 20)
        buf = resistance * 0.001
        candidate = resistance + buf
        if candidate > entry_price:
            sr_stop = candidate

    if sr_stop is not None:
        final_stop = max(atr_stop, sr_stop) if side == "long" else min(atr_stop, sr_stop)
    else:
        final_stop = atr_stop

    if side == "long" and final_stop >= entry_price:
        final_stop = entry_price * (1 - min_stop_pct / 100)
    elif side == "short" and final_stop <= entry_price:
        final_stop = entry_price * (1 + min_stop_pct / 100)

    stop_distance_pct = (
        (entry_price - final_stop) / entry_price * 100 if side == "long"
        else (final_stop - entry_price) / entry_price * 100
    )

    quality_score = 50
    if 1.5 <= atr_pct <= 3.0:
        quality_score += 20
    elif atr_pct < 1.5:
        quality_score += 10
    if 1.5 <= stop_distance_pct <= 3.0:
        quality_score += 20
    elif stop_distance_pct < 1.5:
        quality_score += 10
    quality_score += 10  # S/R contribution (simplified: always some level found)
    quality_score = max(0, min(100, quality_score))

    return {
        "stop_price": final_stop,
        "stop_distance_pct": stop_distance_pct,
        "quality_score": quality_score,
        "volatility_extreme": atr_pct >= 5.0,
    }


def should_open_position(candles, side, entry_price, atr_multiplier=2.0, min_stop_pct=0.5, max_stop_pct=5.0, min_quality=40):
    result = calculate_scientific_stop_loss(candles, side, entry_price, atr_multiplier, min_stop_pct, max_stop_pct)
    if result["stop_distance_pct"] < min_stop_pct:
        return False, result
    if result["stop_distance_pct"] > max_stop_pct:
        return False, result
    if result["volatility_extreme"]:
        return False, result
    if result["quality_score"] < min_quality:
        return False, result
    return True, result


# ---- Take-profit (real enforced logic: fixed 1R/2R/3R @ 33.33/33.33/0,
# volatility-adjusted - confirmed via grep that per-strategy text is never
# actually read by the enforcement code) ----

def analyze_market_volatility(candles):
    if len(candles) < 15:
        return {"level": "NORMAL", "adjustment_factor": 1.0}
    atr14 = atr_wilder(candles, 14)
    current_price = candles["close"].iloc[-1]
    atr_pct = (atr14 / current_price) * 100 if current_price != 0 else 3.0
    if atr_pct < 2:
        return {"level": "LOW", "adjustment_factor": 0.8}
    if atr_pct < 5:
        return {"level": "NORMAL", "adjustment_factor": 1.0}
    if atr_pct < 8:
        return {"level": "HIGH", "adjustment_factor": 1.2}
    return {"level": "EXTREME", "adjustment_factor": 1.5}


def calculate_r_multiple(entry_price, current_price, stop_loss_price, side):
    risk_distance = abs(entry_price - stop_loss_price)
    if risk_distance == 0:
        return 0
    profit_distance = (current_price - entry_price) if side == "long" else (entry_price - current_price)
    return profit_distance / risk_distance


def calculate_target_price(entry_price, stop_loss_price, r_multiple, side):
    risk_distance = abs(entry_price - stop_loss_price)
    target_distance = risk_distance * r_multiple
    return entry_price + target_distance if side == "long" else entry_price - target_distance
