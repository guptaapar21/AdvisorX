// Faithful port of opportunityScorer.ts.

const STRATEGY_SCORE_WEIGHTS = {
  "ultra-short": { signalStrength: 35, trendConsistency: 20, volatilityFit: 20, riskRewardRatio: 10, liquidity: 15, minScore: 65 },
  "aggressive": { signalStrength: 30, trendConsistency: 25, volatilityFit: 20, riskRewardRatio: 12, liquidity: 13, minScore: 70 },
  "balanced": { signalStrength: 30, trendConsistency: 25, volatilityFit: 20, riskRewardRatio: 15, liquidity: 10, minScore: 75 },
  "conservative": { signalStrength: 25, trendConsistency: 30, volatilityFit: 15, riskRewardRatio: 20, liquidity: 10, minScore: 80 },
  "swing-trend": { signalStrength: 20, trendConsistency: 35, volatilityFit: 15, riskRewardRatio: 20, liquidity: 10, minScore: 78 },
};

const STRATEGY_VOLATILITY_PREFS = {
  "ultra-short": { idealMin: 1.0, idealMax: 1.5, acceptableMin: 0.8, acceptableMax: 2.0, penaltyFactor: 0.4 },
  "aggressive": { idealMin: 0.9, idealMax: 1.4, acceptableMin: 0.7, acceptableMax: 1.8, penaltyFactor: 0.5 },
  "balanced": { idealMin: 0.8, idealMax: 1.2, acceptableMin: 0.6, acceptableMax: 1.5, penaltyFactor: 0.5 },
  "conservative": { idealMin: 0.6, idealMax: 1.0, acceptableMin: 0.5, acceptableMax: 1.3, penaltyFactor: 0.7 },
  "swing-trend": { idealMin: 0.7, idealMax: 1.1, acceptableMin: 0.5, acceptableMax: 1.4, penaltyFactor: 0.6 },
};

const TIER1 = ["BTC", "ETH"];
const TIER2 = ["BNB", "SOL", "XRP", "ADA"];
const TIER3 = ["DOGE", "AVAX", "DOT", "MATIC", "LTC", "ARB", "OP"];

function calculateVolatilityFitScore(atrRatio, strategy) {
  const pref = STRATEGY_VOLATILITY_PREFS[strategy];
  if (atrRatio >= pref.idealMin && atrRatio <= pref.idealMax) return 1.0;
  if (atrRatio >= pref.acceptableMin && atrRatio <= pref.acceptableMax) {
    if (atrRatio < pref.idealMin) {
      const distance = pref.idealMin - atrRatio;
      const range = pref.idealMin - pref.acceptableMin;
      return 1.0 - (distance / range) * pref.penaltyFactor;
    }
    const distance = atrRatio - pref.idealMax;
    const range = pref.acceptableMax - pref.idealMax;
    return 1.0 - (distance / range) * pref.penaltyFactor;
  }
  return 0.3;
}

function calculateRiskRewardScore(marketState, leverage, strategy) {
  let baseRR = 0.5;
  if (marketState === "uptrend_oversold" || marketState === "downtrend_overbought") baseRR = 0.9;
  else if (marketState === "uptrend_continuation" || marketState === "downtrend_continuation") baseRR = 0.7;
  else if (marketState === "ranging_oversold" || marketState === "ranging_overbought") baseRR = 0.8;

  if (leverage <= 2) baseRR *= 0.95;
  else if (leverage >= 5) baseRR *= 0.75;

  if (strategy === "conservative" && baseRR < 0.7) baseRR *= 0.8;
  if (strategy === "ultra-short") baseRR = Math.min(1.0, baseRR + 0.1);

  return baseRR;
}

function calculateLiquidityScore(symbol, strategy, volume24h) {
  let baseScore = 0.6;
  if (TIER1.includes(symbol)) baseScore = 1.0;
  else if (TIER2.includes(symbol)) baseScore = 0.85;
  else if (TIER3.includes(symbol)) baseScore = 0.7;

  if (volume24h !== undefined && volume24h > 0) {
    if (volume24h >= 1_000_000_000) baseScore = Math.min(1.0, baseScore + 0.1);
    else if (volume24h >= 500_000_000) baseScore = Math.min(1.0, baseScore + 0.05);
    else if (volume24h < 100_000_000) baseScore = Math.max(0.3, baseScore - 0.1);
  }

  if (strategy === "ultra-short" && baseScore < 0.7) baseScore *= 0.8;
  if (strategy === "swing-trend" && baseScore >= 0.6) baseScore = Math.min(1.0, baseScore + 0.05);

  return baseScore;
}

function calculateWaitScore(strategyResult, marketState, strategy) {
  let baseScore = 0;
  let reason = strategyResult.reason;
  const state = marketState.state;

  if (state === "downtrend_overbought" || state === "uptrend_oversold") {
    baseScore = strategy === "ultra-short" || strategy === "aggressive" ? 60 : 55;
    reason = `best timing but indicators not fully met. ${strategyResult.reason}`;
  } else if (state === "downtrend_continuation" || state === "uptrend_continuation") {
    if (strategy === "swing-trend") { baseScore = 50; reason = `trend continuing, waiting on higher-timeframe confirmation. ${strategyResult.reason}`; }
    else if (strategy === "conservative") { baseScore = 48; reason = `trend clear but waiting for a safer entry. ${strategyResult.reason}`; }
    else { baseScore = 45; reason = `trend clear but no precise entry yet. ${strategyResult.reason}`; }
  } else if (state.startsWith("ranging")) {
    if (strategy === "ultra-short") { baseScore = 35; reason = `ranging market, waiting for boundary break or reversal. ${strategyResult.reason}`; }
    else { baseScore = 30; reason = `ranging market, waiting for a clearer signal. ${strategyResult.reason}`; }
  } else {
    baseScore = 20;
    reason = `no clear market signal, standing aside for now. ${strategyResult.reason}`;
  }

  const alignmentScore = marketState.timeframeAlignment.alignmentScore;
  if (alignmentScore >= 0.8) { baseScore += 10; reason = `multi-timeframe strongly aligned (${(alignmentScore * 100).toFixed(0)}%), ${reason}`; }
  else if (alignmentScore >= 0.6) baseScore += 5;

  const atrRatio = marketState.keyMetrics.atr_ratio;
  const volPref = STRATEGY_VOLATILITY_PREFS[strategy];
  if (atrRatio >= volPref.idealMin && atrRatio <= volPref.idealMax) baseScore += 5;

  return {
    symbol: strategyResult.symbol,
    totalScore: Math.min(baseScore, 70),
    breakdown: { signalStrength: baseScore, trendConsistency: Math.round(alignmentScore * 25), volatilityFit: 0, riskRewardRatio: 0, liquidity: 0 },
    confidence: baseScore >= 55 ? "medium" : "low",
    recommendation: { strategyType: "none", direction: "wait", confidence: baseScore >= 55 ? "medium" : "low", reason },
  };
}

async function scoreOpportunity(strategyResult, marketState, strategy, historicalPenaltyFn) {
  if (strategyResult.action === "wait") {
    return calculateWaitScore(strategyResult, marketState, strategy);
  }

  const weights = STRATEGY_SCORE_WEIGHTS[strategy];

  const signalScore = strategyResult.signalStrength * weights.signalStrength;
  const trendScore = marketState.timeframeAlignment.alignmentScore * weights.trendConsistency;
  const volatilityScore = calculateVolatilityFitScore(marketState.keyMetrics.atr_ratio, strategy) * weights.volatilityFit;
  const rrScore = calculateRiskRewardScore(marketState.state, strategyResult.recommendedLeverage, strategy) * weights.riskRewardRatio;
  const liquidityScore = calculateLiquidityScore(strategyResult.symbol, strategy, strategyResult.keyMetrics.volume24h) * weights.liquidity;

  let historicalPenalty = 0;
  let trendStabilityPenalty = 0;
  let volatilityPenalty = 0;

  if (historicalPenaltyFn) {
    try {
      historicalPenalty = await historicalPenaltyFn(strategyResult.symbol);
    } catch {
      // fall through with 0 penalty on error, matches original
    }
  }

  if (marketState.trendChanges) {
    if (marketState.trendChanges.primary.weakeningSeverity > 40) trendStabilityPenalty += 10;
    if (marketState.trendChanges.confirm.weakeningSeverity > 40) trendStabilityPenalty += 8;
  }

  const atrRatio = marketState.keyMetrics.atr_ratio;
  if (atrRatio > 2.0) volatilityPenalty = 15;
  else if (atrRatio > 1.5) volatilityPenalty = 10;

  const baseScore = signalScore + trendScore + volatilityScore + rrScore + liquidityScore;
  const totalScore = Math.max(0, baseScore - historicalPenalty - trendStabilityPenalty - volatilityPenalty);

  const highThreshold = weights.minScore;
  const mediumThreshold = highThreshold - 15;
  let confidence;
  if (totalScore >= highThreshold) confidence = "high";
  else if (totalScore >= mediumThreshold) confidence = "medium";
  else confidence = "low";

  return {
    symbol: strategyResult.symbol,
    totalScore: Math.round(totalScore),
    breakdown: {
      signalStrength: Math.round(signalScore),
      trendConsistency: Math.round(trendScore),
      volatilityFit: Math.round(volatilityScore),
      riskRewardRatio: Math.round(rrScore),
      liquidity: Math.round(liquidityScore),
    },
    confidence,
    recommendation: {
      strategyType: strategyResult.strategyType,
      direction: strategyResult.action,
      confidence,
      reason: strategyResult.reason,
    },
    isBreakoutExtension: strategyResult.isBreakoutExtension || false,
  };
}

module.exports = {
  STRATEGY_SCORE_WEIGHTS,
  STRATEGY_VOLATILITY_PREFS,
  calculateVolatilityFitScore,
  calculateRiskRewardScore,
  calculateLiquidityScore,
  calculateWaitScore,
  scoreOpportunity,
};
