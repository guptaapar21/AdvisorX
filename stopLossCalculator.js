// Faithful port of stopLossCalculator.ts.
const { atrWilder } = require("./indicators");

const DEFAULT_CONFIG = {
  atrPeriod: 14,
  atrMultiplier: 2.0,
  lookbackPeriod: 20,
  bufferPercent: 0.1,
  useATR: true,
  useSupportResistance: true,
  minStopLossPercent: 0.5,
  maxStopLossPercent: 5.0,
};

function findSupportLevel(candles, lookback = 20) {
  if (candles.length < lookback) return 0;
  const recent = candles.slice(-lookback);
  const lowestLow = Math.min(...recent.map((c) => c.low));
  const localLows = [];
  for (let i = 2; i < recent.length - 2; i++) {
    const current = recent[i].low;
    const leftMin = Math.min(recent[i - 1].low, recent[i - 2].low);
    const rightMin = Math.min(recent[i + 1].low, recent[i + 2].low);
    if (current < leftMin && current < rightMin) localLows.push(current);
  }
  return localLows.length > 0 ? Math.min(...localLows) : lowestLow;
}

function findResistanceLevel(candles, lookback = 20) {
  if (candles.length < lookback) return 0;
  const recent = candles.slice(-lookback);
  const highestHigh = Math.max(...recent.map((c) => c.high));
  const localHighs = [];
  for (let i = 2; i < recent.length - 2; i++) {
    const current = recent[i].high;
    const leftMax = Math.max(recent[i - 1].high, recent[i - 2].high);
    const rightMax = Math.max(recent[i + 1].high, recent[i + 2].high);
    if (current > leftMax && current > rightMax) localHighs.push(current);
  }
  return localHighs.length > 0 ? Math.max(...localHighs) : highestHigh;
}

function assessMarketNoise(atr, currentPrice) {
  const atrPercent = (atr / currentPrice) * 100;
  let volatilityLevel, isNoisy;
  if (atrPercent < 1.5) { volatilityLevel = "LOW"; isNoisy = false; }
  else if (atrPercent < 3.0) { volatilityLevel = "MEDIUM"; isNoisy = false; }
  else if (atrPercent < 5.0) { volatilityLevel = "HIGH"; isNoisy = true; }
  else { volatilityLevel = "EXTREME"; isNoisy = true; }
  return { isNoisy, volatilityLevel, atrPercent };
}

// Computes the hybrid stop-loss price + quality score for a candidate entry.
function calculateScientificStopLoss(candles, side, entryPrice, config = DEFAULT_CONFIG) {
  const currentPrice = candles[candles.length - 1].close;

  const atr = atrWilder(candles, config.atrPeriod);
  const atrPercent = (atr / currentPrice) * 100;
  const atrDistance = atr * config.atrMultiplier;
  let atrStopPrice = side === "long" ? entryPrice - atrDistance : entryPrice + atrDistance;

  let srStopPrice, supportLevel, resistanceLevel;
  if (config.useSupportResistance) {
    if (side === "long") {
      supportLevel = findSupportLevel(candles, config.lookbackPeriod);
      const buffer = supportLevel * (config.bufferPercent / 100);
      srStopPrice = supportLevel - buffer;
      if (srStopPrice >= entryPrice) srStopPrice = undefined;
    } else {
      resistanceLevel = findResistanceLevel(candles, config.lookbackPeriod);
      const buffer = resistanceLevel * (config.bufferPercent / 100);
      srStopPrice = resistanceLevel + buffer;
      if (srStopPrice <= entryPrice) srStopPrice = undefined;
    }
  }

  let finalStopPrice, method;
  if (config.useATR && config.useSupportResistance && srStopPrice !== undefined) {
    finalStopPrice = side === "long" ? Math.max(atrStopPrice, srStopPrice) : Math.min(atrStopPrice, srStopPrice);
    method = "HYBRID";
  } else if (config.useSupportResistance && srStopPrice !== undefined) {
    finalStopPrice = srStopPrice; method = "SUPPORT_RESISTANCE";
  } else {
    finalStopPrice = atrStopPrice; method = "ATR";
  }

  // Sanity-check direction; fall back to ATR, then min distance, if broken
  if (side === "long" && finalStopPrice >= entryPrice) {
    finalStopPrice = atrStopPrice; method = "ATR";
    if (finalStopPrice >= entryPrice) finalStopPrice = entryPrice * (1 - config.minStopLossPercent / 100);
  } else if (side === "short" && finalStopPrice <= entryPrice) {
    finalStopPrice = atrStopPrice; method = "ATR";
    if (finalStopPrice <= entryPrice) finalStopPrice = entryPrice * (1 + config.minStopLossPercent / 100);
  }

  const stopLossDistancePercent = side === "long"
    ? ((entryPrice - finalStopPrice) / entryPrice) * 100
    : ((finalStopPrice - entryPrice) / entryPrice) * 100;

  const noiseAssessment = assessMarketNoise(atr, currentPrice);
  let recommendation;
  if (noiseAssessment.volatilityLevel === "EXTREME") recommendation = "market extremely volatile, consider reducing size or waiting";
  else if (noiseAssessment.isNoisy) recommendation = "market noisy, stop distance auto-widened to avoid getting shaken out";
  else recommendation = "normal volatility, stop distance reasonable";

  let qualityScore = 50;
  if (atrPercent >= 1.5 && atrPercent <= 3.0) qualityScore += 20;
  else if (atrPercent < 1.5) qualityScore += 10;
  if (stopLossDistancePercent >= 1.5 && stopLossDistancePercent <= 3.0) qualityScore += 20;
  else if (stopLossDistancePercent < 1.5) qualityScore += 10;
  if (supportLevel || resistanceLevel) qualityScore += 10;
  qualityScore = Math.max(0, Math.min(100, qualityScore));

  return {
    stopLossPrice: finalStopPrice,
    stopLossDistancePercent,
    method,
    details: { atr, atrPercent, supportLevel, resistanceLevel, atrStopPrice, srStopPrice },
    qualityScore,
    riskAssessment: { isNoisy: noiseAssessment.isNoisy, volatilityLevel: noiseAssessment.volatilityLevel, recommendation },
  };
}

// Pre-entry filter: rejects if stop distance out of bounds, volatility extreme, or quality too low.
function shouldOpenPosition(candles, side, entryPrice, config = DEFAULT_CONFIG) {
  const result = calculateScientificStopLoss(candles, side, entryPrice, config);

  if (result.stopLossDistancePercent < config.minStopLossPercent) {
    return { shouldOpen: false, reason: `stop distance too small (${result.stopLossDistancePercent.toFixed(2)}% < ${config.minStopLossPercent}%), market noise may cause frequent stop-outs`, stopLossResult: result };
  }
  if (result.stopLossDistancePercent > config.maxStopLossPercent) {
    return { shouldOpen: false, reason: `stop distance too wide (${result.stopLossDistancePercent.toFixed(2)}% > ${config.maxStopLossPercent}%), poor risk/reward`, stopLossResult: result };
  }
  if (result.riskAssessment.volatilityLevel === "EXTREME") {
    return { shouldOpen: false, reason: "market volatility extreme, holding off", stopLossResult: result };
  }
  if (result.qualityScore < 40) {
    return { shouldOpen: false, reason: `stop quality score too low (${result.qualityScore}/100), poor trading conditions`, stopLossResult: result };
  }
  return { shouldOpen: true, reason: "stop-loss setup reasonable, ok to open", stopLossResult: result };
}

// Recomputes stop from current price; only allows the stop to move in the favorable direction.
function updateTrailingStopLoss(candles, side, entryPrice, currentPrice, currentStopLoss, config = DEFAULT_CONFIG) {
  const result = calculateScientificStopLoss(candles, side, currentPrice, config);
  const newStopLoss = result.stopLossPrice;
  const isProfitable = side === "long" ? currentPrice > entryPrice : currentPrice < entryPrice;

  if (side === "long") {
    if (newStopLoss > currentStopLoss) {
      const improvement = ((newStopLoss - currentStopLoss) / currentStopLoss) * 100;
      return { shouldUpdate: true, newStopLoss, reason: isProfitable ? `stop moved up ${improvement.toFixed(2)}%, protecting profit` : `stop moved up ${improvement.toFixed(2)}%, improved risk protection` };
    }
    return { shouldUpdate: false, reason: `long stop can't move down (new ${newStopLoss.toFixed(6)} <= current ${currentStopLoss.toFixed(6)})` };
  }
  if (newStopLoss < currentStopLoss) {
    const improvement = ((currentStopLoss - newStopLoss) / currentStopLoss) * 100;
    return { shouldUpdate: true, newStopLoss, reason: isProfitable ? `stop moved down ${improvement.toFixed(2)}%, protecting profit` : `stop moved down ${improvement.toFixed(2)}%, improved risk protection` };
  }
  return { shouldUpdate: false, reason: `short stop can't move up (new ${newStopLoss.toFixed(6)} >= current ${currentStopLoss.toFixed(6)})` };
}

module.exports = { DEFAULT_CONFIG, calculateScientificStopLoss, shouldOpenPosition, updateTrailingStopLoss, findSupportLevel, findResistanceLevel };
