// Faithful port of takeProfitManagement.ts's ACTUAL enforced behavior -
// NOT the per-strategy descriptions in tradingAgent.ts's prompt text, which
// are never actually read by this logic (confirmed: no reference to
// getStrategyParams or partialTakeProfit config anywhere in the source).
// Always 1R/2R/3R at 33.33%/33.33%/0%, adjusted only by volatility.

const { atrWilder } = require("./indicators");

function analyzeMarketVolatility(candles) {
  if (!candles || candles.length < 15) {
    return { level: "NORMAL", atrPercent: 3.0, atr14: 0, adjustmentFactor: 1.0, description: "insufficient data, using default normal volatility" };
  }
  const atr14 = atrWilder(candles, 14);
  const currentPrice = candles[candles.length - 1].close;
  const atrPercent = (atr14 / currentPrice) * 100;

  let level, adjustmentFactor, description;
  if (atrPercent < 2) { level = "LOW"; adjustmentFactor = 0.8; description = "low volatility, tightening take-profit targets, banking gains quickly"; }
  else if (atrPercent < 5) { level = "NORMAL"; adjustmentFactor = 1.0; description = "normal volatility, standard take-profit configuration"; }
  else if (atrPercent < 8) { level = "HIGH"; adjustmentFactor = 1.2; description = "high volatility, widening take-profit targets, letting profit run"; }
  else { level = "EXTREME"; adjustmentFactor = 1.5; description = "extreme volatility, significantly widening take-profit, capturing the big move"; }

  return { level, atrPercent: Number(atrPercent.toFixed(2)), atr14, adjustmentFactor, description };
}

function adjustRMultipleForVolatility(baseRMultiple, volatility) {
  return baseRMultiple * volatility.adjustmentFactor;
}

function calculateRMultiple(entryPrice, currentPrice, stopLossPrice, side) {
  const riskDistance = Math.abs(entryPrice - stopLossPrice);
  if (riskDistance === 0) return 0;
  const profitDistance = side === "long" ? currentPrice - entryPrice : entryPrice - currentPrice;
  return profitDistance / riskDistance;
}

function calculateTargetPrice(entryPrice, stopLossPrice, rMultiple, side) {
  const riskDistance = Math.abs(entryPrice - stopLossPrice);
  const targetDistance = riskDistance * rMultiple;
  return side === "long" ? entryPrice + targetDistance : entryPrice - targetDistance;
}

// Checks a position against the staged plan. `stagesAdvised` is a plain
// object like { "1": true } tracking what's already been done for this
// position (replaces the original's SQLite history query).
function checkPartialTakeProfitOpportunity(entryPrice, currentPrice, originalStopLoss, side, stagesAdvised, candles15m) {
  const volatility = analyzeMarketVolatility(candles15m);
  const currentR = calculateRMultiple(entryPrice, currentPrice, originalStopLoss, side);

  const adjustedR1 = adjustRMultipleForVolatility(1, volatility);
  const adjustedR2 = adjustRMultipleForVolatility(2, volatility);
  const adjustedR3 = adjustRMultipleForVolatility(3, volatility);

  const executedStages = Object.keys(stagesAdvised).filter((k) => stagesAdvised[k]).map(Number);
  let canExecuteStage = null;
  let recommendation = "";

  if (currentR >= adjustedR3 && !executedStages.includes(3) && executedStages.includes(1) && executedStages.includes(2)) {
    canExecuteStage = 3;
    recommendation = `stage 3 ready (${adjustedR3.toFixed(2)}R, ${volatility.description})`;
  } else if (currentR >= adjustedR2 && !executedStages.includes(2) && executedStages.includes(1)) {
    canExecuteStage = 2;
    recommendation = `stage 2 ready (${adjustedR2.toFixed(2)}R, ${volatility.description})`;
  } else if (currentR >= adjustedR1 && !executedStages.includes(1)) {
    canExecuteStage = 1;
    recommendation = `stage 1 ready (${adjustedR1.toFixed(2)}R, ${volatility.description})`;
  }

  if (!canExecuteStage) {
    if (executedStages.includes(3)) recommendation = "all stages complete, use trailing stop for the rest";
    else if (executedStages.includes(2)) recommendation = `R=${currentR.toFixed(2)}, stages 1-2 done, waiting for stage 3 (${adjustedR3.toFixed(2)}R, ${volatility.level} vol)`;
    else if (executedStages.includes(1)) recommendation = `R=${currentR.toFixed(2)}, stage 1 done, waiting for stage 2 (${adjustedR2.toFixed(2)}R, ${volatility.level} vol)`;
    else recommendation = `R=${currentR.toFixed(2)}, waiting for stage 1 (${adjustedR1.toFixed(2)}R, ${volatility.level} vol)`;
  }

  let closePercent = 0;
  let newStop;
  if (canExecuteStage === 1) { closePercent = 33.33; newStop = entryPrice; }
  else if (canExecuteStage === 2) { closePercent = 33.33; newStop = calculateTargetPrice(entryPrice, originalStopLoss, 1, side); }
  else if (canExecuteStage === 3) { closePercent = 0; newStop = calculateTargetPrice(entryPrice, originalStopLoss, 2, side); }

  return {
    canExecute: canExecuteStage !== null,
    stage: canExecuteStage,
    currentR: Number(currentR.toFixed(2)),
    closePercent,
    newStop: newStop !== undefined ? Number(newStop.toFixed(6)) : undefined,
    volatility: { level: volatility.level, atrPercent: volatility.atrPercent, adjustmentFactor: volatility.adjustmentFactor },
    adjustedThresholds: { stage1: Number(adjustedR1.toFixed(2)), stage2: Number(adjustedR2.toFixed(2)), stage3: Number(adjustedR3.toFixed(2)) },
    recommendation,
  };
}

module.exports = { analyzeMarketVolatility, adjustRMultipleForVolatility, calculateRMultiple, calculateTargetPrice, checkPartialTakeProfitOpportunity };
