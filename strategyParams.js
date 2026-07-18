// Faithful port of tradingAgent.ts's getStrategyParams(). Leverage bounds
// are computed as percentages of maxLeverage, exactly matching source.
// Only scientificStopLoss and positionSize fields are used by our agent
// (the legacy fixed-% stopLoss/trailingStop/partialTakeProfit fields in the
// original are either deprecated or - confirmed via grep - never actually
// read by the real execution code, so they're omitted here rather than
// carried over as unused dead config).

function getStrategyParams(strategy, maxLeverage = 15) {
  const conservativeLevMin = Math.max(1, Math.ceil(maxLeverage * 0.3));
  const conservativeLevMax = Math.max(2, Math.ceil(maxLeverage * 0.6));
  const balancedLevMin = Math.max(2, Math.ceil(maxLeverage * 0.6));
  const balancedLevMax = Math.max(3, Math.ceil(maxLeverage * 0.85));
  const aggressiveLevMin = Math.max(3, Math.ceil(maxLeverage * 0.85));
  const aggressiveLevMax = maxLeverage;

  const presets = {
    "ultra-short": {
      name: "Ultra-short",
      leverageMin: Math.max(3, Math.ceil(maxLeverage * 0.5)),
      leverageMax: Math.max(5, Math.ceil(maxLeverage * 0.75)),
      positionSizeMin: 18, positionSizeMax: 25,
      scientificStopLoss: { atrMultiplier: 1.5, minDistance: 0.3, maxDistance: 2.0 },
      volatilityAdjustment: {
        high: { leverageFactor: 0.7, positionFactor: 0.8 },
        normal: { leverageFactor: 1.0, positionFactor: 1.0 },
        low: { leverageFactor: 1.1, positionFactor: 1.0 },
      },
    },
    "swing-trend": {
      name: "Swing-trend",
      leverageMin: Math.max(2, Math.ceil(maxLeverage * 0.2)),
      leverageMax: Math.max(5, Math.ceil(maxLeverage * 0.5)),
      positionSizeMin: 12, positionSizeMax: 20,
      scientificStopLoss: { atrMultiplier: 2.5, minDistance: 1.0, maxDistance: 6.0 },
      volatilityAdjustment: {
        high: { leverageFactor: 0.5, positionFactor: 0.6 },
        normal: { leverageFactor: 1.0, positionFactor: 1.0 },
        low: { leverageFactor: 1.2, positionFactor: 1.1 },
      },
    },
    "conservative": {
      name: "Conservative",
      leverageMin: conservativeLevMin,
      leverageMax: conservativeLevMax,
      positionSizeMin: 15, positionSizeMax: 22,
      scientificStopLoss: { atrMultiplier: 2.5, minDistance: 1.0, maxDistance: 4.0 },
      volatilityAdjustment: {
        high: { leverageFactor: 0.6, positionFactor: 0.7 },
        normal: { leverageFactor: 1.0, positionFactor: 1.0 },
        low: { leverageFactor: 1.0, positionFactor: 1.0 },
      },
    },
    "balanced": {
      name: "Balanced",
      leverageMin: balancedLevMin,
      leverageMax: balancedLevMax,
      positionSizeMin: 10, positionSizeMax: 20,
      scientificStopLoss: { atrMultiplier: 2.0, minDistance: 0.5, maxDistance: 5.0 },
      volatilityAdjustment: {
        high: { leverageFactor: 0.7, positionFactor: 0.8 },
        normal: { leverageFactor: 1.0, positionFactor: 1.0 },
        low: { leverageFactor: 1.1, positionFactor: 1.0 },
      },
    },
    "aggressive": {
      name: "Aggressive",
      leverageMin: aggressiveLevMin,
      leverageMax: aggressiveLevMax,
      positionSizeMin: 25, positionSizeMax: 32,
      scientificStopLoss: { atrMultiplier: 1.5, minDistance: 0.5, maxDistance: 5.0 },
      volatilityAdjustment: {
        high: { leverageFactor: 0.8, positionFactor: 0.85 },
        normal: { leverageFactor: 1.0, positionFactor: 1.0 },
        low: { leverageFactor: 1.2, positionFactor: 1.1 },
      },
    },
  };

  return presets[strategy] || presets.balanced;
}

module.exports = { getStrategyParams };
