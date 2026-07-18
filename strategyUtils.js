// Faithful port of strategyUtils.ts. Field names and formulas match the
// original exactly - do not "simplify" without checking against source.

// Signal strength (0-1): RSI7 extremity (25pt) + MACD direction magnitude (20pt)
// + EMA alignment (25pt) + price position vs EMA20 (15pt) + trend consistency (15pt)
function calculateSignalStrength({ rsi7, macd, macdSignal, emaAlignment, pricePosition, trendConsistency }) {
  let score = 0;
  const maxScore = 100;

  // RSI7 (25pt): oversold <25 or overbought >75 score by extremity; 30-70 gets a flat 15
  if (rsi7 < 25) score += (25 * (25 - rsi7)) / 25;
  else if (rsi7 > 75) score += (25 * (rsi7 - 75)) / 25;
  else if (rsi7 >= 30 && rsi7 <= 70) score += 15;

  // MACD direction confirmation (20pt)
  const macdDiff = macd - macdSignal;
  if (Math.abs(macdDiff) > 0) score += 20 * Math.min(Math.abs(macdDiff) / 100, 1);

  // EMA alignment (25pt) - all or nothing
  if (emaAlignment) score += 25;

  // Price position vs EMA20 (15pt) - closer to 0% deviation scores higher, caps at 3%
  const absDeviation = Math.abs(pricePosition);
  if (absDeviation < 3) score += 15 * (1 - absDeviation / 3);

  // Trend consistency (15pt) - direct scaling of the 0-1 alignment score
  score += 15 * trendConsistency;

  return Math.min(score / maxScore, 1);
}

// Multi-timeframe alignment check (0-1 score, aligned if >= 0.6)
function checkMultiTimeframeAlignment(tf15m, tf1h, direction) {
  let alignmentScore = 0;

  const ema15m = tf15m.ema20 > tf15m.ema50;
  const ema1h = tf1h.ema20 > tf1h.ema50;
  if (direction === "long" && ema15m && ema1h) alignmentScore += 30;
  else if (direction === "short" && !ema15m && !ema1h) alignmentScore += 30;
  else if (direction === "long" && ema1h) alignmentScore += 15;
  else if (direction === "short" && !ema1h) alignmentScore += 15;

  const macd15m = tf15m.macd > 0;
  const macd1h = tf1h.macd > 0;
  if (direction === "long" && macd1h) {
    alignmentScore += 25;
    if (macd15m) alignmentScore += 10;
  } else if (direction === "short" && !macd1h) {
    alignmentScore += 25;
    if (!macd15m) alignmentScore += 10;
  }

  if (direction === "long") {
    if (tf1h.rsi14 < 70) alignmentScore += 15;
    if (tf15m.rsi7 < 30) alignmentScore += 10;
  } else {
    if (tf1h.rsi14 > 30) alignmentScore += 15;
    if (tf15m.rsi7 > 70) alignmentScore += 10;
  }

  if (direction === "long" && tf1h.close > tf1h.ema20) alignmentScore += 10;
  else if (direction === "short" && tf1h.close < tf1h.ema20) alignmentScore += 10;

  const finalScore = alignmentScore / 100;
  return { aligned: finalScore >= 0.6, score: finalScore };
}

// Volatility adjustment based on atrRatio (current ATR / historical ATR, NOT % of price)
function calculateVolatilityAdjustment(atr, atrMa = 1.0) {
  const ratio = atr / atrMa;
  if (ratio < 0.8) return { adjustment: 1.2, leverageMultiplier: 1.0, status: "low" };
  if (ratio < 1.2) return { adjustment: 1.0, leverageMultiplier: 1.0, status: "normal" };
  if (ratio < 1.8) return { adjustment: 0.8, leverageMultiplier: 0.8, status: "high" };
  return { adjustment: 0.6, leverageMultiplier: 0.6, status: "extreme" };
}

// Recommended leverage: base * signalStrength * volatilityMultiplier, clamped 2..max
function calculateRecommendedLeverage(baseLeverage, signalStrength, volatilityAdjustment, maxLeverage = 10) {
  const adjusted = baseLeverage * signalStrength * volatilityAdjustment;
  return Number(Math.max(2, Math.min(adjusted, maxLeverage)).toFixed(1));
}

// MACD histogram reversal detection for mean-reversion (bullish: neg->rising, bearish: pos->falling)
function detectMacdHistogramReversal(currentHist, previousHist, direction) {
  if (direction === "bullish") return currentHist > previousHist && previousHist < 0;
  return currentHist < previousHist && previousHist > 0;
}

// Key support/resistance levels over a lookback window (for breakout strategy)
function identifyKeyLevels(candles, lookback = 20) {
  const recent = candles.slice(-lookback);
  const highs = recent.map((c) => c.high);
  const lows = recent.map((c) => c.low);
  const resistance = Math.max(...highs);
  const support = Math.min(...lows);
  return { resistance, support, range: resistance - support };
}

// Volume spike detection (for breakout confirmation)
function detectVolumeSpike(currentVolume, avgVolume, threshold = 1.5) {
  const ratio = currentVolume / avgVolume;
  const isSpike = ratio >= threshold;
  let level;
  if (ratio >= 3.0) level = "extreme";
  else if (ratio >= 2.0) level = "significant";
  else if (ratio >= 1.5) level = "moderate";
  else level = "normal";
  return { isSpike, ratio: Number(ratio.toFixed(2)), level };
}

module.exports = {
  calculateSignalStrength,
  checkMultiTimeframeAlignment,
  calculateVolatilityAdjustment,
  calculateRecommendedLeverage,
  detectMacdHistogramReversal,
  identifyKeyLevels,
  detectVolumeSpike,
};
