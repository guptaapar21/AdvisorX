// Faithful port of breakoutStrategy.ts.
//
// NOTE: in the original repo, this strategy is fully implemented but never
// actually called by strategyRouter.ts or opportunityAnalysis.ts - it's
// dead code there. Per explicit request, it IS wired in here (see
// strategyRouter.js), and every signal this produces is tagged
// isBreakoutExtension: true so it's clearly flagged in alerts as something
// the original bot never actually did.

const { calculateSignalStrength, checkMultiTimeframeAlignment, calculateVolatilityAdjustment, calculateRecommendedLeverage, identifyKeyLevels, detectVolumeSpike } = require("./strategyUtils");

function extractKeyMetrics(tf15m, tf1h) {
  return {
    rsi7: tf15m.rsi7, rsi14: tf15m.rsi14, macd: tf15m.macd,
    ema20: tf1h.ema20, ema50: tf1h.ema50, price: tf15m.close,
    atrRatio: 1.0,
    priceDeviationFromEma20: ((tf15m.close - tf15m.ema20) / tf15m.ema20) * 100,
  };
}

function breakoutLongSignal(symbol, tf15m, tf1h, marketState, maxLeverage = 10) {
  const warnings = [];
  let signalStrength = 0;

  if (!tf15m.candles || tf15m.candles.length < 20) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "breakout", reason: "insufficient candle history to identify resistance", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }

  const levels = identifyKeyLevels(tf15m.candles, 20);
  const currentPrice = tf15m.close;
  const resistanceBreakout = currentPrice > levels.resistance * 0.998;
  if (!resistanceBreakout) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "breakout", reason: `price ${currentPrice.toFixed(2)} hasn't broken resistance ${levels.resistance.toFixed(2)}`, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }

  let volumeConfirmation = false;
  if (tf15m.volume && tf15m.avgVolume) {
    const v = detectVolumeSpike(tf15m.volume, tf15m.avgVolume, 1.5);
    volumeConfirmation = v.isSpike;
    if (!volumeConfirmation) warnings.push(`volume not confirming (only ${v.ratio}x), breakout may fail`);
    else if (v.level === "extreme") signalStrength += 0.1;
  } else {
    warnings.push("no volume data, can't confirm breakout validity");
  }

  if (!(tf1h.macd > 0)) warnings.push("1h MACD negative, breakout trend weak");

  const rsiInRange = tf15m.rsi7 >= 35 && tf15m.rsi7 <= 75;
  if (!rsiInRange) {
    if (tf15m.rsi7 > 75) { warnings.push(`RSI7 too high (${tf15m.rsi7.toFixed(1)}), may be chasing`); signalStrength *= 0.8; }
    else {
      return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "breakout", reason: `RSI7 too low (${tf15m.rsi7.toFixed(1)}), breakout may fail`, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
    }
  }

  const alignment = checkMultiTimeframeAlignment(tf15m, tf1h, "long");
  signalStrength = calculateSignalStrength({
    rsi7: tf15m.rsi7, macd: tf1h.macd, macdSignal: tf1h.macdSignal,
    emaAlignment: tf1h.ema20 > tf1h.ema50,
    pricePosition: ((currentPrice - levels.resistance) / levels.resistance) * 100,
    trendConsistency: alignment.score,
  });
  if (volumeConfirmation) signalStrength = Math.min(signalStrength * 1.25, 1.0);

  const volAdj = calculateVolatilityAdjustment(marketState.keyMetrics.atr_ratio, 1.0);
  if (volAdj.status === "extreme") { warnings.push("extreme volatility, false-breakout risk high"); signalStrength *= 0.7; }
  else if (volAdj.status === "high") { warnings.push("high volatility"); signalStrength *= 0.85; }

  const recommendedLeverage = calculateRecommendedLeverage(4, signalStrength, volAdj.leverageMultiplier, maxLeverage);

  let reason = `breakout long: broke resistance ${levels.resistance.toFixed(2)}, `;
  if (volumeConfirmation) reason += "volume confirmed, ";
  reason += `signal ${(signalStrength * 100).toFixed(0)}%`;
  if (warnings.length) reason += ` [${warnings.join("; ")}]`;

  return { symbol, action: "long", confidence: signalStrength >= 0.7 ? "high" : signalStrength >= 0.5 ? "medium" : "low", signalStrength, recommendedLeverage, marketState: marketState.state, strategyType: "breakout", reason, warnings, isBreakoutExtension: true, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
}

function breakoutShortSignal(symbol, tf15m, tf1h, marketState, maxLeverage = 10) {
  const warnings = [];
  let signalStrength = 0;

  if (!tf15m.candles || tf15m.candles.length < 20) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "breakout", reason: "insufficient candle history to identify support", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }

  const levels = identifyKeyLevels(tf15m.candles, 20);
  const currentPrice = tf15m.close;
  const supportBreakdown = currentPrice < levels.support * 1.002;
  if (!supportBreakdown) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "breakout", reason: `price ${currentPrice.toFixed(2)} hasn't broken support ${levels.support.toFixed(2)}`, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }

  let volumeConfirmation = false;
  if (tf15m.volume && tf15m.avgVolume) {
    const v = detectVolumeSpike(tf15m.volume, tf15m.avgVolume, 1.5);
    volumeConfirmation = v.isSpike;
    if (!volumeConfirmation) warnings.push(`volume not confirming (only ${v.ratio}x), breakdown may fail`);
    else if (v.level === "extreme") signalStrength += 0.1;
  } else {
    warnings.push("no volume data, can't confirm breakdown validity");
  }

  if (!(tf1h.macd < 0)) warnings.push("1h MACD positive, breakdown trend weak");

  const rsiInRange = tf15m.rsi7 >= 25 && tf15m.rsi7 <= 65;
  if (!rsiInRange) {
    if (tf15m.rsi7 < 25) { warnings.push(`RSI7 too low (${tf15m.rsi7.toFixed(1)}), may bounce`); signalStrength *= 0.8; }
    else {
      return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "breakout", reason: `RSI7 too high (${tf15m.rsi7.toFixed(1)}), breakdown may fail`, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
    }
  }

  const alignment = checkMultiTimeframeAlignment(tf15m, tf1h, "short");
  signalStrength = calculateSignalStrength({
    rsi7: tf15m.rsi7, macd: tf1h.macd, macdSignal: tf1h.macdSignal,
    emaAlignment: tf1h.ema20 < tf1h.ema50,
    pricePosition: ((currentPrice - levels.support) / levels.support) * 100,
    trendConsistency: alignment.score,
  });
  if (volumeConfirmation) signalStrength = Math.min(signalStrength * 1.25, 1.0);

  const volAdj = calculateVolatilityAdjustment(marketState.keyMetrics.atr_ratio, 1.0);
  if (volAdj.status === "extreme") { warnings.push("extreme volatility, false-breakdown risk high"); signalStrength *= 0.7; }
  else if (volAdj.status === "high") { warnings.push("high volatility"); signalStrength *= 0.85; }

  const recommendedLeverage = calculateRecommendedLeverage(4, signalStrength, volAdj.leverageMultiplier, maxLeverage);

  let reason = `breakout short: broke support ${levels.support.toFixed(2)}, `;
  if (volumeConfirmation) reason += "volume confirmed, ";
  reason += `signal ${(signalStrength * 100).toFixed(0)}%`;
  if (warnings.length) reason += ` [${warnings.join("; ")}]`;

  return { symbol, action: "short", confidence: signalStrength >= 0.7 ? "high" : signalStrength >= 0.5 ? "medium" : "low", signalStrength, recommendedLeverage, marketState: marketState.state, strategyType: "breakout", reason, warnings, isBreakoutExtension: true, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
}

// Wrapper matching strategyRouter.ts's call signature
function breakoutStrategy(symbol, direction, marketState, tf15mRaw, tf1hRaw, maxLeverage) {
  const tf15m = {
    close: tf15mRaw.currentPrice, ema20: tf15mRaw.ema20, ema50: tf15mRaw.ema50,
    macd: tf15mRaw.macd, macdSignal: tf15mRaw.macdSignal || 0, rsi7: tf15mRaw.rsi7, rsi14: tf15mRaw.rsi14,
    candles: tf15mRaw.candles || [], volume: tf15mRaw.volume, avgVolume: tf15mRaw.avgVolume,
  };
  const tf1h = { close: tf1hRaw.currentPrice, ema20: tf1hRaw.ema20, ema50: tf1hRaw.ema50, macd: tf1hRaw.macd, macdSignal: tf1hRaw.macdSignal || 0, rsi7: tf1hRaw.rsi7, rsi14: tf1hRaw.rsi14 };
  return direction === "long"
    ? breakoutLongSignal(symbol, tf15m, tf1h, marketState, maxLeverage)
    : breakoutShortSignal(symbol, tf15m, tf1h, marketState, maxLeverage);
}

module.exports = { breakoutLongSignal, breakoutShortSignal, breakoutStrategy };
