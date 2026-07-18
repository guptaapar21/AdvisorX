// Faithful port of meanReversionStrategy.ts.
//
// BUG FIX (per explicit request): the original wrapper sets
// `timeframe15m.macdHistogram` but the signal functions check
// `.macdHist` - a field-name mismatch that means the MACD-reversal
// bonus can never fire in the original. Fixed here by using one
// consistent field name (`macdHist`) throughout.

const { calculateSignalStrength, checkMultiTimeframeAlignment, calculateVolatilityAdjustment, calculateRecommendedLeverage, detectMacdHistogramReversal } = require("./strategyUtils");

function extractKeyMetrics(tf15m, tf1h) {
  return {
    rsi7: tf15m.rsi7, rsi14: tf15m.rsi14, macd: tf15m.macd,
    ema20: tf1h.ema20, ema50: tf1h.ema50, price: tf15m.close,
    atrRatio: 1.0,
    priceDeviationFromEma20: ((tf15m.close - tf15m.ema20) / tf15m.ema20) * 100,
  };
}

function meanReversionLongSignal(symbol, tf15m, tf1h, marketState, maxLeverage = 10) {
  const warnings = [];
  let signalStrength = 0;

  const extremeOversold = tf15m.rsi7 < 35;
  if (!extremeOversold) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "mean_reversion", reason: "15m RSI7 not extreme oversold (<35)", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }

  const nearLowerBB = marketState.keyMetrics.priceVsLowerBB < 0.1;
  if (!nearLowerBB) warnings.push("price hasn't reached lower Bollinger band, not extreme enough");

  let macdReversal = false;
  if (tf15m.macdHist != null && tf15m.prevMacdHist !== undefined) {
    macdReversal = detectMacdHistogramReversal(tf15m.macdHist, tf15m.prevMacdHist, "bullish");
    if (!macdReversal) warnings.push("MACD histogram hasn't shown a bullish reversal yet");
  }

  const strongDowntrend = tf1h.ema20 < tf1h.ema50 && tf1h.macd < -50;
  if (strongDowntrend) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "mean_reversion", reason: "1h strong downtrend, avoiding catching a falling knife", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }

  const alignment = checkMultiTimeframeAlignment(tf15m, tf1h, "long");
  signalStrength = calculateSignalStrength({
    rsi7: tf15m.rsi7, macd: tf15m.macd, macdSignal: tf15m.macdSignal,
    emaAlignment: tf1h.ema20 > tf1h.ema50,
    pricePosition: ((tf15m.close - tf15m.ema20) / tf15m.ema20) * 100,
    trendConsistency: alignment.score * 0.7,
  });

  if (tf15m.rsi7 < 25) signalStrength = Math.min(signalStrength * 1.2, 1.0);
  if (macdReversal) signalStrength = Math.min(signalStrength * 1.15, 1.0);

  const volAdj = calculateVolatilityAdjustment(marketState.keyMetrics.atr_ratio, 1.0);
  if (volAdj.status === "extreme") { warnings.push("extreme volatility, reduce size"); signalStrength *= 0.6; }
  else if (volAdj.status === "high") { warnings.push("high volatility"); signalStrength *= 0.8; }

  const recommendedLeverage = calculateRecommendedLeverage(3, signalStrength, volAdj.leverageMultiplier, Math.min(maxLeverage, 5));

  let reason = `mean-reversion long: 15m RSI7 extreme oversold (${tf15m.rsi7.toFixed(1)}), `;
  if (nearLowerBB) reason += "price at lower Bollinger band, ";
  if (macdReversal) reason += "MACD histogram bullish reversal, ";
  reason += `signal ${(signalStrength * 100).toFixed(0)}%`;
  if (warnings.length) reason += ` [${warnings.join("; ")}]`;

  return { symbol, action: "long", confidence: signalStrength >= 0.7 ? "high" : signalStrength >= 0.5 ? "medium" : "low", signalStrength, recommendedLeverage, marketState: marketState.state, strategyType: "mean_reversion", reason, warnings, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
}

function meanReversionShortSignal(symbol, tf15m, tf1h, marketState, maxLeverage = 10) {
  const warnings = [];
  let signalStrength = 0;

  const extremeOverbought = tf15m.rsi7 > 65;
  if (!extremeOverbought) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "mean_reversion", reason: "15m RSI7 not extreme overbought (>65)", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }

  const nearUpperBB = marketState.keyMetrics.priceVsUpperBB > 0.9;
  if (!nearUpperBB) warnings.push("price hasn't reached upper Bollinger band, not extreme enough");

  let macdReversal = false;
  if (tf15m.macdHist != null && tf15m.prevMacdHist !== undefined) {
    macdReversal = detectMacdHistogramReversal(tf15m.macdHist, tf15m.prevMacdHist, "bearish");
    if (!macdReversal) warnings.push("MACD histogram hasn't shown a bearish reversal yet");
  }

  const strongUptrend = tf1h.ema20 > tf1h.ema50 && tf1h.macd > 50;
  if (strongUptrend) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "mean_reversion", reason: "1h strong uptrend, avoiding shorting into strength", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }

  const alignment = checkMultiTimeframeAlignment(tf15m, tf1h, "short");
  signalStrength = calculateSignalStrength({
    rsi7: tf15m.rsi7, macd: tf15m.macd, macdSignal: tf15m.macdSignal,
    emaAlignment: tf1h.ema20 < tf1h.ema50,
    pricePosition: ((tf15m.close - tf15m.ema20) / tf15m.ema20) * 100,
    trendConsistency: alignment.score * 0.7,
  });

  if (tf15m.rsi7 > 75) signalStrength = Math.min(signalStrength * 1.2, 1.0);
  if (macdReversal) signalStrength = Math.min(signalStrength * 1.15, 1.0);

  const volAdj = calculateVolatilityAdjustment(marketState.keyMetrics.atr_ratio, 1.0);
  if (volAdj.status === "extreme") { warnings.push("extreme volatility, reduce size"); signalStrength *= 0.6; }
  else if (volAdj.status === "high") { warnings.push("high volatility"); signalStrength *= 0.8; }

  const recommendedLeverage = calculateRecommendedLeverage(10, signalStrength, volAdj.leverageMultiplier, Math.min(maxLeverage, 5));

  let reason = `mean-reversion short: 15m RSI7 extreme overbought (${tf15m.rsi7.toFixed(1)}), `;
  if (nearUpperBB) reason += "price at upper Bollinger band, ";
  if (macdReversal) reason += "MACD histogram bearish reversal, ";
  reason += `signal ${(signalStrength * 100).toFixed(0)}%`;
  if (warnings.length) reason += ` [${warnings.join("; ")}]`;

  return { symbol, action: "short", confidence: signalStrength >= 0.7 ? "high" : signalStrength >= 0.5 ? "medium" : "low", signalStrength, recommendedLeverage, marketState: marketState.state, strategyType: "mean_reversion", reason, warnings, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
}

// Wrapper matching strategyRouter.ts's call signature
function meanReversionStrategy(symbol, direction, marketState, tf15mRaw, tf1hRaw, maxLeverage) {
  const tf15m = {
    close: tf15mRaw.currentPrice, ema20: tf15mRaw.ema20, ema50: tf15mRaw.ema50,
    macd: tf15mRaw.macd, macdSignal: tf15mRaw.macdSignal || 0, rsi7: tf15mRaw.rsi7, rsi14: tf15mRaw.rsi14,
    bollingerUpper: tf15mRaw.bollingerUpper, bollingerLower: tf15mRaw.bollingerLower, bollingerMiddle: tf15mRaw.bollingerMiddle,
    macdHist: tf15mRaw.macdHistogram, // fixed field name
    prevMacdHist: tf15mRaw.prevMacdHistogram, // fixed: original never supplied this at all
  };
  const tf1h = { close: tf1hRaw.currentPrice, ema20: tf1hRaw.ema20, ema50: tf1hRaw.ema50, macd: tf1hRaw.macd, macdSignal: tf1hRaw.macdSignal || 0, rsi7: tf1hRaw.rsi7, rsi14: tf1hRaw.rsi14 };
  return direction === "long"
    ? meanReversionLongSignal(symbol, tf15m, tf1h, marketState, maxLeverage)
    : meanReversionShortSignal(symbol, tf15m, tf1h, marketState, maxLeverage);
}

module.exports = { meanReversionLongSignal, meanReversionShortSignal, meanReversionStrategy };
