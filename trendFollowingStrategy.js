// Faithful port of trendFollowingStrategy.ts.
const { calculateSignalStrength, checkMultiTimeframeAlignment, calculateVolatilityAdjustment, calculateRecommendedLeverage } = require("./strategyUtils");

function extractKeyMetrics(tf15m, tf1h) {
  return {
    rsi7: tf15m.rsi7, rsi14: tf15m.rsi14, macd: tf1h.macd,
    ema20: tf1h.ema20, ema50: tf1h.ema50, price: tf15m.close,
    atrRatio: 1.0,
    priceDeviationFromEma20: ((tf15m.close - tf15m.ema20) / tf15m.ema20) * 100,
  };
}

function trendFollowingLongSignal(symbol, tf15m, tf1h, marketState, maxLeverage = 10) {
  const warnings = [];
  let signalStrength = 0;

  const trendConfirmed = tf1h.ema20 > tf1h.ema50;
  if (!trendConfirmed) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "trend_following", reason: "no 1h uptrend", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }
  if (!(tf1h.macd > 0)) warnings.push("1h MACD negative, weak momentum");

  if (marketState.state === "uptrend_continuation" && tf15m.rsi7 >= 45 && tf15m.rsi7 <= 65) {
    signalStrength = 0.5;
    warnings.push("uptrend continuation, RSI neutral, steady long opportunity");
  } else {
    const oversold = tf15m.rsi7 < 40;
    if (!oversold) {
      return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "trend_following", reason: "15m RSI7 not oversold, waiting for pullback", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
    }
    if (!(tf15m.close >= tf15m.ema20 * 0.995)) warnings.push("price still below EMA20, pullback may not be over");

    const alignment = checkMultiTimeframeAlignment(tf15m, tf1h, "long");
    signalStrength = calculateSignalStrength({
      rsi7: tf15m.rsi7, macd: tf1h.macd, macdSignal: tf1h.macdSignal,
      emaAlignment: trendConfirmed,
      pricePosition: ((tf15m.close - tf15m.ema20) / tf15m.ema20) * 100,
      trendConsistency: alignment.score,
    });
  }

  const volAdj = calculateVolatilityAdjustment(marketState.keyMetrics.atr_ratio, 1.0);
  if (volAdj.status === "extreme") { warnings.push("extreme volatility, reduce size or wait"); signalStrength *= 0.7; }
  else if (volAdj.status === "high") { warnings.push("high volatility, be cautious"); signalStrength *= 0.85; }

  const recommendedLeverage = calculateRecommendedLeverage(5, signalStrength, volAdj.leverageMultiplier, maxLeverage);

  let reason = "trend-following long: ";
  if (marketState.state === "uptrend_continuation" && tf15m.rsi7 >= 45 && tf15m.rsi7 <= 65) {
    reason += `uptrend continuation, 1h confirmed, 15m RSI7 neutral (${tf15m.rsi7.toFixed(1)})`;
  } else {
    reason += `1h uptrend confirmed, 15m RSI7 pullback (${tf15m.rsi7.toFixed(1)}), signal ${(signalStrength * 100).toFixed(0)}%`;
  }
  if (warnings.length) reason += ` [${warnings.join("; ")}]`;

  return { symbol, action: "long", confidence: signalStrength >= 0.7 ? "high" : signalStrength >= 0.5 ? "medium" : "low", signalStrength, recommendedLeverage, marketState: marketState.state, strategyType: "trend_following", reason, warnings, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
}

function trendFollowingShortSignal(symbol, tf15m, tf1h, marketState, maxLeverage = 10) {
  const warnings = [];
  let signalStrength = 0;

  const trendConfirmed = tf1h.ema20 < tf1h.ema50;
  if (!trendConfirmed) {
    return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "trend_following", reason: "no 1h downtrend", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
  }
  if (!(tf1h.macd < 0)) warnings.push("1h MACD positive, weak momentum");

  if (marketState.state === "downtrend_continuation" && tf15m.rsi7 >= 35 && tf15m.rsi7 <= 55) {
    signalStrength = 0.5;
    warnings.push("downtrend continuation, RSI neutral, steady short opportunity");
  } else {
    const overbought = tf15m.rsi7 > 60;
    if (!overbought) {
      return { symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0, marketState: marketState.state, strategyType: "trend_following", reason: "15m RSI7 not overbought, waiting for bounce", keyMetrics: extractKeyMetrics(tf15m, tf1h) };
    }
    if (!(tf15m.close <= tf15m.ema20 * 1.005)) warnings.push("price still above EMA20, bounce may not be over");

    const alignment = checkMultiTimeframeAlignment(tf15m, tf1h, "short");
    signalStrength = calculateSignalStrength({
      rsi7: tf15m.rsi7, macd: tf1h.macd, macdSignal: tf1h.macdSignal,
      emaAlignment: trendConfirmed,
      pricePosition: ((tf15m.close - tf15m.ema20) / tf15m.ema20) * 100,
      trendConsistency: alignment.score,
    });
  }

  const volAdj = calculateVolatilityAdjustment(marketState.keyMetrics.atr_ratio, 1.0);
  if (volAdj.status === "extreme") { warnings.push("extreme volatility, reduce size or wait"); signalStrength *= 0.7; }
  else if (volAdj.status === "high") { warnings.push("high volatility, be cautious"); signalStrength *= 0.85; }

  const recommendedLeverage = calculateRecommendedLeverage(5, signalStrength, volAdj.leverageMultiplier, maxLeverage);

  let reason = "trend-following short: ";
  if (marketState.state === "downtrend_continuation" && tf15m.rsi7 >= 35 && tf15m.rsi7 <= 55) {
    reason += `downtrend continuation, 1h confirmed, 15m RSI7 neutral (${tf15m.rsi7.toFixed(1)})`;
  } else {
    reason += `1h downtrend confirmed, 15m RSI7 bounce (${tf15m.rsi7.toFixed(1)}), signal ${(signalStrength * 100).toFixed(0)}%`;
  }
  if (warnings.length) reason += ` [${warnings.join("; ")}]`;

  return { symbol, action: "short", confidence: signalStrength >= 0.7 ? "high" : signalStrength >= 0.5 ? "medium" : "low", signalStrength, recommendedLeverage, marketState: marketState.state, strategyType: "trend_following", reason, warnings, keyMetrics: extractKeyMetrics(tf15m, tf1h) };
}

// Wrapper matching strategyRouter.ts's call signature
function trendFollowingStrategy(symbol, direction, marketState, tf15mRaw, tf1hRaw, maxLeverage) {
  const tf15m = { close: tf15mRaw.currentPrice, ema20: tf15mRaw.ema20, ema50: tf15mRaw.ema50, macd: tf15mRaw.macd, macdSignal: tf15mRaw.macdSignal || 0, rsi7: tf15mRaw.rsi7, rsi14: tf15mRaw.rsi14 };
  const tf1h = { close: tf1hRaw.currentPrice, ema20: tf1hRaw.ema20, ema50: tf1hRaw.ema50, macd: tf1hRaw.macd, macdSignal: tf1hRaw.macdSignal || 0, rsi7: tf1hRaw.rsi7, rsi14: tf1hRaw.rsi14 };
  return direction === "long"
    ? trendFollowingLongSignal(symbol, tf15m, tf1h, marketState, maxLeverage)
    : trendFollowingShortSignal(symbol, tf15m, tf1h, marketState, maxLeverage);
}

module.exports = { trendFollowingLongSignal, trendFollowingShortSignal, trendFollowingStrategy };
