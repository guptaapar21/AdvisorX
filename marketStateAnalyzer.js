// Faithful port of marketStateAnalyzer.ts. Every formula below was copied
// directly from the source file, not reconstructed from memory - includes
// full MACD/RSI divergence detection.

const { ema, rsi, macd, atrRatio, macdHistogramTurn, avgVolume } = require("./indicators");

const OVERSOLD_EXTREME_THRESHOLD = 20;
const OVERSOLD_MILD_THRESHOLD = 30;
const OVERBOUGHT_EXTREME_THRESHOLD = 80;
const OVERBOUGHT_MILD_THRESHOLD = 70;

function buildTimeframeIndicators(candles) {
  const closes = candles.map((c) => c.close);
  const currentPrice = closes[closes.length - 1];
  const ema20Arr = ema(closes, 20);
  const ema50Arr = ema(closes, 50);
  const ema20 = ema20Arr[ema20Arr.length - 1] || 0;
  const ema50 = ema50Arr[ema50Arr.length - 1] || 0;
  const m = macd(closes) || { macd: 0, signal: 0, histogram: 0 };
  // One period back, for MACD-histogram-reversal detection (fixes a gap in
  // the original where this was never computed at all).
  const prevM = closes.length > 1 ? macd(closes.slice(0, -1)) : null;
  const deviationFromEMA20 = ema20 !== 0 ? ((currentPrice - ema20) / ema20) * 100 : 0;
  return {
    currentPrice,
    ema20,
    ema50,
    macd: m.macd,
    macdSignal: m.signal,
    macdHistogram: m.histogram,
    prevMacdHistogram: prevM ? prevM.histogram : undefined,
    macdTurn: macdHistogramTurn(closes),
    rsi7: rsi(closes, 7) ?? 50,
    rsi14: rsi(closes, 14) ?? 50,
    atrRatio: atrRatio(candles, 14),
    deviationFromEMA20,
    volume: candles[candles.length - 1].volume,
    avgVolume: avgVolume(candles, 20),
    candles,
  };
}

function determineTrendStrength(tf) {
  if (tf.ema20 > tf.ema50 && tf.macd > 0) return "trending_up";
  if (tf.ema20 < tf.ema50 && tf.macd < 0) return "trending_down";
  return "ranging";
}

function determineMomentumState(tf) {
  if (tf.rsi7 < OVERSOLD_EXTREME_THRESHOLD) return "oversold_extreme";
  if (tf.rsi7 < OVERSOLD_MILD_THRESHOLD) return "oversold_mild";
  if (tf.rsi7 > OVERBOUGHT_EXTREME_THRESHOLD) return "overbought_extreme";
  if (tf.rsi7 > OVERBOUGHT_MILD_THRESHOLD) return "overbought_mild";
  return "neutral";
}

function determineVolatilityState(tf) {
  if (tf.atrRatio > 1.5) return "high_vol";
  if (tf.atrRatio < 0.7) return "low_vol";
  return "normal_vol";
}

function determineMarketState(trendStrength, momentumState, tf15m) {
  let state = "no_clear_signal";
  let confidence = 0.3;

  if (trendStrength === "trending_up" && momentumState === "oversold_extreme") {
    state = "uptrend_oversold"; confidence = 0.9;
  } else if (trendStrength === "trending_down" && momentumState === "overbought_extreme") {
    state = "downtrend_overbought"; confidence = 0.9;
  } else if (trendStrength === "trending_down" && momentumState === "oversold_extreme") {
    state = "downtrend_oversold"; confidence = 0.6;
  } else if (trendStrength === "trending_up" && momentumState === "overbought_extreme") {
    state = "uptrend_overbought"; confidence = 0.6;
  } else if (trendStrength === "trending_up" && (momentumState === "oversold_mild" || momentumState === "neutral")) {
    state = "uptrend_continuation"; confidence = 0.7;
  } else if (trendStrength === "trending_down" && (momentumState === "overbought_mild" || momentumState === "neutral")) {
    state = "downtrend_continuation"; confidence = 0.7;
  } else if (trendStrength === "trending_down" && momentumState === "oversold_mild") {
    state = "downtrend_oversold"; confidence = 0.5;
  } else if (trendStrength === "trending_up" && momentumState === "overbought_mild") {
    state = "uptrend_overbought"; confidence = 0.5;
  } else if (trendStrength === "ranging" && momentumState === "oversold_extreme") {
    state = "ranging_oversold"; confidence = 0.8;
  } else if (trendStrength === "ranging" && momentumState === "overbought_extreme") {
    state = "ranging_overbought"; confidence = 0.8;
  } else if (trendStrength === "ranging" && momentumState === "neutral") {
    state = "ranging_neutral"; confidence = 0.5;
  }

  if (tf15m.macdTurn === 1 && (state === "uptrend_oversold" || state === "ranging_oversold")) {
    confidence = Math.min(confidence + 0.1, 1.0);
  }
  if (tf15m.macdTurn === -1 && (state === "downtrend_overbought" || state === "ranging_overbought")) {
    confidence = Math.min(confidence + 0.1, 1.0);
  }

  return { state, confidence };
}

function calculateTrendConsistency(ema20A, ema50A, ema20B, ema50B, macdA, macdB) {
  let score = 0;
  const trendA = ema20A > ema50A ? 1 : -1;
  const momentumA = macdA > 0 ? 1 : -1;
  const trendB = ema20B > ema50B ? 1 : -1;
  const momentumB = macdB > 0 ? 1 : -1;
  if (trendA === trendB) score += 0.4;
  if (momentumA === momentumB) score += 0.3;
  if (trendA === momentumA) score += 0.15;
  if (trendB === momentumB) score += 0.15;
  return Math.max(0, Math.min(1, score));
}

function calculateTripleTimeframeConsistency(tfPrimary, tfConfirm, tfFilter) {
  const pc = calculateTrendConsistency(tfPrimary.ema20, tfPrimary.ema50, tfConfirm.ema20, tfConfirm.ema50, tfPrimary.macd, tfConfirm.macd);
  const cf = calculateTrendConsistency(tfConfirm.ema20, tfConfirm.ema50, tfFilter.ema20, tfFilter.ema50, tfConfirm.macd, tfFilter.macd);
  return pc * 0.6 + cf * 0.4;
}

// Trend score (-100..100): EMA gap (40%) + MACD/price normalized (30%) +
// price deviation from EMA20 (20%) + RSI trend (10%). Matches source exactly.
function calculateTrendScore(tf) {
  let score = 0;
  const emaGap = (tf.ema20 - tf.ema50) / tf.ema50;
  score += Math.max(-40, Math.min(40, emaGap * 1000));
  const macdNormalized = tf.macd / tf.currentPrice;
  score += Math.max(-30, Math.min(30, macdNormalized * 10000));
  score += Math.max(-20, Math.min(20, tf.deviationFromEMA20 * 2));
  const rsiTrend = (tf.rsi7 - 50) / 5;
  score += Math.max(-10, Math.min(10, rsiTrend));
  return Math.round(score);
}

// Matches source exactly: 20%-relative-drop weakening, +/-20 crossing = reversing.
function detectTrendWeakening(currentScore, scoreHistory) {
  const previousScore = scoreHistory.length > 0 ? scoreHistory[scoreHistory.length - 1] : currentScore;
  const change = currentScore - previousScore;
  const changePercent = previousScore !== 0 ? (change / Math.abs(previousScore)) * 100 : 0;
  const isWeakening = Math.abs(currentScore) < Math.abs(previousScore) * 0.8;
  const isReversing = (previousScore > 20 && currentScore < -20) || (previousScore < -20 && currentScore > 20);
  const weakeningSeverity = isWeakening ? Math.round((1 - Math.abs(currentScore) / Math.abs(previousScore)) * 100) : 0;
  return { currentScore, previousScore, change, changePercent, isWeakening, isReversing, weakeningSeverity };
}

// ---- MACD / RSI divergence detection (matches source exactly) ----

function detectMACDDivergence(candles, macdValues) {
  if (!candles || candles.length < 20 || !macdValues || macdValues.length < 20) {
    return { type: "none", strength: 0, description: "insufficient data" };
  }
  const prices = candles.map((c) => c.close);
  const halfLen = Math.floor(prices.length / 2);
  const firstHalfPrices = prices.slice(0, halfLen);
  const secondHalfPrices = prices.slice(halfLen);
  const firstHalfMACD = macdValues.slice(0, halfLen);
  const secondHalfMACD = macdValues.slice(halfLen);

  const priceHigh1 = Math.max(...firstHalfPrices);
  const priceHigh2 = Math.max(...secondHalfPrices);
  const priceLow1 = Math.min(...firstHalfPrices);
  const priceLow2 = Math.min(...secondHalfPrices);
  const macdHigh1 = Math.max(...firstHalfMACD);
  const macdHigh2 = Math.max(...secondHalfMACD);
  const macdLow1 = Math.min(...firstHalfMACD);
  const macdLow2 = Math.min(...secondHalfMACD);

  const isPriceHigherHigh = priceHigh2 > priceHigh1 * 1.001;
  const isMACDLowerHigh = macdHigh2 < macdHigh1 * 0.95;
  if (isPriceHigherHigh && isMACDLowerHigh) {
    const priceIncrease = ((priceHigh2 - priceHigh1) / priceHigh1) * 100;
    const macdDecrease = ((macdHigh1 - macdHigh2) / Math.abs(macdHigh1)) * 100;
    const strength = Math.min(100, Math.round((priceIncrease + macdDecrease) * 10));
    return { type: "bearish", strength: Math.max(60, strength), description: `bearish MACD divergence: price new high (${priceHigh2.toFixed(2)}) but MACD didn't confirm (${macdHigh2.toFixed(4)})` };
  }

  const isPriceLowerLow = priceLow2 < priceLow1 * 0.999;
  const isMACDHigherLow = macdLow2 > macdLow1 * 1.05;
  if (isPriceLowerLow && isMACDHigherLow) {
    const priceDecrease = ((priceLow1 - priceLow2) / priceLow1) * 100;
    const macdIncrease = ((macdLow2 - macdLow1) / Math.abs(macdLow1)) * 100;
    const strength = Math.min(100, Math.round((priceDecrease + macdIncrease) * 10));
    return { type: "bullish", strength: Math.max(60, strength), description: `bullish MACD divergence: price new low (${priceLow2.toFixed(2)}) but MACD didn't confirm (${macdLow2.toFixed(4)})` };
  }

  return { type: "none", strength: 0, description: "no clear divergence" };
}

function detectRSIDivergence(candles, rsiValues) {
  if (!candles || candles.length < 20 || !rsiValues || rsiValues.length < 20) {
    return { type: "none", strength: 0, description: "insufficient data" };
  }
  const prices = candles.map((c) => c.close);
  const halfLen = Math.floor(prices.length / 2);
  const firstHalfPrices = prices.slice(0, halfLen);
  const secondHalfPrices = prices.slice(halfLen);
  const firstHalfRSI = rsiValues.slice(0, halfLen);
  const secondHalfRSI = rsiValues.slice(halfLen);

  const priceHigh1 = Math.max(...firstHalfPrices);
  const priceHigh2 = Math.max(...secondHalfPrices);
  const priceLow1 = Math.min(...firstHalfPrices);
  const priceLow2 = Math.min(...secondHalfPrices);
  const rsiHigh1 = Math.max(...firstHalfRSI);
  const rsiHigh2 = Math.max(...secondHalfRSI);
  const rsiLow1 = Math.min(...firstHalfRSI);
  const rsiLow2 = Math.min(...secondHalfRSI);

  const isPriceHigherHigh = priceHigh2 > priceHigh1 * 1.001;
  const isRSILowerHigh = rsiHigh2 < rsiHigh1 - 3;
  if (isPriceHigherHigh && isRSILowerHigh) {
    const priceIncrease = ((priceHigh2 - priceHigh1) / priceHigh1) * 100;
    const rsiDecrease = rsiHigh1 - rsiHigh2;
    const strength = Math.min(100, Math.round((priceIncrease * 5 + rsiDecrease) * 2));
    return { type: "bearish", strength: Math.max(60, strength), description: `bearish RSI divergence: price new high (${priceHigh2.toFixed(2)}) but RSI didn't confirm (${rsiHigh2.toFixed(1)})` };
  }

  const isPriceLowerLow = priceLow2 < priceLow1 * 0.999;
  const isRSIHigherLow = rsiLow2 > rsiLow1 + 3;
  if (isPriceLowerLow && isRSIHigherLow) {
    const priceDecrease = ((priceLow1 - priceLow2) / priceLow1) * 100;
    const rsiIncrease = rsiLow2 - rsiLow1;
    const strength = Math.min(100, Math.round((priceDecrease * 5 + rsiIncrease) * 2));
    return { type: "bullish", strength: Math.max(60, strength), description: `bullish RSI divergence: price new low (${priceLow2.toFixed(2)}) but RSI didn't confirm (${rsiLow2.toFixed(1)})` };
  }

  return { type: "none", strength: 0, description: "no clear divergence" };
}

// Reconstructs an approximate historical MACD/RSI series from candle closes,
// matching the source's own approximation approach (it also estimates
// historical MACD/RSI from price ratios rather than true historical recompute).
function estimateMacdSeries(candles, currentMacd, currentPrice) {
  const closes = candles.map((c) => c.close).slice(-30);
  return closes.map((close) => currentMacd * (close / currentPrice));
}
function estimateRsiSeries(candles, currentRsi7) {
  const closes = candles.map((c) => c.close).slice(-30);
  const out = [];
  for (let i = 0; i < closes.length; i++) {
    const idx = closes.length - 1 - i;
    if (idx >= 0 && idx < closes.length - 1) {
      const priceChange = ((closes[idx + 1] - closes[idx]) / closes[idx]) * 100;
      const rsiAdjust = priceChange * 0.5;
      out.unshift(Math.max(0, Math.min(100, currentRsi7 - rsiAdjust)));
    } else {
      out.unshift(currentRsi7);
    }
  }
  return out;
}

// ---- Persisted trend-score history (replaces the original's in-memory Map) ----

function updateHistory(historyStore, symbol, scores) {
  const now = Date.now();
  const CACHE_EXPIRE_MS = 3600000;
  const HISTORY_SIZE = 5;
  let entry = historyStore[symbol];
  if (!entry || now - entry.lastUpdate > CACHE_EXPIRE_MS) {
    entry = { primary: [scores.primary], confirm: [scores.confirm], filter: [scores.filter], lastUpdate: now };
  } else {
    entry.primary.push(scores.primary);
    entry.confirm.push(scores.confirm);
    entry.filter.push(scores.filter);
    if (entry.primary.length > HISTORY_SIZE) {
      entry.primary.shift(); entry.confirm.shift(); entry.filter.shift();
    }
    entry.lastUpdate = now;
  }
  historyStore[symbol] = entry;
  return historyStore;
}

function getHistory(historyStore, symbol) {
  const entry = historyStore[symbol];
  if (!entry || Date.now() - entry.lastUpdate > 3600000) return { primary: [], confirm: [], filter: [] };
  return { primary: entry.primary.slice(), confirm: entry.confirm.slice(), filter: entry.filter.slice() };
}

// Full reversal score (0-100), matching source weighting:
// primary 40% / confirm 25% / filter 15% / MACD divergence 10% / RSI divergence 10%.
function calculateReversalScore(tfPrimary, tfConfirm, tfFilter, positionDirection, history) {
  let score = 0;
  const details = [];
  const reversedFrames = [];

  const scorePrimary = calculateTrendScore(tfPrimary);
  const scoreConfirm = calculateTrendScore(tfConfirm);
  const scoreFilter = calculateTrendScore(tfFilter);
  const targetSign = positionDirection === "long" ? -1 : 1;
  const targetDivergence = positionDirection === "long" ? "bearish" : "bullish";

  const primaryChange = detectTrendWeakening(scorePrimary, history.primary);
  if (Math.sign(scorePrimary) === targetSign && Math.abs(scorePrimary) > 30) {
    score += 40; details.push(`primary timeframe strongly reversed (score=${scorePrimary})`); reversedFrames.push("primary");
  } else if (primaryChange.isWeakening && primaryChange.weakeningSeverity > 40) {
    score += 20; details.push(`primary timeframe weakening significantly (${primaryChange.weakeningSeverity}%)`);
  } else if (Math.abs(scorePrimary) < 20) {
    score += 12; details.push(`primary timeframe entering range (score=${scorePrimary})`);
  }

  const confirmChange = detectTrendWeakening(scoreConfirm, history.confirm);
  if (Math.sign(scoreConfirm) === targetSign && Math.abs(scoreConfirm) > 30) {
    score += 25; details.push(`confirm timeframe strongly reversed (score=${scoreConfirm})`); reversedFrames.push("confirm");
  } else if (confirmChange.isWeakening && confirmChange.weakeningSeverity > 40) {
    score += 12; details.push(`confirm timeframe weakening significantly (${confirmChange.weakeningSeverity}%)`);
  }

  const filterChange = detectTrendWeakening(scoreFilter, history.filter);
  if (Math.sign(scoreFilter) === targetSign && Math.abs(scoreFilter) > 30) {
    score += 15; details.push(`filter timeframe reversed (score=${scoreFilter})`); reversedFrames.push("filter");
  }

  // MACD divergence (10%) - primary timeframe
  if (tfPrimary.candles && tfPrimary.candles.length >= 20) {
    const macdSeries = estimateMacdSeries(tfPrimary.candles, tfPrimary.macd, tfPrimary.currentPrice);
    const macdDiv = detectMACDDivergence(tfPrimary.candles.slice(-30), macdSeries.slice(-30));
    if (macdDiv.type === targetDivergence && macdDiv.strength >= 60) {
      score += 10; details.push(`${macdDiv.description} (strength ${macdDiv.strength})`);
    }
  }

  // RSI divergence (10%) - confirm timeframe
  if (tfConfirm.candles && tfConfirm.candles.length >= 20) {
    const rsiSeries = estimateRsiSeries(tfConfirm.candles, tfConfirm.rsi7);
    const rsiDiv = detectRSIDivergence(tfConfirm.candles.slice(-30), rsiSeries.slice(-30));
    if (rsiDiv.type === targetDivergence && rsiDiv.strength >= 60) {
      score += 10; details.push(`${rsiDiv.description} (strength ${rsiDiv.strength})`);
    }
  }

  const weakenedFrames = [primaryChange, confirmChange, filterChange].filter((c) => c.weakeningSeverity > 40).length;
  const hasDivergence = details.some((d) => d.includes("divergence"));
  const earlyWarning = weakenedFrames >= 2 || reversedFrames.length >= 2 || hasDivergence;

  let recommendation;
  if (score >= 70) recommendation = "close immediately - multiple timeframes confirm reversal";
  else if (score >= 50) recommendation = "recommend closing - reversal risk elevated";
  else if (earlyWarning && score >= 30) recommendation = "watch closely - trend weakening or divergence detected";
  else recommendation = "trend normal - continue holding";

  return { reversalScore: score, earlyWarning, timeframesReversed: reversedFrames, recommendation, details };
}

module.exports = {
  buildTimeframeIndicators,
  determineTrendStrength,
  determineMomentumState,
  determineVolatilityState,
  determineMarketState,
  calculateTrendConsistency,
  calculateTripleTimeframeConsistency,
  calculateTrendScore,
  detectTrendWeakening,
  getHistory,
  updateHistory,
  calculateReversalScore,
};
