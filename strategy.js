const { rsi, macd, ema, atrPercent, avgVolume, keyLevels } = require("./indicators");

// Classify the broader market regime using the trend timeframe.
function classifyRegime(trendCandles) {
  const closes = trendCandles.map((c) => c.close);
  const ema20 = ema(closes, 20);
  const ema50 = ema(closes, 50);
  const lastEma20 = ema20[ema20.length - 1];
  const lastEma50 = ema50[ema50.length - 1];
  const macdResult = macd(closes);
  const atr = atrPercent(trendCandles);

  if (atr != null && atr > 6) return "volatile";
  if (lastEma20 != null && lastEma50 != null) {
    if (lastEma20 > lastEma50 && macdResult && macdResult.histogram > 0) return "uptrend";
    if (lastEma20 < lastEma50 && macdResult && macdResult.histogram < 0) return "downtrend";
  }
  return "range";
}

// Score one symbol. Returns null if no actionable signal.
function scoreOpportunity(symbol, entryCandles, trendCandles) {
  const closes = entryCandles.map((c) => c.close);
  const currentPrice = closes[closes.length - 1];
  const rsi7 = rsi(closes, 7);
  const rsi14 = rsi(closes, 14);
  const macdEntry = macd(closes);
  const atr = atrPercent(entryCandles);
  const volNow = entryCandles[entryCandles.length - 1].volume;
  const volAvg = avgVolume(entryCandles, 20);
  const { support, resistance } = keyLevels(entryCandles, 20);

  const trendCloses = trendCandles.map((c) => c.close);
  const trendMacd = macd(trendCloses);
  const regime = classifyRegime(trendCandles);

  if (rsi7 == null || macdEntry == null || atr == null) return null;

  let action = null;
  let score = 50;
  const reasons = [];

  // --- Breakout ---
  const brokeResistance = currentPrice > resistance * 0.998;
  const brokeSupport = currentPrice < support * 1.002;
  const volumeSpike = volAvg > 0 && volNow / volAvg > 1.5;

  if (brokeResistance && trendMacd && trendMacd.histogram > 0 && rsi7 < 78) {
    action = "long";
    score += 15;
    reasons.push(`broke resistance ${resistance.toFixed(4)}`);
    if (volumeSpike) { score += 10; reasons.push("volume confirms breakout"); }
  } else if (brokeSupport && trendMacd && trendMacd.histogram < 0 && rsi7 > 22) {
    action = "short";
    score += 15;
    reasons.push(`broke support ${support.toFixed(4)}`);
    if (volumeSpike) { score += 10; reasons.push("volume confirms breakdown"); }
  }

  // --- Trend following (only if no breakout already found) ---
  if (!action && (regime === "uptrend" || regime === "downtrend")) {
    if (regime === "uptrend" && rsi14 > 45 && rsi14 < 70 && macdEntry.histogram > 0) {
      action = "long";
      score += 12;
      reasons.push("aligned with 1h uptrend, MACD rising");
    } else if (regime === "downtrend" && rsi14 < 55 && rsi14 > 30 && macdEntry.histogram < 0) {
      action = "short";
      score += 12;
      reasons.push("aligned with 1h downtrend, MACD falling");
    }
  }

  // --- Mean reversion (only in a range-bound regime) ---
  if (!action && regime === "range") {
    if (rsi7 < 25) {
      action = "long";
      score += 10;
      reasons.push(`oversold RSI7 ${rsi7.toFixed(1)} in a ranging market`);
    } else if (rsi7 > 75) {
      action = "short";
      score += 10;
      reasons.push(`overbought RSI7 ${rsi7.toFixed(1)} in a ranging market`);
    }
  }

  if (!action) return null;

  // Volatility penalty - too wild = less reliable
  if (atr > 6) { score -= 15; reasons.push("very high volatility, wider risk"); }
  else if (atr > 4) { score -= 5; }

  // Multi-timeframe agreement bonus
  const entryDir = macdEntry.histogram > 0 ? "long" : "short";
  if (trendMacd && ((trendMacd.histogram > 0 && entryDir === "long") ||
                     (trendMacd.histogram < 0 && entryDir === "short"))) {
    score += 8;
    reasons.push("15m and 1h momentum agree");
  }

  score = Math.max(0, Math.min(100, Math.round(score)));

  // Simple ATR-based stop suggestion (informational only - you decide sizing)
  const stopDistance = Math.max(atr * 1.5, 0.8); // in %
  const stopPrice = action === "long"
    ? currentPrice * (1 - stopDistance / 100)
    : currentPrice * (1 + stopDistance / 100);

  return {
    symbol,
    action,
    score,
    regime,
    price: currentPrice,
    suggestedStop: stopPrice,
    stopDistancePercent: Number(stopDistance.toFixed(2)),
    atrPercent: Number(atr.toFixed(2)),
    rsi7: Number(rsi7.toFixed(1)),
    reason: reasons.join("; "),
  };
}

module.exports = { scoreOpportunity, classifyRegime };
