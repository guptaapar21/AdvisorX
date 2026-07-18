// Small, dependency-free technical indicator library.
// All functions take an array of closes (or candle objects) oldest -> newest.

function ema(values, period) {
  const k = 2 / (period + 1);
  let emaPrev = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const out = new Array(values.length).fill(null);
  out[period - 1] = emaPrev;
  for (let i = period; i < values.length; i++) {
    emaPrev = values[i] * k + emaPrev * (1 - k);
    out[i] = emaPrev;
  }
  return out;
}

function rsi(closes, period = 14) {
  if (closes.length < period + 1) return null;
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff >= 0) gains += diff; else losses -= diff;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;

  for (let i = period + 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? -diff : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
  }
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

function macd(closes, fast = 12, slow = 26, signalPeriod = 9) {
  if (closes.length < slow + signalPeriod) return null;
  const emaFast = ema(closes, fast);
  const emaSlow = ema(closes, slow);
  const macdLine = closes.map((_, i) =>
    emaFast[i] != null && emaSlow[i] != null ? emaFast[i] - emaSlow[i] : null
  );
  const macdValues = macdLine.filter((v) => v != null);
  const signalLine = ema(macdValues, signalPeriod);
  const lastMacd = macdValues[macdValues.length - 1];
  const lastSignal = signalLine[signalLine.length - 1];
  return {
    macd: lastMacd,
    signal: lastSignal,
    histogram: lastMacd - lastSignal,
  };
}

// Average True Range, as a % of price (volatility measure)
function atrPercent(candles, period = 14) {
  if (candles.length < period + 1) return null;
  const trueRanges = [];
  for (let i = 1; i < candles.length; i++) {
    const cur = candles[i];
    const prevClose = candles[i - 1].close;
    const tr = Math.max(
      cur.high - cur.low,
      Math.abs(cur.high - prevClose),
      Math.abs(cur.low - prevClose)
    );
    trueRanges.push(tr);
  }
  const recent = trueRanges.slice(-period);
  const atr = recent.reduce((a, b) => a + b, 0) / recent.length;
  const lastClose = candles[candles.length - 1].close;
  return (atr / lastClose) * 100;
}

function avgVolume(candles, period = 20) {
  const recent = candles.slice(-period);
  return recent.reduce((a, c) => a + c.volume, 0) / recent.length;
}

// Highest high / lowest low over the last N candles (excluding the most recent one)
function keyLevels(candles, lookback = 20) {
  const slice = candles.slice(-(lookback + 1), -1); // exclude current forming candle
  const resistance = Math.max(...slice.map((c) => c.high));
  const support = Math.min(...slice.map((c) => c.low));
  return { support, resistance };
}

// Bollinger Bands (SMA middle, stdDev bands)
function bollingerBands(closes, period = 20, stdDev = 2) {
  if (closes.length < period) return { upper: 0, middle: 0, lower: 0, bandwidth: 0 };
  const recent = closes.slice(-period);
  const middle = recent.reduce((a, b) => a + b, 0) / period;
  const variance = recent.reduce((sum, p) => sum + Math.pow(p - middle, 2), 0) / period;
  const std = Math.sqrt(variance);
  const upper = middle + stdDev * std;
  const lower = middle - stdDev * std;
  return { upper, middle, lower, bandwidth: upper - lower };
}

// Price position relative to a Bollinger band level, -2..2 (0 = at middle, 1 = at that band)
function priceVsBB(price, bbLevel, bbMiddle) {
  if (bbMiddle === 0 || bbLevel === bbMiddle) return 0;
  const distance = price - bbMiddle;
  const range = Math.abs(bbLevel - bbMiddle);
  if (range === 0) return 0;
  return Math.max(-2, Math.min(2, distance / range));
}

// Detects a MACD histogram turning point: 1 = turned up, -1 = turned down, 0 = none.
// Matches the original's approach: recompute MACD histogram progressively over the
// closes array, then check the last 3 values for a turn.
function macdHistogramTurn(closes) {
  if (closes.length < 30) return 0;
  const histHistory = [];
  for (let i = 26; i <= closes.length; i++) {
    const slice = closes.slice(0, i);
    const m = macd(slice);
    histHistory.push(m ? m.histogram : 0);
  }
  if (histHistory.length < 3) return 0;
  const latest = histHistory[histHistory.length - 1];
  const prev = histHistory[histHistory.length - 2];
  const prevPrev = histHistory[histHistory.length - 3];
  if (prevPrev > prev && prev < latest && latest > 0) return 1;
  if (prevPrev < prev && prev > latest && latest < 0) return -1;
  return 0;
}

// Raw ATR (not %) using a simple average of the last `period` true ranges -
// matches multiTimeframeAnalysis.ts's calculateATR (different from the
// Wilder-smoothed version stopLossCalculator.ts uses for stop distance).
function atrSimple(candles, period = 14) {
  if (!candles || candles.length < period + 1) return 0;
  const trueRanges = [];
  for (let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;
    trueRanges.push(Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose)));
  }
  if (trueRanges.length < period) return 0;
  const recent = trueRanges.slice(-period);
  return recent.reduce((a, b) => a + b, 0) / period;
}

// ATR ratio: current ATR(14) vs ATR(14) computed 20 candles ago - a measure of
// whether volatility is currently expanding or contracting relative to its own
// recent history (NOT the same as ATR-as-%-of-price).
function atrRatio(candles, period = 14) {
  const currentAtr = atrSimple(candles, period);
  const historicalAtr = candles.length >= 40 ? atrSimple(candles.slice(0, -20), period) : currentAtr;
  return historicalAtr !== 0 ? currentAtr / historicalAtr : 1;
}

// Wilder-smoothed ATR (EMA-style), used for stop-loss distance sizing -
// matches stopLossCalculator.ts's calculateATR exactly.
function atrWilder(candles, period = 14) {
  if (candles.length < period + 1) return 0;
  const trueRanges = [];
  for (let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;
    trueRanges.push(Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose)));
  }
  let atr = trueRanges.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < trueRanges.length; i++) {
    atr = (atr * (period - 1) + trueRanges[i]) / period;
  }
  return atr;
}

module.exports = {
  ema, rsi, macd, atrPercent, avgVolume, keyLevels,
  bollingerBands, priceVsBB, macdHistogramTurn, atrSimple, atrRatio, atrWilder,
};
