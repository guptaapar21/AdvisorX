const BASE = "https://public.coindcx.com";
const API_BASE = "https://api.coindcx.com";

let marketsCache = null;

// Resolve the CoinDCX "pair" string (e.g. "B-BTC_USDT") for a given base symbol.
async function resolvePair(symbol, marketType) {
  if (!marketsCache) {
    const res = await fetch(`${API_BASE}/exchange/v1/markets_details`);
    if (!res.ok) throw new Error(`markets_details failed: ${res.status}`);
    marketsCache = await res.json();
  }

  const wantQuote = marketType === "spot" ? "INR" : "USDT";

  // Futures perpetuals typically carry ecode "BM" or "F", spot/margin carry "B"/"I" etc.
  // We just match on base + quote symbol naming rather than hardcoding ecodes,
  // since CoinDCX has added new ecodes over time.
  const match = marketsCache.find((m) => {
    const base = (m.base_currency_short_name || "").toUpperCase();
    const target = (m.target_currency_short_name || "").toUpperCase();
    if (marketType === "futures") {
      return target === symbol.toUpperCase() && base === "USDT" && m.pair;
    }
    return base === symbol.toUpperCase() && target === wantQuote && m.pair;
  });

  if (!match) {
    throw new Error(`Could not resolve a ${marketType} pair for ${symbol}`);
  }
  return match.pair;
}

async function getCandles(pair, interval, limit) {
  const url = `${BASE}/market_data/candles?pair=${encodeURIComponent(pair)}&interval=${interval}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`candles failed for ${pair}: ${res.status}`);
  const raw = await res.json();
  // API returns newest-first; normalize to oldest -> newest for indicator math
  return raw
    .map((c) => ({
      time: c.time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
      volume: c.volume,
    }))
    .sort((a, b) => a.time - b.time);
}

module.exports = { resolvePair, getCandles };
