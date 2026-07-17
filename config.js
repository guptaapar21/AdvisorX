// ---- Edit this file to customize what you watch and alert on ----

module.exports = {
  // Coins to scan. Use the base symbol only (e.g. "BTC", "ETH").
  symbols: ["BTC", "ETH", "SOL", "DOGE", "XRP"],

  // Which CoinDCX market to use for each symbol.
  //   "futures" -> USDT-margined perpetual futures (pair looks like "B-BTC_USDT")
  //   "spot"    -> INR spot market (pair looks like "I-BTC_INR")
  marketType: "futures",

  // Timeframes used for analysis (CoinDCX candle intervals: 1m,5m,15m,1h,4h,1d ...)
  entryTimeframe: "15m",   // used to spot the actual signal
  trendTimeframe: "1h",    // used to confirm the broader trend

  // How many candles to pull for indicator calculation
  candleLimit: 100,

  // Only alert when the opportunity score (0-100) is at or above this
  minScore: 70,

  // Minimum minutes between repeat alerts for the same symbol+direction
  // (prevents spamming you every 15 min while a signal stays valid)
  cooldownMinutes: 120,
};
