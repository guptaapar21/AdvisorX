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

  // Cross-symbol ranking: out of ALL symbols that clear minScore this run,
  // only send alerts for the top N by score.
  maxAlertsPerRun: 3,

  // Minimum minutes between repeat alerts for the same symbol+direction
  cooldownMinutes: 120,

  // Partial take-profit stages, in R-multiples of the initial stop distance.
  partialTakeProfit: {
    stages: [
      { key: "1", r: 1, closePercent: 33, moveStopTo: "entry" },
      { key: "2", r: 2, closePercent: 33, moveStopTo: "stage1" },
      { key: "3", r: 3, closePercent: 34, moveStopTo: "stage2trail" },
    ],
  },

  // Optional LLM layer (used by the simpler Mode A narrator only).
  useLLMAdvisor: false,
  llmProvider: "gemini",

  // If a Gemini key hits its quota (HTTP 429), it's put in cooldown for
  // this many minutes and the next configured key is tried automatically.
  geminiKeyCooldownMinutes: 60,

  // ---- Full agent mode (agentIndex.js) ----
  riskRules: {
    maxPositions: 3,
    leverageMin: 3,
    leverageMax: 10,
    riskPercentPerTrade: 2,       // % of account balance risked per trade
    minStopDistancePercent: 1,    // stop can't be tighter than this
    maxStopDistancePercent: 8,    // stop can't be wider than this
    minBalanceUsdt: 20,           // won't suggest opening below this balance
  },

  // Model used for the full reasoning agent (needs function-calling support).
  agentModel: "gemini-2.5-flash",

  // Safety cap on how many tool-call turns the agent can take in one run.
  agentMaxTurns: 10,
};
