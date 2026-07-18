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
  // only send alerts for the top N by score (like the original engine,
  // which shows the best N opportunities across all coins rather than
  // alerting on every coin that independently clears the bar).
  maxAlertsPerRun: 3,

  // Minimum minutes between repeat alerts for the same symbol+direction
  // (prevents spamming you every 15 min while a signal stays valid)
  cooldownMinutes: 120,

  // Once an entry alert is sent, this tool starts tracking a "virtual"
  // position for that symbol (it assumes you took the trade) so it can
  // guide you through staged take-profit and a trailing stop as price
  // moves. Stages are in R-multiples of the initial stop distance.
  // moveStopTo: "entry" = breakeven, "stage1"/"stage2trail" = trail stop
  // up to the 1R/2R price level. Purely informational either way.
  partialTakeProfit: {
    stages: [
      { key: "1", r: 1, closePercent: 33, moveStopTo: "entry" },
      { key: "2", r: 2, closePercent: 33, moveStopTo: "stage1" },
      { key: "3", r: 3, closePercent: 34, moveStopTo: "stage2trail" },
    ],
  },

  // Optional LLM layer: after the rule-based scorer/position-tracker flags
  // something, an LLM rewrites the structured numbers into a clear,
  // plain-English Telegram message (entry zone, stop, staged TP plan,
  // sizing hint, reasoning). It ONLY writes text - it has no order
  // placement ability and cannot act on your behalf. If every configured
  // key fails, the plain rule-based alert is sent instead, so you're never
  // left without a message.
  useLLMAdvisor: false,

  // "gemini" (default, free tier - see README for GEMINI_API_KEYS setup)
  // or "anthropic" (pay-per-token, needs ANTHROPIC_API_KEY secret instead).
  llmProvider: "gemini",

  // If a Gemini key hits its quota (HTTP 429), it's put in cooldown for
  // this many minutes and the next configured key is tried automatically.
  // With multiple keys, this is really just "how long before retrying a
  // key that was recently rate-limited" — it doesn't block the run.
  geminiKeyCooldownMinutes: 60,

  // ---- Full agent mode (agentIndex.js) ----
  // These are the hard risk rules the agent's system prompt is built from,
  // and that check_open_position enforces regardless of what the model
  // decides. Same spirit as the original engine's per-strategy risk config.
  riskRules: {
    maxPositions: 3,
    leverageMin: 3,
    leverageMax: 10,
    riskPercentPerTrade: 2,       // % of account balance risked per trade
    minStopDistancePercent: 1,    // stop can't be tighter than this
    maxStopDistancePercent: 8,    // stop can't be wider than this
  },

  // Model used for the full reasoning agent (needs function-calling support).
  agentModel: "gemini-3.5-flash",

  // Safety cap on how many tool-call turns the agent can take in one run
  // before the loop is forced to stop (prevents a runaway reasoning loop).
  agentMaxTurns: 10,
};
