// ---- Edit this file to customize what you watch and alert on ----

const { getStrategyParams } = require("./strategyParams");

// Matches RISK_PARAMS.MAX_LEVERAGE's default in the original (env-overridable there).
const MAX_LEVERAGE = 15;

// Which of the original's 5 strategy presets to run. Entry-signal logic is
// identical across all 5 in the original - this only changes leverage
// bounds, position-size guidance, and the stop-loss ATR multiplier/bounds.
const STRATEGY = "balanced";
const strategyParams = getStrategyParams(STRATEGY, MAX_LEVERAGE);

module.exports = {
  // Coins to scan. Use the base symbol only (e.g. "BTC", "ETH").
  symbols: ["BTC", "ETH", "SOL", "DOGE", "XRP"],

  // Which CoinDCX market to use for each symbol.
  //   "futures" -> USDT-margined perpetual futures (pair looks like "B-BTC_USDT")
  //   "spot"    -> INR spot market (pair looks like "I-BTC_INR")
  marketType: "futures",

  strategy: STRATEGY,
  maxLeverage: MAX_LEVERAGE,

  // The original uses 3 timeframes per cycle: primary (trend direction),
  // confirm (momentum/RSI, also where breakout/mean-reversion look for
  // signals), filter (broader volatility/EMA context). This mapping
  // matches the "balanced" preset - change if you change STRATEGY above.
  timeframes: { primary: "5m", confirm: "15m", filter: "1h" },

  // How many candles to pull per timeframe for indicator calculation
  candleLimit: 100,

  // Small delay (ms) between each CoinDCX candle request, as a safety
  // margin - CoinDCX doesn't publish an explicit public rate limit for
  // candles, and our actual volume (15 calls/run) is well within what any
  // comparable exchange allows, but this costs a few seconds and removes
  // the risk entirely.
  candleFetchDelayMs: 300,

  // Only alert when the opportunity score (0-100) is at or above this
  minScore: 70,

  // Cross-symbol ranking: out of ALL symbols that clear minScore this run,
  // only send alerts for the top N by score.
  maxAlertsPerRun: 3,

  // Minimum minutes between repeat alerts for the same symbol+direction
  cooldownMinutes: 120,

  // Optional LLM layer: after the rule-based scorer/position-tracker flags
  // something, an LLM rewrites the structured numbers into a clear,
  // plain-English Telegram message. It ONLY writes text - it has no order
  // placement ability and cannot act on your behalf. If every configured
  // key fails, the plain rule-based alert is sent instead, so you're never
  // left without a message.
  useLLMAdvisor: false,

  // "gemini" (default, free tier) or "anthropic" (pay-per-token).
  llmProvider: "gemini",

  // If a Gemini key hits its quota (HTTP 429), it's put in cooldown for
  // this many minutes and the next configured key is tried automatically.
  geminiKeyCooldownMinutes: 60,

  // ---- Full agent mode (agentIndex.js) ----
  // maxPositions is a hard limit; leverageMin/Max and positionSize% come
  // from the selected strategy preset (see strategyParams.js) - the AI
  // picks a position size as a % of balance within that range, informed
  // by signal strength, rather than a risk-distance formula (that's how
  // the original actually works - confirmed no risk-distance sizing tool
  // exists in the source).
  riskRules: {
    maxPositions: 3,
    leverageMin: strategyParams.leverageMin,
    leverageMax: strategyParams.leverageMax,
    positionSizeMinPercent: strategyParams.positionSizeMin,
    positionSizeMaxPercent: strategyParams.positionSizeMax,
  },

  // Real stop-loss config (from the selected strategy preset)
  stopLoss: {
    atrPeriod: 14,
    atrMultiplier: strategyParams.scientificStopLoss.atrMultiplier,
    lookbackPeriod: 20,
    bufferPercent: 0.1,
    useATR: true,
    useSupportResistance: true,
    minStopLossPercent: strategyParams.scientificStopLoss.minDistance,
    maxStopLossPercent: strategyParams.scientificStopLoss.maxDistance,
    minQualityScore: 40,
  },

  // Model used for the full reasoning agent (needs function-calling support).
  agentModel: "gemini-3.5-flash",

  // Safety cap on how many tool-call turns the agent can take in one run.
  agentMaxTurns: 10,
};
