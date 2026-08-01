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

  // UPDATED: this was originally a single 36h value from an early,
  // general study across BTC/ETH/SOL/XRP/DOGE. Dedicated later sweeps on
  // the actual proven conservative setup found each coin has its own
  // genuine peak shape at very different hold times, not one shared
  // number - SOL peaks sharply at 18h (degrading on both sides: 6h/9h
  // nearly break-even, 48h back down to near-zero), DOGE peaks around
  // 48-60h (9h weak, 72h already declining from its 60h peak), ETH peaks
  // at 60h too (a bumpier curve than SOL/DOGE - a real dip around
  // 18h-30h before rising again - but 60h is still ~$90/year better than
  // the old 36h at the same win rate). All 3 backtested-tier coins now
  // have their own dedicated, tested value - the 36h fallback below is
  // no longer actually used by any currently-active coin.
  maxHoldHoursBySymbol: {
    SOL: 18,
    DOGE: 48,
    ETH: 60,
  },
  maxHoldHours: 36, // fallback for any coin not listed above (none currently - all 3 backtested-tier coins now have their own dedicated value)

  // The original uses 3 timeframes per cycle: primary (trend direction),
  // confirm (momentum/RSI, also where breakout/mean-reversion look for
  // signals), filter (broader volatility/EMA context). This mapping
  // matches the "balanced" preset. CoinDCX's FUTURES candles endpoint only
  // supports 1m/15m/1h/1d directly (confirmed via a real 422: "interval
  // must be one of [1m, 15m, 1h, 1d]"), so "5m" here is transparently
  // synthesized by fetching 1m candles and aggregating 5-at-a-time
  // (coindcxExchangeClient.js's getCandlesForInterval) - not a limitation
  // you need to work around yourself.
  timeframes: { primary: "5m", confirm: "15m", filter: "1h" },

  // How many candles to pull per timeframe for indicator calculation
  candleLimit: 100,

  // Small delay (ms) between each CoinDCX candle request, as a safety
  // margin - CoinDCX doesn't publish an explicit public rate limit for
  // candles, and our actual volume (15 calls/run) is well within what any
  // comparable exchange allows, but this costs a few seconds and removes
  // the risk entirely.
  candleFetchDelayMs: 300,

  // Only alert when the opportunity score (0-100) is at or above this.
  // Matches STRATEGY_SCORE_WEIGHTS.balanced.minScore in opportunityScorer.js
  // exactly - runtimeConfig.js's applyRuntimeOverrides swaps this to the
  // right value automatically when a different strategy is selected via
  // Telegram, so it stays in sync no matter which preset is active.
  minScore: 75,

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
  // maxPositions is now DEAD/unused - the actual rule is "one position
  // per coin" (enforced automatically via checkRealPositionOrOrder in
  // agentTools.js), not a global count across all coins combined. Left
  // here rather than deleted in case anything external still reads it,
  // but nothing in this codebase does anymore. leverageMin/Max and
  // positionSize% come
  // from the selected strategy preset (see strategyParams.js) - the AI
  // picks a position size as a % of balance within that range, informed
  // by signal strength, rather than a risk-distance formula (that's how
  // the original actually works - confirmed no risk-distance sizing tool
  // exists in the source).
  riskRules: {
    maxPositions: 3,
    // Re-entry cooldown after ANY close (manual, stop-loss, take-profit,
    // or /closeposition) - avoids immediately re-suggesting/re-opening
    // the same symbol+direction the moment it re-qualifies. Set to 65
    // min (slightly over 60) so a full 1-hour candle forms before
    // re-evaluation, not just an arbitrary time window.
    reentryCooldownMinutes: 65,
    leverageMin: strategyParams.leverageMin,
    leverageMax: strategyParams.leverageMax,
    positionSizeMinPercent: strategyParams.positionSizeMin,
    positionSizeMaxPercent: strategyParams.positionSizeMax,
    // Real dollar-risk cap: max % of total account balance that can be
    // lost if a trade hits its stop, regardless of what leverage/size the
    // AI picked within its normal ranges. Settled on 7% - a middle ground
    // between the original 5% default and her initial 10% ask, after
    // discussing the drawdown tradeoff (a realistic 5-loss streak at
    // ~45% trade loss rate means roughly -30% drawdown at 7%, vs -23% at
    // 5% and -41% at 10%).
    maxRiskPercentPerTrade: 7,
  },

  // CoinDCX's futures wallet balance is ONLY exposed via a websocket
  // event ("balance-update") - confirmed directly from their official
  // Futures API doc, which has no REST endpoint for it at all (positions/
  // orders/margin/trades are all documented, futures balance isn't among
  // them). This bot runs as a short-lived GitHub Action every 5 min, not
  // a persistent process, so maintaining a live websocket isn't practical
  // here. Rather than guess at a nonexistent endpoint (the earlier bug),
  // set your REAL futures wallet USDT balance here manually - check it in
  // the CoinDCX app's Futures wallet screen and update this when it
  // changes meaningfully (a deposit/withdrawal, or after it's drifted a
  // fair bit from trading P&L). This value is what the 7% risk cap above
  // actually uses.
  manualFuturesBalanceUsdt: null, // e.g. 150 - set this to your real number

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
  // Switched from gemini-3.5-flash (20 RPD free-tier cap) to the Flash Lite
  // variant, which carries a 500 RPD free-tier cap per the quota dashboard -
  // 25x the daily headroom across the same 12 rotating keys, still with
  // function-calling support.
  agentModel: "gemini-3.5-flash-lite",

  // Safety cap on how many tool-call turns the agent can take in one run.
  agentMaxTurns: 10,
};
