// Faithful port of strategyRouter.ts's state->strategy dispatch.
//
// DEVIATION FROM ORIGINAL (explicit, per request): the original's switch
// statement never calls breakoutStrategy - it's fully-written dead code
// there. Here, breakout IS checked for every state (as an independent,
// additional check alongside whatever trend/reversion signal the state
// would normally route to) - and every breakout-sourced result carries
// isBreakoutExtension: true so it's clearly distinguishable in alerts as
// something the original bot never actually did.

const { trendFollowingStrategy } = require("./trendFollowingStrategy");
const { meanReversionStrategy } = require("./meanReversionStrategy");
const { breakoutStrategy } = require("./breakoutStrategy");

function routeStrategy(symbol, marketState, tf15m, tf1h, maxLeverage) {
  let baseResult;

  switch (marketState.state) {
    case "uptrend_oversold":
      baseResult = trendFollowingStrategy(symbol, "long", marketState, tf15m, tf1h, maxLeverage);
      break;
    case "downtrend_overbought":
      baseResult = trendFollowingStrategy(symbol, "short", marketState, tf15m, tf1h, maxLeverage);
      break;
    case "downtrend_oversold":
      baseResult = meanReversionStrategy(symbol, "long", marketState, tf15m, tf1h, maxLeverage);
      break;
    case "uptrend_overbought":
      baseResult = meanReversionStrategy(symbol, "short", marketState, tf15m, tf1h, maxLeverage);
      break;
    case "uptrend_continuation":
      baseResult = trendFollowingStrategy(symbol, "long", marketState, tf15m, tf1h, maxLeverage);
      break;
    case "downtrend_continuation":
      baseResult = trendFollowingStrategy(symbol, "short", marketState, tf15m, tf1h, maxLeverage);
      break;
    case "ranging_oversold":
      baseResult = meanReversionStrategy(symbol, "long", marketState, tf15m, tf1h, maxLeverage);
      break;
    case "ranging_overbought":
      baseResult = meanReversionStrategy(symbol, "short", marketState, tf15m, tf1h, maxLeverage);
      break;
    case "ranging_neutral":
    case "no_clear_signal":
    default:
      baseResult = {
        symbol, action: "wait", confidence: "low", signalStrength: 0, recommendedLeverage: 0,
        marketState: marketState.state, strategyType: "none",
        reason: `market state: ${marketState.state}, no clear signal`,
        keyMetrics: { rsi7: tf15m.rsi7, rsi14: tf15m.rsi14, macd: tf15m.macd, ema20: tf1h.ema20, ema50: tf1h.ema50, price: tf15m.currentPrice, atrRatio: tf1h.atrRatio },
      };
  }

  // Breakout extension: only consider it if the state-routed strategy came
  // back with "wait" (don't override a real trend/reversion signal).
  if (baseResult.action === "wait") {
    const breakoutLong = breakoutStrategy(symbol, "long", marketState, tf15m, tf1h, maxLeverage);
    if (breakoutLong.action === "long") return breakoutLong;
    const breakoutShort = breakoutStrategy(symbol, "short", marketState, tf15m, tf1h, maxLeverage);
    if (breakoutShort.action === "short") return breakoutShort;
  }

  return baseResult;
}

module.exports = { routeStrategy };
