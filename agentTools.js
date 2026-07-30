// The tool suite the agent can call. Read tools (market data, account
// balance, positions, opportunity scoring) use the faithfully-ported
// modules (marketStateAnalyzer.js, strategyRouter.js, opportunityScorer.js,
// stopLossCalculator.js, takeProfitManagement.js). Execution tools
// (open/close position, stop-loss updates, partial take-profit, cancel
// order) are INTERCEPTED - they never call the exchange.

const exchange = require("./coindcxExchangeClient");
const msa = require("./marketStateAnalyzer");
const strategyRouter = require("./strategyRouter");
const opportunityScorer = require("./opportunityScorer");
const stopLossCalculator = require("./stopLossCalculator");
const takeProfitManagement = require("./takeProfitManagement");
const { bollingerBands, priceVsBB, atrWilder } = require("./indicators");
const advisoryStore = require("./advisoryStore");
const { getEffectiveMinScore } = require("./runtimeConfig");
const balanceTracker = require("./balanceTracker");
const tradeOutcomeLog = require("./tradeOutcomeLog");
const { getStrategyParams } = require("./strategyParams");

const path = require("path");
const fs = require("fs");
const TREND_HISTORY_FILE = path.join(__dirname, "trendScoreHistory.json");

function loadTrendHistory() {
  try { return JSON.parse(fs.readFileSync(TREND_HISTORY_FILE, "utf8")); } catch { return {}; }
}
function saveTrendHistory(store) {
  fs.writeFileSync(TREND_HISTORY_FILE, JSON.stringify(store, null, 2));
}

// Gemini is inconsistent about the exact contract string it passes to
// execution tools - sometimes "B-XRP_USDT" (matches the real exchange
// format), sometimes just "XRP_USDT" (no "B-" prefix). advisoryStore and
// fastWatch.js both key advisories on this string EXACTLY, and fastWatch
// compares against the real position's pos.pair (which always has the
// "B-" prefix). A missing prefix means the advisory silently never
// matches the real position - fastWatch logs "not opened by this bot" and
// skips it, so it drops out of the Live Scorecard and reconciliation even
// though the position is real and was genuinely opened off this bot's
// suggestion. Normalizing to the canonical form here, at the point every
// execution tool receives it, makes the stored key always match the real
// exchange format regardless of what Gemini sends.
function normalizeContract(contract) {
  if (!contract) return contract;
  const symbol = contract.replace(/^[A-Z]-/, "").replace(/_USDT$/i, "");
  return `B-${symbol}_USDT`;
}

// ---- Gemini function declarations (JSON Schema) ----

const declarations = [
  {
    name: "get_account_balance",
    description: "Get real wallet balances from the CoinDCX account (read-only).",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "get_positions",
    description: "Get real currently open futures positions from the CoinDCX account (read-only).",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "analyze_opening_opportunities",
    description:
      "Scans all configured symbols across 3 timeframes (primary/confirm/filter), classifies each into one of 10 market states, routes to the matching strategy (trend-following, mean-reversion, or - as an added extension not in the original bot - breakout), scores every opportunity with the real per-strategy weighted formula, filters out symbols with an open position, and returns the top-ranked opportunities that clear the minimum score. Also returns allScores: every scanned symbol's score, including ones below threshold. Any candidate with isBreakoutExtension=true came from the breakout strategy, which the original bot never actually used - flag this clearly when reporting it.",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "check_open_position",
    description:
      "Validates whether a candidate new entry should actually be opened, using the real scientific stop-loss calculator (hybrid ATR + support/resistance, with a quality score) and position-count limit. Call this before open_position.",
    parameters: {
      type: "object",
      properties: {
        symbol: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
      },
      required: ["symbol", "action"],
    },
  },
  {
    name: "calculate_risk",
    description:
      "Reports current account-wide risk exposure across all open positions: total notional value, total margin used, used-margin %, overall risk level (low/medium/high based on margin usage), and return % since the account's tracked starting balance. Takes no parameters - call this to understand overall portfolio risk before sizing a new position, not to size one directly (position size is chosen as a % of balance from the strategy's recommended range, not from a risk-distance formula).",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "check_total_exposure",
    description:
      "Checks whether adding a new position of a given USDT margin amount and leverage would push total account exposure (all positions' notional value combined) over the account's max-leverage limit (total exposure <= balance x maxLeverage). Call this alongside check_open_position before opening.",
    parameters: {
      type: "object",
      properties: {
        amountUsdt: { type: "number", description: "Margin amount (USDT) for the new position" },
        leverage: { type: "number" },
      },
      required: ["amountUsdt", "leverage"],
    },
  },
  {
    name: "check_partial_take_profit_opportunity",
    description:
      "Checks an open position against the real staged take-profit plan (1R/2R/3R at 33.33%/33.33%/0%, adjusted for current volatility 0.8x-1.5x) using the AI's own original entry/stop recommendation for that position, and reports whether a new stage has been reached.",
    parameters: {
      type: "object",
      properties: {
        contract: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        currentPrice: { type: "number" },
      },
      required: ["contract", "action", "currentPrice"],
    },
  },
  {
    name: "check_reversal",
    description:
      "Checks an open position for a trend reversal using the real weighted score (primary timeframe 40%, confirm 25%, filter 15%, MACD divergence 10%, RSI divergence 10%). Score >=70 means close immediately regardless of take-profit stage; 30-70 is an early warning to factor into judgment, not an automatic action.",
    parameters: {
      type: "object",
      properties: {
        symbol: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
      },
      required: ["symbol", "action"],
    },
  },
  {
    name: "check_liquidity",
    description:
      "Checks liquidity conditions for a candidate new position, matching the original's pre-trade checks: (1) time-of-day/weekend low-liquidity position-size reduction (UTC 2-6am, or the weekend window), (2) order-book depth vs. the position's exposure - ask depth for longs, bid depth for shorts, since that's the side a market order actually consumes (the original's own code checks bid depth regardless of direction, which is only correct for shorts - fixed here to check the right side for both), (3) a separate 1h-ATR-based volatility adjustment to leverage and size (different from the take-profit volatility check). Returns adjusted amount/leverage suggestions - apply them before calling open_position.",
    parameters: {
      type: "object",
      properties: {
        symbol: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        amountUsdt: { type: "number", description: "Your proposed position margin (USDT), before adjustment" },
        leverage: { type: "number", description: "Your proposed leverage, before adjustment" },
        totalBalanceUsdt: { type: "number" },
      },
      required: ["symbol", "action", "amountUsdt", "leverage", "totalBalanceUsdt"],
    },
  },
  {
    name: "open_position",
    description:
      "Decide to open a new position. This does NOT place a real order - it sends your decision and reasoning to the user via Telegram for manual execution on CoinDCX.",
    parameters: {
      type: "object",
      properties: {
        contract: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        entryPrice: { type: "number" },
        stopPrice: { type: "number" },
        leverage: { type: "number" },
        positionSizeUsdt: { type: "number" },
        reasoning: { type: "string" },
      },
      required: ["contract", "action", "entryPrice", "stopPrice", "leverage", "positionSizeUsdt", "reasoning"],
    },
  },
  {
    name: "close_position",
    description:
      "Decide to close (fully or partially) an open position. This does NOT close a real position - it sends your decision and reasoning to the user via Telegram for manual execution. Provide currentPrice so this bot can automatically compute and log the outcome of ITS OWN suggested trade (based on its own advised entry price) for future risk decisions - this happens regardless of whether the user actually took the trade, since it tracks the AI's own suggestion quality.",
    parameters: {
      type: "object",
      properties: {
        contract: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        sizePercent: { type: "number", description: "Percent of the position to close, 1-100" },
        currentPrice: { type: "number", description: "Current market price, used to auto-compute the outcome of this bot's own suggested trade" },
        closeReason: { type: "string", enum: ["trend_reversal", "take_profit", "stop_loss", "manual", "other"] },
        reasoning: { type: "string" },
      },
      required: ["contract", "action", "sizePercent", "currentPrice", "reasoning"],
    },
  },
  {
    name: "update_position_stop_loss",
    description:
      "Decide to move the stop-loss. This does NOT modify a real order - it sends the new stop level to the user via Telegram for manual execution.",
    parameters: {
      type: "object",
      properties: {
        contract: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        newStop: { type: "number" },
        reasoning: { type: "string" },
      },
      required: ["contract", "action", "newStop", "reasoning"],
    },
  },
  {
    name: "execute_partial_take_profit",
    description:
      "Decide to take partial profit at a reached R-multiple stage. This does NOT execute a real order - it sends the stage, close percent, and new stop to the user via Telegram for manual execution. Provide currentPrice so the partial outcome can be auto-logged the same way as close_position.",
    parameters: {
      type: "object",
      properties: {
        contract: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        stage: { type: "string" },
        closePercent: { type: "number" },
        newStop: { type: "number" },
        currentPrice: { type: "number" },
        reasoning: { type: "string" },
      },
      required: ["contract", "action", "stage", "closePercent", "newStop", "currentPrice", "reasoning"],
    },
  },
  {
    name: "cancel_order",
    description:
      "Decide to cancel a pending order. This does NOT cancel a real order - it sends the request to the user via Telegram for manual execution.",
    parameters: {
      type: "object",
      properties: {
        orderId: { type: "string" },
        contract: { type: "string" },
        reasoning: { type: "string" },
      },
      required: ["contract", "reasoning"],
    },
  },
];

// Builds the full analysis (market state, strategy result, opportunity
// score) for one symbol. Mutates trendHistoryStore in place.
async function analyzeSymbol(symbol, config, trendHistoryStore) {
  const { primary, confirm, filter } = await exchange.getMultiTimeframeCandles(
    symbol, config.marketType, config.timeframes, config.candleLimit, config.candleFetchDelayMs
  );
  if (primary.length < 55 || confirm.length < 55 || filter.length < 55) {
    return { symbol, error: "insufficient candle history" };
  }

  const tfPrimary = msa.buildTimeframeIndicators(primary);
  const tfConfirm = msa.buildTimeframeIndicators(confirm);
  const tfFilter = msa.buildTimeframeIndicators(filter);

  const confirmCloses = confirm.map((c) => c.close);
  const bb = bollingerBands(confirmCloses, 20, 2);
  tfConfirm.bollingerUpper = bb.upper;
  tfConfirm.bollingerLower = bb.lower;
  tfConfirm.bollingerMiddle = bb.middle;
  const priceVsUpperBB = priceVsBB(tfConfirm.currentPrice, bb.upper, bb.middle);
  const priceVsLowerBB = priceVsBB(tfConfirm.currentPrice, bb.lower, bb.middle);

  const trendStrength = msa.determineTrendStrength(tfPrimary);
  const momentumState = msa.determineMomentumState(tfConfirm);
  const volatilityState = msa.determineVolatilityState(tfFilter);
  const { state, confidence } = msa.determineMarketState(trendStrength, momentumState, tfConfirm);
  const alignmentScore = msa.calculateTripleTimeframeConsistency(tfPrimary, tfConfirm, tfFilter);

  const history = msa.getHistory(trendHistoryStore, symbol);
  const trendScores = { primary: msa.calculateTrendScore(tfPrimary), confirm: msa.calculateTrendScore(tfConfirm), filter: msa.calculateTrendScore(tfFilter) };
  const trendChanges = history.primary.length > 0 ? {
    primary: msa.detectTrendWeakening(trendScores.primary, history.primary),
    confirm: msa.detectTrendWeakening(trendScores.confirm, history.confirm),
    filter: msa.detectTrendWeakening(trendScores.filter, history.filter),
  } : null;
  msa.updateHistory(trendHistoryStore, symbol, trendScores);

  const marketState = {
    state, trendStrength, momentumState, volatilityState, confidence,
    timeframeAlignment: { alignmentScore, is15mAnd1hAligned: alignmentScore > 0.6 },
    keyMetrics: { atr_ratio: tfFilter.atrRatio, priceVsLowerBB, priceVsUpperBB, distanceToEMA20: tfConfirm.deviationFromEMA20, price: tfConfirm.currentPrice },
    trendChanges,
  };

  const strategyResult = strategyRouter.routeStrategy(symbol, marketState, tfConfirm, tfFilter, config.riskRules.leverageMax);
  // "backtested" is a reporting label, not a real scoring formula -
  // opportunityScorer.js only knows the 5 original presets. Every
  // /backtested threshold was found using conservative's actual formula,
  // so that's what gets used here regardless of the label.
  const scoringStrategy = config.strategy === "backtested" ? "conservative" : config.strategy;
  const opportunity = await opportunityScorer.scoreOpportunity(strategyResult, marketState, scoringStrategy, tradeOutcomeLog.historicalPenaltyFn);

  return { symbol, marketState, strategyResult, opportunity, tfPrimary, tfConfirm, tfFilter, currentPrice: tfConfirm.currentPrice };
}

// CoinDCX's getPositions response includes an entry per contract even when
// flat (size 0) - counting raw array length treats every configured symbol
// as "an open position" regardless of whether anything is actually open.
// Checks whether contract+action already exists as a REAL, executed
// position OR a pending, unfilled order on the exchange. Single shared
// implementation used by both check_open_position and open_position -
// previously this exact matching logic was copy-pasted independently in
// both places (a real maintenance risk: a future fix to one copy could
// easily miss the other).
//
// How callers MUST use the result:
//   - verified: false -> the core position check itself failed (e.g.
//     network error). Genuinely unknown whether a duplicate exists.
//     Callers MUST fail closed (treat as if a duplicate might exist),
//     not fail open - the whole point of this check is duplicate
//     prevention, so silently assuming "no duplicate" on an error
//     defeats its purpose exactly when it matters most.
//   - verified: true, exists: true -> confirmed real duplicate
//     (position or pending order), block.
//   - verified: true, exists: false, note: non-null -> positions
//     confirmed clear, but the pending-orders check (best-effort, path
//     not independently verified against CoinDCX's official docs)
//     failed - proceed, but the note should be surfaced so this gap is
//     visible rather than silently assumed away.
async function checkRealPositionOrOrder(exchange, creds, contract, action) {
  let positionMatch;
  try {
    const positionsRaw = await exchange.getPositions(creds);
    const activePositions = getActivePositions(positionsRaw);
    positionMatch = activePositions.some((p) => {
      const posContract = p.pair ?? p.contract;
      const posDirection = Number(p.active_pos ?? p.size ?? 0) > 0 ? "long" : "short";
      return posContract === contract && posDirection === action;
    });
  } catch (err) {
    return { verified: false, exists: null, error: `position check failed: ${err.message}` };
  }
  if (positionMatch) return { verified: true, exists: true, via: "position" };

  let orderMatch = false;
  let note = null;
  try {
    const ordersRaw = await exchange.getActiveOrders(creds);
    const orders = Array.isArray(ordersRaw) ? ordersRaw : (ordersRaw?.data || []);
    orderMatch = orders.some((o) => {
      const orderContract = o.pair ?? o.contract;
      const orderSide = (o.side || "").toLowerCase(); // CoinDCX order convention: buy/sell
      const orderDirection = orderSide === "buy" ? "long" : orderSide === "sell" ? "short" : null;
      return orderContract === contract && orderDirection === action;
    });
  } catch (err) {
    note = `pending-order check failed (${err.message}) - only confirmed no FILLED position exists; a pending unfilled limit order, if any, was not verified`;
  }
  return { verified: true, exists: orderMatch, via: orderMatch ? "pending_order" : null, note };
}

// This filters down to genuinely active positions only, matching the size
// check already used correctly elsewhere (calculate_risk, check_total_exposure).
function getActivePositions(positionsRaw) {
  const positions = Array.isArray(positionsRaw) ? positionsRaw : (positionsRaw?.data || []);
  return positions.filter((p) => Math.abs(Number(p.active_pos ?? p.size ?? 0)) > 0);
}

// Single source of truth for "what balance should risk/exposure checks use"
// - used by BOTH open_position's risk-cap AND check_total_exposure, so
// they can never drift apart again (this exact drift was a real bug: this
// fix used to live only inside open_position, while check_total_exposure
// kept calling the old spot-wallet API directly and blocking trades with
// "zero USDT balance" even after a real manual/tracked balance was set).
//
// CoinDCX doesn't expose futures wallet balance over REST at all
// (confirmed from their official Futures API doc - only a websocket
// event, which doesn't fit this bot's short-lived Action-per-run
// architecture). Prefers her manually-set real balance (auto-tracked
// from real P&L after that); only falls back to the spot-wallet API
// check as a last resort, clearly labeled as unreliable.
async function getEffectiveBalance(config, creds) {
  if (typeof config.manualFuturesBalanceUsdt === "number" && config.manualFuturesBalanceUsdt > 0) {
    const trackedBalance = balanceTracker.getCurrentBalance(config.manualFuturesBalanceUsdt);
    if (typeof trackedBalance === "number" && trackedBalance > 0) {
      return { totalBalance: trackedBalance, balanceSource: "tracked" };
    }
  }
  try {
    const balances = await exchange.getBalances(creds);
    const usdtBalance = Array.isArray(balances) ? balances.find((b) => (b.currency || "").toUpperCase() === "USDT") : null;
    const apiBalance = usdtBalance ? Number(usdtBalance.balance ?? usdtBalance.available_balance ?? 0) : 0;
    // MIN_PLAUSIBLE_BALANCE guards against the spot-wallet-near-zero case -
    // if it looks implausibly small to be real trading capital, don't
    // trust it at all rather than act on a wrong near-zero number.
    const MIN_PLAUSIBLE_BALANCE = 1; // USDT
    if (apiBalance >= MIN_PLAUSIBLE_BALANCE) {
      return { totalBalance: apiBalance, balanceSource: "api_unreliable" }; // SPOT wallet, not futures - fallback only
    }
  } catch {
    // fall through
  }
  return { totalBalance: 0, balanceSource: null };
}

function buildTools(config, creds) {
  const advisories = advisoryStore.loadAdvisories();
  let advisoriesDirty = false;
  const runWarnings = [];

  const handlers = {
    async get_account_balance() {
      // Was calling exchange.getBalances(creds) directly - the raw SPOT
      // wallet API, which has nothing to do with the futures wallet this
      // bot actually trades from (no REST endpoint exists for that, see
      // getEffectiveBalance above). That drift is exactly what caused
      // Gemini to see "0 USDT" and refuse to open positions even with a
      // real tracked/manual balance set - it never saw it, because this
      // tool bypassed getEffectiveBalance entirely while open_position and
      // check_total_exposure used it correctly. Now consistent across all
      // three.
      const { totalBalance, balanceSource } = await getEffectiveBalance(config, creds);
      return { usdt_balance: totalBalance, source: balanceSource };
    },

    async get_positions() {
      const positionsRaw = await exchange.getPositions(creds);
      const activePositions = getActivePositions(positionsRaw);
      if (advisoryStore.reconcileWithRealPositions(advisories, activePositions)) {
        advisoriesDirty = true;
      }
      // Return the FILTERED list, not the raw CoinDCX response - CoinDCX
      // returns a flat/zero-size entry for every configured symbol
      // regardless of whether it's actually open, and the model was
      // seeing all of these as if they were real positions (e.g.
      // reporting "4 active position records" when 0 were actually
      // open) - this was the exact bug pattern already fixed everywhere
      // else via getActivePositions, just missed on this one tool that's
      // directly exposed to the model.
      return activePositions;
    },

    async analyze_opening_opportunities() {
      const trendHistoryStore = loadTrendHistory();
      const candidates = [];
      const allScores = [];

      for (const symbol of config.symbols) {
        const cooldown = tradeOutcomeLog.isSymbolInCooldown(symbol);
        if (cooldown.inCooldown) {
          allScores.push({ symbol, score: 0, note: `in cooldown: ${cooldown.reason} (${cooldown.remainingHours}h remaining)` });
          continue;
        }
        try {
          const analysis = await analyzeSymbol(symbol, config, trendHistoryStore);
          if (analysis.error) {
            allScores.push({ symbol, score: 0, note: analysis.error });
            continue;
          }
          allScores.push({
            symbol, score: analysis.opportunity.totalScore, action: analysis.strategyResult.action,
            setupType: analysis.strategyResult.strategyType, isBreakoutExtension: analysis.opportunity.isBreakoutExtension,
          });
          if (analysis.strategyResult.action !== "wait" && analysis.opportunity.totalScore >= getEffectiveMinScore(config, symbol)) {
            candidates.push(analysis);
          }
        } catch (err) {
          allScores.push({ symbol, score: 0, note: `error: ${err.message}` });
          runWarnings.push(`${symbol}: scan failed - ${err.message}`);
        }
      }

      saveTrendHistory(trendHistoryStore);

      candidates.sort((a, b) => b.opportunity.totalScore - a.opportunity.totalScore);
      const top = candidates.slice(0, config.maxAlertsPerRun);

      return {
        opportunities: top.map((c) => ({
          symbol: c.symbol,
          score: c.opportunity.totalScore,
          action: c.strategyResult.action,
          setupType: c.strategyResult.strategyType,
          marketState: c.marketState.state,
          reason: c.strategyResult.reason,
          recommendedLeverage: c.strategyResult.recommendedLeverage,
          price: c.currentPrice,
          isBreakoutExtension: c.opportunity.isBreakoutExtension,
          scoreBreakdown: c.opportunity.breakdown,
        })),
        allScores,
      };
    },

    async check_open_position({ symbol, action }) {
      const analysis = await analyzeSymbol(symbol, config, loadTrendHistory());
      if (analysis.error) return { shouldOpen: false, reasons: [analysis.error] };

      const reasons = [];
      let shouldOpen = true;

      // HARD gate, cannot be bypassed by Gemini choosing to call this
      // tool directly on a symbol that never cleared the candidate list.
      // Previously the score threshold only existed as a soft filter
      // inside analyze_opening_opportunities - nothing stopped a direct
      // check_open_position/open_position call on any symbol regardless
      // of its actual score, which is exactly how ETH opened at score 46
      // against a real requirement of 81 under /backtested.
      const requiredScore = getEffectiveMinScore(config, symbol);
      if (analysis.strategyResult.action !== action) {
        shouldOpen = false;
        reasons.push(`current strategy signal for ${symbol} is "${analysis.strategyResult.action}", not "${action}" - direction mismatch, hard block`);
      } else if (requiredScore !== undefined && requiredScore !== null && analysis.opportunity.totalScore < requiredScore) {
        shouldOpen = false;
        reasons.push(`score ${analysis.opportunity.totalScore} is below the required ${requiredScore} for ${symbol} under the current strategy - hard block, cannot be opened regardless of other checks`);
      }

      // Guard against re-suggesting a symbol+direction that's ALREADY a
      // real, executed position (or pending order) on the exchange - but
      // otherwise keep recommending it every cycle for as long as it
      // keeps clearing. Cheap local check FIRST: only pay the real
      // network cost if there's actually a prior advisory to verify
      // against - the common case (a symbol with no suggestion history
      // at all) now makes zero extra API calls, not one guaranteed call
      // every time regardless of relevance.
      //
      // This tool is read-only by design (logged as "[read]" throughout
      // this bot) - it no longer mutates advisory state itself. Clearing
      // a stale advisory is open_position's job now, since that's the
      // tool actually allowed to write.
      const contract = `B-${symbol}_USDT`; // matches this bot's futures pair convention
      const existingAdvisory = advisoryStore.getAdvisory(advisories, contract, action);
      if (existingAdvisory) {
        const check = await checkRealPositionOrOrder(exchange, creds, contract, action);
        if (!check.verified) {
          // Core check failed - genuinely unknown, fail CLOSED (assume a
          // duplicate might exist) rather than silently assuming it's
          // safe to proceed, which would defeat the point of this guard
          // exactly when it matters most.
          shouldOpen = false;
          reasons.push(`Could not verify whether ${contract} ${action} already exists on the exchange (${check.error}) - blocking as a precaution until this can be confirmed.`);
        } else if (check.exists) {
          shouldOpen = false;
          reasons.push(`${contract} ${action} already exists on the exchange (${check.via}) - not a new suggestion.`);
        } else if (check.note) {
          reasons.push(check.note);
        }
      }

      const stopCheck = stopLossCalculator.shouldOpenPosition(analysis.tfFilter.candles, action, analysis.currentPrice, config.stopLoss);
      reasons.push(stopCheck.reason);
      shouldOpen = shouldOpen && stopCheck.shouldOpen;


      try {
        const positionsRaw = await exchange.getPositions(creds);
        const openCount = getActivePositions(positionsRaw).length;
        if (openCount >= config.riskRules.maxPositions) {
          shouldOpen = false;
          reasons.push(`already at max positions (${openCount}/${config.riskRules.maxPositions})`);
        }
      } catch (err) {
        reasons.push(`could not verify position count: ${err.message}`);
        runWarnings.push(`${symbol}: could not verify position count before opening - ${err.message}`);
      }

      return {
        shouldOpen, symbol, action, reasons,
        stopLossPrice: stopCheck.stopLossResult?.stopLossPrice,
        stopLossDistancePercent: stopCheck.stopLossResult?.stopLossDistancePercent,
        qualityScore: stopCheck.stopLossResult?.qualityScore,
        volatilityLevel: stopCheck.stopLossResult?.riskAssessment.volatilityLevel,
      };
    },

    async calculate_risk() {
      const account = await exchange.getBalances(creds); // array of {currency, balance, ...}
      const positionsRaw = await exchange.getPositions(creds);
      const positions = getActivePositions(positionsRaw);

      const usdt = Array.isArray(account) ? account.find((b) => (b.currency || "").toUpperCase() === "USDT") : null;
      const availableBalance = usdt ? Number(usdt.balance ?? usdt.available_balance ?? 0) : 0;

      let totalNotional = 0;
      let totalMargin = 0;
      const positionRisks = positions.map((p) => {
        const size = Math.abs(Number(p.active_pos ?? p.size ?? 0));
        const entryPrice = Number(p.avg_price ?? p.entryPrice ?? 0);
        const leverage = Number(p.leverage ?? 1);
        const notionalValue = size * entryPrice;
        const margin = leverage > 0 ? notionalValue / leverage : notionalValue;
        totalNotional += notionalValue;
        totalMargin += margin;
        return { contract: p.pair ?? p.contract, notionalValue, margin, leverage, pnl: Number(p.pnl ?? p.unrealisedPnl ?? 0) };
      });

      const totalBalance = availableBalance + totalMargin; // approx: available + margin already committed
      const usedMarginPercent = totalBalance > 0 ? (totalMargin / totalBalance) * 100 : 0;
      let riskLevel = "low";
      if (usedMarginPercent > 80) riskLevel = "high";
      else if (usedMarginPercent > 50) riskLevel = "medium";

      return {
        totalBalance: Number(totalBalance.toFixed(2)),
        availableBalance: Number(availableBalance.toFixed(2)),
        totalNotional: Number(totalNotional.toFixed(2)),
        totalMargin: Number(totalMargin.toFixed(2)),
        usedMarginPercent: Number(usedMarginPercent.toFixed(1)),
        positionCount: positionRisks.length,
        positions: positionRisks,
        riskLevel,
      };
    },

    async check_total_exposure({ amountUsdt, leverage }) {
      const positionsRaw = await exchange.getPositions(creds);
      const positions = getActivePositions(positionsRaw);

      const { totalBalance: availableBalance, balanceSource } = await getEffectiveBalance(config, creds);

      let currentExposure = 0;
      for (const p of positions) {
        const size = Math.abs(Number(p.active_pos ?? p.size ?? 0));
        const entryPrice = Number(p.avg_price ?? p.entryPrice ?? 0);
        currentExposure += size * entryPrice;
      }

      const totalBalance = availableBalance + currentExposure / Math.max(1, ...positions.map((p) => Number(p.leverage ?? 1)));
      const newExposure = amountUsdt * leverage;
      const totalExposure = currentExposure + newExposure;
      const maxAllowedExposure = totalBalance * config.riskRules.leverageMax;

      const withinLimit = totalExposure <= maxAllowedExposure;
      const sourceNote = balanceSource === "api_unreliable"
        ? " (⚠️ using your SPOT wallet balance as a fallback - set config.manualFuturesBalanceUsdt or /setbalance for an accurate check)"
        : balanceSource === null
          ? " (⚠️ no real balance available - set config.manualFuturesBalanceUsdt or send /setbalance)"
          : "";
      return {
        withinLimit,
        currentExposure: Number(currentExposure.toFixed(2)),
        newExposure: Number(newExposure.toFixed(2)),
        totalExposure: Number(totalExposure.toFixed(2)),
        maxAllowedExposure: Number(maxAllowedExposure.toFixed(2)),
        reason: (withinLimit
          ? "within total exposure limit"
          : `total exposure ${totalExposure.toFixed(2)} USDT would exceed the limit of ${maxAllowedExposure.toFixed(2)} USDT (balance x max leverage)`) + sourceNote,
      };
    },

    async check_liquidity({ symbol, action, amountUsdt, leverage, totalBalanceUsdt }) {
      const notes = [];
      let adjustedAmountUsdt = amountUsdt;

      // 1. Time-of-day / weekend low-liquidity reduction (matches source
      // exactly, including that both can compound if they overlap).
      const now = new Date();
      const hourUTC = now.getUTCHours();
      const dayOfWeek = now.getUTCDay(); // 0=Sun, 6=Sat

      if (hourUTC >= 2 && hourUTC <= 6) {
        adjustedAmountUsdt = Math.max(10, adjustedAmountUsdt * 0.7);
        notes.push(`low-liquidity UTC hour (${hourUTC}:00) - size reduced to 70%`);
      }
      if ((dayOfWeek === 5 && hourUTC >= 22) || dayOfWeek === 6 || (dayOfWeek === 0 && hourUTC < 20)) {
        adjustedAmountUsdt = Math.max(10, adjustedAmountUsdt * 0.8);
        notes.push("weekend low-liquidity window - size reduced to 80% (of whatever it already was)");
      }

      // 2. Order-book depth check (public, no key needed). A market LONG
      // (buy) consumes the ASK side; a market SHORT (sell) consumes the
      // BID side - checking the wrong side would validate against
      // liquidity that isn't actually relevant to the order direction.
      // Skips gracefully (doesn't block) if the order book can't be
      // fetched or parsed, same as the original's own try/catch behavior.
      let bookDepthUsdt = null;
      let requiredDepthUsdt = null;
      let sufficientLiquidity = true;
      try {
        const pair = `B-${symbol}_USDT`; // matches this bot's futures pair convention
        const book = await exchange.getOrderBook(pair);
        const relevantSide = action === "long" ? book?.asks : book?.bids;
        if (relevantSide && relevantSide.length > 0) {
          bookDepthUsdt = relevantSide.slice(0, 5).reduce((sum, b) => sum + b.price * b.size, 0);
          requiredDepthUsdt = adjustedAmountUsdt * leverage * 5;
          sufficientLiquidity = bookDepthUsdt >= requiredDepthUsdt;
          if (!sufficientLiquidity) notes.push(`order book ${action === "long" ? "ask" : "bid"} depth ${bookDepthUsdt.toFixed(2)} USDT < required ${requiredDepthUsdt.toFixed(2)} USDT`);
        } else {
          notes.push("order book unavailable or empty - liquidity depth check skipped");
        }
      } catch (err) {
        notes.push(`order book check failed (${err.message}) - skipped, not blocking`);
        runWarnings.push(`${symbol}: order book check failed - ${err.message}`);
      }

      // 3. Separate 1h-ATR-based volatility adjustment to leverage/size
      // (distinct from takeProfitManagement's own volatility levels - this
      // one uses 1h candles, >5%=high/<2%=low/else normal, and factors
      // come from the strategy preset, not a fixed 0.8-1.5x scale).
      let volatilityLevel = "normal";
      let adjustedLeverage = leverage;
      let volAdjustedAmountUsdt = adjustedAmountUsdt;
      try {
        const { filter } = await exchange.getMultiTimeframeCandles(symbol, config.marketType, config.timeframes, config.candleLimit, config.candleFetchDelayMs);
        const candles1h = filter; // "filter" timeframe is 1h for the balanced preset
        if (candles1h.length > 14) {
          const atr14 = atrWilder(candles1h, 14);
          const currentPrice = candles1h[candles1h.length - 1].close;
          const atrPercent = (atr14 / currentPrice) * 100;
          if (atrPercent > 5) volatilityLevel = "high";
          else if (atrPercent < 2) volatilityLevel = "low";

          const params = getStrategyParams(config.strategy === "backtested" ? "conservative" : config.strategy, config.maxLeverage);
          const adj = params.volatilityAdjustment[volatilityLevel];
          if (volatilityLevel === "high") {
            adjustedLeverage = Math.max(1, Math.round(leverage * adj.leverageFactor));
            volAdjustedAmountUsdt = Math.max(10, adjustedAmountUsdt * adj.positionFactor);
            notes.push(`high volatility (1h ATR ${atrPercent.toFixed(2)}%) - leverage/size reduced`);
          } else if (volatilityLevel === "low") {
            adjustedLeverage = Math.min(config.maxLeverage, Math.round(leverage * adj.leverageFactor));
            volAdjustedAmountUsdt = Math.min(totalBalanceUsdt * 0.32, adjustedAmountUsdt * adj.positionFactor);
            notes.push(`low volatility (1h ATR ${atrPercent.toFixed(2)}%) - leverage/size may increase, capped at 32% of balance`);
          }
        }
      } catch (err) {
        notes.push(`volatility adjustment check failed (${err.message}) - using unadjusted values`);
        runWarnings.push(`${symbol}: volatility adjustment check failed - ${err.message}`);
      }

      return {
        originalAmountUsdt: amountUsdt,
        originalLeverage: leverage,
        suggestedAmountUsdt: Number(volAdjustedAmountUsdt.toFixed(2)),
        suggestedLeverage: adjustedLeverage,
        bookDepthUsdt: bookDepthUsdt !== null ? Number(bookDepthUsdt.toFixed(2)) : null,
        requiredDepthUsdt: requiredDepthUsdt !== null ? Number(requiredDepthUsdt.toFixed(2)) : null,
        sufficientLiquidity,
        volatilityLevel,
        notes,
      };
    },

    async check_partial_take_profit_opportunity({ contract: rawContract, action, currentPrice }) {
      const contract = normalizeContract(rawContract);
      const adv = advisoryStore.getAdvisory(advisories, contract, action);
      if (!adv) return { canExecute: false, reason: "no recorded entry advisory for this position - was it opened by this bot?" };

      const symbol = contract.replace(/^[A-Z]-/, "").replace(/_USDT$/, "");
      let candles15m = [];
      try {
        const { confirm } = await exchange.getMultiTimeframeCandles(symbol, config.marketType, config.timeframes, config.candleLimit, config.candleFetchDelayMs);
        candles15m = confirm;
      } catch (err) {
        runWarnings.push(`${symbol}: couldn't fetch candles for take-profit volatility check - ${err.message} (used normal-volatility default)`);
      }

      return takeProfitManagement.checkPartialTakeProfitOpportunity(
        adv.entryPrice, currentPrice, adv.initialStop, action, adv.stagesAdvised, candles15m
      );
    },

    async check_reversal({ symbol, action }) {
      const trendHistoryStore = loadTrendHistory();
      const analysis = await analyzeSymbol(symbol, config, trendHistoryStore);
      saveTrendHistory(trendHistoryStore);
      if (analysis.error) return { reversalScore: 0, error: analysis.error };

      const history = msa.getHistory(trendHistoryStore, symbol);
      return msa.calculateReversalScore(analysis.tfPrimary, analysis.tfConfirm, analysis.tfFilter, action, history);
    },

    // Real, evidence-based safety net: the original engine force-closes any
    // position after 36 hours regardless of P&L - this was missing
    // entirely until backtested across 5 coins confirmed 36h as the best
    // or near-best hold length (see config.js's maxHoldHours comment).
    // Without this, a position with no clear stop/target/reversal signal
    // could otherwise sit open indefinitely.
    async check_max_hold_time({ symbol, action }) {
      const contract = `B-${symbol}_USDT`;
      const adv = advisoryStore.getAdvisory(advisories, contract, action);
      if (!adv) {
        return { exceededMaxHold: false, note: "no advisory on record for this position - can't verify open time" };
      }
      const hoursOpen = (Date.now() - adv.openedAt) / (60 * 60 * 1000);
      return {
        hoursOpen: Math.round(hoursOpen * 10) / 10,
        maxHoldHours: config.maxHoldHours,
        exceededMaxHold: hoursOpen >= config.maxHoldHours,
      };
    },

    // ---- Execution tools: intercepted, Telegram-only ----

    async open_position({ contract: rawContract, action, entryPrice, stopPrice, leverage, positionSizeUsdt, reasoning }) {
      const contract = normalizeContract(rawContract);

      // HARD score gate, same "move it here so it can't be bypassed"
      // reasoning as the duplicate guard right below - check_open_position
      // reporting shouldOpen=false only helps if the model chooses to
      // respect that result. This directly caused a real, confirmed bug:
      // ETH opened at score 46 against a real requirement of 81 under
      // /backtested, because nothing on the execution path itself ever
      // re-verified the score - the threshold only ever existed as a soft
      // filter on analyze_opening_opportunities' candidate list.
      const symbolForScoreCheck = contract.replace(/^[A-Z]-/, "").replace(/_USDT$/, "");
      const requiredScore = getEffectiveMinScore(config, symbolForScoreCheck);
      if (requiredScore !== undefined && requiredScore !== null) {
        try {
          const freshAnalysis = await analyzeSymbol(symbolForScoreCheck, config, loadTrendHistory());
          if (!freshAnalysis.error && freshAnalysis.opportunity.totalScore < requiredScore) {
            return {
              telegramMessage: null, // silent block, matches the duplicate guard's convention
              resultForModel: {
                status: "blocked_below_threshold",
                note: `${symbolForScoreCheck} scored ${freshAnalysis.opportunity.totalScore}, below the required ${requiredScore} for the current strategy - hard block, this cannot be opened. Do not retry - look for a different candidate instead.`,
              },
            };
          }
        } catch (err) {
          runWarnings.push(`${symbolForScoreCheck}: could not re-verify score before opening - ${err.message}`);
        }
      }

      // HARD duplicate guard - this used to only live inside
      // check_open_position (a separate read tool), which only protects
      // against a duplicate IF the model happens to call that check first
      // this cycle. That's a convention, not an enforced rule - if the
      // model's reasoning goes straight to open_position without it, the
      // advisory for an ALREADY-being-tracked position gets silently
      // overwritten with new entry/stop/size/leverage, even though the
      // real position (from the earlier suggestion) is still what's
      // actually open. This directly caused a real, confusing bug: a
      // tracked stop that jumped to an arbitrary value within minutes of
      // the original suggestion, matching neither the original stop nor
      // any legitimate stage-trail level. Moving the SAME check here makes
      // it impossible to bypass, regardless of tool-call order.
      //
      // Cheap local check first (only pays the real network cost if an
      // advisory already exists to verify), shared helper (same matching
      // logic as check_open_position, not a second independent copy),
      // and fails CLOSED on error rather than silently allowing a
      // possible duplicate through.
      const existingAdvisoryForOpen = advisoryStore.getAdvisory(advisories, contract, action);
      if (existingAdvisoryForOpen) {
        const check = await checkRealPositionOrOrder(exchange, creds, contract, action);
        if (!check.verified) {
          return {
            telegramMessage: null,
            resultForModel: {
              status: "blocked_unverified",
              note: `Could not verify whether ${contract} ${action} already exists on the exchange (${check.error}) - blocking as a precaution until this can be confirmed. Do not retry immediately.`,
            },
          };
        }
        if (check.exists) {
          return {
            telegramMessage: null, // no Telegram spam for a blocked duplicate - this should be silent/internal
            resultForModel: {
              status: "blocked_duplicate",
              note: `${contract} ${action} already exists on the exchange (${check.via}) - refusing to open again/overwrite that tracking. Do not call open_position again for this symbol+direction.`,
            },
          };
        }
        // Confirmed no real duplicate (positions, and orders if that
        // check succeeded) - this suggestion is genuinely allowed to
        // proceed and overwrite the stale advisory with fresh values,
        // which is open_position's legitimate job as the write/action tool.
        if (check.note) runWarnings.push(`${symbol}: ${check.note}`);
      }

      // Real dollar-risk cap: this was a flagged gap that sat too long
      // without being built - regardless of what leverage/size/stop the AI
      // picked, this caps the ACTUAL dollar loss at the stop to a fixed %
      // of total account balance. Auto-scales the position size down to
      // fit (same pattern as check_liquidity's auto-adjustment), rather
      // than blocking a good signal outright.
      const MAX_RISK_PERCENT_OF_BALANCE = config.riskRules.maxRiskPercentPerTrade ?? 5;
      let finalPositionSizeUsdt = positionSizeUsdt;
      const riskNotes = [];

      const { totalBalance, balanceSource } = await getEffectiveBalance(config, creds);

      if (totalBalance > 0) {
        const stopDistancePercent = Math.abs(entryPrice - stopPrice) / entryPrice * 100;
        const notional = positionSizeUsdt * leverage;
        const dollarRisk = notional * (stopDistancePercent / 100);
        const riskPercentOfBalance = (dollarRisk / totalBalance) * 100;

        if (riskPercentOfBalance > MAX_RISK_PERCENT_OF_BALANCE) {
          const scaleFactor = MAX_RISK_PERCENT_OF_BALANCE / riskPercentOfBalance;
          finalPositionSizeUsdt = Number((positionSizeUsdt * scaleFactor).toFixed(2));
          const sourceNote = balanceSource === "tracked" ? "" : " (⚠️ using your SPOT wallet balance as a fallback - set config.manualFuturesBalanceUsdt for an accurate check)";
          riskNotes.push(
            `⚠️ Size auto-reduced from ${positionSizeUsdt} to ${finalPositionSizeUsdt} USDT - the original would have risked ` +
            `${riskPercentOfBalance.toFixed(1)}% of your account at the stop, above the ${MAX_RISK_PERCENT_OF_BALANCE}% cap.${sourceNote}`
          );
        }
      } else {
        riskNotes.push(
          `⚠️ No real futures balance available to check risk % - set config.manualFuturesBalanceUsdt to your real futures wallet balance ` +
          `(CoinDCX doesn't expose this over REST). Using the suggested size as-is, double check your real risk yourself before executing.`
        );
      }

      await exchange.placeOrder({ contract, action, entryPrice, stopPrice, leverage, positionSizeUsdt: finalPositionSizeUsdt });
      advisoryStore.recordOpen(advisories, contract, action, entryPrice, stopPrice, finalPositionSizeUsdt, leverage);
      advisoriesDirty = true;
      const dirEmoji = action === "long" ? "🟢 LONG" : "🔴 SHORT";

      // Show the actual 1R/2R/3R staged take-profit price levels upfront,
      // not just entry/stop - so there's a concrete target to watch on
      // CoinDCX yourself, without waiting for a later follow-up message.
      const r = Math.abs(entryPrice - stopPrice);
      const dir = action === "long" ? 1 : -1;
      const target1 = entryPrice + dir * r * 1;
      const target2 = entryPrice + dir * r * 2;
      const target3 = entryPrice + dir * r * 3;
      // Use 2 decimals for higher-value assets (BTC/ETH-style prices),
      // more precision for sub-$1 coins where 2 decimals would round away
      // all the meaningful movement.
      const decimals = entryPrice >= 1 ? 2 : 6;
      const fmt = (n) => n.toFixed(decimals);

      return {
        telegramMessage: [
          `🤖 *AI SUGGESTS OPENING* ${dirEmoji} *${contract}*`,
          `Entry: ${fmt(entryPrice)} | Stop: ${fmt(stopPrice)} | Leverage: ${leverage}x`,
          `Suggested size: ${finalPositionSizeUsdt} USDT margin`,
          ...riskNotes,
          `Targets (staged take-profit): 1R ${fmt(target1)} (close ~33%) | 2R ${fmt(target2)} (close ~33%) | 3R ${fmt(target3)} (trail rest)`,
          `Reasoning: ${reasoning}`,
          ``,
          `_No order has been placed. Execute manually on CoinDCX if you agree. You'll get a follow-up message when a target is reached, but you can also watch these levels yourself - your live P&L is always visible on CoinDCX's Positions tab, it doesn't wait for this bot._`,
        ].join("\n"),
        resultForModel: { status: "queued_for_manual_execution", note: "Sent to user via Telegram. Not executed." },
      };
    },

    async close_position({ contract: rawContract, action, sizePercent, currentPrice, closeReason, reasoning }) {
      const contract = normalizeContract(rawContract);
      await exchange.closePosition(contract, sizePercent);

      // Auto-record the outcome based on THIS bot's own advised entry price
      // vs currentPrice - happens regardless of whether the user actually
      // acted on the suggestion, since this tracks the AI's own suggestion
      // quality, not the user's real trades.
      const adv = advisoryStore.getAdvisory(advisories, contract, action);
      if (adv && currentPrice) {
        const dir = action === "long" ? 1 : -1;
        const pnlPercent = ((currentPrice - adv.entryPrice) * dir / adv.entryPrice) * 100;
        const symbol = contract.replace(/^[A-Z]-/, "").replace(/_USDT$/, "");
        tradeOutcomeLog.recordOutcome(symbol, Number(pnlPercent.toFixed(2)), closeReason || "manual");

        // Automatic balance tracking: compute the REAL dollar P&L of the
        // portion just closed (using the leverage/size recorded at open)
        // and feed it into the running balance tracker, so it updates
        // itself on every close this bot advises - no manual re-entry needed.
        const closeInfo = advisoryStore.recordPartialClose(advisories, contract, action, sizePercent);
        if (closeInfo) {
          const dollarPnl = closeInfo.closedSizeUsdt * closeInfo.leverage * (pnlPercent / 100);
          balanceTracker.applyPnl(dollarPnl);
        }
      }

      if (sizePercent >= 100) {
        advisoryStore.clearAdvisory(advisories, contract, action);
        advisoriesDirty = true;
      }
      const dirEmoji = action === "long" ? "🟢 LONG" : "🔴 SHORT";
      return {
        telegramMessage: [
          `🤖 *AI SUGGESTS CLOSING* ${dirEmoji} *${contract}* (${sizePercent}%)`,
          `Reasoning: ${reasoning}`,
          ``,
          `_No order has been placed. Execute manually on CoinDCX if you agree._`,
        ].join("\n"),
        resultForModel: { status: "queued_for_manual_execution", note: "Sent to user via Telegram. Not executed." },
      };
    },

    async update_position_stop_loss({ contract: rawContract, action, newStop, reasoning }) {
      const contract = normalizeContract(rawContract);
      await exchange.setPositionStopLoss(contract, newStop);
      advisoryStore.recordStopUpdate(advisories, contract, action, newStop);
      advisoriesDirty = true;
      const dirEmoji = action === "long" ? "🟢 LONG" : "🔴 SHORT";
      return {
        telegramMessage: [
          `🤖 *AI wants to move STOP* ${dirEmoji} *${contract}* → ${newStop}`,
          `Reasoning: ${reasoning}`,
          ``,
          `_No order has been modified. Execute manually on CoinDCX if you agree._`,
        ].join("\n"),
        resultForModel: { status: "queued_for_manual_execution", note: "Sent to user via Telegram. Not executed." },
      };
    },

    async execute_partial_take_profit({ contract: rawContract, action, stage, closePercent, newStop, currentPrice, reasoning }) {
      const contract = normalizeContract(rawContract);
      await exchange.closePosition(contract, closePercent);

      // Auto-record the partial outcome too, same logic as close_position.
      const adv = advisoryStore.getAdvisory(advisories, contract, action);
      if (adv && currentPrice) {
        const dir = action === "long" ? 1 : -1;
        const pnlPercent = ((currentPrice - adv.entryPrice) * dir / adv.entryPrice) * 100;
        const symbol = contract.replace(/^[A-Z]-/, "").replace(/_USDT$/, "");
        tradeOutcomeLog.recordOutcome(symbol, Number(pnlPercent.toFixed(2)), "take_profit");

        const closeInfo = advisoryStore.recordPartialClose(advisories, contract, action, closePercent);
        if (closeInfo) {
          const dollarPnl = closeInfo.closedSizeUsdt * closeInfo.leverage * (pnlPercent / 100);
          balanceTracker.applyPnl(dollarPnl);
        }
      }

      advisoryStore.recordStageAdvised(advisories, contract, action, stage);
      advisoryStore.recordStopUpdate(advisories, contract, action, newStop);
      advisoriesDirty = true;
      const dirEmoji = action === "long" ? "🟢 LONG" : "🔴 SHORT";
      return {
        telegramMessage: [
          `🎯 *AI wants PARTIAL TAKE-PROFIT* ${dirEmoji} *${contract}* — stage ${stage}`,
          `Close ~${closePercent}% | New stop: ${newStop}`,
          `Reasoning: ${reasoning}`,
          ``,
          `_No order has been placed. Execute manually on CoinDCX if you agree._`,
        ].join("\n"),
        resultForModel: { status: "queued_for_manual_execution", note: "Sent to user via Telegram. Not executed." },
      };
    },

    async cancel_order({ orderId, contract: rawContract, reasoning }) {
      const contract = normalizeContract(rawContract);
      await exchange.cancelOrder(orderId, contract);
      return {
        telegramMessage: [
          `🤖 *AI wants to CANCEL an order* on *${contract}*${orderId ? ` (order ${orderId})` : ""}`,
          `Reasoning: ${reasoning}`,
          ``,
          `_No order has been cancelled. Execute manually on CoinDCX if you agree._`,
        ].join("\n"),
        resultForModel: { status: "queued_for_manual_execution", note: "Sent to user via Telegram. Not executed." },
      };
    },
  };

  return {
    declarations,
    handlers,
    isExecutionTool: (name) =>
      ["open_position", "close_position", "update_position_stop_loss", "execute_partial_take_profit", "cancel_order"].includes(name),
    persistAdvisories: () => {
      if (advisoriesDirty) advisoryStore.saveAdvisories(advisories);
    },
    getWarnings: () => runWarnings,
  };
}

module.exports = { buildTools, analyzeSymbol, getActivePositions };
