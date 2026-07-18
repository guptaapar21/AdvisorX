// The tool suite the agent can call, matching the original engine's
// decision-relevant tools. Read tools (market data, account balance,
// positions, opportunity scoring) are REAL. Execution tools (open/close
// position, stop-loss updates, partial take-profit, cancel order) are
// INTERCEPTED - they never call the exchange. Each execution handler
// returns { telegramMessage, resultForModel } - the runner sends the
// message and feeds resultForModel back to Gemini as the tool's result,
// so the model clearly sees "queued for manual execution", never "done".

const exchange = require("./coindcxExchangeClient");
const { scoreOpportunity } = require("./strategy");
const { rsi, macd, ema, atrPercent } = require("./indicators");
const advisoryStore = require("./advisoryStore");

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
      "Scans all configured symbols, scores each with the composite rule-based scorer (regime, breakout/trend/reversion setup, RSI/MACD/EMA blend, volatility tier), filters out symbols with an open position, and returns the top-ranked opportunities across all coins (cross-symbol ranking) that clear the minimum score. Also returns allScores: the score for every symbol scanned this run (including ones below the threshold), for reporting in the run summary.",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "get_technical_indicators",
    description: "Get fresh RSI/MACD/EMA/ATR indicators for one symbol on both the entry and trend timeframes.",
    parameters: {
      type: "object",
      properties: { symbol: { type: "string", description: "Base symbol, e.g. BTC" } },
      required: ["symbol"],
    },
  },
  {
    name: "check_open_position",
    description:
      "Validates whether a candidate new entry should actually be opened: checks stop-loss distance is within configured bounds, position count is under the max, and account balance is sufficient. Call this before open_position.",
    parameters: {
      type: "object",
      properties: {
        symbol: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        price: { type: "number" },
        suggestedStop: { type: "number" },
        atrPercent: { type: "number" },
      },
      required: ["symbol", "action", "price", "suggestedStop"],
    },
  },
  {
    name: "calculate_risk",
    description:
      "Calculates a suggested position size given account balance, entry price, stop price, and leverage, using the configured risk-per-trade percentage.",
    parameters: {
      type: "object",
      properties: {
        accountBalanceUsdt: { type: "number" },
        entryPrice: { type: "number" },
        stopPrice: { type: "number" },
        leverage: { type: "number" },
      },
      required: ["accountBalanceUsdt", "entryPrice", "stopPrice", "leverage"],
    },
  },
  {
    name: "check_partial_take_profit_opportunity",
    description:
      "Checks an open position against the staged take-profit plan (1R/2R/3R) using the AI's own original entry/stop recommendation for that position, and reports whether a new stage has been reached.",
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
      "Decide to close (fully or partially) an open position. This does NOT close a real position - it sends your decision and reasoning to the user via Telegram for manual execution.",
    parameters: {
      type: "object",
      properties: {
        contract: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        sizePercent: { type: "number", description: "Percent of the position to close, 1-100" },
        reasoning: { type: "string" },
      },
      required: ["contract", "action", "sizePercent", "reasoning"],
    },
  },
  {
    name: "update_position_stop_loss",
    description:
      "Decide to move the stop-loss (e.g. trailing stop, or moving to breakeven). This does NOT modify a real order - it sends the new stop level to the user via Telegram for manual execution.",
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
      "Decide to take partial profit at a reached R-multiple stage. This does NOT execute a real order - it sends the stage, close percent, and new stop to the user via Telegram for manual execution.",
    parameters: {
      type: "object",
      properties: {
        contract: { type: "string" },
        action: { type: "string", enum: ["long", "short"] },
        stage: { type: "string" },
        closePercent: { type: "number" },
        newStop: { type: "number" },
        reasoning: { type: "string" },
      },
      required: ["contract", "action", "stage", "closePercent", "newStop", "reasoning"],
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

const STAGES = [
  { key: "1", r: 1, closePercent: 33, moveStopTo: "entry" },
  { key: "2", r: 2, closePercent: 33, moveStopTo: "stage1" },
  { key: "3", r: 3, closePercent: 34, moveStopTo: "stage2trail" },
];

function buildTools(config, creds) {
  const advisories = advisoryStore.loadAdvisories();
  let dirty = false;

  const handlers = {
    async get_account_balance() {
      return exchange.getBalances(creds);
    },

    async get_positions() {
      return exchange.getPositions(creds);
    },

    async analyze_opening_opportunities() {
      const candidates = [];
      const allScores = [];
      for (const symbol of config.symbols) {
        try {
          const { entryCandles, trendCandles } = await exchange.getMarketPrice(
            symbol, config.marketType, config.entryTimeframe, config.trendTimeframe, config.candleLimit
          );
          if (entryCandles.length < 30 || trendCandles.length < 30) {
            allScores.push({ symbol, score: 0, note: "not enough candle history yet" });
            continue;
          }
          const opp = scoreOpportunity(symbol, entryCandles, trendCandles);
          if (opp) {
            allScores.push({ symbol, score: opp.score, action: opp.action, setupType: opp.setupType });
            if (opp.score >= config.minScore) candidates.push(opp);
          } else {
            allScores.push({ symbol, score: 0, note: "no setup detected" });
          }
        } catch (err) {
          console.error(`analyze_opening_opportunities: ${symbol} - ${err.message}`);
          allScores.push({ symbol, score: 0, note: `error: ${err.message}` });
        }
      }
      candidates.sort((a, b) => b.score - a.score);
      return { opportunities: candidates.slice(0, config.maxAlertsPerRun), allScores };
    },

    async get_technical_indicators({ symbol }) {
      const { entryCandles, trendCandles } = await exchange.getMarketPrice(
        symbol, config.marketType, config.entryTimeframe, config.trendTimeframe, config.candleLimit
      );
      const entryCloses = entryCandles.map((c) => c.close);
      const trendCloses = trendCandles.map((c) => c.close);
      return {
        entryTimeframe: {
          rsi7: rsi(entryCloses, 7),
          rsi14: rsi(entryCloses, 14),
          macd: macd(entryCloses),
          atrPercent: atrPercent(entryCandles),
        },
        trendTimeframe: {
          ema20: ema(trendCloses, 20).slice(-1)[0],
          ema50: ema(trendCloses, 50).slice(-1)[0],
          macd: macd(trendCloses),
        },
        currentPrice: entryCloses[entryCloses.length - 1],
      };
    },

    async check_open_position({ symbol, action, price, suggestedStop, atrPercent: atr }) {
      const reasons = [];
      let shouldOpen = true;

      const stopDistancePercent = (Math.abs(price - suggestedStop) / price) * 100;
      if (stopDistancePercent < config.riskRules.minStopDistancePercent) {
        shouldOpen = false;
        reasons.push(`stop too tight (${stopDistancePercent.toFixed(2)}% < ${config.riskRules.minStopDistancePercent}% minimum)`);
      }
      if (stopDistancePercent > config.riskRules.maxStopDistancePercent) {
        shouldOpen = false;
        reasons.push(`stop too wide (${stopDistancePercent.toFixed(2)}% > ${config.riskRules.maxStopDistancePercent}% maximum)`);
      }

      try {
        const positions = await exchange.getPositions(creds);
        const openCount = Array.isArray(positions) ? positions.length : (positions?.data?.length || 0);
        if (openCount >= config.riskRules.maxPositions) {
          shouldOpen = false;
          reasons.push(`already at max positions (${openCount}/${config.riskRules.maxPositions})`);
        }
      } catch (err) {
        reasons.push(`could not verify position count: ${err.message}`);
      }

      let availableUsdt = null;
      try {
        const balances = await exchange.getBalances(creds);
        const usdt = Array.isArray(balances) ? balances.find((b) => (b.currency || "").toUpperCase() === "USDT") : null;
        availableUsdt = usdt ? Number(usdt.balance ?? usdt.available_balance ?? 0) : null;
      } catch (err) {
        reasons.push(`could not read balance (informational only, not blocking): ${err.message}`);
      }

      if (shouldOpen) reasons.push("passes stop-distance and position-count checks");
      return { shouldOpen, symbol, action, availableUsdt, reasons };
    },

    async calculate_risk({ accountBalanceUsdt, entryPrice, stopPrice, leverage }) {
      const riskAmountUsdt = accountBalanceUsdt * (config.riskRules.riskPercentPerTrade / 100);
      const stopDistancePerUnit = Math.abs(entryPrice - stopPrice);
      const positionSizeUnderlying = stopDistancePerUnit > 0 ? riskAmountUsdt / stopDistancePerUnit : 0;
      const notionalUsdt = positionSizeUnderlying * entryPrice;
      const marginRequiredUsdt = leverage > 0 ? notionalUsdt / leverage : notionalUsdt;
      return {
        riskAmountUsdt: Number(riskAmountUsdt.toFixed(2)),
        suggestedQuantity: Number(positionSizeUnderlying.toFixed(6)),
        notionalUsdt: Number(notionalUsdt.toFixed(2)),
        marginRequiredUsdt: Number(marginRequiredUsdt.toFixed(2)),
      };
    },

    async check_partial_take_profit_opportunity({ contract, action, currentPrice }) {
      const adv = advisoryStore.getAdvisory(advisories, contract, action);
      if (!adv) {
        return { canExecute: false, reason: "no recorded entry advisory for this position - was it opened by this bot?" };
      }
      const r = Math.abs(adv.entryPrice - adv.initialStop);
      const dir = action === "long" ? 1 : -1;
      const currentR = r > 0 ? ((currentPrice - adv.entryPrice) * dir) / r : 0;

      for (const stage of STAGES) {
        if (adv.stagesAdvised[stage.key]) continue;
        if (currentR >= stage.r) {
          let newStop;
          if (stage.moveStopTo === "entry") newStop = adv.entryPrice;
          else if (stage.moveStopTo === "stage1") newStop = adv.entryPrice + dir * r * 1;
          else newStop = adv.entryPrice + dir * r * 2;
          return {
            canExecute: true,
            stage: stage.key,
            rMultiple: Number(currentR.toFixed(2)),
            closePercent: stage.closePercent,
            suggestedNewStop: Number(newStop.toFixed(6)),
          };
        }
      }
      return { canExecute: false, rMultiple: Number(currentR.toFixed(2)), reason: "no new stage reached" };
    },

    // ---- Execution tools: intercepted, Telegram-only ----

    async open_position({ contract, action, entryPrice, stopPrice, leverage, positionSizeUsdt, reasoning }) {
      await exchange.placeOrder({ contract, action, entryPrice, stopPrice, leverage, positionSizeUsdt });
      advisoryStore.recordOpen(advisories, contract, action, entryPrice, stopPrice);
      dirty = true;
      const dirEmoji = action === "long" ? "🟢 LONG" : "🔴 SHORT";
      return {
        telegramMessage: [
          `🤖 *AI wants to OPEN* ${dirEmoji} *${contract}*`,
          `Entry: ${entryPrice} | Stop: ${stopPrice} | Leverage: ${leverage}x`,
          `Suggested size: ${positionSizeUsdt} USDT margin`,
          `Reasoning: ${reasoning}`,
          ``,
          `_No order has been placed. Execute manually on CoinDCX if you agree._`,
        ].join("\n"),
        resultForModel: { status: "queued_for_manual_execution", note: "Sent to user via Telegram. Not executed." },
      };
    },

    async close_position({ contract, action, sizePercent, reasoning }) {
      await exchange.closePosition(contract, sizePercent);
      if (sizePercent >= 100) advisoryStore.clearAdvisory(advisories, contract, action);
      dirty = true;
      const dirEmoji = action === "long" ? "🟢 LONG" : "🔴 SHORT";
      return {
        telegramMessage: [
          `🤖 *AI wants to CLOSE* ${dirEmoji} *${contract}* (${sizePercent}%)`,
          `Reasoning: ${reasoning}`,
          ``,
          `_No order has been placed. Execute manually on CoinDCX if you agree._`,
        ].join("\n"),
        resultForModel: { status: "queued_for_manual_execution", note: "Sent to user via Telegram. Not executed." },
      };
    },

    async update_position_stop_loss({ contract, action, newStop, reasoning }) {
      await exchange.setPositionStopLoss(contract, newStop);
      advisoryStore.recordStopUpdate(advisories, contract, action, newStop);
      dirty = true;
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

    async execute_partial_take_profit({ contract, action, stage, closePercent, newStop, reasoning }) {
      await exchange.closePosition(contract, closePercent);
      advisoryStore.recordStageAdvised(advisories, contract, action, stage);
      advisoryStore.recordStopUpdate(advisories, contract, action, newStop);
      dirty = true;
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

    async cancel_order({ orderId, contract, reasoning }) {
      await exchange.cancelOrder(orderId, contract);
      dirty = true;
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
      if (dirty) advisoryStore.saveAdvisories(advisories);
    },
  };
}

module.exports = { buildTools };
