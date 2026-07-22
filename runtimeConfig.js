// Lets you switch the active strategy preset via a Telegram command
// (e.g. "/strategy aggressive") instead of editing config.js and
// redeploying. Persisted so it survives across stateless cron runs.

const fs = require("fs");
const path = require("path");
const { getTelegramUpdates, sendTelegramMessage } = require("./telegram");
const { getStrategyParams } = require("./strategyParams");
const { STRATEGY_SCORE_WEIGHTS } = require("./opportunityScorer");

const RUNTIME_FILE = path.join(__dirname, "runtimeConfig.json");
const VALID_STRATEGIES = ["ultra-short", "swing-trend", "conservative", "balanced", "aggressive"];

function loadRuntimeConfig() {
  try {
    return JSON.parse(fs.readFileSync(RUNTIME_FILE, "utf8"));
  } catch {
    return { strategy: null, lastTelegramUpdateId: 0 };
  }
}

function saveRuntimeConfig(state) {
  fs.writeFileSync(RUNTIME_FILE, JSON.stringify(state, null, 2));
}

// Checks for new Telegram commands since the last run and applies any
// valid ones. Always sends a confirmation (or rejection) reply so a
// mistyped command is never silently ignored. Returns the possibly-updated
// runtime state.
async function processIncomingCommands(runtimeState) {
  const { messages, latestUpdateId } = await getTelegramUpdates(runtimeState.lastTelegramUpdateId);

  for (const msg of messages) {
    const match = msg.text.match(/^\/strategy\s+(\S+)/i);
    if (match) {
      const requested = match[1].toLowerCase();
      if (VALID_STRATEGIES.includes(requested)) {
        runtimeState.strategy = requested;
        await sendTelegramMessage(`✅ Strategy switched to *${requested}*. Takes effect from the next run.`);
      } else {
        await sendTelegramMessage(`❌ Unknown strategy "${requested}". Valid options: ${VALID_STRATEGIES.join(", ")}`);
      }
    } else if (msg.text.trim() === "/status") {
      const current = runtimeState.strategy || "balanced (default)";
      await sendTelegramMessage(`ℹ️ Current strategy: *${current}*`);
    }

    // Lets her correct/update the tracked futures balance straight from
    // Telegram (e.g. after a deposit, or if it's drifted) instead of
    // needing to edit config.js and redeploy. Accepts "/setbalance 120",
    // "/updatebalance 120", or "/update bal 120" - a few reasonable
    // variations on the same intent.
    const balMatch = msg.text.match(/^\/(?:set|update)\s*bal(?:ance)?\s*(\d+(?:\.\d+)?)/i);
    if (balMatch) {
      const newBalance = Number(balMatch[1]);
      if (newBalance > 0) {
        runtimeState.manualBalanceOverride = newBalance;
        await sendTelegramMessage(`✅ Futures balance updated to *${newBalance} USDT*. Takes effect from the next run.`);
      } else {
        await sendTelegramMessage(`❌ Couldn't parse a valid balance from "${msg.text}". Try: /setbalance 120`);
      }
    }
  }

  runtimeState.lastTelegramUpdateId = latestUpdateId;
  return runtimeState;
}

// Rebuilds the effective config for this run, applying any runtime
// strategy and/or balance override on top of the base config from config.js.
function applyRuntimeOverrides(baseConfig, runtimeState) {
  let effectiveConfig = baseConfig;

  if (runtimeState.strategy && runtimeState.strategy !== baseConfig.strategy) {
    const params = getStrategyParams(runtimeState.strategy, baseConfig.maxLeverage);
    effectiveConfig = {
      ...effectiveConfig,
      strategy: runtimeState.strategy,
      // This was a real gap: config.minScore is the actual gate deciding what
      // shows up as a candidate at all, but it previously stayed fixed at
      // config.js's value regardless of which strategy was active - so
      // switching strategy silently didn't change the threshold you'd see,
      // even though each preset has its own real minScore. Fixed here.
      minScore: STRATEGY_SCORE_WEIGHTS[runtimeState.strategy]?.minScore ?? baseConfig.minScore,
      riskRules: {
        ...effectiveConfig.riskRules,
        leverageMin: params.leverageMin,
        leverageMax: params.leverageMax,
        positionSizeMinPercent: params.positionSizeMin,
        positionSizeMaxPercent: params.positionSizeMax,
      },
      stopLoss: {
        ...effectiveConfig.stopLoss,
        atrMultiplier: params.scientificStopLoss.atrMultiplier,
        minStopLossPercent: params.scientificStopLoss.minDistance,
        maxStopLossPercent: params.scientificStopLoss.maxDistance,
      },
    };
  }

  if (typeof runtimeState.manualBalanceOverride === "number" && runtimeState.manualBalanceOverride > 0) {
    // Overriding this here (rather than touching balanceTracker.js at all)
    // means a Telegram-set balance is treated exactly like editing
    // config.js's manualFuturesBalanceUsdt herself - the tracker's existing
    // "seed value changed, re-seed fresh from it" logic just naturally
    // picks this up, no special-casing needed.
    effectiveConfig = { ...effectiveConfig, manualFuturesBalanceUsdt: runtimeState.manualBalanceOverride };
  }

  return effectiveConfig;
}

module.exports = { loadRuntimeConfig, saveRuntimeConfig, processIncomingCommands, applyRuntimeOverrides, VALID_STRATEGIES };
