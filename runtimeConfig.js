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
  }

  runtimeState.lastTelegramUpdateId = latestUpdateId;
  return runtimeState;
}

// Rebuilds the effective config for this run, applying any runtime
// strategy override on top of the base config from config.js.
function applyRuntimeOverrides(baseConfig, runtimeState) {
  if (!runtimeState.strategy || runtimeState.strategy === baseConfig.strategy) return baseConfig;

  const params = getStrategyParams(runtimeState.strategy, baseConfig.maxLeverage);
  return {
    ...baseConfig,
    strategy: runtimeState.strategy,
    // This was a real gap: config.minScore is the actual gate deciding what
    // shows up as a candidate at all, but it previously stayed fixed at
    // config.js's value regardless of which strategy was active - so
    // switching strategy silently didn't change the threshold you'd see,
    // even though each preset has its own real minScore. Fixed here.
    minScore: STRATEGY_SCORE_WEIGHTS[runtimeState.strategy]?.minScore ?? baseConfig.minScore,
    riskRules: {
      ...baseConfig.riskRules,
      leverageMin: params.leverageMin,
      leverageMax: params.leverageMax,
      positionSizeMinPercent: params.positionSizeMin,
      positionSizeMaxPercent: params.positionSizeMax,
    },
    stopLoss: {
      ...baseConfig.stopLoss,
      atrMultiplier: params.scientificStopLoss.atrMultiplier,
      minStopLossPercent: params.scientificStopLoss.minDistance,
      maxStopLossPercent: params.scientificStopLoss.maxDistance,
    },
  };
}

module.exports = { loadRuntimeConfig, saveRuntimeConfig, processIncomingCommands, applyRuntimeOverrides, VALID_STRATEGIES };
