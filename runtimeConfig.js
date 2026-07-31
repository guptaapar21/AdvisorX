// Lets you switch the active strategy preset via a Telegram command
// (e.g. "/strategy aggressive") instead of editing config.js and
// redeploying. Persisted so it survives across stateless cron runs.

const fs = require("fs");
const path = require("path");
const { getTelegramUpdates, sendTelegramMessage } = require("./telegram");
const { getStrategyParams } = require("./strategyParams");
const { STRATEGY_SCORE_WEIGHTS } = require("./opportunityScorer");
const { BACKTESTED_COINS, buildScanThresholdMap, getTierLabel } = require("./backtestedStrategy");
const exchange = require("./coindcxExchangeClient");
const { getActivePositions } = require("./positionUtils");
const advisoryStore = require("./advisoryStore");

const RUNTIME_FILE = path.join(__dirname, "runtimeConfig.json");
const VALID_STRATEGIES = ["ultra-short", "swing-trend", "conservative", "balanced", "aggressive", "backtested"];

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
async function processIncomingCommands(runtimeState, creds) {
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
    }

    // /backtested - one single strategy, no sub-mode to pick. Each coin is
    // scanned at its own loosest tier (found via the real cloud backtest
    // sweep); whichever tier a real score actually clears (aggressive or
    // balanced) is just reported in the message afterward, not selected
    // up front.
    if (msg.text.trim().match(/^\/backtested\s*$/i)) {
      runtimeState.strategy = "backtested";
      const thresholds = buildScanThresholdMap();
      const lines = BACKTESTED_COINS.map((c) => `${c}: scans from ${thresholds[c]}`);
      await sendTelegramMessage(
        `✅ Strategy switched to *backtested*. Takes effect from the next run.\n` +
        `${lines.join(", ")}\n` +
        `BTC and XRP are excluded from scanning entirely under this strategy.`
      );
    } else if (msg.text.trim() === "/status") {
      const current = runtimeState.strategy || "balanced (default)";
      // Show the REAL currently-used balance (auto-tracked, may have
      // drifted from whatever was last manually set via P&L on trades
      // this bot has closed) - not just the raw seed value, since that's
      // what actually gets used in the risk-cap calculation each run.
      const balanceTracker = require("./balanceTracker");
      const seedBalance = runtimeState.manualBalanceOverride;
      let balanceLine = "ℹ️ Futures balance: not set - send /setbalance <amount> to enable the risk cap.";
      if (typeof seedBalance === "number" && seedBalance > 0) {
        const trackedBalance = balanceTracker.getCurrentBalance(seedBalance);
        if (typeof trackedBalance === "number") {
          const drifted = Math.abs(trackedBalance - seedBalance) > 0.01;
          balanceLine = drifted
            ? `ℹ️ Futures balance in use: *${trackedBalance.toFixed(2)} USDT* (auto-tracked from your last /setbalance of ${seedBalance}, adjusted by trades this bot has closed since)`
            : `ℹ️ Futures balance in use: *${trackedBalance.toFixed(2)} USDT* (as last set via /setbalance, no closed trades yet to adjust it)`;
        }
      }
      const autoTradeLine = runtimeState.autoTradingPaused
        ? "⏸️ Automatic trading: PAUSED (send /resumeauto to re-enable)"
        : "▶️ Automatic trading: ACTIVE";

      // Fix #7: previously no single place showed a consolidated view of
      // everything currently open at once - each position reported
      // individually via its own updates, but "3 positions open right
      // now: SOL, DOGE, ETH, $X total risk" didn't exist anywhere.
      let portfolioLine = "ℹ️ Positions: unable to check right now.";
      if (creds && creds.apiKey) {
        try {
          const { computePortfolioRiskAndMargin } = require("./positionUtils");
          const portfolio = await computePortfolioRiskAndMargin(exchange, creds);
          if (portfolio.positions.length === 0) {
            portfolioLine = "ℹ️ Positions: none currently open.";
          } else {
            const lines = portfolio.positions.map((p) =>
              `  ${p.contract}: ${p.quantity.toFixed(2)} qty @ ${p.leverage}x${p.riskUsdt !== null ? ` (risk: ${p.riskUsdt.toFixed(2)} USDT)` : " (stop not found)"}`
            );
            portfolioLine = `ℹ️ ${portfolio.positions.length} position(s) open, ${portfolio.totalRiskUsdt.toFixed(2)} USDT total risk:\n${lines.join("\n")}`;
          }
        } catch (err) {
          portfolioLine = `ℹ️ Positions: couldn't verify (${err.message}).`;
        }
      }

      await sendTelegramMessage(`ℹ️ Current strategy: *${current}*\n${balanceLine}\n${autoTradeLine}\n${portfolioLine}`);
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
    // /pauseauto and /resumeauto - stops/restarts AUTOMATIC real order
    // placement for NEW entries only. An already-open position's resting
    // SL/TP bracket orders are untouched either way - they live on the
    // exchange independently of the bot, which is the whole point of
    // placing them immediately as real orders rather than relying on the
    // bot's own uptime to manage the exit.
    if (msg.text.trim().match(/^\/pause(auto)?\s*$/i)) {
      runtimeState.autoTradingPaused = true;
      await sendTelegramMessage(
        "⏸️ Automatic trading PAUSED. No new real orders will be placed until /resumeauto. " +
        "Any position already open keeps its existing stop-loss/take-profit exactly as placed - this does not touch that."
      );
    }
    if (msg.text.trim().match(/^\/resume(auto)?\s*$/i)) {
      runtimeState.autoTradingPaused = false;
      await sendTelegramMessage("▶️ Automatic trading RESUMED. Takes effect from the next run.");
    }

    // /closeposition - manual override, executes a REAL close
    // immediately (not queued for the next cycle), and cleans up
    // whichever resting bracket order (SL or TP) is left over - closing
    // manually doesn't cancel those automatically, they'd otherwise sit
    // on the exchange with nothing to act on.
    if (msg.text.trim().match(/^\/close\s*(position|all)?\s*$/i)) {
      if (!creds || !creds.apiKey) {
        await sendTelegramMessage("❌ /closeposition needs exchange credentials, which aren't available in this context. Try again from the main run.");
      } else {
        try {
          const positionsRaw = await exchange.getPositions(creds);
          const activePositions = getActivePositions(positionsRaw);
          if (activePositions.length === 0) {
            await sendTelegramMessage("ℹ️ No open position found to close.");
          } else {
            for (const pos of activePositions) {
              const contract = pos.pair ?? pos.contract;
              const rawSize = Number(pos.active_pos ?? pos.size ?? 0);
              const direction = rawSize > 0 ? "long" : "short";
              const quantity = Math.abs(rawSize);
              const leverage = Number(pos.leverage) || 1;
              await exchange.closePosition(creds, { pair: contract, direction, quantity, leverage });

              // Clean up any resting bracket order for this exact
              // contract - a manual close doesn't remove these
              // automatically, and a stale SL/TP sitting on the
              // exchange with no position behind it is a real hazard
              // (it would fire against whatever position happens to
              // exist there next).
              try {
                const advisories = advisoryStore.loadAdvisories();
                const adv = advisoryStore.getAdvisory(advisories, contract, direction);
                const knownOrderIds = [adv?.stopOrderId, adv?.takeProfitOrderId].filter(Boolean);
                let legacyEndpointUnverifiable = false;
                if (knownOrderIds.length > 0) {
                  for (const orderId of knownOrderIds) {
                    try { await exchange.cancelOrder(creds, orderId); } catch { /* likely already filled/cancelled - expected for one side */ }
                  }
                } else {
                  try {
                    const ordersRaw = await exchange.getActiveOrders(creds);
                    const orders = Array.isArray(ordersRaw) ? ordersRaw : (ordersRaw?.data || []);
                    const staleOrders = orders.filter((o) => (o.pair ?? o.contract) === contract);
                    for (const order of staleOrders) {
                      await exchange.cancelOrder(creds, order.id);
                    }
                  } catch {
                    legacyEndpointUnverifiable = true;
                  }
                }
                if (adv) {
                  advisoryStore.clearAdvisory(advisories, contract, direction);
                  advisoryStore.saveAdvisories(advisories);
                }
                if (legacyEndpointUnverifiable) {
                  await sendTelegramMessage(`ℹ️ Closed ${contract} but couldn't verify any leftover SL/TP order via CoinDCX's order-listing API (a known limitation for legacy positions, not a confirmed stale order) - worth a quick manual check, but likely nothing to do.`);
                }
              } catch (cleanupErr) {
                await sendTelegramMessage(`⚠️ Closed ${contract} but couldn't confirm/cancel any leftover SL/TP orders (${cleanupErr.message}) - please check manually on CoinDCX.`);
              }

              await sendTelegramMessage(`🛑 Manually closed ${contract} ${direction} (${quantity} qty) per /closeposition.`);
            }
          }
        } catch (err) {
          await sendTelegramMessage(`❌ /closeposition failed: ${err.message}. Please check and close manually on CoinDCX if needed.`);
        }
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

  if (runtimeState.strategy === "backtested") {
    // Every /backtested threshold was found using CONSERVATIVE's actual
    // scoring formula and risk parameters (5-9x leverage, 2.5x ATR stop,
    // 1.0-4.0% stop bounds) - only the score bar itself varies by coin.
    const params = getStrategyParams("conservative", baseConfig.maxLeverage);
    const perCoinMinScore = buildScanThresholdMap(); // loosest tier per coin - always scan here, tier label is reported after the fact
    effectiveConfig = {
      ...effectiveConfig,
      strategy: "backtested",
      // Only scan coins with a real, evidence-based edge - BTC/XRP are
      // excluded outright, not defaulted to some other threshold.
      symbols: BACKTESTED_COINS.filter((c) => perCoinMinScore[c] !== undefined),
      perCoinMinScore,
      minScore: null, // no single global bar under this strategy - see perCoinMinScore
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
  } else if (runtimeState.strategy && runtimeState.strategy !== baseConfig.strategy) {
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

  // Applies regardless of which branch ran above - the pause/resume
  // toggle is orthogonal to strategy selection. This is what
  // open_position actually checks before ever placing a real order.
  effectiveConfig = { ...effectiveConfig, autoTradingPaused: !!runtimeState.autoTradingPaused };

  return effectiveConfig;
}

// Resolves the actual score bar for a given symbol. Every normal preset
// just uses config.minScore, the same for every coin - only /backtested
// sets a real per-coin map (config.perCoinMinScore), which this checks
// first. Centralized here so preFilter.js, agentTools.js, and
// agentIndex.js all read this the same way rather than duplicating the
// fallback logic three times.
function getEffectiveMinScore(config, symbol) {
  return config.perCoinMinScore?.[symbol] ?? config.minScore;
}

module.exports = {
  loadRuntimeConfig, saveRuntimeConfig, processIncomingCommands, applyRuntimeOverrides,
  VALID_STRATEGIES, getEffectiveMinScore,
};
