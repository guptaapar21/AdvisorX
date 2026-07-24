// Runs on its OWN separate fast schedule (~1 minute), completely
// independent of the main 5-minute agent. Zero Gemini calls - just real
// price vs. already-known stop/target levels (pure arithmetic). This is a
// SPEED layer on top of the 5-minute agent, not a replacement for it: the
// 5-minute cycle still does the real reasoning (staged take-profit
// decisions, reversal scoring, new entries). This only answers "has price
// already crossed a level we know about, right now" and pings immediately
// if so - so you're not waiting up to 5 minutes to find out.
//
// IMPORTANT: this never places or modifies any real order. It's purely
// informational, same as everything else in this bot. Your REAL protection
// against a bad move is still the stop-loss order you place on CoinDCX
// yourself - this just tells you faster when a level has been crossed.

const exchange = require("./coindcxExchangeClient");
const advisoryStore = require("./advisoryStore");
const { sendTelegramMessage } = require("./telegram");
const { getActivePositions } = require("./agentTools");
const scorecard = require("./scorecard");

const fs = require("fs");
const path = require("path");
const WATCH_STATE_FILE = path.join(__dirname, "fastWatchState.json");

function loadWatchState() {
  try { return JSON.parse(fs.readFileSync(WATCH_STATE_FILE, "utf8")); } catch { return {}; }
}
function saveWatchState(state) {
  fs.writeFileSync(WATCH_STATE_FILE, JSON.stringify(state, null, 2));
}

function contractToSymbol(contract) {
  return contract.replace(/^[A-Z]-/, "").replace(/_USDT$/, "");
}

async function run(config, creds) {
  const positionsRaw = await exchange.getPositions(creds);
  const activePositions = getActivePositions(positionsRaw);

  if (activePositions.length === 0) {
    console.log("Fast watch: no open positions, nothing to check.");
    // Only push a scorecard update if we just transitioned from having a
    // position to being flat (one final "closed out" refresh). Otherwise
    // every 1-minute fastwatch cycle would spam a fresh "No open positions"
    // message forever while flat.
    const watchState = loadWatchState();
    if (watchState.__hadOpenPositions) {
      await scorecard.updateScorecard([], config.strategy);
      watchState.__hadOpenPositions = false;
      saveWatchState(watchState);
    }
    return;
  }

  const advisories = advisoryStore.loadAdvisories();
  const reconciled = advisoryStore.reconcileWithRealPositions(advisories, activePositions);
  if (reconciled) {
    advisoryStore.saveAdvisories(advisories);
    console.log("Fast watch: reconciled advisory entry price(s) with real CoinDCX fill price.");
  }

  const watchState = loadWatchState();
  let watchStateDirty = false;
  const scorecardPositions = [];

  for (const pos of activePositions) {
    const contract = pos.pair ?? pos.contract;
    const rawSize = Number(pos.active_pos ?? pos.size ?? 0);
    const action = rawSize > 0 ? "long" : "short";
    const symbol = contractToSymbol(contract);
    const key = `${contract}:${action}`;

    const adv = advisoryStore.getAdvisory(advisories, contract, action);
    if (!adv) {
      console.log(`Fast watch: ${contract} has no recorded advisory (not opened by this bot) - skipping.`);
      continue;
    }

    let currentPrice;
    try {
      currentPrice = await exchange.getCurrentPrice(symbol, config.marketType);
    } catch (err) {
      console.log(`Fast watch: ${symbol} price fetch failed - ${err.message}`);
      continue;
    }

    const dir = action === "long" ? 1 : -1;
    const r = Math.abs(adv.entryPrice - adv.initialStop);
    const currentStop = adv.lastAdvisedStop;
    const stopCrossed = action === "long" ? currentPrice <= currentStop : currentPrice >= currentStop;

    scorecardPositions.push({
      contract, action, entryPrice: adv.entryPrice, currentPrice, currentStop,
      // CoinDCX shows ROE (return on margin, i.e. leveraged) - this was
      // showing raw unleveraged price movement instead, so it never
      // matched what's actually on screen in the app (e.g. 0.12% here vs
      // 1.90% ROE on CoinDCX for the same move, at 10x leverage). Multiply
      // by leverage to match CoinDCX's own convention.
      pnlPercent: ((currentPrice - adv.entryPrice) * dir / adv.entryPrice) * 100 * (adv.leverage || 1),
    });

    // Next target the AI hasn't already advised on
    const stages = [
      { key: "1", r: 1 }, { key: "2", r: 2 }, { key: "3", r: 3 },
    ];
    const currentR = r > 0 ? ((currentPrice - adv.entryPrice) * dir) / r : 0;
    const nextStage = stages.find((s) => !adv.stagesAdvised?.[s.key] && currentR >= s.r);

    const prevState = watchState[key] || {};

    if (stopCrossed && !prevState.stopAlerted) {
      await sendTelegramMessage(
        `⚡ *Fast check*: ${contract} (${action}) has crossed its stop level (${currentStop}). Current price: ${currentPrice}.\n` +
        `_This is a faster heads-up only - the full agent will reason about this on its next 5-min cycle. If you have a real stop-loss on CoinDCX, it should already be handling this._`
      );
      watchState[key] = { ...prevState, stopAlerted: true };
      watchStateDirty = true;
    } else if (!stopCrossed && prevState.stopAlerted) {
      // Price came back - clear so a future crossing alerts again
      watchState[key] = { ...prevState, stopAlerted: false };
      watchStateDirty = true;
    }

    if (nextStage && prevState.lastStageAlerted !== nextStage.key) {
      await sendTelegramMessage(
        `⚡ *Fast check*: ${contract} (${action}) has reached ${nextStage.r}R (current price: ${currentPrice}). A take-profit stage may be ready.\n` +
        `_This is a faster heads-up only - the full agent will confirm and act on its next 5-min cycle._`
      );
      watchState[key] = { ...watchState[key], lastStageAlerted: nextStage.key };
      watchStateDirty = true;
    }
  }

  if (!watchState.__hadOpenPositions) {
    watchState.__hadOpenPositions = true;
    watchStateDirty = true;
  }
  if (watchStateDirty) saveWatchState(watchState);
  await scorecard.updateScorecard(scorecardPositions, config.strategy);
  console.log("Fast watch run complete.");
}

module.exports = { run };
