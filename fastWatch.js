// Runs on its OWN separate fast schedule (~2 minutes), completely
// independent of the main 5-minute agent. Zero Gemini calls - just real
// price vs. already-known stop/target levels (pure arithmetic). This is a
// SPEED layer on top of the 5-minute agent, not a replacement for it: the
// 5-minute cycle still does the real reasoning (staged take-profit
// decisions, reversal scoring, new entries). This only answers "has price
// already crossed a level we know about, right now" and pings immediately
// if so - so you're not waiting up to 5 minutes to find out - PLUS a
// plain recurring position status (entry/current/stop/PnL) every cycle
// while a position is open.
//
// UPDATED: this bot now DOES place real orders (entry, and a bracket
// stop-loss/take-profit immediately after). That bracket is two
// INDEPENDENT orders, not a native one-cancels-other pair - when one side
// fills and the position closes, the other side is left resting on the
// exchange with nothing left to act on. THIS is the layer responsible for
// catching that: every cycle, it checks whether any advisory it's
// tracking no longer has a matching real position, and if so, cancels
// whatever's left over and records the real outcome.
//
// Rebuilt: no more liveSnapshot.json / KWGT widget feed, no more edited
// scorecard message, no more coin scores here. Flat = totally silent.
// Open = a fresh Telegram message every cycle with just this position's
// numbers.

const exchange = require("./coindcxExchangeClient");
const advisoryStore = require("./advisoryStore");
const { sendTelegramMessage } = require("./telegram");
const { getActivePositions } = require("./positionUtils");
const scorecard = require("./scorecard");
const tradeOutcomeLog = require("./tradeOutcomeLog");
const balanceTracker = require("./balanceTracker");

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

  // Orphaned-bracket-order cleanup - THE actual gap this file was always
  // supposed to cover (per the header comment) but never actually
  // implemented. Runs BEFORE the early-return-on-flat check below, since
  // the exact case that matters is "a position just closed" - meaning
  // activePositions may now be empty or missing this specific contract,
  // which is precisely when a leftover bracket order needs cancelling.
  const advisories = advisoryStore.loadAdvisories();
  let advisoriesDirty = false;
  for (const [key, adv] of Object.entries(advisories)) {
    const lastColon = key.lastIndexOf(":");
    const contract = key.slice(0, lastColon);
    const action = key.slice(lastColon + 1);
    const stillOpen = activePositions.some((p) => {
      const posContract = p.pair ?? p.contract;
      const posDirection = Number(p.active_pos ?? p.size ?? 0) > 0 ? "long" : "short";
      return posContract === contract && posDirection === action;
    });
    if (stillOpen) continue;

    // This advisory's position is gone - it closed since we last
    // checked (via one side of its bracket, or some other real close).
    // Cancel whatever's left over for this contract, since there's no
    // position for it to act on anymore.
    let cancelledCount = 0;
    let cancelError = null;
    let cancelErrorSource = null; // "confirmed_cancel_failed" | "legacy_endpoint_unverifiable"
    const knownOrderIds = [adv.stopOrderId, adv.takeProfitOrderId].filter(Boolean);
    if (knownOrderIds.length > 0) {
      // Preferred path: cancel the exact orders placed for this advisory,
      // by ID, via the confirmed /orders/cancel endpoint. Exactly one of
      // these already triggered (that's why we're here) and will come
      // back as "not found"/already-filled - that's expected, not an
      // error, so we don't let one failed cancel stop the other.
      for (const orderId of knownOrderIds) {
        try {
          await exchange.cancelOrder(creds, orderId);
          cancelledCount++;
        } catch (err) {
          // Expected for whichever side already triggered. Only surface
          // this as a real problem if BOTH cancels fail below.
          cancelError = err.message;
        }
      }
      if (cancelledCount > 0) {
        cancelError = null;
      } else {
        // Both known-ID cancels failed on the CONFIRMED /orders/cancel
        // endpoint - this is a genuine signal something's wrong, not an
        // endpoint-availability artifact.
        cancelErrorSource = "confirmed_cancel_failed";
      }
    } else {
      // Legacy advisory recorded before order-ID tracking was added -
      // fall back to the best-effort list+filter approach, which depends
      // on an endpoint known to 404 (see coindcxExchangeClient.js). A
      // failure here means "couldn't check," not "confirmed a stale order."
      try {
        const ordersRaw = await exchange.getActiveOrders(creds);
        const orders = Array.isArray(ordersRaw) ? ordersRaw : (ordersRaw?.data || []);
        const staleOrders = orders.filter((o) => (o.pair ?? o.contract) === contract);
        for (const order of staleOrders) {
          await exchange.cancelOrder(creds, order.id);
          cancelledCount++;
        }
      } catch (err) {
        cancelError = err.message;
        cancelErrorSource = "legacy_endpoint_unverifiable";
      }
    }

    // Fix: only treat this as a real, order-backed position that
    // genuinely closed if we have actual evidence of that (a known order
    // ID recorded when it was opened). Without that, this advisory is
    // either an advisory-only suggestion that was never executed, or
    // predates real execution entirely - reporting a fabricated P&L
    // percentage and feeding it into the real balance tracker / trade
    // outcome log would corrupt both with fictional trades that never
    // happened. Previously this ran unconditionally for every advisory
    // with no matching position, regardless of whether it was ever real.
    if (knownOrderIds.length === 0) {
      console.log(`Fast watch: clearing stale advisory for ${contract} ${action} - no real order IDs on record, so no real outcome to report (likely a never-executed suggestion or pre-dates order tracking).`);
      advisoryStore.clearAdvisory(advisories, contract, action);
      advisoriesDirty = true;
      continue;
    }

    // Record the real outcome using the last known price (close enough -
    // this runs every ~2 minutes, so price shouldn't have moved far from
    // whatever level actually triggered the close).
    const symbol = contractToSymbol(contract);
    let outcomeNote = "";
    try {
      const currentPrice = await exchange.getCurrentPrice(symbol, config.marketType);
      const dir = action === "long" ? 1 : -1;
      const pnlPercent = ((currentPrice - adv.entryPrice) * dir / adv.entryPrice) * 100;
      tradeOutcomeLog.recordOutcome(symbol, Number(pnlPercent.toFixed(2)), "bracket_order");
      const dollarPnl = adv.positionSizeUsdt * adv.leverage * (pnlPercent / 100);
      balanceTracker.applyPnl(dollarPnl);
      outcomeNote = `~${pnlPercent >= 0 ? "+" : ""}${pnlPercent.toFixed(2)}% (approx, based on current price - exact bracket trigger price isn't directly known)`;
    } catch (err) {
      outcomeNote = `couldn't compute (${err.message})`;
    }

    await sendTelegramMessage(
      `🔔 *${contract} ${action}* closed via its bracket order (stop or take-profit triggered).\n` +
      `Outcome: ${outcomeNote}\n` +
      (cancelErrorSource === "confirmed_cancel_failed"
        ? `⚠️ Tried to cancel the other resting order but the cancel itself failed (${cancelError}) - please check CoinDCX for a stale order on this contract.`
        : cancelErrorSource === "legacy_endpoint_unverifiable"
          ? `ℹ️ Couldn't verify the other resting order was cancelled (CoinDCX's order-listing API returned an error for this legacy position, a known limitation - not a confirmed stale order). Worth a quick manual check on CoinDCX, but likely nothing to do.`
          : cancelledCount > 0
            ? `Cancelled ${cancelledCount} leftover order(s) for this contract.`
            : `No leftover order found to cancel.`)
    );

    advisoryStore.clearAdvisory(advisories, contract, action);
    advisoriesDirty = true;
  }
  if (advisoriesDirty) {
    advisoryStore.saveAdvisories(advisories);
  }

  if (activePositions.length === 0) {
    // Flat: no interaction at all, no Telegram message of any kind.
    console.log("Fast watch: no open positions, nothing to check, staying silent.");
    return;
  }

  const reconciled = advisoryStore.reconcileWithRealPositions(advisories, activePositions);
  if (reconciled) {
    advisoryStore.saveAdvisories(advisories);
    console.log("Fast watch: reconciled advisory entry price(s) with real CoinDCX fill price.");
  }

  const watchState = loadWatchState();
  let watchStateDirty = false;
  const statusPositions = [];

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

    // CoinDCX shows ROE (return on margin, i.e. leveraged) - multiply by
    // leverage to match CoinDCX's own convention on-screen.
    const pnlPercent = ((currentPrice - adv.entryPrice) * dir / adv.entryPrice) * 100 * (adv.leverage || 1);

    statusPositions.push({
      contract, action, entryPrice: adv.entryPrice, currentPrice, currentStop, pnlPercent,
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

  if (watchStateDirty) saveWatchState(watchState);
  await scorecard.sendPositionStatus(statusPositions, config.strategy);
  console.log("Fast watch run complete.");
}

module.exports = { run };
