// Since positions are now REAL (read from CoinDCX via get_positions), we no
// longer need to guess whether a trade was taken. What we still can't get
// from the exchange is the AI's own original stop-loss recommendation and
// which take-profit stages it has already advised on - because no real
// conditional order was ever placed (every execution tool is advisory
// only). This file is just that: a small record of what the AI itself last
// told the user, per open position, so follow-up R-multiple guidance stays
// consistent across runs.

const fs = require("fs");
const path = require("path");

const ADVISORY_FILE = path.join(__dirname, "advisories.json");

function loadAdvisories() {
  try {
    return JSON.parse(fs.readFileSync(ADVISORY_FILE, "utf8"));
  } catch {
    return {};
  }
}

function saveAdvisories(advisories) {
  fs.writeFileSync(ADVISORY_FILE, JSON.stringify(advisories, null, 2));
}

function keyFor(contract, action) {
  return `${contract}:${action}`;
}

// Called when the agent decides to open a position - records its own
// entry/stop recommendation as the reference point for later stage/R math.
function recordOpen(advisories, contract, action, entryPrice, stopPrice, positionSizeUsdt, leverage) {
  advisories[keyFor(contract, action)] = {
    entryPrice,
    initialStop: stopPrice,
    lastAdvisedStop: stopPrice,
    stagesAdvised: {},
    openedAt: Date.now(),
    // Needed to compute real dollar P&L on close, so the running balance
    // tracker can update automatically without asking her to do it by
    // hand. remainingSizeUsdt shrinks as partial take-profits close
    // portions of the position.
    positionSizeUsdt,
    leverage,
    remainingSizeUsdt: positionSizeUsdt,
  };
  return advisories;
}

// Call when a portion of a position closes (partial take-profit or a full
// close). Returns the dollar amount that was just closed, for computing
// its P&L contribution - and shrinks remainingSizeUsdt so the NEXT close
// is computed against what's actually still open, not the original size.
function recordPartialClose(advisories, contract, action, closePercent) {
  const rec = advisories[keyFor(contract, action)];
  if (!rec || typeof rec.remainingSizeUsdt !== "number") return null;
  const closedSizeUsdt = rec.remainingSizeUsdt * (closePercent / 100);
  rec.remainingSizeUsdt = Math.max(0, rec.remainingSizeUsdt - closedSizeUsdt);
  return { closedSizeUsdt, leverage: rec.leverage };
}

function recordStopUpdate(advisories, contract, action, newStop) {
  const rec = advisories[keyFor(contract, action)];
  if (rec) rec.lastAdvisedStop = newStop;
  return advisories;
}

function recordStageAdvised(advisories, contract, action, stageKey) {
  const rec = advisories[keyFor(contract, action)];
  if (rec) rec.stagesAdvised[stageKey] = true;
  return advisories;
}

function clearAdvisory(advisories, contract, action) {
  delete advisories[keyFor(contract, action)];
  return advisories;
}

function getAdvisory(advisories, contract, action) {
  return advisories[keyFor(contract, action)] || null;
}

// Once a real position shows up on CoinDCX, its avg_price is the price you
// ACTUALLY paid - not the price the AI suggested when it made the call.
// Execution delay, slippage, or just taking a few minutes to act all mean
// these can differ. This corrects the tracked entry price to your real
// fill, so R-multiple/target math is based on what you actually paid.
//
// The stop price is deliberately left UNCHANGED: if you placed your real
// stop-loss order on CoinDCX at the exact price the bot suggested, that
// real order sits at that price regardless of your exact fill - shifting
// it here would describe a stop you didn't actually place. R is simply
// recomputed fresh from (real entry, unchanged stop).
//
// Only corrects meaningfully-different prices (>0.05%) to avoid noisy
// no-op updates from float rounding.
function reconcileWithRealPositions(advisories, activePositions) {
  let changed = false;
  for (const pos of activePositions) {
    const contract = pos.pair ?? pos.contract;
    const rawSize = Number(pos.active_pos ?? pos.size ?? 0);
    const action = rawSize > 0 ? "long" : "short";
    const realEntry = Number(pos.avg_price ?? 0);
    if (!contract || !realEntry) continue;

    const key = keyFor(contract, action);
    const adv = advisories[key];
    if (!adv) continue;

    const diffPercent = Math.abs(realEntry - adv.entryPrice) / adv.entryPrice * 100;
    if (diffPercent > 0.05) {
      if (!adv.reconciledFrom) adv.reconciledFrom = adv.entryPrice;
      adv.entryPrice = realEntry;
      changed = true;
    }
  }
  return changed;
}

module.exports = {
  loadAdvisories,
  saveAdvisories,
  recordOpen,
  recordPartialClose,
  recordStopUpdate,
  recordStageAdvised,
  clearAdvisory,
  getAdvisory,
  reconcileWithRealPositions,
};
