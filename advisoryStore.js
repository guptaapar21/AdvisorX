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
function recordOpen(advisories, contract, action, entryPrice, stopPrice) {
  advisories[keyFor(contract, action)] = {
    entryPrice,
    initialStop: stopPrice,
    lastAdvisedStop: stopPrice,
    stagesAdvised: {},
    openedAt: Date.now(),
  };
  return advisories;
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

module.exports = {
  loadAdvisories,
  saveAdvisories,
  recordOpen,
  recordStopUpdate,
  recordStageAdvised,
  clearAdvisory,
  getAdvisory,
};
