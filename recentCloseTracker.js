// Tracks when a symbol+direction was last closed (by any means - manual,
// stop-loss, take-profit, or /closeposition), so a fresh setup on the
// same coin+direction doesn't get immediately re-suggested/re-opened the
// moment it re-qualifies. Kept as its own file rather than folded into
// advisories.json, since other code (fastWatch.js's orphan-cleanup loop)
// iterates every key in that file assuming each one is a real position
// record - adding a differently-shaped entry there would corrupt that.

const fs = require("fs");
const path = require("path");

const STATE_FILE = path.join(__dirname, "recentCloses.json");

function loadRecentCloses() {
  try {
    return JSON.parse(fs.readFileSync(STATE_FILE, "utf8"));
  } catch {
    return {};
  }
}

function saveRecentCloses(state) {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function keyFor(contract, action) {
  return `${contract}:${action}`;
}

function recordClose(contract, action) {
  const state = loadRecentCloses();
  state[keyFor(contract, action)] = Date.now();
  saveRecentCloses(state);
}

// Returns { onCooldown: boolean, minutesRemaining: number|null }
function checkCooldown(contract, action, cooldownMs) {
  const state = loadRecentCloses();
  const closedAt = state[keyFor(contract, action)];
  if (!closedAt) return { onCooldown: false, minutesRemaining: null };
  const elapsedMs = Date.now() - closedAt;
  if (elapsedMs >= cooldownMs) return { onCooldown: false, minutesRemaining: null };
  return { onCooldown: true, minutesRemaining: Math.ceil((cooldownMs - elapsedMs) / 60000) };
}

module.exports = { recordClose, checkCooldown };
