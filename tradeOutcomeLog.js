// Faithful port of coinCooldownManager.ts's logic, backed by a local JSON
// log instead of SQLite (we have no database - this bot never places real
// orders, so it can only log outcomes for trades it advised on and that
// were later reported closed via close_position/execute_partial_take_profit).

const fs = require("fs");
const path = require("path");

const LOG_FILE = path.join(__dirname, "tradeOutcomeLog.json");

function loadLog() {
  try {
    return JSON.parse(fs.readFileSync(LOG_FILE, "utf8"));
  } catch {
    return [];
  }
}

function saveLog(entries) {
  // Keep the log bounded - only the last 200 entries are ever needed
  // (48h lookback window at realistic trade volume).
  const trimmed = entries.slice(-200);
  fs.writeFileSync(LOG_FILE, JSON.stringify(trimmed, null, 2));
}

// Records a closed trade's outcome. pnlPercent should be negative for a loss.
function recordOutcome(symbol, pnlPercent, closeReason) {
  const entries = loadLog();
  entries.push({ symbol, pnlPercent, closeReason, closedAt: Date.now() });
  saveLog(entries);
}

function getSymbolLossStats(symbol) {
  const entries = loadLog();
  const now = Date.now();
  const losses24h = entries.filter((e) => e.symbol === symbol && e.pnlPercent < 0 && now - e.closedAt < 24 * 3600 * 1000);
  const losses48h = entries.filter((e) => e.symbol === symbol && e.pnlPercent < 0 && now - e.closedAt < 48 * 3600 * 1000);
  const avgLossPercent24h = losses24h.length > 0
    ? losses24h.reduce((sum, e) => sum + Math.abs(e.pnlPercent), 0) / losses24h.length
    : 0;
  const hasReversalLoss = losses24h.some((e) => e.closeReason === "trend_reversal");
  return { losses24h: losses24h.length, losses48h: losses48h.length, avgLossPercent24h, hasReversalLoss };
}

// Matches calculateHistoricalLossPenalty exactly.
function calculateHistoricalLossPenalty(stats) {
  let penalty = 0;
  if (stats.losses24h > 0) {
    penalty += 20;
    if (stats.avgLossPercent24h >= 20) penalty += 15;
    else if (stats.avgLossPercent24h >= 15) penalty += 10;
    else if (stats.avgLossPercent24h >= 10) penalty += 5;
  }
  if (stats.losses48h >= 2) penalty += 20;
  if (stats.hasReversalLoss) penalty += 15;
  return penalty;
}

async function historicalPenaltyFn(symbol) {
  const stats = getSymbolLossStats(symbol);
  return calculateHistoricalLossPenalty(stats);
}

// Matches isSymbolInCooldown exactly (single loss >=15% -> 12h,
// 2 losses/24h -> 24h, 3+ losses/24h -> 48h, trend-reversal loss -> +6h extra).
function isSymbolInCooldown(symbol) {
  const entries = loadLog();
  const now = Date.now();
  const losses24h = entries
    .filter((e) => e.symbol === symbol && e.pnlPercent < 0 && now - e.closedAt < 24 * 3600 * 1000)
    .sort((a, b) => b.closedAt - a.closedAt);

  if (losses24h.length === 0) return { inCooldown: false };

  const mostRecent = losses24h[0];
  const check = (hours, reason) => {
    const cooldownUntil = mostRecent.closedAt + hours * 3600 * 1000;
    if (now < cooldownUntil) {
      return { inCooldown: true, reason, remainingHours: Number(((cooldownUntil - now) / 3600000).toFixed(1)) };
    }
    return null;
  };

  if (Math.abs(mostRecent.pnlPercent) >= 15) {
    const r = check(12, `single loss of ${Math.abs(mostRecent.pnlPercent).toFixed(1)}% exceeded 15% threshold`);
    if (r) return r;
  }
  if (losses24h.length >= 3) {
    const r = check(48, `${losses24h.length} losses in 24h, extended cooldown`);
    if (r) return r;
  }
  if (losses24h.length >= 2) {
    const r = check(24, `${losses24h.length} losses in 24h`);
    if (r) return r;
  }
  if (mostRecent.closeReason === "trend_reversal") {
    const r = check(6, "trend-reversal loss, waiting for market to stabilize");
    if (r) return r;
  }
  return { inCooldown: false };
}

module.exports = { recordOutcome, getSymbolLossStats, calculateHistoricalLossPenalty, historicalPenaltyFn, isSymbolInCooldown };
