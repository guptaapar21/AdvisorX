// Automatically tracks account balance so it doesn't need manual updates
// after every trade. Seeds ONCE from config.manualFuturesBalanceUsdt (the
// real starting figure she provides), then updates itself automatically
// as positions open and close - no need to keep re-entering a number.
//
// HONEST LIMITATION: this can only track P&L for trades the bot itself
// advised opening/closing. If she executes something on CoinDCX the bot
// never suggested (or exits early on her own without the bot's
// close_position ever firing), this running balance will drift from her
// real one. If it ever looks off, she can still correct it by updating
// config.manualFuturesBalanceUsdt - that will re-seed the tracker fresh
// next run.

const fs = require("fs");
const path = require("path");
const STATE_FILE = path.join(__dirname, "balanceState.json");

function loadState() {
  try { return JSON.parse(fs.readFileSync(STATE_FILE, "utf8")); } catch { return null; }
}
function saveState(state) {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

// Returns the current tracked balance. If the tracker has never been
// seeded (first run) OR the config's manual figure has been deliberately
// changed since (she's correcting drift or updating after a deposit/
// withdrawal), re-seeds from the config value. Otherwise uses the
// self-updating tracked value, ignoring the config figure so automatic
// P&L updates aren't overwritten every run.
function getCurrentBalance(manualBaseFromConfig) {
  const state = loadState();

  if (typeof manualBaseFromConfig !== "number" || manualBaseFromConfig <= 0) {
    // No manual base configured - use tracked value if we have one, else nothing.
    return state ? state.balance : null;
  }

  if (!state || state.seededFrom !== manualBaseFromConfig) {
    // First run, or she's changed the manual config value since - treat
    // this as a deliberate reset/correction and re-seed from it.
    const fresh = { balance: manualBaseFromConfig, seededFrom: manualBaseFromConfig, lastUpdatedAt: Date.now() };
    saveState(fresh);
    return fresh.balance;
  }

  return state.balance;
}

// Applies a realized dollar P&L (positive or negative) to the tracked
// balance. Call this every time a real close/partial-close is recorded.
function applyPnl(dollarPnl) {
  const state = loadState();
  if (!state) return; // nothing to update against - getCurrentBalance must run first to seed it
  state.balance = Math.max(0, state.balance + dollarPnl);
  state.lastUpdatedAt = Date.now();
  saveState(state);
}

module.exports = { getCurrentBalance, applyPnl };
