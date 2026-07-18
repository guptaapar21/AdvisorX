// The cron itself runs every 5 minutes (cheap - public repo, unlimited
// Actions minutes). But sending a Telegram message every 5 minutes when
// there's genuinely nothing happening would be spammy. This throttles
// ONLY the routine "idle" message to once per 15 minutes; the moment
// there's an open position or the AI takes any action, every run messages
// immediately with no throttling - that's when tight monitoring matters.

const fs = require("fs");
const path = require("path");

const THROTTLE_FILE = path.join(__dirname, "idleThrottleState.json");
const IDLE_INTERVAL_MS = 15 * 60 * 1000;

function loadThrottleState() {
  try {
    return JSON.parse(fs.readFileSync(THROTTLE_FILE, "utf8"));
  } catch {
    return { lastIdleMessageSentAt: 0 };
  }
}

function saveThrottleState(state) {
  fs.writeFileSync(THROTTLE_FILE, JSON.stringify(state, null, 2));
}

// Returns true if an idle message should actually be SENT this run (either
// because enough time has passed, or because there's no prior record at
// all - e.g. first run ever). Updates and persists the timestamp when it
// returns true, so the caller doesn't need to manage that separately.
function shouldSendIdleMessage() {
  const state = loadThrottleState();
  const now = Date.now();
  if (now - (state.lastIdleMessageSentAt || 0) >= IDLE_INTERVAL_MS) {
    state.lastIdleMessageSentAt = now;
    saveThrottleState(state);
    return true;
  }
  return false;
}

// Called whenever there WAS activity (position open, action taken) - resets
// the idle timer so that as soon as things go quiet again, the next idle
// run always messages immediately rather than waiting out a stale timer.
function resetIdleThrottle() {
  saveThrottleState({ lastIdleMessageSentAt: 0 });
}

module.exports = { shouldSendIdleMessage, resetIdleThrottle };
