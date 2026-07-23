const fs = require("fs");
const path = require("path");

const KEY_STATE_FILE = path.join(__dirname, "geminiKeyState.json");

function loadKeyState() {
  try {
    const parsed = JSON.parse(fs.readFileSync(KEY_STATE_FILE, "utf8"));
    if (!parsed.cooldowns) parsed.cooldowns = {};
    if (typeof parsed.lastIndex !== "number") parsed.lastIndex = -1;
    return parsed;
  } catch {
    return { lastIndex: -1, cooldowns: {} };
  }
}

function saveKeyState(state) {
  fs.writeFileSync(KEY_STATE_FILE, JSON.stringify(state, null, 2));
}

// Supports GEMINI_API_KEYS="key1,key2,...", GEMINI_API_KEY_1 through _30
// (raised from the old hard cap of _10 - the exact cause of "added more
// keys, still failing": individually-numbered keys past #10 were being
// silently ignored), and a single GEMINI_API_KEY - any combination, de-duplicated.
function getGeminiKeys() {
  const keys = [];
  if (process.env.GEMINI_API_KEYS) {
    keys.push(...process.env.GEMINI_API_KEYS.split(",").map((k) => k.trim()).filter(Boolean));
  }
  for (let i = 1; i <= 30; i++) {
    const k = process.env[`GEMINI_API_KEY_${i}`];
    if (k) keys.push(k.trim());
  }
  if (process.env.GEMINI_API_KEY) keys.push(process.env.GEMINI_API_KEY.trim());
  const uniqueKeys = [...new Set(keys)];
  console.log(`Gemini keys detected: ${uniqueKeys.length}`);
  return uniqueKeys;
}

// Calls `attemptFn(key)` for each configured key in round-robin order
// (starting after whichever key was used last), skipping any key still in
// a quota cooldown. attemptFn should throw an Error with `.rateLimited =
// true` on an HTTP 429 so this can put that key in cooldown and move on.
// Returns whatever attemptFn returns on the first success, or null if every
// key is unavailable/fails.
async function withKeyRotation(attemptFn, cooldownMinutes) {
  const keys = getGeminiKeys();
  if (keys.length === 0) return { result: null, diagnosis: "no_keys_configured", details: [] };

  const cooldownMs = (cooldownMinutes || 60) * 60 * 1000;
  const now = Date.now();
  const keyState = loadKeyState();

  const tryOrder = [];
  for (let i = 1; i <= keys.length; i++) {
    tryOrder.push((keyState.lastIndex + i) % keys.length);
  }

  const details = []; // one entry per key this run actually touched, for real diagnosis
  let skippedInCooldown = 0;

  for (const idx of tryOrder) {
    const cooldownUntil = keyState.cooldowns[idx];
    if (cooldownUntil && cooldownUntil > now) {
      skippedInCooldown++;
      continue;
    }

    try {
      const result = await attemptFn(keys[idx]);
      keyState.lastIndex = idx;
      delete keyState.cooldowns[idx];
      saveKeyState(keyState);
      return { result, diagnosis: "success", details };
    } catch (err) {
      if (err.rateLimited) {
        console.error(`Gemini key #${idx + 1}/${keys.length}: quota hit, cooling down ${cooldownMinutes || 60}m`);
        keyState.cooldowns[idx] = now + cooldownMs;
        saveKeyState(keyState);
        details.push({ keyIndex: idx + 1, type: "rate_limited" });
      } else {
        console.error(`Gemini key #${idx + 1}/${keys.length}: call failed - ${err.message}`);
        details.push({ keyIndex: idx + 1, type: "error", message: err.message });
      }
    }
  }

  // Real diagnosis: distinguish "genuinely all out of quota" (expected,
  // recoverable by waiting) from "something is actually broken" (a real
  // bug/config issue, won't fix itself by waiting) - the old code
  // collapsed both into one vague "no key available" message, which is
  // exactly why this kept looking unexplained across multiple sessions.
  const allRateLimited = details.length > 0 && details.every((d) => d.type === "rate_limited");
  const anyGenuineError = details.some((d) => d.type === "error");
  let diagnosis;
  if (details.length === 0 && skippedInCooldown === keys.length) {
    diagnosis = "all_keys_in_cooldown_from_earlier";
  } else if (allRateLimited) {
    diagnosis = "all_keys_quota_exhausted";
  } else if (anyGenuineError) {
    diagnosis = "genuine_errors_not_quota";
  } else {
    diagnosis = "unknown";
  }

  console.error(`All ${keys.length} Gemini key(s) unavailable this run. Diagnosis: ${diagnosis}. Skipped (still in cooldown): ${skippedInCooldown}.`);
  return { result: null, diagnosis, details, keyCount: keys.length, skippedInCooldown };
}

module.exports = { getGeminiKeys, withKeyRotation };
