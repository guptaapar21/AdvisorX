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

// Supports GEMINI_API_KEYS="key1,key2,...", GEMINI_API_KEY_1..10, and a
// single GEMINI_API_KEY - any combination, de-duplicated.
function getGeminiKeys() {
  const keys = [];
  if (process.env.GEMINI_API_KEYS) {
    keys.push(...process.env.GEMINI_API_KEYS.split(",").map((k) => k.trim()).filter(Boolean));
  }
  for (let i = 1; i <= 10; i++) {
    const k = process.env[`GEMINI_API_KEY_${i}`];
    if (k) keys.push(k.trim());
  }
  if (process.env.GEMINI_API_KEY) keys.push(process.env.GEMINI_API_KEY.trim());
  return [...new Set(keys)];
}

// Calls `attemptFn(key)` for each configured key in round-robin order
// (starting after whichever key was used last), skipping any key still in
// a quota cooldown. attemptFn should throw an Error with `.rateLimited =
// true` on an HTTP 429 so this can put that key in cooldown and move on.
// Returns whatever attemptFn returns on the first success, or null if every
// key is unavailable/fails.
async function withKeyRotation(attemptFn, cooldownMinutes) {
  const keys = getGeminiKeys();
  if (keys.length === 0) return null;

  const cooldownMs = (cooldownMinutes || 60) * 60 * 1000;
  const now = Date.now();
  const keyState = loadKeyState();

  const tryOrder = [];
  for (let i = 1; i <= keys.length; i++) {
    tryOrder.push((keyState.lastIndex + i) % keys.length);
  }

  for (const idx of tryOrder) {
    const cooldownUntil = keyState.cooldowns[idx];
    if (cooldownUntil && cooldownUntil > now) continue;

    try {
      const result = await attemptFn(keys[idx]);
      keyState.lastIndex = idx;
      delete keyState.cooldowns[idx];
      saveKeyState(keyState);
      return result;
    } catch (err) {
      if (err.rateLimited) {
        console.error(`Gemini key #${idx + 1}/${keys.length}: quota hit, cooling down ${cooldownMinutes || 60}m`);
        keyState.cooldowns[idx] = now + cooldownMs;
        saveKeyState(keyState);
      } else {
        console.error(`Gemini key #${idx + 1}/${keys.length}: call failed - ${err.message}`);
      }
    }
  }

  console.error(`All ${keys.length} Gemini key(s) unavailable this run.`);
  return null;
}

module.exports = { getGeminiKeys, withKeyRotation };
