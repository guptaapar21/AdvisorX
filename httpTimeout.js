// Every outbound fetch() in this repo previously had no timeout, so a
// single stalled network call (e.g. Telegram) could hang for minutes and
// trigger GitHub Actions' own "internal error" job-kill (confirmed root
// cause of the fastwatch failures around ~5 minute run durations). This
// wraps fetch with a hard ceiling so a stall fails fast with a real,
// readable error instead of hanging indefinitely.
//
// Currently only used by telegram.js. coindcxExchangeClient.js, coindcx.js,
// geminiAgent.js, and llmAdvisor.js have the same unguarded-fetch issue but
// are intentionally NOT patched yet - holding on those per instruction.
function fetchWithTimeout(url, options = {}, timeoutMs = 15000) {
  return fetch(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
}

module.exports = { fetchWithTimeout };
