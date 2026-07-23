// Narrator-only LLM layer for the lighter alert-only bot (index.js).
// Takes structured numbers the rule-based scorer/position-tracker already
// computed and asks an LLM to write a clear Telegram message about them.
// No tools, no exchange access - text only. Falls back to the rule-based
// message if every key/provider fails.

const { withKeyRotation } = require("./geminiKeys");
const { parseQuotaInfo } = require("./geminiQuotaInfo");

const SYSTEM_PROMPT = `You are a crypto trading assistant that writes short Telegram guidance messages for a retail trader in India who trades manually on CoinDCX.

You have no ability to place, modify, or cancel any order on any exchange, and no tools of any kind - you only produce message text that a human reads and acts on themselves. Never imply you are executing, have executed, or will execute anything.

You will be given structured, already-computed technical data (either a new potential trade or an update on a trade the user may have taken). Write ONE concise Telegram message (under 130 words) that:
- states the coin and direction plainly (long/short)
- gives the current price, the stop-loss level, and - for a new trade - the staged take-profit plan in R-multiples if provided
- gives the position-size hint if provided, in one line
- gives 1-2 sentences of plain-English reasoning using ONLY the numbers and reasons given in the data - never invent prices, indicators, or reasons not present in the input
- ends with one short line reminding the user this is guidance only and any order must be placed, adjusted, or closed by them manually on CoinDCX

Plain text only, no markdown headers or code blocks, Telegram-friendly line breaks, at most 1-2 emoji.`;

async function callGemini(prompt, apiKey, model) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: [{ role: "user", parts: [{ text: prompt }] }],
    }),
  });
  if (res.status === 429) {
    // See geminiQuotaInfo.js - read the real quota details instead of
    // assuming a flat cooldown, so a per-day cap doesn't get retried every
    // 60 minutes and re-cool-down all day long.
    const { period, retryDelayMs, quotaId } = await parseQuotaInfo(res);
    const err = new Error(
      `Gemini quota/rate limit exceeded${quotaId ? ` (${quotaId})` : ""}`
    );
    err.rateLimited = true;
    err.quotaPeriod = period;
    err.retryDelayMs = retryDelayMs;
    throw err;
  }
  if (res.status === 404) {
    const body = await res.text();
    throw new Error(
      `Gemini call failed: 404 (model "${model}" not found/unavailable - set the GEMINI_MODEL secret to override, e.g. gemini-3.5-flash) ${body}`
    );
  }
  if (!res.ok) {
    throw new Error(`Gemini call failed: ${res.status} ${await res.text()}`);
  }
  const json = await res.json();
  const text = (json.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join("\n").trim();
  return text || null;
}

async function craftMessageWithGemini(kind, data, config) {
  const model = process.env.GEMINI_MODEL || "gemini-3.5-flash";
  const prompt = `Message type: ${kind}\n\nData:\n${JSON.stringify(data, null, 2)}`;
  return withKeyRotation((key) => callGemini(prompt, key, model), config && config.geminiKeyCooldownMinutes);
}

async function craftMessageWithAnthropic(kind, data) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return null;
  const model = process.env.ANTHROPIC_MODEL || "claude-haiku-4-5-20251001";
  const prompt = `Message type: ${kind}\n\nData:\n${JSON.stringify(data, null, 2)}`;

  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": apiKey, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model, max_tokens: 400, system: SYSTEM_PROMPT, messages: [{ role: "user", content: prompt }] }),
    });
    if (!res.ok) {
      console.error(`Anthropic advisor call failed: ${res.status} ${await res.text()}`);
      return null;
    }
    const json = await res.json();
    return (json.content || []).map((b) => b.text || "").join("\n").trim() || null;
  } catch (err) {
    console.error(`Anthropic advisor error: ${err.message}`);
    return null;
  }
}

async function craftMessage(kind, data, config) {
  const provider = (config && config.llmProvider) || "gemini";
  if (provider === "anthropic") return craftMessageWithAnthropic(kind, data);
  return craftMessageWithGemini(kind, data, config);
}

module.exports = { craftMessage };
