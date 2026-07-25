// Trimmed down from the earlier version: editTelegramMessage() and
// pinTelegramMessage() are gone. Those existed only for the old
// edit-in-place live scorecard (the KWGT-era design), which has been
// removed - fastwatch now sends a fresh, plain message every cycle
// instead of editing one message in place. Every call here now goes
// through fetchWithTimeout so a stalled request fails fast instead of
// hanging for minutes.

const { fetchWithTimeout } = require("./httpTimeout");

async function sendTelegramMessage(text) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  if (!token || !chatId) {
    throw new Error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars");
  }

  const url = `https://api.telegram.org/bot${token}/sendMessage`;

  async function attempt(parseMode) {
    const body = { chat_id: chatId, text };
    if (parseMode) body.parse_mode = parseMode;
    return fetchWithTimeout(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  let res = await attempt("Markdown");

  if (!res.ok) {
    const body = await res.text();
    // Dynamic content (symbol names, error messages, reasoning text) can
    // contain characters that break Telegram's Markdown parser (stray _
    // or * that don't form a valid pair). Rather than let one bad
    // character crash the entire run's notification, retry once as plain
    // text - this can never fail the same way since there's no parse_mode.
    const isEntityParseError = res.status === 400 && /can't (parse entities|find end of the entity)/i.test(body);
    if (isEntityParseError) {
      res = await attempt(null);
      if (!res.ok) {
        const retryBody = await res.text();
        throw new Error(`Telegram send failed even as plain text: ${res.status} ${retryBody}`);
      }
      const json = await res.json();
      return json.result?.message_id;
    }
    throw new Error(`Telegram send failed: ${res.status} ${body}`);
  }

  const json = await res.json();
  return json.result?.message_id;
}

// Fetches new incoming messages since `sinceUpdateId` (exclusive). Returns
// { messages: [{updateId, text}], latestUpdateId }. Never throws - a
// failure here should never block the run, just means no commands were
// picked up this cycle.
async function getTelegramUpdates(sinceUpdateId) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  if (!token) return { messages: [], latestUpdateId: sinceUpdateId };

  try {
    const offset = sinceUpdateId ? sinceUpdateId + 1 : undefined;
    const url = `https://api.telegram.org/bot${token}/getUpdates${offset ? `?offset=${offset}` : ""}`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return { messages: [], latestUpdateId: sinceUpdateId };
    const json = await res.json();
    const results = json.result || [];
    const messages = results
      .filter((u) => u.message && u.message.text)
      .map((u) => ({ updateId: u.update_id, text: u.message.text.trim() }));
    const latestUpdateId = results.length > 0 ? results[results.length - 1].update_id : sinceUpdateId;
    return { messages, latestUpdateId };
  } catch {
    return { messages: [], latestUpdateId: sinceUpdateId };
  }
}

module.exports = { sendTelegramMessage, getTelegramUpdates };
