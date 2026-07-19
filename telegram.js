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
    return fetch(url, {
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

// Edits an existing message in place (used for the live scorecard, so it
// refreshes instead of spamming a new message every cycle). Returns true
// on success, false if the edit failed for any reason (e.g. the message
// was deleted, or is too old to edit) - the caller should fall back to
// sending a fresh message and tracking its new ID in that case.
async function editTelegramMessage(messageId, text) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  if (!token || !chatId) {
    throw new Error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars");
  }

  const url = `https://api.telegram.org/bot${token}/editMessageText`;

  async function attempt(parseMode) {
    const body = { chat_id: chatId, message_id: messageId, text };
    if (parseMode) body.parse_mode = parseMode;
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  let res = await attempt("Markdown");

  if (!res.ok) {
    const body = await res.text();
    const isEntityParseError = res.status === 400 && /can't (parse entities|find end of the entity)/i.test(body);
    if (isEntityParseError) {
      res = await attempt(null);
      if (res.ok) return true;
    }
    // "message is not modified" happens when the content is identical to
    // last time - that's fine, not a real failure, just nothing changed.
    if (/message is not modified/i.test(body)) return true;
    return false;
  }
  return true;
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
    const res = await fetch(url);
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

module.exports = { sendTelegramMessage, editTelegramMessage, getTelegramUpdates };
