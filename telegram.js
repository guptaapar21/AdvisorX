async function sendTelegramMessage(text) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  if (!token || !chatId) {
    throw new Error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars");
  }

  const url = `https://api.telegram.org/bot${token}/sendMessage`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chat_id: chatId,
      text,
      parse_mode: "Markdown",
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Telegram send failed: ${res.status} ${body}`);
  }
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

module.exports = { sendTelegramMessage, getTelegramUpdates };
