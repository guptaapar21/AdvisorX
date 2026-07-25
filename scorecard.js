// Fully rebuilt. The old version edited one Telegram message in place
// every fastwatch cycle, pinned it, tracked its message ID, and folded in
// coin scores + a liveSnapshot.json feed for the KWGT widget / browser
// dashboard. All of that is gone now that KWGT and the dashboard are
// retired: this just sends a fresh, plain status message, ONLY when a
// position is actually open. Flat = completely silent. No scores, no
// editing, no pinning, no widget feed.

const { sendTelegramMessage } = require("./telegram");

function formatPositionLine(p) {
  const directionArrow = p.action === "long" ? "▲" : "▼";
  const pnlEmoji = p.pnlPercent !== null ? (p.pnlPercent >= 0 ? "🟢" : "🔴") : "⚪";
  const pnlStr = p.pnlPercent !== null ? `${p.pnlPercent >= 0 ? "+" : ""}${p.pnlPercent.toFixed(2)}%` : "n/a";
  return `${pnlEmoji} *${p.contract}* ${directionArrow} ${p.action} | Entry: ${p.entryPrice} | Now: ${p.currentPrice} | ROE: ${pnlStr} | Stop: ${p.currentStop}`;
}

// Called by fastWatch.js every cycle, but only ever does anything when
// positions.length > 0 - the caller already guarantees this, but the
// early return here is a second, explicit guard against ever going back
// to spamming a message while flat.
async function sendPositionStatus(positions, strategyName) {
  if (!positions || positions.length === 0) return;
  const lines = [
    `📊 *Position update* (strategy: ${strategyName})`,
    `_${new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour12: true })} IST_`,
    "",
  ];
  positions.forEach((p) => lines.push(formatPositionLine(p)));
  await sendTelegramMessage(lines.join("\n"));
}

module.exports = { sendPositionStatus };
