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
  // entryPrice/currentPrice come straight from the exchange, already
  // clean (e.g. 0.07314) - currentStop is computed via an ATR formula in
  // JS floating-point and previously showed raw (e.g. 0.0718977269405086).
  // Same rounding convention as agentTools.js's open-suggestion message:
  // 2 decimals for prices >= 1 (e.g. BTC/ETH), 6 decimals for sub-$1
  // coins (e.g. DOGE) - a fixed rule, not decimal-counting from a
  // price's string form, which is fragile (JS drops trailing zeros, so
  // 65432.10 becomes "65432.1" and would undercount to 1 decimal).
  const decimals = p.entryPrice >= 1 ? 2 : 6;
  const roundedStop = Number(p.currentStop).toFixed(decimals);
  return `${pnlEmoji} *${p.contract}* ${directionArrow} ${p.action} | Entry: ${p.entryPrice} | Now: ${p.currentPrice} | ROE: ${pnlStr} | Stop: ${roundedStop}`;
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
