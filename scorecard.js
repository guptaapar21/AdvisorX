// A single Telegram message that gets EDITED in place every fast-watch
// cycle (~1 min) instead of a new message being sent each time - this is
// what "live" means here: not instant push updates, but a refreshed
// snapshot on a tight, predictable cadence. Shows real position P&L (using
// the reconciled real entry price) and the most recent scan scores from
// the main 5-min cycle.

const fs = require("fs");
const path = require("path");
const { editTelegramMessage, sendTelegramMessage, pinTelegramMessage } = require("./telegram");

const SCORECARD_STATE_FILE = path.join(__dirname, "scorecardState.json");
const LATEST_SCORES_FILE = path.join(__dirname, "latestScores.json");

function loadScorecardState() {
  try { return JSON.parse(fs.readFileSync(SCORECARD_STATE_FILE, "utf8")); } catch { return { messageId: null }; }
}
function saveScorecardState(state) {
  fs.writeFileSync(SCORECARD_STATE_FILE, JSON.stringify(state, null, 2));
}

function loadLatestScores() {
  try { return JSON.parse(fs.readFileSync(LATEST_SCORES_FILE, "utf8")); } catch { return null; }
}

// Called by preFilter.js after every scan so the scorecard always has
// something reasonably fresh to show, even though the scorecard itself
// only runs the cheap fast-watch cycle.
function saveLatestScores(allScores) {
  fs.writeFileSync(LATEST_SCORES_FILE, JSON.stringify({ scores: allScores, timestamp: Date.now() }, null, 2));
}

const LIVE_SNAPSHOT_FILE = path.join(__dirname, "liveSnapshot.json");

// Written every fast-watch cycle (~1 min) with everything a simple
// external dashboard needs in one file - open positions with live P&L,
// the most recent scan scores, and a timestamp. This is what the
// home-screen dashboard page reads from raw.githubusercontent.com.
function saveLiveSnapshot(positions, latestScores, strategyName) {
  // Defensive on purpose: this file write is a nice-to-have for the
  // external dashboard, not part of the Telegram flow. It must never be
  // able to throw and block the actual scorecard message from sending -
  // same reasoning as the try/catch already used elsewhere in this file
  // for loadLatestScores/loadScorecardState.
  try {
    fs.writeFileSync(LIVE_SNAPSHOT_FILE, JSON.stringify({
      updatedAt: Date.now(),
      strategyName,
      positions,
      latestScores: latestScores ? latestScores.scores : null,
      latestScoresTimestamp: latestScores ? latestScores.timestamp : null,
    }, null, 2));
  } catch (err) {
    console.log(`Scorecard: couldn't write liveSnapshot.json (dashboard-only, non-fatal) - ${err.message}`);
  }
}

function formatPositionLine(p) {
  // Direction (long/short) and P&L (winning/losing) are two DIFFERENT
  // things - a long position can easily be sitting at a loss. The old
  // version used 🟢/🔴 for direction only, which looked like a P&L color
  // even when it wasn't (e.g. a losing long still showed green). Now
  // direction gets a neutral arrow, and 🟢/🔴 reflects the actual P&L sign.
  const directionArrow = p.action === "long" ? "▲" : "▼";
  const pnlEmoji = p.pnlPercent !== null ? (p.pnlPercent >= 0 ? "🟢" : "🔴") : "⚪";
  const pnlStr = p.pnlPercent !== null ? `${p.pnlPercent >= 0 ? "+" : ""}${p.pnlPercent.toFixed(2)}%` : "n/a";
  return `${pnlEmoji} *${p.contract}* ${directionArrow} ${p.action} | Entry: ${p.entryPrice} | Now: ${p.currentPrice} | ROE: ${pnlStr} | Stop: ${p.currentStop}`;
}

// `positions` is an array of { contract, action, entryPrice, currentPrice,
// currentStop, pnlPercent } - built by fastWatch.js during its own loop,
// since it already fetches all of this. `strategyName` is just for
// context in the header.
async function updateScorecard(positions, strategyName) {
  const state = loadScorecardState();
  const latestScores = loadLatestScores();

  const lines = [];
  lines.push(`📊 *Live Scorecard* (strategy: ${strategyName})`);
  lines.push(`_Updated: ${new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour12: true })} IST_`);
  lines.push("");

  if (positions.length > 0) {
    lines.push("*Open positions:*");
    positions.forEach((p) => lines.push(formatPositionLine(p)));
  } else {
    lines.push("_No open positions._");
  }

  if (latestScores && latestScores.scores && latestScores.scores.length > 0) {
    lines.push("");
    lines.push("*Latest scan scores:*");
    const sorted = latestScores.scores.slice().sort((a, b) => b.score - a.score);
    lines.push(sorted.map((s) => `${s.symbol} ${s.score}`).join(", "));
    const ageMin = Math.round((Date.now() - latestScores.timestamp) / 60000);
    lines.push(`_(from ${ageMin} min ago)_`);
  }

  const text = lines.join("\n");
  saveLiveSnapshot(positions, latestScores, strategyName);

  if (state.messageId) {
    const edited = await editTelegramMessage(state.messageId, text);
    if (edited) {
      console.log("Scorecard: updated existing message.");
      return;
    }
    console.log("Scorecard: edit failed (message may be too old/deleted) - sending a fresh one.");
  }

  const newId = await sendTelegramMessage(text);
  if (newId) {
    saveScorecardState({ messageId: newId });
    await pinTelegramMessage(newId);
    console.log("Scorecard: sent new message, pinned it, and tracking its ID for future edits.");
  }
}

// Writes JUST the JSON snapshot, without touching Telegram at all - for
// the "flat, nothing changed" case where sending a fresh scorecard
// message every cycle would be spam, but the external JSON snapshot
// (used by a dashboard/widget) still needs to reflect "no positions"
// rather than staying stuck on stale data indefinitely.
function saveLiveSnapshotOnly(positions, strategyName) {
  const latestScores = loadLatestScores();
  saveLiveSnapshot(positions, latestScores, strategyName);
}

module.exports = { updateScorecard, saveLatestScores, saveLiveSnapshotOnly };
