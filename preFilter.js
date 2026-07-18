// Runs every 5 minutes with ZERO Gemini calls - just real CoinDCX data and
// the existing rule-based scorer (marketStateAnalyzer/strategyRouter/
// opportunityScorer, all pure JS). Only when this finds something worth
// reasoning about (a real open position, or a candidate that clears
// minScore) does agentIndex.js go on to invoke the full Gemini agent.
//
// Why this exists: at 5-minute cadence, calling Gemini on every run vastly
// exceeds the free tier's daily request quota (confirmed from her Google AI
// Studio dashboard: 20 requests/day per key, ~10 keys = ~200/day total,
// but 288 runs/day x several Gemini calls per run needs 500-1400+). This
// keeps the market-awareness at 5 minutes while only spending Gemini quota
// on the runs that actually need a decision.

const exchange = require("./coindcxExchangeClient");
const { analyzeSymbol, getActivePositions } = require("./agentTools");
const tradeOutcomeLog = require("./tradeOutcomeLog");

const path = require("path");
const fs = require("fs");
const TREND_HISTORY_FILE = path.join(__dirname, "trendScoreHistory.json");

function loadTrendHistory() {
  try { return JSON.parse(fs.readFileSync(TREND_HISTORY_FILE, "utf8")); } catch { return {}; }
}
function saveTrendHistory(store) {
  fs.writeFileSync(TREND_HISTORY_FILE, JSON.stringify(store, null, 2));
}

// Returns { hasOpenPositions, activePositions, candidates, allScores, isActive }.
// `candidates` are analysis objects (same shape agentTools.js's
// analyze_opening_opportunities produces) that cleared config.minScore,
// sorted best-first. `isActive` is true if there's a real position OR at
// least one qualifying candidate - that's the signal for whether to spend
// a Gemini call this run.
async function runPreFilter(config, creds) {
  const positionsRaw = await exchange.getPositions(creds);
  const activePositions = getActivePositions(positionsRaw);
  const hasOpenPositions = activePositions.length > 0;

  const trendHistoryStore = loadTrendHistory();
  const candidates = [];
  const allScores = [];

  for (const symbol of config.symbols) {
    const cooldown = tradeOutcomeLog.isSymbolInCooldown(symbol);
    if (cooldown.inCooldown) {
      allScores.push({ symbol, score: 0, note: `in cooldown: ${cooldown.reason} (${cooldown.remainingHours}h remaining)` });
      continue;
    }
    try {
      const analysis = await analyzeSymbol(symbol, config, trendHistoryStore);
      if (analysis.error) {
        allScores.push({ symbol, score: 0, note: analysis.error });
        continue;
      }
      allScores.push({
        symbol, score: analysis.opportunity.totalScore, action: analysis.strategyResult.action,
        setupType: analysis.strategyResult.strategyType, isBreakoutExtension: analysis.opportunity.isBreakoutExtension,
      });
      if (analysis.strategyResult.action !== "wait" && analysis.opportunity.totalScore >= config.minScore) {
        candidates.push(analysis);
      }
    } catch (err) {
      allScores.push({ symbol, score: 0, note: `error: ${err.message}` });
    }
  }

  saveTrendHistory(trendHistoryStore);
  candidates.sort((a, b) => b.opportunity.totalScore - a.opportunity.totalScore);

  return {
    hasOpenPositions,
    activePositions,
    candidates,
    allScores,
    isActive: hasOpenPositions || candidates.length > 0,
  };
}

module.exports = { runPreFilter };
