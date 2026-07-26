// The /backtested strategy: a coin-specific set of min_score thresholds,
// found by running the actual cloud backtest sweep (365 days, real CoinDCX
// data, real fees modeled) across every integer from 65 to 85, per coin.
// Every threshold here uses conservative's underlying scoring formula and
// risk parameters (5-9x leverage, 2.5x ATR stop, 1.0-4.0% stop bounds) -
// what changes per coin is ONLY the score bar required to act, not the
// formula itself.
//
// BTC and XRP are deliberately excluded entirely: every threshold from 65
// through 85 was tested on both, and neither ever produced a real,
// consistent profitable zone - BTC had one breakeven point (75, +$10 on
// $500/5%) surrounded by losses on both sides (not a real edge, just
// noise), and XRP was negative at every single value tested. Rather than
// silently fall back to a default that's already proven not to work,
// /backtested skips scanning these two coins entirely.

const BACKTESTED_TIERS = {
  SOL: { aggressive: 75, balanced: 80 },
  DOGE: { aggressive: 73, balanced: 79 },
  ETH: { balanced: 81 }, // no aggressive tier exists - nothing below 77 was ever profitable
};

const BACKTESTED_COINS = Object.keys(BACKTESTED_TIERS);

// The actual gate used for scanning: the LOOSEST tier available per coin
// (aggressive if it exists, else balanced) - so nothing is ever missed.
// There's no mode to select; /backtested always scans at this threshold,
// and getTierLabel() below reports after the fact which tier a real score
// actually cleared.
function getScanThreshold(symbol) {
  const tiers = BACKTESTED_TIERS[symbol];
  if (!tiers) return null;
  return tiers.aggressive ?? tiers.balanced;
}

function buildScanThresholdMap() {
  const map = {};
  for (const symbol of BACKTESTED_COINS) {
    const t = getScanThreshold(symbol);
    if (t !== null) map[symbol] = t;
  }
  return map;
}

// Given a real achieved score, returns the HIGHEST tier it actually
// clears for this coin ("balanced" > "aggressive"), or null if it's below
// even the loosest tier. This is purely a reporting label - it never
// gates anything, since scanning already happens at the loosest tier.
function getTierLabel(symbol, score) {
  const tiers = BACKTESTED_TIERS[symbol];
  if (!tiers) return null;
  if (tiers.balanced !== undefined && score >= tiers.balanced) return "balanced";
  if (tiers.aggressive !== undefined && score >= tiers.aggressive) return "aggressive";
  return null;
}

module.exports = {
  BACKTESTED_TIERS, BACKTESTED_COINS,
  getScanThreshold, buildScanThresholdMap, getTierLabel,
};
