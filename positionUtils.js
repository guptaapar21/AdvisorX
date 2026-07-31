// Extracted into its own module specifically to avoid a circular
// dependency: agentTools.js already requires runtimeConfig.js (for
// getEffectiveMinScore), so runtimeConfig.js requiring agentTools.js back
// (e.g. for the /closeposition command needing real position data) would
// create a genuine circular require - both files import this instead.

// Fixes two real bugs at once:
// 1. There was previously no cumulative/portfolio-level risk check at
//    all - each trade's risk cap was checked in total isolation, so 3
//    simultaneous positions (now the normal case, one per coin) could
//    each independently pass a 7% check while collectively risking 21%+
//    of the account at once.
// 2. check_total_exposure's old margin-reconstruction divided the WHOLE
//    portfolio's exposure by a SINGLE (max) leverage value, which is
//    only correct if every position shares the same leverage - wrong the
//    moment two coins use genuinely different leverage, which is now the
//    expected case, not an edge case.
//
// Computes each open position's REAL dollar risk (quantity x distance to
// ITS OWN actual resting stop-loss order, not a guess) and its REAL
// margin (using ITS OWN leverage, not a portfolio-wide average).
async function computePortfolioRiskAndMargin(exchange, creds) {
  const positionsRaw = await exchange.getPositions(creds);
  const activePositions = getActivePositions(positionsRaw);
  if (activePositions.length === 0) return { totalRiskUsdt: 0, totalMarginUsdt: 0, positions: [], hasUnknownRisk: false };

  const ordersRaw = await exchange.getActiveOrders(creds);
  const orders = Array.isArray(ordersRaw) ? ordersRaw : (ordersRaw?.data || []);

  const details = activePositions.map((p) => {
    const contract = p.pair ?? p.contract;
    const rawSize = Number(p.active_pos ?? p.size ?? 0);
    const quantity = Math.abs(rawSize);
    const entryPrice = Number(p.avg_price ?? p.entryPrice ?? 0);
    const leverage = Number(p.leverage) || 1;
    const marginUsdt = (quantity * entryPrice) / leverage; // THIS position's own margin, own leverage

    const stopOrder = orders.find((o) => (o.pair ?? o.contract) === contract && o.order_type === "stop_market");
    const stopPrice = stopOrder ? Number(stopOrder.price) : null;
    const riskUsdt = stopPrice !== null ? quantity * Math.abs(entryPrice - stopPrice) : null;

    return { contract, quantity, entryPrice, leverage, marginUsdt, stopPrice, riskUsdt };
  });

  const totalMarginUsdt = details.reduce((sum, d) => sum + d.marginUsdt, 0);
  // Positions with no resolvable stop order (e.g. the getActiveOrders
  // best-effort check failed) can't have their risk computed - counted
  // as null, not silently as zero, so callers can tell "no risk" from
  // "unknown risk" and choose to be cautious rather than assume safety.
  const knownRisks = details.filter((d) => d.riskUsdt !== null);
  const hasUnknownRisk = knownRisks.length < details.length;
  const totalRiskUsdt = knownRisks.reduce((sum, d) => sum + d.riskUsdt, 0);

  return { totalRiskUsdt, totalMarginUsdt, positions: details, hasUnknownRisk };
}

// This filters down to genuinely active positions only, matching the size
// check already used correctly elsewhere (calculate_risk, check_total_exposure).
// Confirmed directly from CoinDCX's own official documentation: the
// position object itself carries take_profit_trigger/stop_loss_trigger
// fields - TP/SL is a native attribute of the position, not a separate
// resting order. The "not set" representation is inconsistent in
// CoinDCX's own sample data (shows as the string "None" in one example,
// numeric 0.0 in another) - this checks for both.
function hasRealBracket(position) {
  const tp = position.take_profit_trigger;
  const sl = position.stop_loss_trigger;
  const isSet = (v) => v !== undefined && v !== null && v !== "None" && Number(v) !== 0;
  return isSet(tp) && isSet(sl);
}

function getActivePositions(positionsRaw) {
  const positions = Array.isArray(positionsRaw) ? positionsRaw : (positionsRaw?.data || []);
  return positions.filter((p) => Math.abs(Number(p.active_pos ?? p.size ?? 0)) > 0);
}

module.exports = { getActivePositions, computePortfolioRiskAndMargin, hasRealBracket };
