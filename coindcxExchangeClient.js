// CoinDCX exchange client.
//
// READ methods (ticker, candles, balances, positions) hit the real CoinDCX
// API and return real data, so the agent reasons on accurate numbers.
//
// WRITE/execution methods (placeOrder, cancelOrder, setPositionStopLoss,
// closePosition) NEVER call the exchange. They return an "intercepted
// action" descriptor instead - agentTools.js turns that into a Telegram
// message for the user to act on manually. This is the one deliberate
// difference from the original engine: every execution tool call becomes a
// message, not a trade.
//
// ENDPOINT COVERAGE: getBalances and getPositions use endpoints confirmed
// against CoinDCX's official docs/repo (docs.coindcx.com,
// github.com/coindcx-official/rest-api) and independent sources. Only
// functions actually called by agentTools.js are kept here - no unused or
// unverified endpoints. If you want the agent to also see pending
// spot/margin orders, the confirmed endpoint for that is
// POST /exchange/v1/orders/active_orders (note: futures take-profit/stop
// is attached to the position object itself per CoinDCX's docs, not a
// separate pending order, so get_positions already covers that case for
// futures).

const crypto = require("crypto");
const { resolvePair, getCandles } = require("./coindcx");

const API_BASE = "https://api.coindcx.com";

function sign(body, secret) {
  const jsonBody = JSON.stringify(body);
  const signature = crypto.createHmac("sha256", secret).update(jsonBody).digest("hex");
  return { jsonBody, signature };
}

async function privatePost(path, bodyExtra, creds) {
  if (!creds || !creds.apiKey || !creds.apiSecret) {
    throw new Error(`CoinDCX private call to ${path} needs COINDCX_API_KEY / COINDCX_API_SECRET`);
  }
  const body = { timestamp: Date.now(), ...bodyExtra };
  const { jsonBody, signature } = sign(body, creds.apiSecret);

  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-AUTH-APIKEY": creds.apiKey,
      "X-AUTH-SIGNATURE": signature,
    },
    body: jsonBody,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`CoinDCX private call failed (${path}): ${res.status} ${errText}`);
  }
  return res.json();
}

// ---- READ: market data (public, no key needed) ----

async function getMarketPrice(symbol, marketType, entryTimeframe, trendTimeframe, candleLimit) {
  const pair = await resolvePair(symbol, marketType);
  const [entryCandles, trendCandles] = await Promise.all([
    getCandles(pair, entryTimeframe, candleLimit),
    getCandles(pair, trendTimeframe, candleLimit),
  ]);
  return { pair, entryCandles, trendCandles, currentPrice: entryCandles[entryCandles.length - 1]?.close };
}

// ---- READ: account (private, needs a READ-ONLY key) ----

// Wallet balances. CoinDCX's balances endpoint returns all currencies in
// the account; we return the raw list and let the caller pick out what it
// needs (e.g. the USDT balance for futures margin).
async function getBalances(creds) {
  return privatePost("/exchange/v1/users/balances", {}, creds);
}

// Open futures positions - includes any attached take-profit/stop-loss,
// since CoinDCX attaches those to the position itself rather than as a
// separate order.
async function getPositions(creds, page = "1", size = "50") {
  return privatePost("/exchange/v1/derivatives/futures/positions", { page, size }, creds);
}

// ---- WRITE / EXECUTION: intercepted, never touches the exchange ----
// Each returns a plain descriptor of the intended action. agentTools.js
// turns this into a Telegram message and a synthetic tool result for the
// model - it never calls the real exchange.

async function placeOrder(params) {
  return { intercepted: true, action: "open_position", params };
}

async function cancelOrder(orderId, contract) {
  return { intercepted: true, action: "cancel_order", params: { orderId, contract } };
}

async function setPositionStopLoss(contract, stopLoss, takeProfit) {
  return { intercepted: true, action: "set_position_stop_loss", params: { contract, stopLoss, takeProfit } };
}

async function closePosition(contract, sizePercent) {
  return { intercepted: true, action: "close_position", params: { contract, sizePercent: sizePercent ?? 100 } };
}

module.exports = {
  getMarketPrice,
  getBalances,
  getPositions,
  placeOrder,
  cancelOrder,
  closePosition,
  setPositionStopLoss,
};
