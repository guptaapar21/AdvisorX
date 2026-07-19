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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// CoinDCX's FUTURES candles endpoint only supports 1m, 15m, 1h, 1d
// (confirmed via a real 422: "interval must be one of [1m, 15m, 1h, 1d]" -
// their general docs list 5m too, but that's evidently spot-only). For any
// other interval, fetch the base interval and aggregate N candles into one.
const SYNTHETIC_INTERVALS = {
  "5m": { base: "1m", factor: 5 },
  "30m": { base: "15m", factor: 2 },
  "4h": { base: "1h", factor: 4 },
};

// Combines `factor` consecutive base candles into one - standard OHLCV
// aggregation: first open, last close, max high, min low, summed volume.
// Assumes candles are already sorted oldest -> newest (as getCandles returns).
function aggregateCandles(candles, factor) {
  const out = [];
  for (let i = 0; i + factor <= candles.length; i += factor) {
    const group = candles.slice(i, i + factor);
    out.push({
      time: group[0].time,
      open: group[0].open,
      close: group[group.length - 1].close,
      high: Math.max(...group.map((c) => c.high)),
      low: Math.min(...group.map((c) => c.low)),
      volume: group.reduce((sum, c) => sum + (c.volume || 0), 0),
    });
  }
  return out;
}

// Fetches candles for one interval, transparently synthesizing it from a
// supported base interval if CoinDCX futures doesn't offer it directly.
async function getCandlesForInterval(pair, interval, limit) {
  const synthetic = SYNTHETIC_INTERVALS[interval];
  if (!synthetic) return getCandles(pair, interval, limit);

  // Need `limit * factor` base candles to produce `limit` aggregated ones,
  // plus a little extra buffer in case of any gaps.
  const baseLimit = Math.min(limit * synthetic.factor + synthetic.factor, 1000);
  const baseCandles = await getCandles(pair, synthetic.base, baseLimit);
  const aggregated = aggregateCandles(baseCandles, synthetic.factor);
  return aggregated.slice(-limit);
}

// Cheap current-price check (single 1m candle, not a full 3-timeframe
// fetch) - for the fast position watcher, which only needs "what's the
// price right now", not a full re-analysis.
async function getCurrentPrice(symbol, marketType) {
  const pair = await resolvePair(symbol, marketType);
  const candles = await getCandles(pair, "1m", 2);
  return candles[candles.length - 1].close;
}

// Fetches primary/confirm/filter candles for one symbol, sequentially with
// a small delay between requests (safety margin - see config.candleFetchDelayMs).
async function getMultiTimeframeCandles(symbol, marketType, timeframes, candleLimit, delayMs = 300) {
  const pair = await resolvePair(symbol, marketType);

  async function fetchLabeled(label, interval) {
    try {
      return await getCandlesForInterval(pair, interval, candleLimit);
    } catch (err) {
      // Re-throw with which of the 3 timeframes actually failed - a bare
      // "candles failed" error looks identical whether it was primary,
      // confirm, or filter that broke.
      throw new Error(`[${label}/${interval}] ${err.message}`);
    }
  }

  const primary = await fetchLabeled("primary", timeframes.primary);
  await sleep(delayMs);
  const confirm = await fetchLabeled("confirm", timeframes.confirm);
  await sleep(delayMs);
  const filter = await fetchLabeled("filter", timeframes.filter);
  return { pair, primary, confirm, filter };
}

// Order book depth (public, no key needed). NOTE: CoinDCX's own docs show
// two slightly different response shapes across sources (array of
// {p,s}/{price,size} objects vs. a price-keyed object) - this parses
// defensively for either and returns null (not throws) if the shape is
// unrecognized, so a liquidity check can skip gracefully rather than break
// the run, matching how the original itself wraps this in a try/catch.
async function getOrderBook(pair) {
  const url = `https://public.coindcx.com/market_data/orderbook?pair=${encodeURIComponent(pair)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`orderbook failed for ${pair}: ${res.status}`);
  const raw = await res.json();

  function normalizeSide(side) {
    if (!side) return null;
    if (Array.isArray(side)) {
      return side
        .map((entry) => {
          if (Array.isArray(entry)) return { price: Number(entry[0]), size: Number(entry[1]) };
          const price = Number(entry.p ?? entry.price ?? 0);
          const size = Number(entry.s ?? entry.size ?? entry.q ?? entry.quantity ?? 0);
          return { price, size };
        })
        .filter((e) => Number.isFinite(e.price) && Number.isFinite(e.size) && e.price > 0 && e.size > 0);
    }
    if (typeof side === "object") {
      // price-keyed object: { "12345.6": "0.5", ... }
      return Object.entries(side)
        .map(([price, size]) => ({ price: Number(price), size: Number(size) }))
        .filter((e) => Number.isFinite(e.price) && Number.isFinite(e.size) && e.price > 0 && e.size > 0);
    }
    return null;
  }

  const bids = normalizeSide(raw.bids);
  const asks = normalizeSide(raw.asks);
  if (!bids && !asks) return null; // unrecognized shape - caller should skip the check
  return { bids: bids || [], asks: asks || [] };
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
  getMultiTimeframeCandles,
  getCurrentPrice,
  aggregateCandles,
  getOrderBook,
  getBalances,
  getPositions,
  placeOrder,
  cancelOrder,
  closePosition,
  setPositionStopLoss,
};
