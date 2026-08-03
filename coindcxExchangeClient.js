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
const { fetchWithTimeout } = require("./httpTimeout");

const API_BASE = "https://api.coindcx.com";

// Cache within a single run - instrument specs rarely change, no need to
// refetch for every order in the same process.
const instrumentDetailsCache = {};

// Confirmed real, needed fix: CoinDCX rejected an order with "Quantity
// should be divisible by 1.0" - different coins require different
// quantity increments (confirmed via CoinDCX's own docs: e.g. one sample
// instrument required increments of 1000). A single fixed rounding rule
// would fix DOGE but likely break other coins needing fractional
// quantities (BTC, ETH) - this fetches the REAL required increment per
// instrument instead of guessing one rule for all.
async function getInstrumentDetails(pair) {
  if (instrumentDetailsCache[pair]) return instrumentDetailsCache[pair];
  const res = await fetchWithTimeout(`${API_BASE}/exchange/v1/derivatives/futures/data/instrument?pair=${pair}`);
  if (!res.ok) {
    throw new Error(`Could not fetch instrument details for ${pair}: ${res.status}`);
  }
  const data = await res.json();
  instrumentDetailsCache[pair] = data.instrument;
  return data.instrument;
}

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

  const res = await fetchWithTimeout(`${API_BASE}${path}`, {
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
    // CoinDCX's response body is sometimes empty/unhelpful on a 400 -
    // including the actual request we sent (which we know completely)
    // makes this diagnosable even when their response isn't. Wrapped in
    // a code block (backticks) so Telegram's Markdown renderer shows it
    // literally - without this, an even number of underscores in the
    // JSON gets silently parsed as italic markers and eaten, making a
    // perfectly well-formed request (e.g. "B-DOGE_USDT") display as if
    // it were malformed ("B-DOGEUSDT") - exactly the kind of thing that
    // makes a real bug undiagnosable from the Telegram message alone.
    throw new Error(`CoinDCX private call failed (${path}): ${res.status} ${errText || "(empty response body)"} | Request sent: \`${jsonBody}\``);
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

// Same "don't fetch what you don't need" precedent as getCurrentPrice
// above, for the opposite case: a full timeframe's worth of candles, but
// only ONE timeframe - not all 3. Added for the BTC trend confirmation
// bonus (ETH-only), which only reads the primary timeframe - calling the
// full getMultiTimeframeCandles for this was wasting 2 extra real
// exchange API calls plus 2 extra candleFetchDelayMs sleeps every single
// cycle, for confirm/filter data that was fetched and then immediately
// discarded.
async function getSingleTimeframeCandles(symbol, marketType, interval, limit) {
  const pair = await resolvePair(symbol, marketType);
  return getCandlesForInterval(pair, interval, limit);
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
  const res = await fetchWithTimeout(url);
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

// Best-effort - this endpoint path follows the same convention as
// /positions above and is confirmed to exist as a real CoinDCX feature
// via third-party API wrapper libraries, but the exact path was NOT
// independently verified against CoinDCX's own official documentation.
// Callers MUST wrap this in a try/catch and treat a failure as "unknown,"
// not as "no pending orders" - added specifically so a pending, unfilled
// limit order doesn't get silently missed by a duplicate-position check
// that only looks at filled positions.
async function getActiveOrders(creds, page = "1", size = "50") {
  return privatePost("/exchange/v1/derivatives/futures/orders/active", { page, size }, creds);
}

// ---- WRITE / EXECUTION: REAL calls to the CoinDCX exchange ----
// Schema confirmed directly against CoinDCX's own official API
// documentation (not a third-party guess) before writing any of this.

async function createOrder(creds, { side, pair, orderType, price, totalQuantity, leverage, timeInForce }) {
  // Confirmed directly against CoinDCX's own official documentation
  // (fetched and read directly, not inferred) - the entire order payload
  // must be nested inside an "order" key, with only "timestamp" at the
  // top level. The previous version sent every field flat at the top
  // level, which is a structural mismatch CoinDCX rejects outright -
  // this is the actual root cause of the persistent empty-body 400,
  // not the contract naming or quantity precision fixed earlier (those
  // were real, separate issues, but this was the blocking one).
  //
  // "price" is required even for market orders per the same
  // documentation's own example (which includes it alongside
  // order_type: "market_order") - previously omitted for market orders.
  const order = {
    side, // "buy" or "sell"
    pair, // e.g. "B-DOGE_USDT"
    order_type: orderType, // "market_order" | "limit_order" | "stop_market" | "take_profit_market"
    price: String(price),
    total_quantity: totalQuantity,
    leverage,
    notification: "no_notification",
    time_in_force: timeInForce || "good_till_cancel",
    hidden: false,
    post_only: false,
  };
  return privatePost("/exchange/v1/derivatives/futures/orders/create", { order }, creds);
}

async function placeOrder(creds, { pair, direction, quantity, leverage, entryPrice, orderType }) {
  const side = direction === "long" ? "buy" : "sell";
  return createOrder(creds, {
    side, pair, orderType: orderType || "market_order",
    price: entryPrice, // required even for market orders - see createOrder's notes above
    totalQuantity: quantity, leverage,
  });
}

// Rebuilt to use the REAL documented mechanism - CoinDCX has a dedicated
// endpoint (positions/create_tpsl) that sets both stop-loss and
// take-profit in ONE call, tied natively to the position by its real ID,
// not two independent generic orders placed via orders/create. This was
// confirmed directly against CoinDCX's own documentation, which
// describes this as "position TPSL" that "closes the entire position
// when the trigger price is reached" - meaning this is very likely also
// the fix for the orphaned-sibling-order problem flagged earlier, since
// it's natively tied to the position rather than two free-floating orders.
async function placeBracketOrders(creds, { positionId, stopPrice, takeProfitPrice }) {
  const body = { id: positionId };
  if (stopPrice !== undefined && stopPrice !== null) {
    body.stop_loss = { stop_price: String(stopPrice), order_type: "stop_market" };
  }
  if (takeProfitPrice !== undefined && takeProfitPrice !== null) {
    body.take_profit = { stop_price: String(takeProfitPrice), order_type: "take_profit_market" };
  }
  return privatePost("/exchange/v1/derivatives/futures/positions/create_tpsl", body, creds);
}

async function cancelOrder(creds, orderId) {
  return privatePost("/exchange/v1/derivatives/futures/orders/cancel", { id: orderId }, creds);
}

async function closePosition(creds, { pair, direction, quantity, leverage, currentPrice }) {
  // Closing a position is just an order in the opposite direction.
  const closingSide = direction === "long" ? "sell" : "buy";
  return createOrder(creds, { side: closingSide, pair, orderType: "market_order", price: currentPrice, totalQuantity: quantity, leverage });
}

module.exports = {
  getMultiTimeframeCandles,
  getSingleTimeframeCandles,
  getCurrentPrice,
  aggregateCandles,
  getOrderBook,
  getBalances,
  getPositions,
  getActiveOrders,
  getInstrumentDetails,
  createOrder,
  placeOrder,
  placeBracketOrders,
  cancelOrder,
  closePosition,
};
