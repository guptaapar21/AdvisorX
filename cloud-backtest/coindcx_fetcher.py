"""
Fetches historical OHLCV data from CoinDCX's PUBLIC candles endpoint -
confirmed free, no API key required. Switched from Binance after a real
Binance API call from Colab returned HTTP 451 "Unavailable For Legal
Reasons" - this is a well-documented, longstanding restriction: Binance
blocks its main API from certain server regions, and Google Colab's
servers fall in that range (confirmed via multiple independent reports,
including one that literally says "Google Colab is not available for the
Binance API"). CoinDCX doesn't have this issue, and it's also more
representative since it's the actual exchange you trade on.

Confirmed directly from CoinDCX's own official docs
(https://docs.coindcx.com/): GET /market_data/candles supports pair,
interval, startTime, endTime (both in ms), and limit (max 1000) - this
genuinely supports historical range queries, not just "most recent N".

IMPORTANT DISCOVERY: CoinDCX's docs list 5m/15m/30m/1h/2h/4h/6h/8h/1d/... as
generally valid intervals for this endpoint, but that list is apparently
generic across pair types - "B-" prefixed pairs (futures/margin-type, which
is what B-BTC_USDT is) actually only support 1m/15m/1h/1d in practice
(confirmed via a real 422 error requesting 5m directly - this matches
EXACTLY the same restriction found while building the live bot's own
candle-fetching code). Fix is the same one already used there: fetch 1m
(which works) and aggregate up to whatever finer granularity is needed via
resample_candles() below - OHLCV aggregation composes correctly through
intermediate levels, so 1m->5m->15m gives identical results to 1m->15m
directly.

NOTE: this sandbox has no network access, so this still couldn't be
live-tested here either. Do a small test pull first in Colab before
running a long backtest, same caution as before.
"""
import time
import random
import requests
import pandas as pd

BASE = "https://public.coindcx.com/market_data/candles"

INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def fetch_coindcx_klines(symbol="BTC", interval="5m", start_time=None, end_time=None, limit_per_call=1000,
                          stagger_delay=True):
    """
    symbol: base symbol, e.g. "BTC" (builds the futures-style pair
    "B-{symbol}_USDT" - matches the live bot's own pair convention)
    interval: one of CoinDCX's documented intervals (1m, 5m, 15m, 30m, 1h,
    4h, 1d, ...)
    start_time / end_time: pandas.Timestamp-parseable or None (None
    end_time = now)
    Returns a DataFrame indexed by open_time, columns: open, high, low,
    close, volume
    """
    pair = f"B-{symbol}_USDT"

    # GitHub Actions matrix jobs all start within the same second or two -
    # meaning many jobs' FIRST request naturally lands on CoinDCX at
    # nearly the same instant. The failure pattern (multiple SOL jobs all
    # failing at ~76-77s = exactly 3 retries x 25s timeout, while other
    # SOL jobs in the same run succeeded normally) points to request
    # contention, not a per-request fluke - retrying alone can't fix
    # that, since all 3 retries hit the same collision window. A random
    # startup delay spreads out when different jobs' requests actually
    # reach CoinDCX, so parallel jobs stop landing on the exact same
    # instant.
    time.sleep(random.uniform(0, 8) if stagger_delay else 0)

    if end_time is None:
        end_time = pd.Timestamp.utcnow()
    start_ms = int(pd.Timestamp(start_time).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_time).timestamp() * 1000)
    step_ms = INTERVAL_MS[interval] * limit_per_call

    all_rows = []
    cursor = start_ms
    request_count = 0
    start_wall_time = time.time()
    total_span_ms = end_ms - start_ms

    while cursor < end_ms:
        params = {
            "pair": pair,
            "interval": interval,
            "startTime": cursor,
            "endTime": min(cursor + step_ms, end_ms),
            "limit": limit_per_call,
        }
        # 3 retries with backoff on a transient failure (timeout, connection
        # reset, momentary 5xx) - a single unprotected request repeated
        # 500+ times per full-year fetch was near-guaranteed to fail
        # somewhere eventually, which is exactly what kept happening
        # (recurring SOL timeouts killing the whole job on one bad request
        # out of hundreds). Slightly longer timeout (25s, up from 15s) too,
        # but the retry is what actually matters here - a timeout that
        # happens 1 time in 200 requests will still happen eventually
        # over enough calls, retry is what survives that, not a bigger
        # number alone.
        last_err = None
        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(BASE, params=params, timeout=25)
                resp.raise_for_status()
                last_err = None
                break
            except (requests.exceptions.RequestException,) as e:
                last_err = e
                if attempt < 2:
                    print(f"    {symbol}: request {request_count+1} failed ({e}), retrying (attempt {attempt+2}/3)...")
                    time.sleep(2 * (attempt + 1) + random.uniform(0, 2))
        if last_err is not None:
            raise last_err
        rows = resp.json()
        request_count += 1

        if not rows:
            cursor += step_ms
            time.sleep(0.3)
            continue
        all_rows.extend(rows)
        cursor = max(r["time"] for r in rows) + INTERVAL_MS[interval]
        time.sleep(0.3)  # polite pacing

        # Progress every 20 requests - so a genuinely slow-but-working fetch
        # is visibly distinguishable from one that's actually stuck. If you
        # don't see this print advancing every few seconds, something's
        # actually wrong (network issue, rate limiting) rather than just slow.
        if request_count % 20 == 0:
            pct_done = min(100, round((cursor - start_ms) / total_span_ms * 100)) if total_span_ms > 0 else 100
            elapsed = time.time() - start_wall_time
            print(f"  ...{request_count} requests done, ~{pct_done}% through the date range, "
                  f"{len(all_rows)} candles so far, {elapsed:.0f}s elapsed")

    if not all_rows:
        raise ValueError(
            f"No candles returned for {pair} {interval} in this range - double check the pair "
            f"exists on CoinDCX and the date range isn't before it was listed."
        )

    df = pd.DataFrame(all_rows)
    df["open_time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].sort_index()
    return df[~df.index.duplicated(keep="first")]


def resample_candles(base_candles, target_interval):
    """Resamples finer candles up to a coarser interval - e.g. 5m -> 1h -
    same aggregation logic as the live bot's aggregateCandles (first open,
    last close, max high, min low, summed volume), using pandas resample.

    target_interval accepts either a legacy string key (backward
    compatible: "5m","15m","1h","4h","1d") or a raw integer number of
    minutes (e.g. 1, 3, 30) - added to test genuinely different timeframe
    combinations (e.g. 1m/5m/15m or 3m/15m/30m instead of the fixed
    5m/15m/1h), not just skip or cache the existing ones."""
    legacy_rule_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    if isinstance(target_interval, str):
        rule = legacy_rule_map[target_interval]
    else:
        rule = f"{int(target_interval)}min"
    out = base_candles.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    return out.dropna()
