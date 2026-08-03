"""
Fetches REAL taker buy/sell volume (genuine aggressor-side split) from
Binance's public /api/v3/klines endpoint - confirmed via Binance's own
docs to include, per candle: volume, taker_buy_base_volume (taker sell
= volume - taker_buy_base_volume). No API key needed for market data.
Fully paginated historical range queries are supported (startTime/
endTime in ms, max 1000 rows per call) - unlike CoinDCX's public trades
endpoint, which per CoinDCX's own docs only returns recent trades with
no historical pagination.

THIS IS NOT COINDCX'S OWN ORDER FLOW. It's Binance's - used purely as
an external proxy for real buy/sell pressure on the SAME underlying
asset. For majors (SOL/ETH/DOGE) with deep cross-exchange arbitrage,
this is a standard, defensible substitute - but it is measuring a
different venue's flow, not CoinDCX's own. Flagged here, not hidden.

KNOWN RISK, NOT YET CONFIRMED EITHER WAY: coindcx_fetcher.py documents
that a real Binance API call from Google Colab's servers returned HTTP
451 (blocked for legal/regional reasons) - that's why this project
switched to CoinDCX for OHLCV in the first place. GitHub Actions runners
are a different cloud pool than Colab's, so this MAY or MAY NOT hit the
same block - untested from this sandbox (no network access here either).
Every function below fails LOUD and returns None rather than crashing
silently, specifically so a 451 (or any other failure) can be caught by
the caller and gracefully fall back to the existing OBV proxy instead of
breaking the whole backtest run.

Usage:
    from binance_taker_volume_fetcher import fetch_binance_taker_volume
    taker_df = fetch_binance_taker_volume("SOL", "5m", "2025-08-03", "2026-08-03")
    if taker_df is None:
        print("Binance unreachable (451 or other) - falling back to OBV proxy")
"""
import time
import requests
import pandas as pd
from datetime import datetime, timezone

BASE = "https://api.binance.com/api/v3/klines"

# CoinDCX coin code -> Binance USDT spot symbol. Binance doesn't list every
# CoinDCX pair identically, but all 3 live coins here trade as plain
# <COIN>USDT spot pairs on Binance.
COIN_TO_BINANCE_SYMBOL = {
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "ETH": "ETHUSDT",
    "BTC": "BTCUSDT",
    "XRP": "XRPUSDT",
}

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def fetch_binance_taker_volume(coin, interval, start_date, end_date, max_retries=3):
    """
    Returns a DataFrame indexed by UTC timestamp (matching the candle's
    OPEN time, same convention as coindcx_fetcher.py) with columns:
      volume              total base-asset volume in this candle
      taker_buy_volume     base-asset volume where the TAKER was a buyer
                           (i.e. resting sell orders were lifted - real
                           buying pressure)
      taker_sell_volume    volume - taker_buy_volume (taker was a seller,
                           real selling pressure)

    Returns None (does NOT raise) if the symbol isn't mapped, if Binance
    is unreachable, or if a non-200 response comes back (including a 451)
    - see module docstring for why this fails soft instead of hard.
    """
    symbol = COIN_TO_BINANCE_SYMBOL.get(coin.upper())
    if symbol is None:
        print(f"[binance_taker_volume_fetcher] No Binance symbol mapping for coin={coin!r} - skipping.")
        return None
    if interval not in INTERVAL_MS:
        print(f"[binance_taker_volume_fetcher] Unsupported interval={interval!r} - skipping.")
        return None

    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    step_ms = INTERVAL_MS[interval]

    all_rows = []
    cursor = start_ms
    consecutive_failures = 0

    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(BASE, params=params, timeout=20)
        except requests.RequestException as e:
            consecutive_failures += 1
            print(f"[binance_taker_volume_fetcher] Request error ({e}), "
                  f"attempt {consecutive_failures}/{max_retries}")
            if consecutive_failures >= max_retries:
                print("[binance_taker_volume_fetcher] Giving up after repeated failures - "
                      "caller should fall back to the OBV proxy.")
                return None
            time.sleep(2 * consecutive_failures)
            continue

        if resp.status_code == 451:
            print("[binance_taker_volume_fetcher] HTTP 451 - Binance is blocking this server's "
                  "region/IP (the SAME restriction that originally forced this project onto "
                  "CoinDCX for OHLCV - see coindcx_fetcher.py). Real CVD is not usable from "
                  "this environment. Falling back to the OBV proxy is the correct move here, "
                  "not a retry - this will not resolve itself.")
            return None

        if resp.status_code != 200:
            consecutive_failures += 1
            print(f"[binance_taker_volume_fetcher] HTTP {resp.status_code}: {resp.text[:200]}, "
                  f"attempt {consecutive_failures}/{max_retries}")
            if consecutive_failures >= max_retries:
                return None
            time.sleep(2 * consecutive_failures)
            continue

        consecutive_failures = 0
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        last_open_time = batch[-1][0]
        if last_open_time <= cursor:
            break  # safety against an infinite loop if Binance ever returns a non-advancing page
        cursor = last_open_time + step_ms
        time.sleep(0.15)  # be polite - well under Binance's public rate limit

    if not all_rows:
        print(f"[binance_taker_volume_fetcher] No data returned for {symbol} {interval} "
              f"{start_date}..{end_date}")
        return None

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "num_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["volume"] = df["volume"].astype(float)
    df["taker_buy_volume"] = df["taker_buy_base_volume"].astype(float)
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
    df = df.set_index("timestamp")[["volume", "taker_buy_volume", "taker_sell_volume"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df
