from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import requests

# CoinDCX documents the spot candles REST API at api.coindcx.com.
BASE = "https://api.coindcx.com/market_data/candles"


def fetch_1m(symbol: str, start_time, end_time, limit: int = 1000, retries: int = 4) -> pd.DataFrame:
    s = pd.Timestamp(start_time)
    e = pd.Timestamp(end_time)
    s = s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")
    e = e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
    if e <= s:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if limit <= 0 or limit > 1000:
        raise ValueError("CoinDCX candles limit must be between 1 and 1000")

    cur = int(s.timestamp() * 1000)
    end_ms = int(e.timestamp() * 1000)
    step = 60_000 * limit
    rows = []

    for _page in range(10000):
        if cur >= end_ms:
            break

        params = {
            "pair": f"B-{symbol}_USDT",
            "interval": "1m",
            "startTime": cur,
            "endTime": min(cur + step, end_ms),
            "limit": limit,
        }

        data = None
        last_error = None

        for attempt in range(retries):
            try:
                response = requests.get(BASE, params=params, timeout=25)
                response.raise_for_status()
                data = response.json()
                last_error = None
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 >= retries:
                    raise RuntimeError(
                        f"CoinDCX candles request failed for {symbol}: {exc}"
                    ) from exc
                time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1.5))

        if last_error is not None or data is None:
            raise RuntimeError(
                f"CoinDCX returned no candles for {symbol}: {last_error}"
            )

        if not isinstance(data, list):
            raise ValueError(
                f"CoinDCX candles response for {symbol} was not a list: {str(data)[:500]}"
            )

        if not data:
            cur += step
            continue

        valid_rows = [row for row in data if isinstance(row, dict) and "time" in row]
        if not valid_rows:
            raise ValueError(
                f"CoinDCX candles response for {symbol} contained no valid candle rows: {str(data)[:500]}"
            )

        rows.extend(valid_rows)
        max_time = max(int(row["time"]) for row in valid_rows)
        next_cur = max(cur + 60_000, max_time + 60_000)
        if next_cur <= cur:
            raise RuntimeError(f"CoinDCX candles pagination made no progress for {symbol}")
        cur = next_cur
        time.sleep(0.15)
    else:
        raise RuntimeError(f"CoinDCX candles pagination exceeded safety limit for {symbol}")

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(rows)
    frame["open_time"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
    frame = frame.set_index("open_time")[["open", "high", "low", "close", "volume"]].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame.dropna(subset=["open", "high", "low", "close", "volume"])


# Retained for compatibility; ResearchLab does not persist candles to Git.
def store_candles(df: pd.DataFrame, symbol: str, root, retention_days: int) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{symbol}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=df.columns)
    both = pd.concat([old, df]).sort_index()
    both.index = pd.to_datetime(both.index, utc=True)
    both = both[~both.index.duplicated(keep="last")]
    if not both.empty:
        cutoff = both.index.max() - pd.Timedelta(days=retention_days)
        both = both.loc[both.index >= cutoff]
    both.to_parquet(path, index=True)


def load_candles(symbol: str, root) -> pd.DataFrame:
    path = Path(root) / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()
