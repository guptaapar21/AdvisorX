
from __future__ import annotations
import random, time
from pathlib import Path
import pandas as pd
import requests

BASE = "https://public.coindcx.com/market_data/candles"

def fetch_1m(symbol: str, start_time, end_time, limit: int = 1000, retries: int = 4) -> pd.DataFrame:
    s = pd.Timestamp(start_time)
    e = pd.Timestamp(end_time)
    s = s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")
    e = e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
    if e <= s:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    cur = int(s.timestamp() * 1000)
    end_ms = int(e.timestamp() * 1000)
    step = 60_000 * limit
    rows = []

    while cur < end_ms:
        params = {
            "pair": f"B-{symbol}_USDT",
            "interval": "1m",
            "startTime": cur,
            "endTime": min(cur + step, end_ms),
            "limit": limit,
        }
        data = None
        last = None

        for attempt in range(retries):
            try:
                r = requests.get(BASE, params=params, timeout=25)
                r.raise_for_status()
                data = r.json()
                last = None
                break
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt + 1 >= retries:
                    raise
                time.sleep(
                    1.5 * (attempt + 1)
                    + random.uniform(0, 1.5)
                )

        if last is not None or data is None:
            raise last or RuntimeError("CoinDCX returned no response")

        if not data:
            cur += step
            continue

        rows.extend(data)
        max_time = max(int(x["time"]) for x in data)
        cur = max(cur + 60_000, max_time + 60_000)
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    frame = pd.DataFrame(rows)
    frame["open_time"] = pd.to_datetime(
        frame["time"], unit="ms", utc=True
    )
    frame = frame.set_index("open_time")[
        ["open","high","low","close","volume"]
    ].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    for col in ["open","high","low","close","volume"]:
        frame[col] = pd.to_numeric(
            frame[col],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=["open","high","low","close","volume"]
    )
    return frame

# Retained for compatibility; research_lab workflow does NOT persist candles to Git.
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
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()
