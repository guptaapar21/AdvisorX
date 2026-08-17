
from __future__ import annotations
import pandas as pd

def utc_ts(value=None) -> pd.Timestamp:
    t = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

def completed_minute(now=None) -> pd.Timestamp:
    # Current minute may still be forming; last fully closed minute is used.
    return utc_ts(now).floor("min") - pd.Timedelta(minutes=1)

def canonical_minute(value) -> pd.Timestamp:
    return utc_ts(value).floor("min")
