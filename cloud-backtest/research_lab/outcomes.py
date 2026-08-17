
from __future__ import annotations
import pandas as pd
from .research_config import HORIZONS_MIN, TAKER_FEE_RATE
from .research_time import canonical_minute, completed_minute

def label_observation(candles: pd.DataFrame, feature_time, now=None, fee_rate=TAKER_FEE_RATE):
    t = canonical_minute(feature_time)
    cutoff = completed_minute(now)
    c = candles.copy()
    c.index = pd.to_datetime(c.index, utc=True)
    c = c.loc[c.index <= cutoff].sort_index()

    past = c.loc[c.index <= t]
    if past.empty:
        return []

    entry = float(past["close"].iloc[-1])
    out = []
    cost = 2.0 * float(fee_rate)

    for horizon in HORIZONS_MIN:
        end = t + pd.Timedelta(minutes=horizon)
        if end > cutoff:
            continue

        fut = c.loc[(c.index > t) & (c.index <= end)]
        if fut.empty or fut.index.max() < end:
            continue

        # A label is valid only when every expected 1-minute bar exists.
        expected_idx = pd.date_range(
            t + pd.Timedelta(minutes=1),
            end,
            freq="1min",
            tz="UTC",
        )
        if len(fut) != horizon or not expected_idx.isin(fut.index).all():
            continue

        exit_price = float(fut["close"].iloc[-1])
        gl = exit_price / entry - 1.0
        gs = entry / exit_price - 1.0

        out.append({
            "feature_time": t.isoformat(),
            "horizon_min": horizon,
            "outcome_time": end.isoformat(),
            "return_long": float(gl),
            "return_short": float(gs),
            "net_return_long": float(gl - cost),
            "net_return_short": float(gs - cost),
            "mfe_long": float(fut["high"].max() / entry - 1.0),
            "mae_long": float(1.0 - fut["low"].min() / entry),
            "mfe_short": float(1.0 - fut["low"].min() / entry),
            "mae_short": float(fut["high"].max() / entry - 1.0),
            "label_schema_version": "2026-08-17-r4",
        })

    return out
