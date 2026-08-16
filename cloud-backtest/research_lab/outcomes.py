from __future__ import annotations
import pandas as pd
from .research_config import HORIZONS_MIN
from .research_time import canonical_minute, completed_minute


def label_observation(candles: pd.DataFrame, feature_time, now=None, fee_rate=0.00075):
    t = canonical_minute(feature_time); cutoff = completed_minute(now)
    c = candles.copy(); c.index = pd.to_datetime(c.index, utc=True); c = c.sort_index()
    c = c.loc[c.index <= cutoff]
    past = c.loc[c.index <= t]
    if past.empty: return []
    entry = float(past["close"].iloc[-1]); out=[]; cost=2.0*float(fee_rate)
    for h in HORIZONS_MIN:
        end = t + pd.Timedelta(minutes=h)
        # Critical causality boundary: the endpoint candle itself must be closed.
        if end > cutoff: continue
        fut = c.loc[(c.index > t) & (c.index <= end)]
        if fut.empty or fut.index.max() < end: continue
        exit_price=float(fut["close"].iloc[-1])
        gl=exit_price/entry-1; gs=entry/exit_price-1
        out.append({
            "feature_time":t.isoformat(),"horizon_min":h,"outcome_time":end.isoformat(),
            "return_long":float(gl),"return_short":float(gs),
            "net_return_long":float(gl-cost),"net_return_short":float(gs-cost),
            "mfe_long":float(fut["high"].max()/entry-1),"mae_long":float(1-fut["low"].min()/entry),
            "mfe_short":float(1-fut["low"].min()/entry),"mae_short":float(fut["high"].max()/entry-1),
        })
    return out
