
from __future__ import annotations
import pandas as pd
from .research_config import PURGE_MIN

def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

def split_times(feature_times):
    times = sorted({_ts(t) for t in feature_times})
    if len(times) < 60:
        return None
    n = len(times)
    d = times[int(n * 0.60) - 1]
    v = times[int(n * 0.80) - 1]
    h = times[-1]
    if not (d < v < h):
        return None
    return d, v, h

def _row_key(row):
    oid = row.get("observation_id")
    if oid is not None:
        return (str(oid), int(row.get("horizon_min", -1)))
    return (
        _ts(row["feature_time"]).isoformat(),
        str(row.get("symbol", "?")),
        int(row.get("horizon_min", -1)),
    )

def split_purged(rows, discovery_end=None, validation_end=None, holdout_end=None):
    if discovery_end is None:
        bounds = split_times([r["feature_time"] for r in rows])
        if bounds is None:
            return [], [], []
        discovery_end, validation_end, holdout_end = bounds

    d, v, h = map(_ts, (discovery_end, validation_end, holdout_end))
    if not (d < v < h):
        raise ValueError("Boundaries must be strictly increasing")

    disc = [r for r in rows if _ts(r["feature_time"]) < d]
    val = [
        r for r in rows
        if _ts(r["feature_time"]) >= d + pd.Timedelta(minutes=PURGE_MIN)
        and _ts(r["feature_time"]) < v
    ]
    hold = [
        r for r in rows
        if _ts(r["feature_time"]) >= v + pd.Timedelta(minutes=PURGE_MIN)
        and _ts(r["feature_time"]) < h
    ]

    assert_clean_boundaries(disc, val, hold, d, v, h)
    return disc, val, hold

def assert_clean_boundaries(disc, val, hold, discovery_end, validation_end, holdout_end):
    def times(rows):
        return {_ts(r["feature_time"]) for r in rows}

    a, b, c = times(disc), times(val), times(hold)
    assert a.isdisjoint(b) and b.isdisjoint(c) and a.isdisjoint(c)

    if a and b:
        assert min(b) - max(a) >= pd.Timedelta(minutes=PURGE_MIN)
    if b and c:
        assert min(c) - max(b) >= pd.Timedelta(minutes=PURGE_MIN)

    for rows in (disc, val, hold):
        keys = [_row_key(row) for row in rows]
        assert len(keys) == len(set(keys)), "Duplicate observation/horizon key inside partition"

def assert_observation_partitions(rows):
    seen = set()
    for row in rows:
        oid = row.get("observation_id")
        if oid is None:
            raise AssertionError("Missing observation_id")
        if oid in seen:
            raise AssertionError(f"Duplicate observation_id: {oid}")
        seen.add(oid)
    return seen
