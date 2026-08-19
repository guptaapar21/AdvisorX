
from __future__ import annotations

import pandas as pd

from run_trade_path_replay import (
    parse_filter_text,
    chronological_three_way,
    simulate_one,
)


def test_filter_parse():
    atoms = parse_filter_text(
        "rvol20 > 1.5 AND trend_long == true"
    )
    assert atoms == [
        ("rvol20", ">", 1.5),
        ("trend_long", "==", True),
    ]


def test_stop_first_on_ambiguous_candle():
    raw = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.0, 103.0],
            "low": [100.0, 97.0],
            "close": [100.0, 100.0],
            "volume": [1.0, 1.0],
            "atr14": [1.0, 1.0],
        }
    )

    out = simulate_one(
        raw,
        0,
        "long",
        1.2,
        1.5,
    )
    assert out["outcome"] == "stop"


def test_three_way_has_recent_oos_boundary():
    # More than 3 days of history are required so the chronological
    # discovery, validation, holdout and frozen 3-day OOS partitions
    # are all non-empty.
    t = pd.date_range(
        "2026-01-01",
        periods=10000,
        freq="min",
        tz="UTC",
    )
    df = pd.DataFrame(
        {
            "signal_time": t,
            "net_return": 0.001,
            "r_multiple": 0.1,
            "outcome": "target",
        }
    )
    d, v, h, recent = chronological_three_way(
        df,
        recent_oos_days=3,
    )
    assert not d.empty
    assert not v.empty
    assert not h.empty
    assert not recent.empty

    # Every partition must be strictly chronological.
    assert d["signal_time"].max() < v["signal_time"].min()
    assert v["signal_time"].max() < h["signal_time"].min()
    assert h["signal_time"].max() < recent["signal_time"].min()


if __name__ == "__main__":
    test_filter_parse()
    test_stop_first_on_ambiguous_candle()
    test_three_way_has_recent_oos_boundary()
    print("3 tests passed")
