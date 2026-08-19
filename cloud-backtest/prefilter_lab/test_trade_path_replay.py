
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
    t = pd.date_range(
        "2026-01-01",
        periods=1000,
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
    assert recent["signal_time"].min() > h["signal_time"].max()


if __name__ == "__main__":
    test_filter_parse()
    test_stop_first_on_ambiguous_candle()
    test_three_way_has_recent_oos_boundary()
    print("3 tests passed")
