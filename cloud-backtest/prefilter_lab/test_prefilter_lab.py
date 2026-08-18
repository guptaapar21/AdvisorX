
from __future__ import annotations

import pandas as pd

from run_prefilter_lab import (
    HORIZONS,
    evaluate_rule,
    split_research_and_recent,
    split_research_pool,
)


def synthetic():
    per_coin = 6000
    t = pd.date_range(
        "2026-01-01",
        periods=per_coin,
        freq="min",
        tz="UTC",
    )

    rows = []
    for symbol in ("BTC", "ETH", "SOL", "XRP"):
        d = pd.DataFrame(
            {
                "symbol": [symbol] * per_coin,
                "feature_time": t,
                "rsi14": [50.0] * per_coin,
                "adx14": [30.0] * per_coin,
                "adx_slope_5m": [1.0] * per_coin,
                "rvol20": [1.5] * per_coin,
                "vwap_gap_pct": [0.0] * per_coin,
                "ema9_gap_pct": [0.0] * per_coin,
                "ema21_gap_pct": [0.0] * per_coin,
                "ema50_gap_pct": [0.0] * per_coin,
                "ema9_21_gap_pct": [0.001] * per_coin,
                "ema21_50_gap_pct": [0.001] * per_coin,
                "return_5m": [0.001] * per_coin,
                "return_15m": [0.002] * per_coin,
                "range_position_20": [0.5] * per_coin,
                "range_extension_atr": [0.2] * per_coin,
                "trend_long": [True] * per_coin,
                "trend_short": [False] * per_coin,
            }
        )
        for h in HORIZONS:
            d[f"long_return_{h}m"] = 0.001
            d[f"short_return_{h}m"] = -0.001
        rows.append(d)

    return pd.concat(rows, ignore_index=True)


def test_frozen_recent_split():
    d = synthetic()
    research, recent = split_research_and_recent(d)
    assert research.feature_time.max() < recent.feature_time.min()
    assert (
        recent.feature_time.max()
        - recent.feature_time.min()
    ) >= pd.Timedelta(days=2, hours=23)


def test_research_split_is_chronological_and_purged():
    d = synthetic()
    research, _ = split_research_and_recent(d)
    discovery, validation, holdout = split_research_pool(research)
    purge = pd.Timedelta(minutes=max(HORIZONS))
    assert discovery.feature_time.max() + purge <= validation.feature_time.min()
    assert validation.feature_time.max() + purge <= holdout.feature_time.min()


def test_persistent_condition_is_sampled_every_horizon():
    from run_prefilter_lab import evaluate_rule

    d = synthetic()
    metric = evaluate_rule(
        d,
        [("trend_long", "==", True)],
        "long",
        15,
    )
    assert metric is not None
    # 6000 minutes / 15-minute event spacing ≈ 400 events per coin.
    # Four coins -> roughly 1600 independent events.
    assert metric["n"] >= 1500
    assert metric["n"] <= 1700


def test_short_and_long_labels_are_distinct():
    d = synthetic()
    long_m = evaluate_rule(
        d,
        [("trend_long", "==", True)],
        "long",
        15,
    )
    short_m = evaluate_rule(
        d,
        [("trend_short", "==", False)],
        "short",
        15,
    )
    assert long_m["avg"] > 0
    assert short_m["avg"] < 0
