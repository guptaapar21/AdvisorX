import json
from pathlib import Path

import pandas as pd
import pytest

from .research_features import feature_snapshot
from .outcomes import label_observation
from .research_walkforward import split_purged, assert_clean_boundaries
from .research_hypotheses import evaluate_one, _group_rows, discover
from . import research_miner
from . import run as runmod
from . import research_store


def candles(
    n=300,
    start="2026-01-01T00:00:00Z",
):
    i = pd.date_range(
        start,
        periods=n,
        freq="1min",
        tz="UTC",
    )
    close = pd.Series(
        range(n),
        index=i,
        dtype=float,
    ) + 100.0

    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
        },
        index=i,
    )


def _production_rows(
    n=800,
    symbols=("BTC", "ETH", "SOL", "ADA", "LINK"),
):
    rows = []

    for i in range(n):
        t = (
            pd.Timestamp(
                "2026-01-01",
                tz="UTC",
            )
            + pd.Timedelta(minutes=i)
        ).isoformat()

        for symbol in symbols:
            for horizon in (
                5,
                10,
                15,
                30,
                60,
            ):
                edge = (
                    0.0015
                    if i % 5 == 0
                    else -0.00005
                )

                rows.append(
                    {
                        "observation_id": (
                            f"{symbol}:{i:04d}"
                        ),
                        "symbol": symbol,
                        "feature_time": t,
                        "feature": {
                            "rvol20": 1.5,
                            "return_5m": 0.001,
                        },
                        "outcome": {
                            "horizon_min": horizon,
                            "net_return_long": edge,
                            "net_return_short": -edge,
                            "mfe_long": max(
                                edge,
                                0.0,
                            ),
                            "mae_long": 0.0005,
                            "mfe_short": max(
                                -edge,
                                0.0,
                            ),
                            "mae_short": max(
                                edge,
                                0.0,
                            ),
                        },
                    }
                )

    return rows


def test_feature_cutoff_ignores_future_mutation():
    c = candles()
    t = c.index[120]

    before = feature_snapshot(c, t)

    mutated = c.copy()
    mutated.loc[
        mutated.index > t,
        ["close", "volume"],
    ] = 9999.0

    after = feature_snapshot(
        mutated,
        t,
    )

    assert before == after


def test_outcome_strict_future_and_maturity():
    c = candles()
    t = c.index[100]

    assert (
        label_observation(
            c,
            t,
            now=t + pd.Timedelta(minutes=4),
        )
        == []
    )

    outcomes = label_observation(
        c,
        t,
        now=t + pd.Timedelta(minutes=61),
    )

    assert {
        x["horizon_min"]
        for x in outcomes
    } == {5, 10, 15, 30, 60}

    assert all(
        pd.Timestamp(
            x["outcome_time"]
        )
        > t
        for x in outcomes
    )


def test_current_forming_endpoint_is_rejected():
    c = candles()
    t = c.index[100]

    now = (
        t
        + pd.Timedelta(
            minutes=5,
            seconds=10,
        )
    )

    # completed_minute(now) is T+4, so the T+5 endpoint is not yet closed.
    assert (
        label_observation(
            c,
            t,
            now=now,
        )
        == []
    )


def test_closed_endpoint_is_accepted():
    c = candles()
    t = c.index[100]

    now = t + pd.Timedelta(minutes=6)

    outcomes = label_observation(
        c,
        t,
        now=now,
    )

    assert 5 in {
        x["horizon_min"]
        for x in outcomes
    }


def test_missing_future_bar_invalidates_that_horizon():
    c = candles()
    t = c.index[100]

    # Remove the T+5 bar only.
    c_gap = c.drop(
        c.index[105]
    )

    outcomes = label_observation(
        c_gap,
        t,
        now=t + pd.Timedelta(minutes=61),
    )

    assert 5 not in {
        x["horizon_min"]
        for x in outcomes
    }

    # The missing T+5 bar lies inside every longer horizon as well.
    assert not {10, 15, 30, 60}.intersection(
        {x["horizon_min"] for x in outcomes}
    )


def test_multihorizon_purge_real_shape():
    rows = []

    for i in range(500):
        t = (
            pd.Timestamp(
                "2026-01-01",
                tz="UTC",
            )
            + pd.Timedelta(minutes=i)
        )

        for horizon in (
            5,
            10,
            15,
            30,
            60,
        ):
            rows.append(
                {
                    "observation_id": (
                        f"BTC:{i}"
                    ),
                    "symbol": "BTC",
                    "feature_time": t.isoformat(),
                    "horizon_min": horizon,
                }
            )

    discovery, validation, holdout = split_purged(
        rows,
        "2026-01-01T05:00Z",
        "2026-01-01T10:00Z",
        "2026-01-01T12:00Z",
    )

    assert_clean_boundaries(
        discovery,
        validation,
        holdout,
        pd.Timestamp(
            "2026-01-01T05:00Z"
        ),
        pd.Timestamp(
            "2026-01-01T10:00Z"
        ),
        pd.Timestamp(
            "2026-01-01T12:00Z"
        ),
    )


def test_grouped_rows_are_one_observation_for_discovery():
    rows = []

    for i in range(20):
        t = (
            pd.Timestamp(
                "2026-01-01",
                tz="UTC",
            )
            + pd.Timedelta(minutes=i)
        ).isoformat()

        for horizon in (
            5,
            10,
            15,
            30,
            60,
        ):
            rows.append(
                {
                    "observation_id": f"O{i}",
                    "symbol": "BTC",
                    "feature_time": t,
                    "feature": {
                        "rvol20": 1.5
                    },
                    "outcome": {
                        "horizon_min": horizon,
                        "net_return_long": 0.01,
                        "net_return_short": -0.01,
                    },
                }
            )

    grouped = _group_rows(
        rows
    )

    assert len(grouped) == 20
    assert all(
        len(x["outcomes"]) == 5
        for x in grouped
    )


def test_discovery_actual_path_handles_production_multihorizon_shape(
    monkeypatch,
):
    rows = _production_rows(
        800,
        symbols=(
            "BTC",
            "ETH",
            "SOL",
            "ADA",
            "LINK",
        ),
    )

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"hypotheses": ['
                                        '{"name":"rvol",'
                                        '"direction":"LONG",'
                                        '"horizon_min":15,'
                                        '"conditions":['
                                        '{"feature":"rvol20",'
                                        '"op":">=",'
                                        '"value":1.0}],'
                                        '"rationale":"test"}]}'
                                    )
                                }
                            ]
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: Response(),
    )

    result = discover(
        "key",
        rows,
        "test-model",
    )

    assert result["status"] == "ok"
    assert result["quant_candidate_count"] > 0
    assert result["hypotheses"]


def test_discovery_works_without_gemini():
    rows = _production_rows(
        800,
        symbols=(
            "BTC",
            "ETH",
            "SOL",
            "ADA",
            "LINK",
        ),
    )

    result = discover(
        "",
        rows,
        "test-model",
    )

    assert result["status"] == "ok"
    assert result["quant_candidate_count"] > 0
    assert result["hypotheses"] == []


def test_evaluation_production_shape():
    rule = {
        "name": "test",
        "direction": "LONG",
        "horizon_min": 15,
        "conditions": [
            {
                "feature": "rvol20",
                "op": ">=",
                "value": 1.0,
            }
        ],
    }

    metrics = evaluate_one(
        rule,
        _group_rows(
            _production_rows(
                800,
                symbols=(
                    "BTC",
                    "ETH",
                    "SOL",
                    "ADA",
                    "LINK",
                ),
            )
        ),
    )

    assert metrics["n"] == 4000
    assert metrics["avg_net_return"] > 0
    assert metrics["profit_factor"] > 1


def test_integration_maturity_backfill(
    tmp_path,
    monkeypatch,
):
    obs = tmp_path / "observations.jsonl"
    out = tmp_path / "outcomes.jsonl"
    errors = tmp_path / "errors.jsonl"
    state = tmp_path / "state.json"

    monkeypatch.setattr(
        runmod,
        "OBS_FILE",
        obs,
    )
    monkeypatch.setattr(
        runmod,
        "OUTCOME_FILE",
        out,
    )
    monkeypatch.setattr(
        runmod,
        "ERROR_FILE",
        errors,
    )
    monkeypatch.setattr(
        runmod,
        "CANDLE_DIR",
        tmp_path / "candles",
    )
    monkeypatch.setattr(
        runmod,
        "COINS",
        ("BTC",),
    )
    monkeypatch.setattr(
        runmod,
        "TAKER_FEE_RATE",
        0.00075,
    )

    monkeypatch.setattr(
        research_store,
        "STATE_FILE",
        state,
    )
    monkeypatch.setattr(
        research_store,
        "ROOT",
        tmp_path,
    )

    c = candles(
        360,
        "2026-01-01T00:00:00Z",
    )

    def fake_fetch(
        symbol,
        start,
        end,
    ):
        return c.loc[
            (c.index >= start)
            & (c.index <= end)
        ].copy()

    # First run creates observations from the available 00:00–00:50 candles.
    runmod.collect_once(
        pd.Timestamp(
            "2026-01-01T00:50:10Z"
        ),
        fetch_fn=fake_fetch,
    )

    # The current collector waits until the maximum 60-minute horizon is
    # mature before it attempts backfill. These follow-up runs keep the
    # observations inside the 120-minute fetch window while allowing all
    # horizons to mature.
    runmod.collect_once(
        pd.Timestamp(
            "2026-01-01T01:51:10Z"
        ),
        fetch_fn=fake_fetch,
    )
    runmod.collect_once(
        pd.Timestamp(
            "2026-01-01T01:57:10Z"
        ),
        fetch_fn=fake_fetch,
    )
    runmod.collect_once(
        pd.Timestamp(
            "2026-01-01T02:02:10Z"
        ),
        fetch_fn=fake_fetch,
    )

    outrows = runmod.read_jsonl(
        out
    )

    assert outrows

    horizons = {
        int(x["horizon_min"])
        for x in outrows
    }

    assert {
        5,
        10,
        15,
        30,
        60,
    }.issubset(horizons)


def test_delayed_run_backfills_unresolved_observation_inside_fetch_window(
    tmp_path,
    monkeypatch,
):
    obs = tmp_path / "observations.jsonl"
    out = tmp_path / "outcomes.jsonl"
    errors = tmp_path / "errors.jsonl"
    state = tmp_path / "state.json"

    monkeypatch.setattr(
        runmod,
        "OBS_FILE",
        obs,
    )
    monkeypatch.setattr(
        runmod,
        "OUTCOME_FILE",
        out,
    )
    monkeypatch.setattr(
        runmod,
        "ERROR_FILE",
        errors,
    )
    monkeypatch.setattr(
        runmod,
        "CANDLE_DIR",
        tmp_path / "candles",
    )
    monkeypatch.setattr(
        runmod,
        "COINS",
        ("BTC",),
    )

    monkeypatch.setattr(
        research_store,
        "STATE_FILE",
        state,
    )
    monkeypatch.setattr(
        research_store,
        "ROOT",
        tmp_path,
    )

    c = candles(
        360,
        "2026-01-01T00:00:00Z",
    )

    feature_time = c.index[60]

    obs.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    obs.write_text(
        json.dumps(
            {
                "observation_id": "BTC:test",
                "symbol": "BTC",
                "features": feature_snapshot(
                    c,
                    feature_time,
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_fetch(
        symbol,
        start,
        end,
    ):
        return c.loc[
            (c.index >= start)
            & (c.index <= end)
        ].copy()

    # The observation is old enough to have matured but remains inside the
    # 120-minute fetch window, so the current production code can backfill it
    # using only the current fetch-window backfill path.
    runmod.collect_once(
        pd.Timestamp(
            "2026-01-01T02:01:10Z"
        ),
        fetch_fn=fake_fetch,
    )

    outrows = runmod.read_jsonl(
        out
    )

    assert outrows

    assert {
        5,
        10,
        15,
        30,
        60,
    }.issubset(
        {
            int(x["horizon_min"])
            for x in outrows
        }
    )
