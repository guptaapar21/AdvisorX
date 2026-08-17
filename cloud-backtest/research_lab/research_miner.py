from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter

import numpy as np
import pandas as pd

from .research_config import (
    HORIZONS_MIN,
    MIN_MINER_RULE_N,
    MIN_MINER_COIN_COUNT,
    MIN_MINER_AVG_RETURN,
    MIN_MINER_PF,
    MIN_MINER_UNIVARIATE_KEEP,
    MIN_MINER_PAIR_KEEP,
    MIN_MINER_TRIPLE_KEEP,
)
from .research_hypotheses import _match


FEATURES = (
    "return_1m", "return_3m", "return_5m", "return_10m", "return_15m",
    "return_30m", "return_60m",
    "volatility_5m", "volatility_15m", "volatility_30m", "volatility_60m",
    "rvol20", "rvol60", "atr14_pct", "candle_range_pct", "body_to_range",
    "upper_wick_to_range", "lower_wick_to_range", "close_location",
    "ema9_gap_pct", "ema20_gap_pct", "ema50_gap_pct", "ema9_20_gap_pct",
    "rsi14", "adx14", "plus_di14", "minus_di14", "adx_slope_5m",
    "vwap_gap_pct", "distance_from_30m_high", "distance_from_30m_low",
    "distance_from_60m_high", "distance_from_60m_low", "efficiency_20",
    "hour_utc", "minute_of_day", "day_of_week", "is_weekend",
    "session_asia_utc", "session_europe_utc", "session_us_utc",
    "rsi14_delta_5m", "adx14_delta_5m", "rvol20_delta_5m",
    "ema9_gap_pct_delta_5m", "ema20_gap_pct_delta_5m",
    "vwap_gap_pct_delta_5m", "return_5m_delta_5m",
    "return_15m_delta_5m", "return_5m_accel",
    "xsec_breadth_up_5m", "xsec_breadth_up_15m",
    "xsec_trend_breadth", "xsec_median_return_15m",
    "xsec_median_rvol20", "xsec_median_adx14",
    "xsec_dispersion_return_15m", "bar_coverage_60m", "gap_count_60m",
)


def _profit_factor(values):
    if len(values) == 0:
        return 0.0

    values = np.asarray(values, dtype=float)
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()

    if losses == 0:
        return 999.0 if wins > 0 else 0.0

    return float(wins / losses)


def _drawdown(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0

    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(
        np.maximum(equity, 0.0)
    )
    return float(np.max(peaks - equity))


def _thresholds(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) < MIN_MINER_RULE_N:
        return []

    thresholds = []

    for quantile in (
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90,
    ):
        value = float(
            values.quantile(quantile)
        )
        if math.isfinite(value):
            thresholds.append(
                round(value, 10)
            )

    if values.min() < 0 < values.max():
        thresholds.append(0.0)

    return sorted(
        set(thresholds)
    )


def _frame(rows, horizon, direction):
    return_key = (
        "net_return_short"
        if direction == "SHORT"
        else "net_return_long"
    )

    records = []

    for row in rows:
        outcome = row["outcomes"].get(
            str(horizon)
        )
        if not outcome:
            continue

        record = dict(
            row["features"]
        )
        record["_symbol"] = str(
            row.get("symbol", "?")
        )
        record["_return"] = float(
            outcome.get(
                return_key,
                0.0,
            )
        )
        records.append(record)

    return pd.DataFrame(
        records
    )


def _condition_mask(
    arrays,
    condition,
):
    feature = condition["feature"]

    if feature not in arrays:
        return None

    values = arrays[feature]
    valid = np.isfinite(values)
    op = condition["op"]
    threshold = condition["value"]

    if op == ">":
        return valid & (
            values > float(threshold)
        )

    if op == ">=":
        return valid & (
            values >= float(threshold)
        )

    if op == "<":
        return valid & (
            values < float(threshold)
        )

    if op == "<=":
        return valid & (
            values <= float(threshold)
        )

    if op == "between":
        lo = float(
            threshold[0]
        )
        hi = float(
            threshold[1]
        )
        return valid & (
            (values >= lo)
            & (values <= hi)
        )

    return None


def _stats_from_mask(
    frame,
    mask,
):
    if mask is None:
        return None

    if int(mask.sum()) < MIN_MINER_RULE_N:
        return None

    coverage = frame.loc[mask, "bar_coverage_60m"].astype(float) if "bar_coverage_60m" in frame.columns else None
    if coverage is not None and len(coverage) and float(coverage.mean()) < 0.95:
        return None

    values = frame.loc[
        mask,
        "_return",
    ].to_numpy(
        dtype=float
    )

    average = float(
        values.mean()
    )
    profit_factor = _profit_factor(
        values
    )

    if (
        average < MIN_MINER_AVG_RETURN
        or profit_factor < MIN_MINER_PF
    ):
        return None

    symbols = frame.loc[
        mask,
        "_symbol",
    ].astype(str)

    coin_means = (
        pd.Series(
            values,
            index=symbols,
        )
        .groupby(level=0)
        .mean()
    )

    coin_count = int(
        len(coin_means)
    )

    if coin_count < MIN_MINER_COIN_COUNT:
        return None

    coin_positive_fraction = float(
        (coin_means > 0).mean()
    )

    score = (
        average
        * math.sqrt(len(values))
        * (
            0.5
            + 0.5
            * min(
                1.0,
                coin_count / 10.0,
            )
        )
        * (
            0.5
            + 0.5
            * coin_positive_fraction
        )
        * min(
            2.0,
            profit_factor,
        )
    )

    return {
        "n": int(len(values)),
        "avg_net_return": average,
        "median_net_return": float(
            np.median(values)
        ),
        "profit_factor": float(
            profit_factor
        ),
        "positive_rate": float(
            np.mean(values > 0)
        ),
        "max_drawdown": _drawdown(
            values
        ),
        "coin_count": coin_count,
        "coin_positive_fraction": coin_positive_fraction,
        "discovery_score": float(score),
    }


def _record(
    direction,
    horizon,
    conditions,
    stats,
    family,
):
    conditions = sorted(
        conditions,
        key=lambda condition: (
            str(
                condition["feature"]
            ),
            str(
                condition["op"]
            ),
            json.dumps(
                condition["value"],
                sort_keys=True,
            ),
        ),
    )

    return {
        "name": (
            f"MINER_{family}_"
            f"{direction}_{horizon}m_"
            f"{len(conditions)}C"
        ),
        "direction": direction,
        "horizon_min": horizon,
        "conditions": conditions,
        "rationale": (
            "Python quant-miner lead generated "
            "from discovery-only data. "
            "Validation and holdout are mandatory."
        ),
        "discovery": stats,
        "source": "python_quant_miner",
        "discovery_only": True,
    }


def _rank(candidate):
    stats = candidate[
        "discovery"
    ]
    return (
        stats["discovery_score"],
        stats["avg_net_return"],
        stats["profit_factor"],
    )


def _public_internal(item):
    result = dict(item)
    result.pop(
        "_mask",
        None,
    )
    return result


def mine_discovery(
    grouped_rows,
):
    """
    Broad numerical discovery over the FULL discovery split.

    The important design change is that Gemini is no longer the numerical
    search engine. Python searches thousands of inexpensive threshold and
    interaction candidates first. Gemini only reviews a compact top-candidate
    representation. Therefore a Gemini quota failure does not blind the lab.

    All masks are computed from discovery rows only.
    """
    candidates = []

    for horizon in HORIZONS_MIN:
        for direction in (
            "LONG",
            "SHORT",
        ):
            frame = _frame(
                grouped_rows,
                horizon,
                direction,
            )

            if frame.empty:
                continue

            arrays = {
                feature: pd.to_numeric(
                    frame[feature],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
                for feature in FEATURES
                if feature in frame.columns
            }

            univariate = []

            for feature in arrays:
                thresholds = _thresholds(
                    frame[feature]
                )

                for threshold in thresholds:
                    for op in (
                        ">=",
                        "<=",
                    ):
                        condition = {
                            "feature": feature,
                            "op": op,
                            "value": threshold,
                        }

                        mask = _condition_mask(
                            arrays,
                            condition,
                        )
                        stats = _stats_from_mask(
                            frame,
                            mask,
                        )

                        if stats is None:
                            continue

                        univariate.append(
                            {
                                **_record(
                                    direction,
                                    horizon,
                                    [condition],
                                    stats,
                                    "UNI",
                                ),
                                "_mask": mask,
                            }
                        )

            univariate.sort(
                key=_rank,
                reverse=True,
            )

            kept_uni = univariate[
                :MIN_MINER_UNIVARIATE_KEEP
            ]

            candidates.extend(
                _public_internal(
                    item
                )
                for item in kept_uni
            )

            pairs = []

            for left, right in itertools.combinations(
                kept_uni,
                2,
            ):
                left_feature = left[
                    "conditions"
                ][0]["feature"]
                right_feature = right[
                    "conditions"
                ][0]["feature"]

                if left_feature == right_feature:
                    continue

                mask = (
                    left["_mask"]
                    & right["_mask"]
                )

                stats = _stats_from_mask(
                    frame,
                    mask,
                )

                if stats is None:
                    continue

                conditions = [
                    left[
                        "conditions"
                    ][0],
                    right[
                        "conditions"
                    ][0],
                ]

                pairs.append(
                    {
                        **_record(
                            direction,
                            horizon,
                            conditions,
                            stats,
                            "PAIR",
                        ),
                        "_mask": mask,
                    }
                )

            pairs.sort(
                key=_rank,
                reverse=True,
            )

            kept_pairs = pairs[
                :MIN_MINER_PAIR_KEEP
            ]

            candidates.extend(
                _public_internal(
                    item
                )
                for item in kept_pairs
            )

            triples = []

            for pair in kept_pairs[
                :MIN_MINER_TRIPLE_KEEP
            ]:
                used = {
                    condition["feature"]
                    for condition in pair[
                        "conditions"
                    ]
                }

                for uni in kept_uni:
                    extra = uni[
                        "conditions"
                    ][0]

                    if (
                        extra["feature"]
                        in used
                    ):
                        continue

                    mask = (
                        pair["_mask"]
                        & uni["_mask"]
                    )

                    stats = _stats_from_mask(
                        frame,
                        mask,
                    )

                    if stats is None:
                        continue

                    triples.append(
                        {
                            **_record(
                                direction,
                                horizon,
                                [
                                    *pair[
                                        "conditions"
                                    ],
                                    extra,
                                ],
                                stats,
                                "TRIPLE",
                            ),
                            "_mask": mask,
                        }
                    )

            triples.sort(
                key=_rank,
                reverse=True,
            )

            candidates.extend(
                _public_internal(
                    item
                )
                for item in triples[
                    :MIN_MINER_TRIPLE_KEEP
                ]
            )

    unique = {}

    for candidate in candidates:
        signature = (
            candidate["direction"],
            int(
                candidate["horizon_min"]
            ),
            json.dumps(
                candidate["conditions"],
                sort_keys=True,
            ),
        )

        previous = unique.get(
            signature
        )

        if (
            previous is None
            or _rank(candidate)
            > _rank(previous)
        ):
            unique[
                signature
            ] = candidate

    result = list(
        unique.values()
    )
    result.sort(
        key=_rank,
        reverse=True,
    )

    return result
