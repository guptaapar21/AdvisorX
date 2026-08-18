from __future__ import annotations

import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines, resample_candles


OUT = Path("prefilter_output")
OUT.mkdir(parents=True, exist_ok=True)

COINS = tuple(
    x.strip().upper()
    for x in os.getenv(
        "PREFILTER_COINS",
        "BTC,ETH,BNB,SOL,XRP,DOGE,LTC,LINK,TRX,AVAX,HYPE,ZEC,ADA,ACE,PAXG",
    ).split(",")
    if x.strip()
)

DAYS = max(15, min(30, int(os.getenv("PREFILTER_DAYS", "15"))))
HORIZON = int(os.getenv("PREFILTER_HORIZON_MIN", "15"))
MIN_TRADES = int(os.getenv("PREFILTER_MIN_TRADES", "40"))
MIN_COINS = int(os.getenv("PREFILTER_MIN_COINS", "4"))
MIN_BLOCKS = int(os.getenv("PREFILTER_MIN_BLOCKS", "8"))
MIN_BLOCK_POS = float(os.getenv("PREFILTER_MIN_BLOCK_POSITIVE", "0.60"))
MIN_COIN_POS = float(os.getenv("PREFILTER_MIN_COIN_POSITIVE", "0.60"))
FEE = float(os.getenv("PREFILTER_FEE_RATE", "0.00059"))


def wilder_rsi(close: pd.Series, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def wilder_atr(df: pd.DataFrame, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n,
    ).mean()


def adx14(df: pd.DataFrame, n=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0,
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0,
    )

    atr = wilder_atr(df, n).replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1/n,
            adjust=False,
            min_periods=n,
        ).mean()
        / atr
    )
    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1/n,
            adjust=False,
            min_periods=n,
        ).mean()
        / atr
    )

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom

    return dx.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n,
    ).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["ema9"] = x["close"].ewm(
        span=9,
        adjust=False,
    ).mean()
    x["ema21"] = x["close"].ewm(
        span=21,
        adjust=False,
    ).mean()
    x["ema50"] = x["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    x["rsi14"] = wilder_rsi(x["close"])
    x["atr14"] = wilder_atr(x)
    x["adx14"] = adx14(x)
    x["adx_slope_5m"] = x["adx14"] - x["adx14"].shift(5)

    avg_vol = x["volume"].rolling(20).mean()
    x["rvol20"] = x["volume"] / avg_vol.replace(0, np.nan)

    # Rolling VWAP, using only candles up to the current timestamp.
    pv = (x["close"] * x["volume"]).rolling(20).sum()
    vv = x["volume"].rolling(20).sum().replace(0, np.nan)
    x["vwap_gap_pct"] = x["close"] / (pv / vv) - 1.0

    x["ema9_gap_pct"] = x["close"] / x["ema9"] - 1.0
    x["ema21_gap_pct"] = x["close"] / x["ema21"] - 1.0
    x["ema50_gap_pct"] = x["close"] / x["ema50"] - 1.0
    x["ema9_21_gap_pct"] = x["ema9"] / x["ema21"] - 1.0
    x["ema21_50_gap_pct"] = x["ema21"] / x["ema50"] - 1.0

    x["return_5m"] = x["close"].pct_change(5)
    x["return_15m"] = x["close"].pct_change(15)

    high20 = x["high"].rolling(20).max()
    low20 = x["low"].rolling(20).min()
    width = (high20 - low20).replace(0, np.nan)
    x["range_position_20"] = (
        x["close"] - low20
    ) / width

    x["range_extension_atr"] = (
        (x["close"] - (high20 + low20) / 2).abs()
        / x["atr14"].replace(0, np.nan)
    )

    # These are deterministic tags, calculated only from completed bars.
    x["trend_long"] = (
        (x["ema9"] > x["ema21"])
        & (x["ema21"] > x["ema50"])
    )
    x["trend_short"] = (
        (x["ema9"] < x["ema21"])
        & (x["ema21"] < x["ema50"])
    )

    return x.replace(
        [np.inf, -np.inf],
        np.nan,
    )


FEATURES = [
    "rsi14",
    "adx14",
    "adx_slope_5m",
    "rvol20",
    "vwap_gap_pct",
    "ema9_gap_pct",
    "ema21_gap_pct",
    "ema50_gap_pct",
    "ema9_21_gap_pct",
    "ema21_50_gap_pct",
    "return_5m",
    "return_15m",
    "range_position_20",
    "range_extension_atr",
]


def make_examples(df: pd.DataFrame, symbol: str):
    x = build_features(df)

    # Outcome is deliberately created AFTER features.
    # Features at T never read any candle after T.
    future = x["close"].shift(-HORIZON)

    x["long_return"] = (
        future / x["close"]
        - 1.0
        - 2 * FEE
    )
    x["short_return"] = (
        x["close"] / future
        - 1.0
        - 2 * FEE
    )

    x["symbol"] = symbol
    x["feature_time"] = x.index

    keep = [
        "symbol",
        "feature_time",
        *FEATURES,
        "trend_long",
        "trend_short",
        "long_return",
        "short_return",
    ]

    return (
        x[keep]
        .dropna()
        .reset_index(drop=True)
    )


def chronological_split(df: pd.DataFrame):
    times = np.array(
        sorted(df["feature_time"].unique())
    )

    if len(times) < 100:
        raise RuntimeError(
            "Insufficient timestamps for chronological split"
        )

    discovery_idx = int(len(times) * 0.60)
    validation_idx = int(len(times) * 0.80)

    discovery_end = pd.Timestamp(
        times[discovery_idx - 1]
    )
    validation_end = pd.Timestamp(
        times[validation_idx - 1]
    )

    validation_start = (
        discovery_end
        + pd.Timedelta(minutes=HORIZON)
    )
    holdout_start = (
        validation_end
        + pd.Timedelta(minutes=HORIZON)
    )

    discovery = df[
        df.feature_time <= discovery_end
    ].copy()

    validation = df[
        (df.feature_time >= validation_start)
        & (df.feature_time <= validation_end)
    ].copy()

    holdout = df[
        df.feature_time >= holdout_start
    ].copy()

    return (
        discovery,
        validation,
        holdout,
    )


def candidate_thresholds(
    df: pd.DataFrame,
    feature: str,
):
    s = df[feature].dropna()

    # Thresholds are learned from discovery only.
    return sorted(
        {
            float(s.quantile(q))
            for q in (
                0.15,
                0.25,
                0.35,
                0.50,
                0.65,
                0.75,
                0.85,
            )
        }
    )


def match_atom(df, atom):
    feature, operator, value = atom

    if operator == ">":
        return df[feature] > value
    if operator == ">=":
        return df[feature] >= value
    if operator == "<":
        return df[feature] < value
    if operator == "<=":
        return df[feature] <= value

    raise ValueError(
        f"Unknown operator: {operator}"
    )


def evaluate_rule(
    df: pd.DataFrame,
    atoms,
    direction: str,
):
    mask = np.ones(
        len(df),
        dtype=bool,
    )

    for atom in atoms:
        mask &= match_atom(
            df,
            atom,
        ).to_numpy()

    value_col = (
        "long_return"
        if direction == "long"
        else "short_return"
    )

    z = df.loc[
        mask,
        [
            "symbol",
            "feature_time",
            value_col,
        ],
    ].copy()

    if z.empty:
        return None

    # A persistent condition can produce many consecutive bars.
    # Treat a new qualifying event as one observation per symbol per horizon.
    z = z.sort_values(
        ["symbol", "feature_time"]
    )
    gap = (
        z.groupby("symbol")["feature_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    z = z[
        gap.isna() | (gap >= HORIZON)
    ].copy()

    if len(z) < MIN_TRADES:
        return None

    ret = z[value_col].astype(float)

    positive = ret[ret > 0]
    negative = ret[ret < 0]

    gross_profit = float(
        positive.sum()
    )
    gross_loss = float(
        -negative.sum()
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            999.0
            if gross_profit > 0
            else 0.0
        )
    )

    coin_mean = (
        z.groupby("symbol")[value_col]
        .mean()
    )

    block_mean = (
        z.assign(
            block=z.feature_time.dt.floor("30min")
        )
        .groupby("block")[value_col]
        .mean()
    )

    if (
        len(coin_mean) < MIN_COINS
        or len(block_mean) < MIN_BLOCKS
    ):
        return None

    return {
        "n": int(len(z)),
        "coins": int(len(coin_mean)),
        "positive_rate": float(
            (ret > 0).mean()
        ),
        "avg_return": float(
            ret.mean()
        ),
        "profit_factor": float(pf),
        "positive_coin_fraction": float(
            (coin_mean > 0).mean()
        ),
        "positive_block_fraction": float(
            (block_mean > 0).mean()
        ),
        "worst_return": float(
            ret.min()
        ),
    }


def make_atoms(discovery):
    atoms = []

    # Fixed trend-direction atoms are especially useful because the eventual
    # production use is a pre-filter before Gemini, not a standalone strategy.
    for f in FEATURES:
        for threshold in candidate_thresholds(
            discovery,
            f,
        ):
            atoms.extend(
                [
                    (f, ">", threshold),
                    (f, "<", threshold),
                ]
            )

    return atoms


def mine_three_filters(
    discovery,
    direction,
):
    atoms = make_atoms(
        discovery
    )

    single = []

    for atom in atoms:
        metric = evaluate_rule(
            discovery,
            [atom],
            direction,
        )

        if (
            metric
            and metric["avg_return"] > 0
            and metric["profit_factor"] >= 1.0
        ):
            single.append(
                (atom, metric)
            )

    single.sort(
        key=lambda item: (
            item[1]["positive_block_fraction"],
            item[1]["positive_coin_fraction"],
            item[1]["avg_return"],
            item[1]["profit_factor"],
        ),
        reverse=True,
    )

    # Keep the candidate search bounded.
    single = single[:24]

    pairs = []

    for (a, _), (b, _) in itertools.combinations(
        single,
        2,
    ):
        if a[0] == b[0]:
            continue

        metric = evaluate_rule(
            discovery,
            [a, b],
            direction,
        )

        if (
            metric
            and metric["avg_return"] > 0
            and metric["profit_factor"] >= 1.02
        ):
            pairs.append(
                ((a, b), metric)
            )

    pairs.sort(
        key=lambda item: (
            item[1]["positive_block_fraction"],
            item[1]["positive_coin_fraction"],
            item[1]["avg_return"],
            item[1]["profit_factor"],
        ),
        reverse=True,
    )

    pairs = pairs[:48]

    triples = []
    seen = set()

    for (a, b), _ in pairs:
        for c, _ in single:
            if c[0] in (a[0], b[0]):
                continue

            atoms3 = (
                a,
                b,
                c,
            )

            key = tuple(
                sorted(atoms3)
            )

            if key in seen:
                continue

            metric = evaluate_rule(
                discovery,
                atoms3,
                direction,
            )

            if (
                metric
                and metric["avg_return"] > 0
                and metric["profit_factor"] >= 1.05
            ):
                triples.append(
                    (atoms3, metric)
                )
                seen.add(key)

    triples.sort(
        key=lambda item: (
            item[1]["positive_block_fraction"],
            item[1]["positive_coin_fraction"],
            item[1]["avg_return"],
            item[1]["profit_factor"],
        ),
        reverse=True,
    )

    return triples[:60]


def atom_text(atom):
    feature, op, value = atom
    return (
        f"{feature} {op} "
        f"{value:.8g}"
    )


def main():
    end = pd.Timestamp.now(
        tz="UTC"
    ).floor("min")

    start = (
        end
        - pd.Timedelta(days=DAYS)
    )

    all_examples = []
    download_failures = {}

    manifest = {
        "research_only": True,
        "production_changes": False,
        "fetcher": "cloud-backtest/coindcx_fetcher.py",
        "indicator_reference": "cloud-backtest/indicators.py",
        "fee_reference": "cloud-backtest/fee_model.py",
        "window_days": DAYS,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "horizon_min": HORIZON,
    }

    for symbol in COINS:
        try:
            candles_1m = (
                fetch_coindcx_klines(
                    symbol=symbol,
                    interval="1m",
                    start_time=start,
                    end_time=end,
                    limit_per_call=1000,
                    stagger_delay=True,
                )
            )

            # Store the actual fetched raw data in the research artifact.
            candles_1m.to_csv(
                OUT
                / f"{symbol}_1m.csv"
            )

            examples = make_examples(
                candles_1m,
                symbol,
            )

            all_examples.append(
                examples
            )

            print(
                f"{symbol}: "
                f"{len(candles_1m):,} "
                f"1m candles -> "
                f"{len(examples):,} examples"
            )

        except Exception as exc:
            download_failures[symbol] = repr(
                exc
            )
            print(
                f"{symbol}: "
                f"FAILED: {exc}"
            )

    if not all_examples:
        raise RuntimeError(
            "No CoinDCX symbols were downloaded"
        )

    data = (
        pd.concat(
            all_examples,
            ignore_index=True,
        )
        .sort_values(
            "feature_time"
        )
        .reset_index(
            drop=True
        )
    )

    data.to_csv(
        OUT/"feature_dataset.csv",
        index=False,
    )

    (
        discovery,
        validation,
        holdout,
    ) = chronological_split(
        data
    )

    results = []

    for direction in (
        "long",
        "short",
    ):
        candidates = mine_three_filters(
            discovery,
            direction,
        )

        for atoms, discovery_metric in candidates:
            validation_metric = evaluate_rule(
                validation,
                atoms,
                direction,
            )

            holdout_metric = evaluate_rule(
                holdout,
                atoms,
                direction,
            )

            if (
                validation_metric is None
                or holdout_metric is None
            ):
                continue

            passed = (
                validation_metric[
                    "avg_return"
                ] > 0
                and holdout_metric[
                    "avg_return"
                ] > 0
                and validation_metric[
                    "profit_factor"
                ] >= 1.05
                and holdout_metric[
                    "profit_factor"
                ] >= 1.02
                and validation_metric[
                    "positive_block_fraction"
                ] >= MIN_BLOCK_POS
                and holdout_metric[
                    "positive_block_fraction"
                ] >= MIN_BLOCK_POS
                and validation_metric[
                    "positive_coin_fraction"
                ] >= MIN_COIN_POS
                and holdout_metric[
                    "positive_coin_fraction"
                ] >= MIN_COIN_POS
            )

            results.append(
                {
                    "direction": direction,
                    "filter_1": atom_text(
                        atoms[0]
                    ),
                    "filter_2": atom_text(
                        atoms[1]
                    ),
                    "filter_3": atom_text(
                        atoms[2]
                    ),
                    "discovery_n": discovery_metric["n"],
                    "discovery_avg": discovery_metric["avg_return"],
                    "discovery_pf": discovery_metric["profit_factor"],
                    "validation_n": validation_metric["n"],
                    "validation_avg": validation_metric["avg_return"],
                    "validation_pf": validation_metric["profit_factor"],
                    "validation_block_positive": validation_metric[
                        "positive_block_fraction"
                    ],
                    "validation_coin_positive": validation_metric[
                        "positive_coin_fraction"
                    ],
                    "holdout_n": holdout_metric["n"],
                    "holdout_avg": holdout_metric["avg_return"],
                    "holdout_pf": holdout_metric["profit_factor"],
                    "holdout_block_positive": holdout_metric[
                        "positive_block_fraction"
                    ],
                    "holdout_coin_positive": holdout_metric[
                        "positive_coin_fraction"
                    ],
                    "passed": passed,
                }
            )

    result_df = pd.DataFrame(
        results
    )

    if not result_df.empty:
        result_df[
            "robust_score"
        ] = (
            result_df["holdout_avg"]
            * 1000
            + result_df["holdout_pf"]
            * 0.10
            + result_df[
                "holdout_block_positive"
            ]
            * 0.20
            + result_df[
                "holdout_coin_positive"
            ]
            * 0.20
        )

        result_df = result_df.sort_values(
            [
                "passed",
                "robust_score",
            ],
            ascending=False,
        )

    result_df.to_csv(
        OUT/"three_filter_candidates.csv",
        index=False,
    )

    approved = (
        result_df[
            result_df["passed"]
        ]
        .head(10)
        .to_dict("records")
        if not result_df.empty
        else []
    )

    manifest.update(
        {
            "symbols_succeeded": [
                x
                for x in COINS
                if x
                not in download_failures
            ],
            "symbols_failed": download_failures,
            "example_rows": int(
                len(data)
            ),
            "discovery_rows": int(
                len(discovery)
            ),
            "validation_rows": int(
                len(validation)
            ),
            "holdout_rows": int(
                len(holdout)
            ),
            "candidate_count": int(
                len(result_df)
            ),
            "approved_count": int(
                len(approved)
            ),
        }
    )

    (
        OUT/"manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = [
        "# AdvisorX CoinDCX Pre-Filter Lab",
        "",
        f"Window: **{start.isoformat()} → {end.isoformat()}**",
        f"Days: **{DAYS}**",
        f"Symbols succeeded: **{len(manifest['symbols_succeeded'])}/{len(COINS)}**",
        f"Examples: **{len(data):,}**",
        (
            "Chronological "
            f"discovery/validation/holdout: "
            f"**{len(discovery):,} / "
            f"{len(validation):,} / "
            f"{len(holdout):,}**"
        ),
        f"Three-filter candidates: **{len(result_df):,}**",
        f"Holdout-passed: **{len(approved)}**",
        "",
        "## Isolation",
        "- Production scanner untouched.",
        "- Existing ResearchLab untouched.",
        "- No Gemini prompt/state changes.",
        "- No automatic production promotion.",
        "",
        "## Data source",
        "This lab reuses the repository's existing `coindcx_fetcher.py` rather than maintaining a second CoinDCX downloader.",
        "",
        "## Top holdout-passed filters",
    ]

    if approved:
        for row in approved:
            summary.append(
                "- "
                f"{row['direction'].upper()} | "
                f"{row['filter_1']} AND "
                f"{row['filter_2']} AND "
                f"{row['filter_3']} | "
                f"holdout N={row['holdout_n']} "
                f"PF={row['holdout_pf']:.2f} "
                f"avg={row['holdout_avg']:.6f}"
            )
    else:
        summary.append(
            "- None yet; no candidate passed "
            "the holdout gates."
        )

    (
        OUT/"SUMMARY.md"
    ).write_text(
        "\n".join(summary)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
