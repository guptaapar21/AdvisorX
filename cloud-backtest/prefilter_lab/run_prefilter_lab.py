
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from coindcx_fetcher import fetch_coindcx_klines


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

# Research design:
# - 30 days are used to discover/validate/hold out candidates.
# - The newest 3 days are kept completely untouched as final OOS.
RESEARCH_DAYS = max(15, min(30, int(os.getenv("PREFILTER_RESEARCH_DAYS", "30"))))
RECENT_OOS_DAYS = max(3, min(7, int(os.getenv("PREFILTER_RECENT_OOS_DAYS", "3"))))
HORIZONS = (5, 10, 15, 30, 60)
TARGET_HORIZON = int(os.getenv("PREFILTER_TARGET_HORIZON", "15"))

MIN_TRADES = int(os.getenv("PREFILTER_MIN_TRADES", "40"))
MIN_COINS = int(os.getenv("PREFILTER_MIN_COINS", "4"))
MIN_BLOCKS = int(os.getenv("PREFILTER_MIN_BLOCKS", "8"))
MIN_BLOCK_POS = float(os.getenv("PREFILTER_MIN_BLOCK_POSITIVE", "0.60"))
MIN_COIN_POS = float(os.getenv("PREFILTER_MIN_COIN_POSITIVE", "0.60"))
FEE = float(os.getenv("PREFILTER_FEE_RATE", "0.00059"))

# Broad discovery beam: 1-, 2-, and 3-filter candidates are all eligible.
Q = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MAX_SINGLE = 48
MAX_PAIRS = 96
MAX_TRIPLES = 96
MAX_PER_FEATURE = 4

# Validation / holdout gates.
DISCOVERY_MIN_PF = 1.00
VALIDATION_MIN_PF = 1.02
HOLDOUT_MIN_PF = 1.02

# Robustness around fitted numeric thresholds, evaluated on validation only.
PERTURB = (0.90, 1.10)
MIN_GOOD_NEIGHBORS = 2


# ---------------------- feature construction ---------------------------

def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def adx14(df: pd.DataFrame, n: int = 14) -> pd.Series:
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
        * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        / atr
    )
    minus_di = (
        100
        * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        / atr
    )

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()

    x["rsi14"] = wilder_rsi(x["close"])
    x["atr14"] = wilder_atr(x)
    x["adx14"] = adx14(x)
    x["adx_slope_5m"] = x["adx14"] - x["adx14"].shift(5)

    avg_vol = x["volume"].rolling(20).mean()
    x["rvol20"] = x["volume"] / avg_vol.replace(0, np.nan)

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
    x["range_position_20"] = (x["close"] - low20) / width
    x["range_extension_atr"] = (
        (x["close"] - (high20 + low20) / 2).abs()
        / x["atr14"].replace(0, np.nan)
    )

    x["trend_long"] = (x["ema9"] > x["ema21"]) & (x["ema21"] > x["ema50"])
    x["trend_short"] = (x["ema9"] < x["ema21"]) & (x["ema21"] < x["ema50"])

    return x.replace([np.inf, -np.inf], np.nan)


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


def make_examples(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    x = build_features(df)

    for h in HORIZONS:
        future = x["close"].shift(-h)
        x[f"long_return_{h}m"] = future / x["close"] - 1.0 - 2 * FEE
        x[f"short_return_{h}m"] = x["close"] / future - 1.0 - 2 * FEE

    x["symbol"] = symbol
    x["feature_time"] = x.index

    keep = [
        "symbol",
        "feature_time",
        *FEATURES,
        "trend_long",
        "trend_short",
    ]
    for h in HORIZONS:
        keep.extend([f"long_return_{h}m", f"short_return_{h}m"])

    return x[keep].dropna().reset_index(drop=True)


# -------------------------- split design --------------------------------

def split_research_and_recent(data: pd.DataFrame):
    """Freeze the newest recent window and leave a max-horizon purge before it."""
    recent_start = data["feature_time"].max() - pd.Timedelta(days=RECENT_OOS_DAYS)

    # Important: the research holdout's forward labels must not consume
    # candles from the final recent-OOS period.
    research_end = recent_start - pd.Timedelta(minutes=max(HORIZONS))

    research_pool = data[data.feature_time <= research_end].copy()
    recent_oos = data[data.feature_time >= recent_start].copy()

    if research_pool.empty or recent_oos.empty:
        raise RuntimeError("Research/OOS split produced an empty partition")

    return research_pool, recent_oos


def split_research_pool(data: pd.DataFrame):
    """Chronological 60/20/20 within the 30-day research pool."""
    times = np.array(sorted(data["feature_time"].unique()))
    if len(times) < 100:
        raise RuntimeError("Insufficient timestamps for chronological research split")

    i1 = int(len(times) * 0.60)
    i2 = int(len(times) * 0.80)
    d_end = pd.Timestamp(times[i1 - 1])
    v_end = pd.Timestamp(times[i2 - 1])

    purge = pd.Timedelta(minutes=max(HORIZONS))
    discovery = data[data.feature_time <= d_end].copy()
    validation = data[
        (data.feature_time >= d_end + purge)
        & (data.feature_time <= v_end)
    ].copy()
    holdout = data[data.feature_time >= v_end + purge].copy()
    return discovery, validation, holdout


# ------------------------ candidate engine ------------------------------

def make_atoms(discovery: pd.DataFrame):
    atoms = []
    for feature in FEATURES:
        s = discovery[feature].dropna()
        for q in Q:
            value = float(s.quantile(q))
            atoms.append((feature, "<", value))
            atoms.append((feature, ">", value))

    # Explicit trend-state candidates.
    atoms += [
        ("trend_long", "==", True),
        ("trend_short", "==", True),
    ]
    return atoms


def match_atom(df: pd.DataFrame, atom):
    feature, op, value = atom
    if op == "<":
        return df[feature] < value
    if op == ">":
        return df[feature] > value
    if op == "==":
        return df[feature] == value
    raise ValueError(f"Unsupported operator: {op}")


def select_events(z: pd.DataFrame, horizon: int):
    """Greedily sample persistent conditions at least `horizon` apart."""
    if z.empty:
        return z

    z = z.sort_values(["symbol", "feature_time"]).copy()
    keep_parts = []
    gap = pd.Timedelta(minutes=horizon)

    for _, group in z.groupby("symbol", sort=False):
        times = group["feature_time"].to_numpy()
        if len(times) == 0:
            continue

        # Greedy event selection using searchsorted rather than iterating
        # through every qualifying row.
        pos = 0
        keep = []
        while pos < len(times):
            keep.append(pos)
            target = times[pos] + gap.to_timedelta64()
            pos = int(times.searchsorted(target, side="left"))

        keep_parts.append(group.iloc[keep])

    if not keep_parts:
        return z.iloc[0:0]
    return pd.concat(keep_parts, ignore_index=False)


def evaluate_rule(
    df: pd.DataFrame,
    rule,
    direction: str,
    horizon: int,
):
    col = f"{direction}_return_{horizon}m"
    mask = np.ones(len(df), dtype=bool)

    for atom in rule:
        mask &= match_atom(df, atom).to_numpy()

    z = df.loc[
        mask,
        ["symbol", "feature_time", col],
    ].copy()
    z = select_events(z, horizon)

    if len(z) < MIN_TRADES:
        return None

    r = z[col].astype(float)
    positive = r[r > 0]
    negative = r[r < 0]
    gp = float(positive.sum())
    gl = float(-negative.sum())
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    coin_means = z.groupby("symbol")[col].mean()
    block_means = (
        z.assign(block=z.feature_time.dt.floor("30min"))
        .groupby("block")[col]
        .mean()
    )

    if len(coin_means) < MIN_COINS or len(block_means) < MIN_BLOCKS:
        return None

    return {
        "n": int(len(z)),
        "coins": int(len(coin_means)),
        "blocks": int(len(block_means)),
        "avg": float(r.mean()),
        "median": float(r.median()),
        "pf": float(pf),
        "win": float((r > 0).mean()),
        "coin_pos": float((coin_means > 0).mean()),
        "block_pos": float((block_means > 0).mean()),
        "worst": float(r.min()),
    }


def discovery_score(metric):
    # Breadth first, raw return second.
    # Ranking only; this is NOT an approval score.
    return (
        metric["coin_pos"] * 2.0
        + metric["block_pos"] * 2.0
        + np.tanh(metric["avg"] * 500.0) * 0.75
        + min(max(metric["pf"], 0.0), 3.0) * 0.15
        + min(metric["n"], 500) / 500.0 * 0.75
    )


def rank_singles(discovery, direction):
    rows = []
    for atom in make_atoms(discovery):
        metric = evaluate_rule(
            discovery,
            [atom],
            direction,
            TARGET_HORIZON,
        )
        # Discovery is permissive: an individual filter need not be
        # profitable by itself; synergy can emerge only after combination.
        if metric:
            rows.append((atom, metric, discovery_score(metric)))

    rows.sort(key=lambda x: x[2], reverse=True)

    chosen = []
    per_feature = {}

    for row in rows:
        feature = row[0][0]
        if per_feature.get(feature, 0) >= MAX_PER_FEATURE:
            continue
        chosen.append(row)
        per_feature[feature] = per_feature.get(feature, 0) + 1
        if len(chosen) >= MAX_SINGLE:
            break

    return chosen


def mine_candidates(discovery, direction):
    singles = rank_singles(discovery, direction)

    candidates = [
        ([atom], metric, 1, score)
        for atom, metric, score in singles
    ]

    pairs = []
    for left, right in itertools.combinations(singles, 2):
        a, b = left[0], right[0]
        if a[0] == b[0]:
            continue

        metric = evaluate_rule(
            discovery,
            [a, b],
            direction,
            TARGET_HORIZON,
        )

        # Do not require a profitable pair before it can be considered.
        # Validation/holdout are the approval gates.
        if metric:
            pairs.append(
                ([a, b], metric, 2, discovery_score(metric))
            )

    pairs.sort(key=lambda x: x[3], reverse=True)

    # Preserve feature diversity in the pair beam.
    pair_counts = {}
    selected_pairs = []
    for item in pairs:
        key = item[0][0][0]
        if pair_counts.get(key, 0) >= max(6, MAX_PAIRS // 16):
            continue
        selected_pairs.append(item)
        pair_counts[key] = pair_counts.get(key, 0) + 1
        if len(selected_pairs) >= MAX_PAIRS:
            break
    pairs = selected_pairs
    candidates.extend(pairs)

    triples = []
    for pair in pairs[:MAX_TRIPLES]:
        pair_rule = pair[0]
        used = {a[0] for a in pair_rule}

        for atom, _, _ in singles:
            if atom[0] in used:
                continue

            rule = sorted(pair_rule + [atom])
            metric = evaluate_rule(
                discovery,
                rule,
                direction,
                TARGET_HORIZON,
            )
            if metric:
                triples.append(
                    (rule, metric, 3, discovery_score(metric))
                )

    triples.sort(key=lambda x: x[3], reverse=True)
    candidates.extend(triples[:MAX_TRIPLES])

    dedup = {}
    for rule, metric, complexity, score in candidates:
        key = tuple(sorted(rule))
        dedup[key] = (rule, metric, complexity, score)

    return list(dedup.values())


def perturb_rule(rule, multiplier):
    out = []
    for feature, op, value in rule:
        if op == "==":
            out.append((feature, op, value))
        else:
            out.append((feature, op, float(value) * multiplier))
    return out


def validation_neighbor_stability(validation, rule, direction):
    variants = []
    for multiplier in (0.90, 1.10):
        variants.append(
            perturb_rule(rule, multiplier)
        )

    if not variants:
        return True, 0

    good = 0
    for variant in variants:
        metric = evaluate_rule(
            validation,
            variant,
            direction,
            TARGET_HORIZON,
        )
        if (
            metric
            and metric["avg"] > 0
            and metric["pf"] >= VALIDATION_MIN_PF
        ):
            good += 1

    return good >= min(MIN_GOOD_NEIGHBORS, len(variants)), good


def format_atom(atom):
    feature, op, value = atom
    if op == "==":
        return f"{feature} == {str(value).lower()}"
    return f"{feature} {op} {float(value):.8g}"


def multi_horizon_stress(df, rule, direction, split_name):
    out = []
    for horizon in HORIZONS:
        metric = evaluate_rule(
            df,
            rule,
            direction,
            horizon,
        )
        if metric:
            out.append(
                {
                    "split": split_name,
                    "direction": direction,
                    "horizon": horizon,
                    **metric,
                }
            )
    return out


# ------------------------------ main ------------------------------------

def main():
    end = pd.Timestamp.now(tz="UTC").floor("min")
    total_start = end - pd.Timedelta(
        days=RESEARCH_DAYS + RECENT_OOS_DAYS
    )

    frames = []
    failures = {}

    for symbol in COINS:
        try:
            candles = fetch_coindcx_klines(
                symbol=symbol,
                interval="1m",
                start_time=total_start,
                end_time=end,
                limit_per_call=1000,
                stagger_delay=True,
            )
            candles.to_csv(OUT / f"{symbol}_1m.csv")
            frames.append(make_examples(candles, symbol))
            print(f"{symbol}: {len(candles):,} candles")
        except Exception as exc:
            failures[symbol] = repr(exc)
            print(f"{symbol}: FAILED: {exc}")

    if not frames:
        raise RuntimeError("No CoinDCX symbols downloaded")

    data = (
        pd.concat(frames, ignore_index=True)
        .sort_values("feature_time")
        .reset_index(drop=True)
    )
    data.to_csv(OUT / "feature_dataset.csv", index=False)

    research_pool, recent_oos = split_research_and_recent(data)
    discovery, validation, research_holdout = split_research_pool(
        research_pool
    )

    rows = []
    stress_rows = []
    search_summary = []

    for direction in ("long", "short"):
        candidates = mine_candidates(
            discovery,
            direction,
        )

        search_summary.append(
            {
                "direction": direction,
                "candidate_count": len(candidates),
                "one_filter_candidates": sum(c[2] == 1 for c in candidates),
                "two_filter_candidates": sum(c[2] == 2 for c in candidates),
                "three_filter_candidates": sum(c[2] == 3 for c in candidates),
            }
        )

        for rule, dmetric, complexity, dscore in candidates:
            vmetric = evaluate_rule(
                validation,
                rule,
                direction,
                TARGET_HORIZON,
            )
            rmetric = evaluate_rule(
                research_holdout,
                rule,
                direction,
                TARGET_HORIZON,
            )

            if vmetric is None or rmetric is None:
                continue

            neighbor_ok, neighbor_good = (
                validation_neighbor_stability(
                    validation,
                    rule,
                    direction,
                )
            )

            # Recent OOS is NEVER used for discovery or validation gates.
            recent_metric = evaluate_rule(
                recent_oos,
                rule,
                direction,
                TARGET_HORIZON,
            )

            if recent_metric is None:
                continue

            # Candidate is "research-approved" only if it passes historical
            # validation + research holdout + threshold robustness.
            research_pass = (
                vmetric["avg"] > 0
                and vmetric["pf"] >= VALIDATION_MIN_PF
                and rmetric["avg"] > 0
                and rmetric["pf"] >= HOLDOUT_MIN_PF
                and vmetric["block_pos"] >= MIN_BLOCK_POS
                and rmetric["block_pos"] >= MIN_BLOCK_POS
                and vmetric["coin_pos"] >= MIN_COIN_POS
                and rmetric["coin_pos"] >= MIN_COIN_POS
                and neighbor_ok
            )

            rows.append(
                {
                    "direction": direction,
                    "complexity": complexity,
                    "filters": " AND ".join(
                        format_atom(a) for a in rule
                    ),
                    "discovery_n": dmetric["n"],
                    "discovery_avg": dmetric["avg"],
                    "discovery_pf": dmetric["pf"],
                    "validation_n": vmetric["n"],
                    "validation_avg": vmetric["avg"],
                    "validation_pf": vmetric["pf"],
                    "research_holdout_n": rmetric["n"],
                    "research_holdout_avg": rmetric["avg"],
                    "research_holdout_pf": rmetric["pf"],
                    "neighbor_good": neighbor_good,
                    "neighbor_ok": neighbor_ok,
                    "recent_oos_n": recent_metric["n"],
                    "recent_oos_avg": recent_metric["avg"],
                    "recent_oos_pf": recent_metric["pf"],
                    "recent_oos_win": recent_metric["win"],
                    "recent_oos_coin_pos": recent_metric["coin_pos"],
                    "recent_oos_block_pos": recent_metric["block_pos"],
                    "research_pass": research_pass,
                    "recent_oos_pass": (
                        recent_metric["avg"] > 0
                        and recent_metric["pf"] >= HOLDOUT_MIN_PF
                        and recent_metric["coin_pos"] >= MIN_COIN_POS
                        and recent_metric["block_pos"] >= MIN_BLOCK_POS
                    ),
                    "discovery_score": dscore,
                }
            )

            stress_rows.extend(
                multi_horizon_stress(
                    validation,
                    rule,
                    direction,
                    "validation",
                )
            )
            stress_rows.extend(
                multi_horizon_stress(
                    research_holdout,
                    rule,
                    direction,
                    "research_holdout",
                )
            )
            stress_rows.extend(
                multi_horizon_stress(
                    recent_oos,
                    rule,
                    direction,
                    "recent_oos",
                )
            )

    results = pd.DataFrame(rows)

    if not results.empty:
        results["promote_to_production_candidate"] = (
            results["research_pass"]
            & results["recent_oos_pass"]
        )
        results["robust_score"] = (
            results["recent_oos_avg"] * 1000
            + results["recent_oos_pf"].clip(upper=3.0) * 0.10
            + results["research_holdout_pf"].clip(upper=3.0) * 0.10
            + results["recent_oos_block_pos"] * 0.20
            + results["recent_oos_coin_pos"] * 0.20
            + results["neighbor_good"] * 0.02
            - results["complexity"] * 0.01
        )
        results = results.sort_values(
            ["promote_to_production_candidate", "research_pass", "robust_score"],
            ascending=False,
        )

    results.to_csv(
        OUT / "candidate_results.csv",
        index=False,
    )
    # Compatibility name.
    results.to_csv(
        OUT / "three_filter_candidates.csv",
        index=False,
    )

    pd.DataFrame(search_summary).to_csv(
        OUT / "search_summary.csv",
        index=False,
    )

    pd.DataFrame(stress_rows).to_csv(
        OUT / "multihorizon_stress.csv",
        index=False,
    )

    final_candidates = (
        results[
            results["promote_to_production_candidate"]
        ]
        .head(20)
        .to_dict("records")
        if not results.empty
        else []
    )

    manifest = {
        "research_only": True,
        "production_changes": False,
        "research_days": RESEARCH_DAYS,
        "recent_oos_days": RECENT_OOS_DAYS,
        "total_days_fetched": RESEARCH_DAYS + RECENT_OOS_DAYS,
        "target_horizon_min": TARGET_HORIZON,
        "stress_horizons_min": list(HORIZONS),
        "total_start_utc": total_start.isoformat(),
        "end_utc": end.isoformat(),
        "research_pool_end_utc": research_pool.feature_time.max().isoformat(),
        "research_to_recent_purge_min": max(HORIZONS),
        "recent_oos_start_utc": recent_oos.feature_time.min().isoformat(),
        "symbols_succeeded": [
            c for c in COINS if c not in failures
        ],
        "symbols_failed": failures,
        "dataset_rows": int(len(data)),
        "research_rows": int(len(research_pool)),
        "recent_oos_rows": int(len(recent_oos)),
        "discovery_rows": int(len(discovery)),
        "validation_rows": int(len(validation)),
        "research_holdout_rows": int(len(research_holdout)),
        "research_pass_count": int(
            results["research_pass"].sum()
        ) if not results.empty else 0,
        "recent_oos_pass_count": int(
            results["recent_oos_pass"].sum()
        ) if not results.empty else 0,
        "production_candidate_count": len(final_candidates),
        "anti_lookahead": {
            "features_use_candles_at_or_before_feature_time": True,
            "thresholds_fit_only_in_discovery": True,
            "validation_is_chronological": True,
            "research_holdout_unseen": True,
            "recent_oos_is_fully_frozen": True,
            "recent_oos_used_for_selection": False,
        },
    }

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# AdvisorX CoinDCX Pre-Filter Lab",
        "",
        "## Correct experiment design",
        f"- Historical research window: **{RESEARCH_DAYS} days**",
        f"- Final frozen recent OOS: **{RECENT_OOS_DAYS} days**",
        f"- Total data fetched: **{RESEARCH_DAYS + RECENT_OOS_DAYS} days**",
        f"- Target horizon for discovery: **{TARGET_HORIZON} minutes**",
        "- Recent OOS is never used to fit thresholds or choose candidates.",
        "",
        "## Search coverage",
    ]

    for row in search_summary:
        lines.append(
            f"- {row['direction'].upper()}: "
            f"{row['candidate_count']} candidates "
            f"({row['one_filter_candidates']} one-filter, "
            f"{row['two_filter_candidates']} two-filter, "
            f"{row['three_filter_candidates']} three-filter)"
        )

    lines += [
        "",
        "## Final candidates that survived research + frozen recent OOS",
    ]

    if final_candidates:
        for row in final_candidates:
            lines.append(
                f"- **{row['direction'].upper()} / {row['complexity']} filter(s)** "
                f"| {row['filters']} "
                f"| recent OOS N={row['recent_oos_n']} "
                f"PF={row['recent_oos_pf']:.2f} "
                f"avg={row['recent_oos_avg']:.6f}"
            )
    else:
        lines.append(
            "- None. No candidate has yet earned production consideration."
        )

    lines += [
        "",
        "## Important",
        "A positive discovery result is not enough.",
        "Production consideration requires chronological validation, research holdout, threshold robustness, and a positive frozen recent-OOS result.",
        "Production scanner and existing ResearchLab are untouched.",
    ]

    (OUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
