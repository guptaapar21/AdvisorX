
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("trade_path_output")
OUT.mkdir(parents=True, exist_ok=True)

# These values are taken from the current AdvisorX Gemini risk constants:
# taker fee 0.075% each side, minimum RR 1.5, minimum stop 1.2x ATR,
# maximum stop 8%, 2h unresolved expiry.
TAKER_FEE = float(os.getenv("TRADEPATH_TAKER_FEE", "0.00075"))
MIN_RR = float(os.getenv("TRADEPATH_MIN_RR", "1.5"))
MIN_STOP_ATR = float(os.getenv("TRADEPATH_MIN_STOP_ATR", "1.2"))
MAX_STOP_PCT = float(os.getenv("TRADEPATH_MAX_STOP_PCT", "0.08"))
HOLD_MINUTES = int(os.getenv("TRADEPATH_HOLD_MINUTES", "120"))
COOLDOWN_MINUTES = int(os.getenv("TRADEPATH_COOLDOWN_MINUTES", "30"))

ATR_MULTS = (1.2, 1.5)
RRS = (1.5, 2.0)


_FILTER_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(==|<=|>=|<|>)\s*(true|false|[-+0-9.eE]+)\s*$"
)


def parse_filter_text(text: str):
    atoms = []
    for part in str(text).split(" AND "):
        m = _FILTER_RE.match(part)
        if not m:
            raise ValueError(f"Unable to parse research filter: {part!r}")
        feature, op, value = m.groups()
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            value = float(value)
        atoms.append((feature, op, value))
    return atoms


def compute_atr14(raw: pd.DataFrame) -> pd.Series:
    prev = raw["close"].shift(1)
    tr = pd.concat(
        [
            raw["high"] - raw["low"],
            (raw["high"] - prev).abs(),
            (raw["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def evaluate_atom(df: pd.DataFrame, atom):
    feature, op, value = atom
    if feature not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[feature]
    if op == "==":
        return s == value
    if op == "<":
        return s < value
    if op == ">":
        return s > value
    if op == "<=":
        return s <= value
    if op == ">=":
        return s >= value
    raise ValueError(op)


def evaluate_rule(df: pd.DataFrame, atoms):
    mask = np.ones(len(df), dtype=bool)
    for atom in atoms:
        mask &= evaluate_atom(df, atom).to_numpy()
    return mask


def simulate_one(
    raw: pd.DataFrame,
    entry_idx: int,
    direction: str,
    atr_mult: float,
    rr: float,
):
    entry = float(raw["close"].iloc[entry_idx])
    atr = float(raw["atr14"].iloc[entry_idx])

    if not np.isfinite(entry) or not np.isfinite(atr) or entry <= 0 or atr <= 0:
        return None

    risk_distance = max(atr * atr_mult, MIN_STOP_ATR * atr)
    risk_pct = risk_distance / entry

    if risk_pct > MAX_STOP_PCT:
        return {
            "outcome": "risk_rejected",
            "net_return": np.nan,
            "r_multiple": np.nan,
            "hold_minutes": 0,
        }

    if direction == "long":
        stop = entry - risk_distance
        target = entry + rr * risk_distance
    else:
        stop = entry + risk_distance
        target = entry - rr * risk_distance

    end_idx = min(
        len(raw) - 1,
        entry_idx + HOLD_MINUTES,
    )

    for j in range(entry_idx + 1, end_idx + 1):
        high = float(raw["high"].iloc[j])
        low = float(raw["low"].iloc[j])

        stop_hit = (
            low <= stop
            if direction == "long"
            else high >= stop
        )
        target_hit = (
            high >= target
            if direction == "long"
            else low <= target
        )

        # Conservative treatment of ambiguous 1m candles:
        # if both are touched in the same candle, stop wins.
        if stop_hit:
            exit_price = stop
            outcome = "stop"
        elif target_hit:
            exit_price = target
            outcome = "target"
        else:
            continue

        gross = (
            exit_price / entry - 1
            if direction == "long"
            else entry / exit_price - 1
        )
        net = gross - 2 * TAKER_FEE

        return {
            "outcome": outcome,
            "net_return": float(net),
            "r_multiple": float(net / risk_pct),
            "hold_minutes": int(j - entry_idx),
            "entry": entry,
            "stop": stop,
            "target": target,
        }

    exit_price = float(raw["close"].iloc[end_idx])
    gross = (
        exit_price / entry - 1
        if direction == "long"
        else entry / exit_price - 1
    )
    net = gross - 2 * TAKER_FEE

    return {
        "outcome": "expiry",
        "net_return": float(net),
        "r_multiple": float(net / risk_pct),
        "hold_minutes": int(end_idx - entry_idx),
        "entry": entry,
        "stop": stop,
        "target": target,
    }


def select_events(times, cooldown_minutes: int):
    selected = []
    last = None

    for idx, ts in enumerate(times):
        if last is None or ts >= last + pd.Timedelta(minutes=cooldown_minutes):
            selected.append(idx)
            last = ts

    return selected


def metrics(trades: pd.DataFrame):
    if trades.empty:
        return None

    t = trades[
        trades["outcome"] != "risk_rejected"
    ].copy()

    if t.empty:
        return None

    ret = t["net_return"]
    positive = ret[ret > 0]
    negative = ret[ret < 0]

    gp = float(positive.sum())
    gl = float(-negative.sum())
    pf = gp / gl if gl > 0 else (
        999.0 if gp > 0 else 0.0
    )

    equity = ret.cumsum()
    drawdown = equity - equity.cummax()

    counts = t["outcome"].value_counts()

    return {
        "n": int(len(t)),
        "wins": int((ret > 0).sum()),
        "win_rate": float((ret > 0).mean()),
        "avg_net_return": float(ret.mean()),
        "median_net_return": float(ret.median()),
        "pf": float(pf),
        "avg_r": float(t["r_multiple"].mean()),
        "median_r": float(t["r_multiple"].median()),
        "max_drawdown_return": float(drawdown.min()),
        "targets": int(counts.get("target", 0)),
        "stops": int(counts.get("stop", 0)),
        "expiries": int(counts.get("expiry", 0)),
    }


def chronological_three_way(
    trades: pd.DataFrame,
    recent_oos_days: int = 3,
):
    max_time = trades["signal_time"].max()
    recent_start = (
        max_time
        - pd.Timedelta(days=recent_oos_days)
    )

    research = trades[
        trades["signal_time"] < recent_start
    ].copy()

    recent = trades[
        trades["signal_time"] >= recent_start
    ].copy()

    times = np.array(
        sorted(research["signal_time"].unique())
    )

    if len(times) < 100:
        return (
            research,
            research.iloc[0:0].copy(),
            research.iloc[0:0].copy(),
            recent,
        )

    i1 = int(len(times) * 0.60)
    i2 = int(len(times) * 0.80)

    discovery_end = pd.Timestamp(
        times[i1 - 1]
    )
    validation_end = pd.Timestamp(
        times[i2 - 1]
    )

    purge = pd.Timedelta(minutes=HOLD_MINUTES)

    discovery = research[
        research["signal_time"] <= discovery_end
    ].copy()

    validation = research[
        (research["signal_time"] >= discovery_end + purge)
        & (research["signal_time"] <= validation_end)
    ].copy()

    holdout = research[
        research["signal_time"] >= validation_end + purge
    ].copy()

    return (
        discovery,
        validation,
        holdout,
        recent,
    )


def load_inputs():
    root = Path("prefilter_output")
    cand_path = root / "candidate_results.csv"

    if not cand_path.exists():
        raise FileNotFoundError(
            "prefilter_output/candidate_results.csv not found. "
            "Run the discovery engine first."
        )

    candidates = pd.read_csv(cand_path)

    raw = {}
    feature = {}

    for p in root.glob("*_1m.csv"):
        symbol = p.stem.replace("_1m", "")
        df = pd.read_csv(p)
        df["time"] = pd.to_datetime(
            df["open_time"],
            utc=True,
        )

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df = (
            df.sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        df["atr14"] = compute_atr14(df)
        raw[symbol] = df

        # Features are already generated by the discovery engine.
        feature_path = root / "feature_dataset.csv"
        if not feature_path.exists():
            raise FileNotFoundError(
                "prefilter_output/feature_dataset.csv not found."
            )

    all_features = pd.read_csv(
        root / "feature_dataset.csv"
    )
    all_features["feature_time"] = pd.to_datetime(
        all_features["feature_time"],
        utc=True,
    )

    for symbol, group in all_features.groupby("symbol"):
        feature[symbol] = (
            group
            .sort_values("feature_time")
            .reset_index(drop=True)
        )

    return candidates, raw, feature


def replay_candidate(
    row,
    raw,
    feature,
    atr_mult,
    rr,
):
    rule = parse_filter_text(row["filters"])
    direction = row["direction"]

    trades = []

    for symbol, fdf in feature.items():
        if symbol not in raw:
            continue

        mask = evaluate_rule(
            fdf,
            rule,
        )

        candidate_rows = fdf.loc[
            mask,
            ["feature_time"],
        ].copy()

        if candidate_rows.empty:
            continue

        rawdf = raw[symbol]
        raw_ns = (
            rawdf["time"]
            .astype("int64")
            .to_numpy()
        )

        selected = select_events(
            list(
                candidate_rows[
                    "feature_time"
                ]
            ),
            COOLDOWN_MINUTES,
        )

        for pos in selected:
            signal_time = candidate_rows[
                "feature_time"
            ].iloc[pos]

            signal_ns = int(
                signal_time.value
            )

            entry_idx = int(
                np.searchsorted(
                    raw_ns,
                    signal_ns,
                    side="left",
                )
            )

            if (
                entry_idx >= len(rawdf)
                or raw_ns[entry_idx] != signal_ns
            ):
                continue

            sim = simulate_one(
                rawdf,
                entry_idx,
                direction,
                atr_mult,
                rr,
            )

            if sim is None:
                continue

            sim.update(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "signal_time": signal_time,
                }
            )

            trades.append(sim)

    return pd.DataFrame(trades)


def main():
    recent_days = int(
        os.getenv(
            "TRADEPATH_RECENT_OOS_DAYS",
            "3",
        )
    )

    candidates, raw, feature = load_inputs()

    # Do not pretend every candidate deserves an expensive path replay.
    # Include:
    #  - anything that passed historical discovery gates,
    #  - anything with both validation/holdout PF > 1,
    #  - strongest recent-OOS candidates,
    #  - strongest historical-holdout candidates.
    selected = candidates[
        (
            candidates.get(
                "research_pass",
                False,
            ).astype(bool)
        )
        | (
            (
                pd.to_numeric(
                    candidates["validation_pf"],
                    errors="coerce",
                ) > 1.0
            )
            & (
                pd.to_numeric(
                    candidates["research_holdout_pf"],
                    errors="coerce",
                ) > 1.0
            )
        )
        | (
            pd.to_numeric(
                candidates["recent_oos_pf"],
                errors="coerce",
            ) >= 1.20
        )
        | (
            pd.to_numeric(
                candidates["research_holdout_pf"],
                errors="coerce",
            ) >= 1.20
        )
    ].copy()

    # Always include at least the top 50 discovery rows if the filter above
    # somehow produces no rows.
    if selected.empty:
        selected = candidates.head(50).copy()

    # De-duplicate exact filter definitions.
    selected = selected.drop_duplicates(
        subset=["direction", "filters"]
    )

    output_rows = []

    for candidate_index, row in selected.iterrows():
        for atr_mult in ATR_MULTS:
            for rr in RRS:
                trades = replay_candidate(
                    row,
                    raw,
                    feature,
                    atr_mult,
                    rr,
                )

                if trades.empty:
                    continue

                (
                    discovery,
                    validation,
                    holdout,
                    recent,
                ) = chronological_three_way(
                    trades,
                    recent_days,
                )

                for split_name, split_df in [
                    ("discovery", discovery),
                    ("validation", validation),
                    ("research_holdout", holdout),
                    ("recent_oos", recent),
                ]:
                    m = metrics(split_df)

                    if not m:
                        continue

                    output_rows.append(
                        {
                            "candidate_index": int(
                                candidate_index
                            ),
                            "direction": row[
                                "direction"
                            ],
                            "complexity": row[
                                "complexity"
                            ],
                            "filters": row[
                                "filters"
                            ],
                            "atr_mult": atr_mult,
                            "rr": rr,
                            "split": split_name,
                            **m,
                        }
                    )

    result = pd.DataFrame(
        output_rows
    )

    result.to_csv(
        OUT / "trade_path_replay_results.csv",
        index=False,
    )

    # Aggregate each candidate/risk configuration into one row.
    summary = []

    if not result.empty:
        for key, g in result.groupby(
            [
                "candidate_index",
                "atr_mult",
                "rr",
            ]
        ):
            direction = g["direction"].iloc[0]
            filters = g["filters"].iloc[0]

            def get(split):
                x = g[
                    g["split"] == split
                ]
                return (
                    x.iloc[0].to_dict()
                    if not x.empty
                    else None
                )

            d = get("discovery")
            v = get("validation")
            h = get("research_holdout")
            o = get("recent_oos")

            if not all([d, v, h, o]):
                continue

            historical_pass = (
                v["pf"] >= 1.02
                and h["pf"] >= 1.02
                and v["avg_r"] > 0
                and h["avg_r"] > 0
            )

            recent_pass = (
                o["pf"] >= 1.02
                and o["avg_r"] > 0
            )

            summary.append(
                {
                    "candidate_index": key[0],
                    "direction": direction,
                    "filters": filters,
                    "atr_mult": key[1],
                    "rr": key[2],
                    "validation_pf": v["pf"],
                    "validation_avg_r": v["avg_r"],
                    "validation_n": v["n"],
                    "holdout_pf": h["pf"],
                    "holdout_avg_r": h["avg_r"],
                    "holdout_n": h["n"],
                    "recent_oos_pf": o["pf"],
                    "recent_oos_avg_r": o["avg_r"],
                    "recent_oos_n": o["n"],
                    "historical_pass": historical_pass,
                    "recent_oos_pass": recent_pass,
                    "production_candidate": (
                        historical_pass
                        and recent_pass
                    ),
                }
            )

    summary_df = pd.DataFrame(
        summary
    )

    if not summary_df.empty:
        summary_df["rank_score"] = (
            summary_df["recent_oos_avg_r"] * 10
            + summary_df["recent_oos_pf"].clip(
                upper=3
            )
            + summary_df["holdout_pf"].clip(
                upper=3
            )
        )

        summary_df = summary_df.sort_values(
            [
                "production_candidate",
                "historical_pass",
                "rank_score",
            ],
            ascending=False,
        )

    summary_df.to_csv(
        OUT / "trade_path_candidate_summary.csv",
        index=False,
    )

    approved = (
        summary_df[
            summary_df["production_candidate"]
        ].head(20)
        if not summary_df.empty
        else pd.DataFrame()
    )

    lines = [
        "# AdvisorX Trade-Path Replay",
        "",
        "This is a research-only replay. It does not modify production.",
        "",
        "## Production-style execution assumptions",
        "- Entry: close of the qualifying 1-minute candle.",
        "- Stop: ATR-based risk envelope using 1.2x/1.5x ATR variants.",
        "- Target: 1.5R/2.0R variants.",
        "- Round-trip taker fee: 0.075% each side.",
        "- Ambiguous same-candle stop+target: conservative stop-first.",
        "- Expiry: 120 minutes at the last observed close.",
        "- Per-symbol entry cooldown: 30 minutes.",
        "- Latest 3 days remain frozen OOS and are not used to fit filters.",
        "",
        f"Candidates replayed: **{len(selected)}**",
        f"Risk configurations: **{len(ATR_MULTS) * len(RRS)}**",
        f"Production candidates after trade-path replay: **{len(approved)}**",
        "",
    ]

    if approved.empty:
        lines.append(
            "No candidate passed the full trade-path validation + historical holdout + recent OOS gates."
        )
    else:
        lines.append(
            "## Candidates that survived the full trade path"
        )
        for _, r in approved.iterrows():
            lines.append(
                f"- {r['direction'].upper()} | "
                f"{r['filters']} | "
                f"ATR={r['atr_mult']} RR={r['rr']} | "
                f"validation PF={r['validation_pf']:.2f} | "
                f"holdout PF={r['holdout_pf']:.2f} | "
                f"recent OOS PF={r['recent_oos_pf']:.2f}"
            )

    (OUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
