#!/usr/bin/env python3
"""Research-to-formulation engine for AdvisorX.

The analyst consumes append-only market-behaviour observation/outcome shards,
creates a bounded family of falsifiable candidate rules, controls multiple
comparisons on the full eligible hypothesis family, validates candidates on a
chronological development split, and promotes only candidates that also survive
a frozen holdout and cost-stress gate.

This module is research-only. It never changes scanner thresholds, never creates
TAKE decisions, and never mutates live trade state. The only live-facing output
is a read-only context file containing formulations that have passed all gates.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "research_formulation_v1"
ROUND_TRIP_FEE_PCT_DEFAULT = 0.15
HORIZONS = (5, 15, 30, 60, 120, 180)
MIN_DISCOVERY_N = 10
MIN_VALIDATION_N = 30
MIN_HOLDOUT_N = 15
MIN_VALIDATION_PF = 1.10
MIN_HOLDOUT_PF = 1.10
MIN_VALIDATION_AVG = 0.02
MIN_HOLDOUT_AVG = 0.02
MIN_POSITIVE_BLOCK_FRACTION = 0.55
MIN_POSITIVE_COIN_FRACTION = 0.55
MAX_VALIDATED = 20
FDR_Q = 0.05
BOOTSTRAP_REPS = 1000
BOOTSTRAP_BLOCK_LEN = 3

@dataclass(frozen=True)
class Row:
    observed_at: datetime
    symbol: str
    direction: str
    horizon_min: int
    event_families: tuple[str, ...]
    activity_ratio: float
    activity_percentile: float
    delta_abs: float
    event_score: float
    wick_ratio: float
    signed_forward_return_pct: float


def _float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _dt(v: Any) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def load_rows(root: Path) -> list[Row]:
    observations: dict[str, dict[str, Any]] = {}
    observation_root = root / "observations"
    outcome_root = root / "outcomes"
    for path in sorted(observation_root.rglob("*.jsonl")):
        for obs in _read_jsonl(path):
            event_id = str(obs.get("event_id") or "")
            when = _dt(obs.get("observed_at"))
            if event_id and when is not None:
                observations[event_id] = obs

    rows: list[Row] = []
    for path in sorted(outcome_root.rglob("*.jsonl")):
        for out in _read_jsonl(path):
            event_id = str(out.get("event_id") or "")
            obs = observations.get(event_id)
            if not obs:
                continue
            when = _dt(obs.get("observed_at"))
            if when is None:
                continue
            horizon = int(_float(out.get("horizon_min"), 0))
            if horizon not in HORIZONS:
                continue
            direction = str(out.get("direction") or obs.get("direction") or "").lower()
            rows.append(Row(
                observed_at=when,
                symbol=str(obs.get("symbol") or out.get("symbol") or ""),
                direction=direction,
                horizon_min=horizon,
                event_families=tuple(str(x) for x in (obs.get("event_families") or [])),
                activity_ratio=_float(obs.get("activity_ratio_vs_60m_median")),
                activity_percentile=_float(obs.get("activity_percentile_60m")),
                delta_abs=abs(_float(obs.get("delta_proxy_ratio"))),
                event_score=_float(obs.get("event_score")),
                wick_ratio=(
                    _float(obs.get("lower_wick_ratio"))
                    if str(obs.get("direction") or "").lower() == "bullish"
                    else _float(obs.get("upper_wick_ratio"))
                ),
                signed_forward_return_pct=_float(out.get("signed_forward_return_pct")),
            ))
    rows.sort(key=lambda r: r.observed_at)
    return rows


def _block_avgs(rows: list[Row], fee_pct: float) -> list[float]:
    blocks: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.signed_forward_return_pct - fee_pct
        key = row.observed_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")
        blocks[key].append(value)
    return [sum(values) / len(values) for values in blocks.values() if values]


def _one_sided_sign_p_value(block_avgs: list[float]) -> float:
    """Exact one-sided sign-test p-value for positive hourly-block means."""
    n = len(block_avgs)
    positives = sum(x > 0 for x in block_avgs)
    negatives = sum(x < 0 for x in block_avgs)
    effective_n = positives + negatives
    if effective_n == 0:
        return 1.0
    # Ties do not contribute evidence in either direction.
    tail = sum(math.comb(effective_n, k) for k in range(positives, effective_n + 1))
    return min(1.0, tail / (2.0 ** effective_n))


def _moving_block_bootstrap_ci_low(
    block_avgs: list[float],
    reps: int = BOOTSTRAP_REPS,
    block_len: int = BOOTSTRAP_BLOCK_LEN,
) -> float:
    """Two-sided 95% lower endpoint from a non-wrapping moving-block bootstrap."""
    n = len(block_avgs)
    if n == 0:
        return -1e9
    if n == 1:
        return block_avgs[0]
    block_len = max(1, min(block_len, n))
    starts = list(range(0, n - block_len + 1))
    # Stable seed makes diagnostics reproducible without using historical data to
    # choose the random seed.
    rng = random.Random(20260912 + n * 1009)
    means: list[float] = []
    for _ in range(max(200, reps)):
        sample: list[float] = []
        while len(sample) < n:
            start = starts[rng.randrange(len(starts))]
            sample.extend(block_avgs[start:start + block_len])
        sample = sample[:n]
        means.append(sum(sample) / n)
    means.sort()
    return means[max(0, min(len(means) - 1, int(0.025 * len(means))))]


def _stats(rows: list[Row], fee_pct: float, include_bootstrap: bool = False) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0, "positive_rate": 0.0, "avg_net_return": 0.0,
            "median_net_return": 0.0, "profit_factor": 0.0, "worst_return": 0.0,
            "max_drawdown": 0.0, "coin_count": 0, "time_block_count": 0,
            "positive_block_fraction": 0.0, "positive_coin_fraction": 0.0,
            "block_sign_p_value": 1.0, "bootstrap_block_ci95_low": -1e9 if include_bootstrap else None,
        }

    vals = [r.signed_forward_return_pct - fee_pct for r in rows]
    gains = sum(v for v in vals if v > 0)
    losses = -sum(v for v in vals if v < 0)
    blocks: dict[str, list[float]] = defaultdict(list)
    coins: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, vals):
        key = row.observed_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")
        blocks[key].append(value)
        coins[row.symbol].append(value)

    block_avgs = [sum(values) / len(values) for values in blocks.values() if values]
    coin_avgs = [sum(values) / len(values) for values in coins.values() if values]
    positive_blocks = sum(x > 0 for x in block_avgs)
    positive_coins = sum(x > 0 for x in coin_avgs)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in vals:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    result: dict[str, Any] = {
        "n": len(vals),
        "positive_rate": sum(v > 0 for v in vals) / len(vals),
        "avg_net_return": sum(vals) / len(vals),
        "median_net_return": statistics.median(vals),
        "profit_factor": gains / losses if losses > 0 else (99.0 if gains > 0 else 0.0),
        "worst_return": min(vals),
        "max_drawdown": max_dd,
        "coin_count": len(coins),
        "time_block_count": len(blocks),
        "positive_block_fraction": positive_blocks / len(blocks) if blocks else 0.0,
        "positive_coin_fraction": positive_coins / len(coins) if coins else 0.0,
        "block_sign_p_value": _one_sided_sign_p_value(block_avgs),
    }
    if include_bootstrap:
        result["bootstrap_block_ci95_low"] = _moving_block_bootstrap_ci_low(block_avgs)
    return result


def _candidate_signature(h: dict[str, Any]) -> str:
    return json.dumps(h, sort_keys=True, separators=(",", ":"))


def generate_candidates(rows: list[Row]) -> list[dict[str, Any]]:
    families = sorted({family for row in rows for family in row.event_families})
    candidates: list[dict[str, Any]] = []
    modifiers = [
        ("none", None),
        ("high_activity", {"activity_ratio_min": 1.8}),
        ("extreme_activity", {"activity_percentile_min": 85}),
        ("very_extreme_activity", {"activity_percentile_min": 95}),
        ("strong_delta_proxy", {"delta_abs_min": 0.65}),
        ("very_strong_delta_proxy", {"delta_abs_min": 0.90}),
        ("strong_event", {"event_score_min": 4}),
        ("strong_wick", {"wick_ratio_min": 0.45}),
    ]
    for family in families:
        for direction in ("bullish", "bearish"):
            for horizon in HORIZONS:
                for label, conditions in modifiers:
                    candidates.append({
                        "name": f"{family} | {label}",
                        "event_family": family,
                        "direction": direction,
                        "horizon_min": horizon,
                        "conditions": conditions or {},
                    })
    for family in families:
        for direction in ("bullish", "bearish"):
            for horizon in (15, 30, 60):
                for conditions, label in [
                    ({"activity_ratio_min": 1.8, "delta_abs_min": 0.65}, "high_activity_and_delta"),
                    ({"activity_percentile_min": 85, "wick_ratio_min": 0.45}, "extreme_activity_and_wick"),
                    ({"event_score_min": 4, "delta_abs_min": 0.65}, "strong_event_and_delta"),
                ]:
                    candidates.append({
                        "name": f"{family} | {label}",
                        "event_family": family,
                        "direction": direction,
                        "horizon_min": horizon,
                        "conditions": conditions,
                    })
    return candidates


def _match(row: Row, hypothesis: dict[str, Any]) -> bool:
    conditions = hypothesis.get("conditions") or {}
    return (
        hypothesis.get("event_family") in row.event_families
        and row.direction == hypothesis.get("direction")
        and row.horizon_min == int(hypothesis.get("horizon_min"))
        and row.activity_ratio >= _float(conditions.get("activity_ratio_min"), -1e99)
        and row.activity_percentile >= _float(conditions.get("activity_percentile_min"), -1e99)
        and row.delta_abs >= _float(conditions.get("delta_abs_min"), -1e99)
        and row.event_score >= _float(conditions.get("event_score_min"), -1e99)
        and row.wick_ratio >= _float(conditions.get("wick_ratio_min"), -1e99)
    )


def _evaluate(hypothesis: dict[str, Any], rows: list[Row], fee_pct: float, include_bootstrap: bool = False) -> dict[str, Any]:
    matched = [row for row in rows if _match(row, hypothesis)]
    return {"rows": matched, "metrics": _stats(matched, fee_pct, include_bootstrap=include_bootstrap)}


def _bh_fdr(
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]],
    q: float = FDR_Q,
) -> tuple[set[str], dict[str, float]]:
    """Benjamini-Hochberg on the complete eligible hypothesis family.

    No performance-based filtering or nested-threshold selection occurs before
    this correction. q-values are reported for every eligible hypothesis.
    """
    ranked = sorted(
        evaluated,
        key=lambda item: (item[1]["block_sign_p_value"], _candidate_signature(item[0])),
    )
    m = len(ranked)
    if m == 0:
        return set(), {}

    q_by_sig: dict[str, float] = {}
    running = 1.0
    for rank in range(m, 0, -1):
        hypothesis, metrics = ranked[rank - 1]
        raw_q = metrics["block_sign_p_value"] * m / rank
        running = min(running, raw_q)
        q_by_sig[_candidate_signature(hypothesis)] = min(1.0, running)

    passed = {sig for sig, value in q_by_sig.items() if value <= q}
    return passed, q_by_sig


def _collapse_nested_activity_after_fdr(
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]],
    fdr_passed: set[str],
) -> set[str]:
    """Collapse only nested 85/95 percentile activity candidates after FDR.

    The choice is deterministic and never uses return performance: when both
    nested thresholds pass FDR for the same family/direction/horizon, keep the
    more specific 95th-percentile hypothesis.
    """
    selected = set(fdr_passed)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for hypothesis, _ in evaluated:
        conditions = hypothesis.get("conditions") or {}
        if set(conditions) == {"activity_percentile_min"}:
            key = (hypothesis.get("event_family"), hypothesis.get("direction"), hypothesis.get("horizon_min"))
            groups[key].append(hypothesis)
    for hypotheses in groups.values():
        passing = [h for h in hypotheses if _candidate_signature(h) in fdr_passed]
        if len(passing) <= 1:
            continue
        passing.sort(key=lambda h: float((h.get("conditions") or {}).get("activity_percentile_min", 0)), reverse=True)
        for hypothesis in passing[1:]:
            selected.discard(_candidate_signature(hypothesis))
    return selected


def analyse(rows: list[Row], fee_pct: float) -> dict[str, Any]:
    if not rows:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "insufficient_data",
            "rows": 0,
            "tested_candidates": 0,
            "deduplicated_candidates": 0,
            "fdr_discovery_candidates": 0,
            "validated": [],
            "rejected": [],
            "discovery_leads": [],
        }

    split = max(1, min(len(rows) - 1, int(len(rows) * 0.70))) if len(rows) > 1 else 1
    validation = rows[:split]
    holdout = rows[split:]

    # Every hypothesis with a minimally meaningful development sample enters the
    # same multiple-testing family. Do not rank/select by return before FDR.
    eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for hypothesis in generate_candidates(rows):
        metrics = _evaluate(hypothesis, validation, fee_pct)["metrics"]
        if metrics["n"] >= MIN_DISCOVERY_N:
            eligible.append((hypothesis, metrics))

    fdr_passed, q_by_sig = _bh_fdr(eligible, FDR_Q)
    post_fdr_selected = _collapse_nested_activity_after_fdr(eligible, fdr_passed)

    enriched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for hypothesis, metrics in eligible:
        enriched_metrics = dict(metrics)
        enriched_metrics["fdr_q"] = q_by_sig[_candidate_signature(hypothesis)]
        enriched.append((hypothesis, enriched_metrics))

    # Discovery leads remain observational and are ranked only for display.
    enriched.sort(
        key=lambda item: (
            item[1]["avg_net_return"],
            item[1]["profit_factor"],
            item[1]["positive_coin_fraction"],
        ),
        reverse=True,
    )
    discovery_leads = [{"hypothesis": h, "validation": v} for h, v in enriched[:30]]

    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for hypothesis, metrics in enriched:
        signature = _candidate_signature(hypothesis)
        fdr_ok = signature in post_fdr_selected and metrics["fdr_q"] <= FDR_Q
        base_ok = (
            metrics["n"] >= MIN_VALIDATION_N
            and metrics["profit_factor"] >= MIN_VALIDATION_PF
            and metrics["avg_net_return"] >= MIN_VALIDATION_AVG
            and metrics["positive_block_fraction"] >= MIN_POSITIVE_BLOCK_FRACTION
            and metrics["positive_coin_fraction"] >= MIN_POSITIVE_COIN_FRACTION
        )
        if not (base_ok and fdr_ok):
            rejected.append({
                "hypothesis": hypothesis,
                "validation": metrics,
                "status": "validation_failed",
                "fdr_passed": fdr_ok,
                "fdr_q": metrics["fdr_q"],
            })
            continue

        # Only FDR-passing + base-gate candidates pay the bootstrap cost.
        boot_validation = _evaluate(hypothesis, validation, fee_pct, include_bootstrap=True)["metrics"]
        if boot_validation["bootstrap_block_ci95_low"] <= 0:
            rejected.append({
                "hypothesis": hypothesis,
                "validation": boot_validation,
                "status": "validation_failed_bootstrap_ci",
                "fdr_passed": True,
                "fdr_q": metrics["fdr_q"],
            })
            continue

        holdout_metrics = _evaluate(hypothesis, holdout, fee_pct, include_bootstrap=True)["metrics"]
        stress_metrics = _evaluate(hypothesis, holdout, max(fee_pct, 0.20), include_bootstrap=True)["metrics"]
        passed = (
            holdout_metrics["n"] >= MIN_HOLDOUT_N
            and holdout_metrics["profit_factor"] >= MIN_HOLDOUT_PF
            and holdout_metrics["avg_net_return"] >= MIN_HOLDOUT_AVG
            and holdout_metrics["positive_block_fraction"] >= MIN_POSITIVE_BLOCK_FRACTION
            and holdout_metrics["positive_coin_fraction"] >= MIN_POSITIVE_COIN_FRACTION
            and holdout_metrics["bootstrap_block_ci95_low"] > 0
            and stress_metrics["avg_net_return"] > 0
        )
        item = {
            "hypothesis": hypothesis,
            "validation": boot_validation,
            "holdout": holdout_metrics,
            "stress": stress_metrics,
            "status": "HOLDOUT_PASSED" if passed else "holdout_failed",
            "fdr_q": metrics["fdr_q"],
        }
        (validated if passed else rejected).append(item)
        if len(validated) >= MAX_VALIDATED:
            break

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "rows": len(rows),
        "validation_rows": len(validation),
        "holdout_rows": len(holdout),
        "tested_candidates": len(generate_candidates(rows)),
        "deduplicated_candidates": len(eligible),
        "fdr_discovery_candidates": len(post_fdr_selected),
        "fdr_q_threshold": FDR_Q,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_block_len": BOOTSTRAP_BLOCK_LEN,
        "fee_pct": fee_pct,
        "discovery_leads": discovery_leads,
        "validated": validated,
        "rejected": rejected[:100],
    }


def render_live_context(result: dict[str, Any], path: Path) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "validated_formulations": [
            {
                "name": item["hypothesis"]["name"],
                "event_family": item["hypothesis"]["event_family"],
                "direction": item["hypothesis"]["direction"],
                "horizon_min": item["hypothesis"]["horizon_min"],
                "conditions": item["hypothesis"]["conditions"],
                "validation": item["validation"],
                "holdout": item["holdout"],
                "stress": item["stress"],
                "status": item["status"],
                "fdr_q": item["fdr_q"],
            }
            for item in result.get("validated", [])
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="market_behavior_research")
    parser.add_argument("--output", default="research_formulations.json")
    parser.add_argument("--fee-pct", type=float, default=ROUND_TRIP_FEE_PCT_DEFAULT)
    args = parser.parse_args()
    result = analyse(load_rows(Path(args.root)), args.fee_pct)
    print(json.dumps({
        "status": result.get("status"),
        "rows": result.get("rows", 0),
        "tested_candidates": result.get("tested_candidates", 0),
        "deduplicated_candidates": result.get("deduplicated_candidates", 0),
        "fdr_discovery_candidates": result.get("fdr_discovery_candidates", 0),
        "discovery_leads": len(result.get("discovery_leads", [])),
        "validated_formulations": len(result.get("validated", [])),
        "rejected": len(result.get("rejected", [])),
    }, indent=2))
    render_live_context(result, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
