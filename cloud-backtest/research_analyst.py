#!/usr/bin/env python3
"""Research-to-formulation engine for AdvisorX."""
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
MIN_VALIDATION_N = 30
MIN_HOLDOUT_N = 15
MIN_VALIDATION_PF = 1.10
MIN_HOLDOUT_PF = 1.10
MIN_VALIDATION_AVG = 0.02
MIN_HOLDOUT_AVG = 0.02
MIN_POSITIVE_BLOCK_FRACTION = 0.55
MIN_POSITIVE_COIN_FRACTION = 0.55
MAX_VALIDATED = 20
BLOCK_MINUTES = 60
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
    for path in sorted((root / "observations").rglob("*.jsonl")):
        for obs in _read_jsonl(path):
            event_id = str(obs.get("event_id") or "")
            when = _dt(obs.get("observed_at"))
            if not event_id or when is None:
                continue
            observations[event_id] = obs

    rows: list[Row] = []
    for path in sorted((root / "outcomes").rglob("*.jsonl")):
        for out in _read_jsonl(path):
            obs = observations.get(str(out.get("event_id") or ""))
            if not obs:
                continue
            when = _dt(obs.get("observed_at"))
            if when is None:
                continue
            horizon = int(_float(out.get("horizon_min"), 0))
            if horizon not in HORIZONS:
                continue
            rows.append(Row(
                observed_at=when,
                symbol=str(obs.get("symbol") or out.get("symbol") or ""),
                direction=str(out.get("direction") or obs.get("direction") or "").lower(),
                horizon_min=horizon,
                event_families=tuple(str(x) for x in (obs.get("event_families") or [])),
                activity_ratio=_float(obs.get("activity_ratio_vs_60m_median")),
                activity_percentile=_float(obs.get("activity_percentile_60m")),
                delta_abs=abs(_float(obs.get("delta_proxy_ratio"))),
                event_score=_float(obs.get("event_score")),
                wick_ratio=_float(obs.get("lower_wick_ratio")) if str(obs.get("direction")) == "bullish" else _float(obs.get("upper_wick_ratio")),
                signed_forward_return_pct=_float(out.get("signed_forward_return_pct")),
            ))
    rows.sort(key=lambda r: r.observed_at)
    return rows


def _normal_one_sided_p(block_avgs: list[float]) -> float:
    if len(block_avgs) < 2:
        return 1.0
    mean = statistics.mean(block_avgs)
    sd = statistics.stdev(block_avgs)
    if sd <= 0:
        return 0.0 if mean > 0 else 1.0
    z = mean / (sd / math.sqrt(len(block_avgs)))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _bootstrap_block_ci_low(block_avgs: list[float], reps: int = BOOTSTRAP_REPS,
                            block_len: int = BOOTSTRAP_BLOCK_LEN) -> float:
    """Two-sided 95% lower endpoint from a deterministic moving-block bootstrap."""
    n = len(block_avgs)
    if n == 0:
        return -1e9
    if n == 1:
        return block_avgs[0]
    block_len = max(1, min(block_len, n))
    rng = random.Random(20260912 + n * 1009 + round(sum(block_avgs) * 1_000_000))
    means: list[float] = []
    for _ in range(max(200, reps)):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n)
            for offset in range(block_len):
                sample.append(block_avgs[(start + offset) % n])
                if len(sample) >= n:
                    break
        means.append(sum(sample) / n)
    means.sort()
    return means[max(0, min(len(means) - 1, int(0.025 * len(means))))]


def _stats(rows: list[Row], fee_pct: float) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "positive_rate": 0.0, "avg_net_return": 0.0, "median_net_return": 0.0,
                "profit_factor": 0.0, "worst_return": 0.0, "max_drawdown": 0.0,
                "coin_count": 0, "time_block_count": 0, "positive_block_fraction": 0.0,
                "positive_coin_fraction": 0.0, "block_ci95_low": -1e9,
                "block_p_value": 1.0, "bootstrap_block_ci95_low": -1e9}
    vals = [r.signed_forward_return_pct - fee_pct for r in rows]
    gains = sum(v for v in vals if v > 0)
    losses = -sum(v for v in vals if v < 0)
    blocks: dict[str, list[float]] = defaultdict(list)
    coins: dict[str, list[float]] = defaultdict(list)
    for r, v in zip(rows, vals):
        key = r.observed_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")
        blocks[key].append(v)
        coins[r.symbol].append(v)
    block_avgs = [sum(v) / len(v) for v in blocks.values() if v]
    positive_blocks = sum(x > 0 for x in block_avgs)
    coin_avgs = [sum(v) / len(v) for v in coins.values() if v]
    positive_coins = sum(x > 0 for x in coin_avgs)
    normal_p = _normal_one_sided_p(block_avgs)
    bootstrap_ci = _bootstrap_block_ci_low(block_avgs)
    if len(block_avgs) > 1:
        mean = statistics.mean(block_avgs)
        se = statistics.stdev(block_avgs) / math.sqrt(len(block_avgs))
        normal_ci = mean - 1.96 * se
    else:
        normal_ci = block_avgs[0] if block_avgs else -1e9
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in vals:
        equity += v
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "n": len(vals), "positive_rate": sum(v > 0 for v in vals) / len(vals),
        "avg_net_return": sum(vals) / len(vals), "median_net_return": statistics.median(vals),
        "profit_factor": gains / losses if losses > 0 else (99.0 if gains > 0 else 0.0),
        "worst_return": min(vals), "max_drawdown": max_dd, "coin_count": len(coins),
        "time_block_count": len(blocks),
        "positive_block_fraction": positive_blocks / len(blocks) if blocks else 0.0,
        "positive_coin_fraction": positive_coins / len(coins) if coins else 0.0,
        "block_ci95_low": normal_ci, "block_p_value": normal_p,
        "bootstrap_block_ci95_low": bootstrap_ci,
    }


def _candidate_signature(h: dict[str, Any]) -> str:
    return json.dumps(h, sort_keys=True, separators=(",", ":"))


def generate_candidates(rows: list[Row]) -> list[dict[str, Any]]:
    families = sorted({family for r in rows for family in r.event_families})
    candidates: list[dict[str, Any]] = []
    modifiers = [
        ("none", None), ("high_activity", {"activity_ratio_min": 1.8}),
        ("extreme_activity", {"activity_percentile_min": 85}),
        ("very_extreme_activity", {"activity_percentile_min": 95}),
        ("strong_delta_proxy", {"delta_abs_min": 0.65}),
        ("very_strong_delta_proxy", {"delta_abs_min": 0.90}),
        ("strong_event", {"event_score_min": 4}), ("strong_wick", {"wick_ratio_min": 0.45}),
    ]
    for family in families:
        for direction in ("bullish", "bearish"):
            for horizon in HORIZONS:
                for label, cond in modifiers:
                    candidates.append({"name": f"{family} | {label}", "event_family": family,
                                       "direction": direction, "horizon_min": horizon,
                                       "conditions": cond or {}})
    for family in families:
        for direction in ("bullish", "bearish"):
            for horizon in (15, 30, 60):
                for cond, label in [
                    ({"activity_ratio_min": 1.8, "delta_abs_min": 0.65}, "high_activity_and_delta"),
                    ({"activity_percentile_min": 85, "wick_ratio_min": 0.45}, "extreme_activity_and_wick"),
                    ({"event_score_min": 4, "delta_abs_min": 0.65}, "strong_event_and_delta"),
                ]:
                    candidates.append({"name": f"{family} | {label}", "event_family": family,
                                       "direction": direction, "horizon_min": horizon, "conditions": cond})
    return candidates


def _match(row: Row, h: dict[str, Any]) -> bool:
    c = h.get("conditions") or {}
    return (h.get("event_family") in row.event_families and row.direction == h.get("direction")
            and row.horizon_min == int(h.get("horizon_min"))
            and row.activity_ratio >= _float(c.get("activity_ratio_min"), -1e99)
            and row.activity_percentile >= _float(c.get("activity_percentile_min"), -1e99)
            and row.delta_abs >= _float(c.get("delta_abs_min"), -1e99)
            and row.event_score >= _float(c.get("event_score_min"), -1e99)
            and row.wick_ratio >= _float(c.get("wick_ratio_min"), -1e99))


def _evaluate(h: dict[str, Any], rows: list[Row], fee_pct: float) -> dict[str, Any]:
    matched = [r for r in rows if _match(r, h)]
    return {"rows": matched, "metrics": _stats(matched, fee_pct)}


def _apply_bh(evaluated: list[tuple[dict[str, Any], dict[str, Any]]], q: float = FDR_Q) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ranked = sorted(enumerate(evaluated), key=lambda x: x[1][1]["block_p_value"])
    m = len(ranked)
    cutoff_rank = 0
    for rank, (_, (_, metrics)) in enumerate(ranked, start=1):
        if metrics["block_p_value"] <= (rank / m) * q:
            cutoff_rank = rank
    passing = {idx for rank, (idx, _) in enumerate(ranked, start=1) if rank <= cutoff_rank}
    return [(h, v) for idx, (h, v) in enumerate(evaluated) if idx in passing]


def _dedupe_nested_activity(evaluated: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Deduplicate only the mathematically nested 85th/95th percentile activity thresholds."""
    groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for h, v in evaluated:
        c = h.get("conditions") or {}
        if set(c) == {"activity_percentile_min"}:
            key = (h.get("event_family"), h.get("direction"), h.get("horizon_min"))
            groups[key].append((h, v))
        else:
            result.append((h, v))
    for values in groups.values():
        values.sort(key=lambda x: (x[1]["avg_net_return"], x[1]["profit_factor"], x[1]["positive_coin_fraction"]), reverse=True)
        result.append(values[0])
    return result


def analyse(rows: list[Row], fee_pct: float) -> dict[str, Any]:
    if not rows:
        return {"schema_version": SCHEMA_VERSION, "status": "insufficient_data", "rows": 0,
                "candidates": [], "validated": [], "rejected": [], "discovery_leads": []}
    split = max(1, int(len(rows) * 0.70))
    validation, holdout = rows[:split], rows[split:]
    evaluated_all = []
    for h in generate_candidates(rows):
        v = _evaluate(h, validation, fee_pct)["metrics"]
        if v["n"] >= 10:
            evaluated_all.append((h, v))
    deduped = _dedupe_nested_activity(evaluated_all)
    fdr_passed = _apply_bh(deduped)
    fdr_keys = {_candidate_signature(h) for h, _ in fdr_passed}
    evaluated = []
    for h, v in deduped:
        item = dict(v)
        item["fdr_q"] = None
        evaluated.append((h, item))
    ranked = sorted(evaluated, key=lambda x: x[1]["block_p_value"])
    m = len(ranked)
    running_q = 1.0
    q_by_sig: dict[str, float] = {}
    for rank, (h, v) in reversed(list(enumerate(ranked, start=1))):
        raw = v["block_p_value"] * m / rank
        running_q = min(running_q, raw)
        q_by_sig[_candidate_signature(h)] = min(1.0, running_q)
    for h, v in evaluated:
        v["fdr_q"] = q_by_sig[_candidate_signature(h)]
    evaluated.sort(key=lambda x: (x[1]["avg_net_return"], x[1]["profit_factor"], x[1]["positive_coin_fraction"]), reverse=True)
    leads = [{"hypothesis": h, "validation": v} for h, v in evaluated[:30]]
    validated, rejected = [], []
    for h, v in evaluated:
        fdr_ok = _candidate_signature(h) in fdr_keys and v["fdr_q"] <= FDR_Q
        base_ok = (v["n"] >= MIN_VALIDATION_N and v["profit_factor"] >= MIN_VALIDATION_PF
                   and v["avg_net_return"] >= MIN_VALIDATION_AVG
                   and v["positive_block_fraction"] >= MIN_POSITIVE_BLOCK_FRACTION
                   and v["positive_coin_fraction"] >= MIN_POSITIVE_COIN_FRACTION
                   and v["bootstrap_block_ci95_low"] > 0)
        if not (base_ok and fdr_ok):
            rejected.append({"hypothesis": h, "validation": v, "status": "validation_failed",
                             "fdr_passed": fdr_ok, "fdr_q": v["fdr_q"]})
            continue
        hm = _evaluate(h, holdout, fee_pct)["metrics"]
        stress = _evaluate(h, holdout, max(fee_pct, 0.20))["metrics"]
        passed = (hm["n"] >= MIN_HOLDOUT_N and hm["profit_factor"] >= MIN_HOLDOUT_PF
                  and hm["avg_net_return"] >= MIN_HOLDOUT_AVG
                  and hm["positive_block_fraction"] >= MIN_POSITIVE_BLOCK_FRACTION
                  and hm["positive_coin_fraction"] >= MIN_POSITIVE_COIN_FRACTION
                  and hm["bootstrap_block_ci95_low"] > 0 and stress["avg_net_return"] > 0)
        item = {"hypothesis": h, "validation": v, "holdout": hm, "stress": stress,
                "status": "HOLDOUT_PASSED" if passed else "holdout_failed", "fdr_q": v["fdr_q"]}
        (validated if passed else rejected).append(item)
        if len(validated) >= MAX_VALIDATED:
            break
    return {"schema_version": SCHEMA_VERSION, "status": "ok", "rows": len(rows),
            "validation_rows": len(validation), "holdout_rows": len(holdout),
            "tested_candidates": len(evaluated_all), "deduplicated_candidates": len(deduped),
            "fdr_discovery_candidates": len(fdr_passed), "fdr_q_threshold": FDR_Q,
            "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_block_len": BOOTSTRAP_BLOCK_LEN,
            "fee_pct": fee_pct, "discovery_leads": leads,
            "validated": validated, "rejected": rejected[:100]}


def render_live_context(result: dict[str, Any], path: Path) -> None:
    payload = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
               "research_only": True, "validated_formulations": [
        {"name": x["hypothesis"]["name"], "event_family": x["hypothesis"]["event_family"],
         "direction": x["hypothesis"]["direction"], "horizon_min": x["hypothesis"]["horizon_min"],
         "conditions": x["hypothesis"]["conditions"], "validation": x["validation"],
         "holdout": x["holdout"], "stress": x["stress"], "status": x["status"], "fdr_q": x["fdr_q"]}
        for x in result.get("validated", [])]}
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
    render_live_context(result, Path(args.output))
    print(json.dumps({"status": result.get("status"), "rows": result.get("rows", 0),
                      "tested_candidates": result.get("tested_candidates", 0),
                      "deduplicated_candidates": result.get("deduplicated_candidates", 0),
                      "fdr_discovery_candidates": result.get("fdr_discovery_candidates", 0),
                      "discovery_leads": len(result.get("discovery_leads", [])),
                      "validated_formulations": len(result.get("validated", [])),
                      "rejected": len(result.get("rejected", []))}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
