"""V2 observational research bridge for the production AdvisorX path.

This module NEVER changes live trading decisions. It records the complete
post-Python-gate decision snapshot after the production scanner has finished
adding decision telemetry, then records Gemini position exits separately.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import candidate_funnel
import portfolio_risk
import trade_counterfactuals


def classify_decision(decision: Dict[str, Any]) -> str:
    return candidate_funnel.classify_decision(decision)


def record_cycle(
    signals: Iterable[Dict[str, Any]],
    flagged: Dict[str, Dict[str, Any]],
    open_positions=None,
) -> Dict[str, Any]:
    """Record one completed fresh-coin Gemini cycle.

    `flagged` is expected to be the same dict returned by Gemini and later
    enriched by the production scanner before this function is called. This
    guarantees that entry-location/recent-signal/risk telemetry is present.
    """
    records = []
    for signal in signals:
        coin = str(signal.get("coin"))
        decision = flagged.get(coin, {}) or {}
        bucket = classify_decision(decision)
        row = candidate_funnel.record(signal, decision)
        # Carry the deterministic opportunity ID into every counterfactual row.
        decision = dict(decision)
        decision["_opportunity_id"] = row["opportunity_id"]
        records.append(row)
        if bucket == "GEMINI_SKIP":
            trade_counterfactuals.log_skip(signal, decision)
        elif bucket == "PYTHON_REJECT":
            trade_counterfactuals.log_python_reject(
                signal,
                decision,
                decision.get("_entry_quality_reject_reason")
                or decision.get("risk_validation_error"),
            )
        elif bucket == "TAKE":
            trade_counterfactuals.log_take(signal, decision)
        elif bucket == "WATCH":
            trade_counterfactuals.log_watch(signal, decision)
    candidate_funnel.append(records)
    buckets = {}
    for row in records:
        bucket = row.get("decision_bucket")
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return {
        "records": records,
        "buckets": buckets,
        "portfolio": portfolio_risk.summarize(open_positions),
    }


def record_position_exits(open_positions, position_updates):
    by_coin = {str(item.get("coin")): item for item in (open_positions or [])}
    for coin, update in (position_updates or {}).items():
        if str(update.get("action") or "").lower() != "exit_now":
            continue
        position = by_coin.get(str(coin))
        if position:
            trade_counterfactuals.log_exit(position, update)
