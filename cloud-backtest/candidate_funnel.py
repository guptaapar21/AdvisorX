"""AdvisorX opportunity-funnel telemetry.

This module is observational. It never changes an entry, exit, stop, target,
or position size. It is designed to be called from the real production scan
path so the research dataset contains the denominator that trade-only logs do
not provide.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

FUNNEL_FILE = os.environ.get("CANDIDATE_FUNNEL_FILE", "candidate_funnel.jsonl")


def _f(value: Any):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _structure(signal: Dict[str, Any]):
    ms = signal.get("market_structure") or {}
    return ms, ms.get("3m") or {}, ms.get("15m") or {}, ms.get("1h") or {}


def classify_event(signal: Dict[str, Any]) -> str:
    """Classify the currently observed event without selecting direction."""
    ms, s3, s15, _ = _structure(signal)
    latest = s3.get("latest_break") or {}
    direction = str(latest.get("direction") or "").lower()
    event_type = str(latest.get("type") or "").upper()
    phase = str(s3.get("phase") or "").lower()
    bias = str(s3.get("structure_bias") or "").lower()
    sweeps = s3.get("liquidity_sweeps") or []
    failed = s3.get("failed_breaks") or []

    # Event precedence is deliberately ordered so a failed/rejected break or
    # a pullback/retest is not flattened into a generic BREAKOUT/BREAKDOWN.
    # This is telemetry only; it never gates a trade.
    if failed:
        return "FAILED_BREAK"

    # A reclaim/sweep with a transition is a better description of a reversal
    # event than a generic break label.
    if sweeps and ("transition" in phase or "reversal" in phase):
        return "LIQUIDITY_REVERSAL"

    # Detect a trend pullback/retest observationally. This is deliberately not
    # a trade trigger; it records a potentially useful setup family.
    ema21 = _f(s3.get("ema21"))
    close = _f(s3.get("close"))
    if ema21 is not None and close is not None and bias in {"bullish", "bearish"}:
        near_ema = abs(close - ema21) / max(abs(ema21), 1e-12) <= 0.006
        if near_ema and phase in {"bull_continuation", "bear_continuation", "neutral"}:
            return "TREND_PULLBACK"

    if direction == "bullish" and event_type in {"BOS", "CHOCH"}:
        return "BREAKOUT"
    if direction == "bearish" and event_type in {"BOS", "CHOCH"}:
        return "BREAKDOWN"

    if ms.get("market_regime") == "RANGE":
        return "RANGE"
    if "exhaust" in phase or s3.get("exhaustion_flags"):
        return "EXHAUSTION"
    if bias == "bullish":
        return "TREND_LONG_CONTEXT"
    if bias == "bearish":
        return "TREND_SHORT_CONTEXT"
    if ms.get("market_regime") in {"BREAKOUT_TRANSITION", "BREAKDOWN_TRANSITION"}:
        return "TRANSITION"
    if s15.get("structure_bias") in {"bullish", "bearish"}:
        return "HTF_CONTEXT"
    return "NONE"


def setup_quality(signal: Dict[str, Any], event: str) -> int:
    """Descriptive setup score; never used as a production gate."""
    ms, s3, s15, s1h = _structure(signal)
    score = 0
    if event not in {"NONE", "HTF_CONTEXT"}:
        score += 15
    if s3.get("latest_break"):
        score += 20
    if s3.get("liquidity_sweeps"):
        score += 10
    if s3.get("failed_breaks"):
        score += 5
    rvol = _f(signal.get("rvol"))
    if rvol is not None and rvol >= 1.2:
        score += 10
    adx = _f(s3.get("adx14"))
    if adx is not None and adx >= 20:
        score += 10
    if str(s3.get("structure_bias") or "").lower() in {"bullish", "bearish"}:
        score += 10
    if str(s15.get("structure_bias") or "").lower() == str(s3.get("structure_bias") or "").lower() and s15.get("structure_bias"):
        score += 10
    if str(s1h.get("structure_bias") or "").lower() == str(s3.get("structure_bias") or "").lower() and s1h.get("structure_bias"):
        score += 5
    return min(100, score)


def entry_quality(signal: Dict[str, Any]) -> int:
    """Descriptive entry-location score; never used as a production gate."""
    ctx = signal.get("entry_quality_context") or {}
    continuation = ctx.get("continuation") or {}
    tel = signal.get("entry_location_telemetry") or {}
    bars = _f(continuation.get("bars_since_break", tel.get("bars_since_break")))
    ext = _f(continuation.get("extension_atr_from_break", tel.get("extension_atr_from_break")))

    score = 60
    if bars is not None:
        if bars <= 2:
            score += 15
        elif bars <= 5:
            score += 5
        elif bars >= 10:
            score -= 15
    if ext is not None:
        if ext <= 0.75:
            score += 10
        elif ext <= 1.5:
            score += 2
        elif ext > 2:
            score -= 20
    return max(0, min(100, score))


def classify_decision(decision: Dict[str, Any]) -> str:
    """Important: inspect Python rejection before take_trade.

    The real production Python risk/quality gate sets take_trade=False while
    attaching _entry_quality_reject_reason. Checking take_trade first would
    incorrectly merge PYTHON_REJECT into GEMINI_SKIP.
    """
    if decision.get("_entry_quality_reject_reason") or decision.get("risk_validation_error"):
        return "PYTHON_REJECT"
    if decision.get("take_trade") is True:
        return "TAKE"
    # WATCH is accepted for future/extended Gemini schemas but is not invented
    # by this module when the current Gemini contract doesn't emit it.
    if str(decision.get("decision") or "").upper() == "WATCH":
        return "WATCH"
    return "GEMINI_SKIP"


def opportunity_id(signal: Dict[str, Any], event: str | None = None) -> str:
    """Stable ID for one coin/candle/event opportunity.

    The ID is deterministic across re-runs, unlike a random UUID, so funnel,
    counterfactual, and later research exports can join records reliably.
    """
    event = event or classify_event(signal)
    coin = str(signal.get("coin") or "UNKNOWN").upper()
    candle = str(signal.get("candle_time") or "UNKNOWN")
    raw = f"{coin}|{candle}|{event}".encode("utf-8")
    return "opp_" + hashlib.sha1(raw).hexdigest()[:20]


def record(signal: Dict[str, Any], decision: Dict[str, Any] | None = None) -> Dict[str, Any]:
    decision = decision or {}
    event = classify_event(signal)
    opp_id = opportunity_id(signal, event)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candle_time": signal.get("candle_time"),
        "coin": signal.get("coin"),
        "opportunity_id": opp_id,
        "event_type": event,
        "market_regime": (signal.get("market_structure") or {}).get("market_regime"),
        "setup_quality": setup_quality(signal, event),
        "entry_quality": entry_quality(signal),
        "decision_bucket": classify_decision(decision),
        "gemini_direction": decision.get("direction"),
        "gemini_take": decision.get("take_trade"),
        "gemini_conviction": decision.get("conviction"),
        "python_rejection": decision.get("_entry_quality_reject_reason") or decision.get("risk_validation_error"),
        "risk_validation_error": decision.get("risk_validation_error"),
        "entry_price": _f(decision.get("entry_price")),
        "stop_loss": _f(decision.get("stop_loss")),
        "target_price": _f(decision.get("target_price")),
        "entry_location_telemetry": decision.get("_entry_location_telemetry") or {},
        "recent_signal_context": decision.get("_recent_signal_context") or {},
        "supporting_tags": decision.get("supporting_tags") or [],
        "risk_tags": decision.get("risk_tags") or [],
    }


def append(records: Iterable[Dict[str, Any]]) -> None:
    records = list(records)
    if not records:
        return
    directory = os.path.dirname(os.path.abspath(FUNNEL_FILE)) or "."
    os.makedirs(directory, exist_ok=True)
    with open(FUNNEL_FILE, "a", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
