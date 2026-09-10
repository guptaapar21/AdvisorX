"""Deterministic execution-quality gate for AdvisorX.

Gemini owns directional judgment, setup selection, and proposed levels.
Python owns the final execution permission. V4 explicitly promotes the
entry-location / continuation telemetry that was previously observational:
range position, break extension/freshness, failed breaks, liquidity sweeps,
and exhaustion are evaluated together so a strong trend is not mistaken for a
high-quality entry at the end of an already extended move.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, Tuple

MAX_ENTRY_DISTANCE_ATR = float(os.environ.get("MAX_ENTRY_DISTANCE_ATR", "1.5"))
BREAKOUT_MAX_BARS = int(os.environ.get("BREAKOUT_MAX_BARS", "12"))
BREAKOUT_MAX_EXTENSION_ATR = float(os.environ.get("BREAKOUT_MAX_EXTENSION_ATR", "1.5"))
RANGE_STRETCHED_PCT = float(os.environ.get("V4_RANGE_STRETCHED_PCT", "85"))
RANGE_EXTREME_PCT = float(os.environ.get("V4_RANGE_EXTREME_PCT", "92"))
RANGE_CRITICAL_PCT = float(os.environ.get("V4_RANGE_CRITICAL_PCT", "97"))
EXTENDED_ATR = float(os.environ.get("V4_EXTENDED_ATR", "1.25"))
CRITICAL_EXTENSION_ATR = float(os.environ.get("V4_CRITICAL_EXTENSION_ATR", "2.0"))
EXHAUSTION_WARN_SCORE = int(os.environ.get("V4_EXHAUSTION_WARN_SCORE", "2"))
EXHAUSTION_HARD_SCORE = int(os.environ.get("V4_EXHAUSTION_HARD_SCORE", "3"))
FAILED_BREAK_WARN_COUNT = int(os.environ.get("V4_FAILED_BREAK_WARN_COUNT", "2"))
FAILED_BREAK_HARD_COUNT = int(os.environ.get("V4_FAILED_BREAK_HARD_COUNT", "4"))
SWEEP_WARN_COUNT = int(os.environ.get("V4_SWEEP_WARN_COUNT", "3"))
MIN_ENTRY_QUALITY_SCORE = int(os.environ.get("V4_MIN_ENTRY_QUALITY_SCORE", "3"))


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _num(*values: Any) -> float | None:
    for value in values:
        if _finite(value):
            return float(value)
    return None


def _opposite(direction: str, bias: str) -> bool:
    d = str(direction or "").lower()
    b = str(bias or "").lower()
    return (d == "long" and b == "bearish") or (d == "short" and b == "bullish")


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _count(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if _finite(value):
        return max(0, int(float(value)))
    return 0


def _extract_metrics(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    cont = _first_dict(
        (snapshot.get("entry_quality_context") or {}).get("continuation"),
        snapshot.get("continuation"),
    )
    loc = _first_dict(snapshot.get("entry_location_telemetry"), snapshot.get("entry_quality_context"))
    rng = ms.get("range") or {}
    range_pct = _num(
        loc.get("entry_range_position_pct"), loc.get("range_position_pct"), loc.get("position_pct"),
        cont.get("entry_range_position_pct"), cont.get("range_position_pct"),
        rng.get("position_pct"), rng.get("entry_position_pct"),
    )
    extension = _num(
        cont.get("extension_atr_from_break"), loc.get("extension_atr_from_break"),
        snapshot.get("extension_atr_from_break"),
    )
    entry_vs_break = _num(
        loc.get("entry_vs_latest_break_atr"), loc.get("distance_from_latest_break_atr"),
        cont.get("entry_vs_latest_break_atr"),
    )
    bars = _num(cont.get("bars_since_break"), loc.get("bars_since_break"), s3.get("bars_since_break"))
    exhaustion = _num(
        s3.get("exhaustion_score_3m"), s3.get("exhaustion_score"),
        loc.get("exhaustion_score_3m"), cont.get("exhaustion_score_3m"),
    )
    failed = max(
        _count(s3.get("failed_breaks")), _count(s3.get("failed_break_count_3m")),
        _count(loc.get("failed_break_count_3m")), _count(cont.get("failed_break_count_3m")),
    )
    sweeps = max(
        _count(s3.get("liquidity_sweeps")), _count(s3.get("liquidity_sweep_count_3m")),
        _count(loc.get("liquidity_sweep_count_3m")), _count(cont.get("liquidity_sweep_count_3m")),
    )
    high_rejections = max(_count(loc.get("high_rejection_sweeps_3m")), _count(cont.get("high_rejection_sweeps_3m")))
    low_reclaims = max(_count(loc.get("low_reclaim_sweeps_3m")), _count(cont.get("low_reclaim_sweeps_3m")))
    cq = str(cont.get("continuation_quality") or loc.get("continuation_quality") or "").lower()
    freshness = str(cont.get("break_freshness") or loc.get("break_freshness") or s3.get("break_freshness") or "").lower()
    return {
        "range_pct": range_pct, "extension": extension, "entry_vs_break": entry_vs_break, "bars": bars,
        "exhaustion": exhaustion, "failed": failed, "sweeps": sweeps,
        "high_rejections": high_rejections, "low_reclaims": low_reclaims,
        "continuation_quality": cq, "freshness": freshness,
        "latest_break": s3.get("latest_break") or {},
    }


def evaluate_entry_quality(signal: Dict[str, Any], snapshot: Dict[str, Any]) -> Tuple[bool, str]:
    d = str(signal.get("direction") or "").lower()
    if d not in {"long", "short"}:
        return False, "invalid direction"
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}; s15 = ms.get("15m") or {}; s1h = ms.get("1h") or {}
    regime = str(ms.get("market_regime") or "UNCLEAR").upper()
    recent = snapshot.get("recent_signal_context") or {}
    m = _extract_metrics(snapshot)
    if recent.get("same_direction_reentry_without_new_break"):
        return False, "same-direction re-entry without a new structural break"
    for tf, st in (("3m", s3), ("15m", s15)):
        if _opposite(d, st.get("structure_bias")):
            return False, f"{tf} structure directly opposes proposed direction"
    if regime not in {"BREAKOUT_TRANSITION", "BREAKDOWN_TRANSITION", "EXHAUSTION_OR_TRANSITION"} and _opposite(d, s1h.get("structure_bias")):
        return False, "1h structure directly opposes proposed direction"
    wanted = "bullish" if d == "long" else "bearish"
    if regime in {"TREND_UP", "TREND_DOWN", "EXHAUSTION_OR_TRANSITION"}:
        if s3.get("structure_bias") != wanted:
            return False, "3m structure is not aligned with proposed direction"
        if regime != "EXHAUSTION_OR_TRANSITION":
            if str(s15.get("structure_bias") or "").lower() not in {wanted, "neutral", "unclear", ""}:
                return False, "15m structure is not aligned or neutral"
    if regime == "RANGE":
        rng = ms.get("range") or {}
        if d == "long" and not rng.get("near_low"):
            return False, "range LONG is not near the lower boundary"
        if d == "short" and not rng.get("near_high"):
            return False, "range SHORT is not near the upper boundary"
    latest_direction = str((m["latest_break"] or {}).get("direction") or "").lower()
    if regime in {"BREAKOUT_TRANSITION", "BREAKDOWN_TRANSITION"}:
        if latest_direction != wanted:
            return False, "latest 3m structural break does not support direction"
        if m["bars"] is not None and m["bars"] > BREAKOUT_MAX_BARS:
            return False, f"breakout is stale (> {BREAKOUT_MAX_BARS} closed 3m bars)"
    if m["extension"] is not None and abs(m["extension"]) > CRITICAL_EXTENSION_ATR:
        return False, f"entry is excessively extended {m['extension']:.2f} ATR from break"
    if m["entry_vs_break"] is not None and abs(m["entry_vs_break"]) > CRITICAL_EXTENSION_ATR:
        return False, f"entry is {m['entry_vs_break']:.2f} ATR from latest break"
    if m["continuation_quality"] == "late_exhausted":
        return False, "continuation is late and exhausted"

    score = 0; reasons = []
    if s3.get("structure_bias") == wanted:
        score += 2; reasons.append("3m aligned")
    if str(s15.get("structure_bias") or "").lower() in {wanted, "neutral", "unclear", ""}:
        score += 1
    if m["freshness"] in {"fresh", "fresh_break", "fresh_continuation"} or (m["bars"] is not None and m["bars"] <= 2):
        score += 2; reasons.append("fresh break")
    elif m["bars"] is not None and m["bars"] <= 5:
        score += 1
    elif m["bars"] is not None and m["bars"] > 10:
        score -= 2; reasons.append("late break")
    if m["extension"] is not None:
        e = abs(m["extension"])
        if e <= 0.75: score += 2
        elif e <= EXTENDED_ATR: score += 1
        elif e > BREAKOUT_MAX_EXTENSION_ATR: score -= 2; reasons.append("extended")
        else: score -= 1
    if m["range_pct"] is not None:
        extreme = m["range_pct"] if d == "long" else 100.0 - m["range_pct"]
        if extreme <= 70: score += 2
        elif extreme <= 85: score += 1
        elif extreme >= RANGE_CRITICAL_PCT: score -= 3; reasons.append("critical range extreme")
        elif extreme >= RANGE_EXTREME_PCT: score -= 2; reasons.append("range extreme")
        else: score -= 1; reasons.append("stretched range location")
    exhaustion = m["exhaustion"] or 0
    if exhaustion >= EXHAUSTION_HARD_SCORE: score -= 3; reasons.append("high exhaustion")
    elif exhaustion >= EXHAUSTION_WARN_SCORE: score -= 2; reasons.append("exhaustion")
    elif exhaustion > 0: score -= 1
    if m["failed"] >= FAILED_BREAK_HARD_COUNT: score -= 2; reasons.append("many failed breaks")
    elif m["failed"] >= FAILED_BREAK_WARN_COUNT: score -= 1; reasons.append("failed breaks")
    if m["sweeps"] >= SWEEP_WARN_COUNT: score -= 1; reasons.append("repeated liquidity sweeps")
    rejection_against = m["high_rejections"] if d == "long" else m["low_reclaims"]
    if rejection_against >= 1: score -= 1; reasons.append("directional rejection sweeps")
    if m["continuation_quality"] == "late_extended": score -= 2; reasons.append("late extended continuation")

    if m["range_pct"] is not None:
        extreme = m["range_pct"] if d == "long" else 100.0 - m["range_pct"]
        if extreme >= RANGE_CRITICAL_PCT and exhaustion >= EXHAUSTION_WARN_SCORE:
            return False, "critical range extreme with exhaustion evidence"
        if extreme >= RANGE_EXTREME_PCT and (
            exhaustion >= EXHAUSTION_WARN_SCORE or m["failed"] >= FAILED_BREAK_WARN_COUNT or
            m["sweeps"] >= SWEEP_WARN_COUNT or (m["extension"] is not None and abs(m["extension"]) >= EXTENDED_ATR)
        ):
            return False, "range extreme combined with exhaustion/failure/extension"
        if extreme >= RANGE_STRETCHED_PCT and m["failed"] >= FAILED_BREAK_HARD_COUNT and exhaustion >= 1:
            return False, "stretched location with repeated failed breaks and exhaustion"
    if m["extension"] is not None and abs(m["extension"]) > BREAKOUT_MAX_EXTENSION_ATR:
        if not (m["freshness"] in {"fresh", "fresh_break"} and exhaustion < EXHAUSTION_WARN_SCORE and m["failed"] < FAILED_BREAK_WARN_COUNT):
            return False, f"continuation is extended {m['extension']:.2f} ATR"
    if score < MIN_ENTRY_QUALITY_SCORE:
        detail = "; ".join(reasons[:4]) or "insufficient entry-quality evidence"
        return False, f"entry-quality score {score} < {MIN_ENTRY_QUALITY_SCORE}: {detail}"
    close = snapshot.get("close"); atr = ms.get("atr14_3m"); entry = signal.get("entry_price")
    if _finite(close) and _finite(atr) and _finite(entry) and float(atr) > 0:
        dist = abs(float(entry) - float(close)) / float(atr)
        if dist > MAX_ENTRY_DISTANCE_ATR:
            return False, f"proposed entry is {dist:.2f} ATR from current price"
    return True, f"ok (quality_score={score})"


def apply_entry_quality_gate(flagged: Dict[str, Dict[str, Any]], snapshots: Iterable[Dict[str, Any]]) -> int:
    by_coin = {s.get("coin"): s for s in snapshots}
    rejected = 0
    for coin, signal in flagged.items():
        if not signal.get("take_trade"):
            continue
        snap = by_coin.get(coin)
        if snap is None:
            signal["take_trade"] = False
            signal["risk_validation_error"] = "entry_quality_gate: missing deterministic market snapshot"
            signal["_entry_quality_reject_reason"] = "missing deterministic market snapshot"
            rejected += 1
            continue
        ok, reason = evaluate_entry_quality(signal, snap)
        signal["entry_quality_gate_reason"] = reason
        signal["entry_quality_gate_version"] = "2026-09-10-v4"
        if not ok:
            signal["take_trade"] = False
            signal["risk_validation_error"] = f"entry_quality_gate: {reason}"
            signal["_entry_quality_reject_reason"] = reason
            rejected += 1
    return rejected
