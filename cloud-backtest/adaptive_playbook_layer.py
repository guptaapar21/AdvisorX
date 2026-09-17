"""Adaptive market-regime/playbook layer for AdvisorX.

This module wraps the existing V4 production launcher without touching the
research engine. It separates:

MARKET REGIME -> LOCAL REGIME -> VOLATILITY/LIQUIDITY -> PLAYBOOK -> EVENT
-> SETUP QUALITY -> WATCH/ARMED -> TRIGGERED -> EXECUTION GATE.

Gemini remains the final discretionary direction/setup judge. Python adds
playbook context, persistent WATCH/ARMED state, regime hysteresis, a durable
calendar-day risk governor, and playbook-aware execution validation.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import trend_alignment_scanner_live as live
import gemini_advisor


ROOT = Path(__file__).resolve().parent
WATCH_FILE = ROOT / os.environ.get("ADAPTIVE_WATCH_FILE", "adaptive_watch_state.json")
REGIME_FILE = ROOT / os.environ.get("ADAPTIVE_REGIME_FILE", "adaptive_regime_state.json")
DAILY_RISK_FILE = ROOT / os.environ.get("ADAPTIVE_DAILY_RISK_FILE", "daily_risk_state.json")
PLAYBOOK_TELEMETRY_FILE = ROOT / os.environ.get(
    "ADAPTIVE_PLAYBOOK_TELEMETRY_FILE", "adaptive_playbook_telemetry.jsonl"
)

WATCH_EXPIRY_MINUTES = float(os.environ.get("ADAPTIVE_WATCH_EXPIRY_MINUTES", "90"))
ARM_DISTANCE_ATR = float(os.environ.get("ADAPTIVE_ARM_DISTANCE_ATR", "0.35"))
REGIME_HYSTERESIS_CYCLES = int(os.environ.get("ADAPTIVE_REGIME_HYSTERESIS_CYCLES", "2"))
DAILY_MAX_LOSS_INR = float(os.environ.get("DAILY_MAX_LOSS_INR", "3000"))
ADAPTIVE_MAX_ENTRY_DISTANCE_ATR = float(os.environ.get("ADAPTIVE_MAX_ENTRY_DISTANCE_ATR", "1.75"))


_BASE_GEMINI_BATCH = live._original_get_trade_suggestions_batch
_BASE_ENTRY_GATE = live.apply_entry_quality_gate
_PATCHED = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    return default


def _write_json(path: Path, payload: Any) -> None:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        print(f"  Adaptive state write WARNING ({path.name}): {exc}")


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    except OSError as exc:
        print(f"  Adaptive telemetry WARNING ({path.name}): {exc}")


def _parse_time(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _normalize_regime(value: Any) -> str:
    r = str(value or "").upper()
    if r in {"TREND_UP", "TREND_DOWN", "RANGE"}:
        return r
    if r in {"BREAKOUT_TRANSITION", "BREAKDOWN_TRANSITION", "EXHAUSTION_OR_TRANSITION", "TRANSITION"}:
        return "TRANSITION"
    return "UNCLEAR"


def _global_regime_candidate(snapshots: Iterable[Dict[str, Any]]) -> Tuple[str, float, Dict[str, Any]]:
    snaps = list(snapshots)
    btc = next((s for s in snaps if s.get("coin") == "BTC"), None)
    btc_regime = _normalize_regime((btc or {}).get("market_structure", {}).get("market_regime"))
    btc_strength = _safe_float((btc or {}).get("momentum_pct_20_3m"))
    valid = [s for s in snaps if s.get("market_structure")]
    regs = [_normalize_regime(s.get("market_structure", {}).get("market_regime")) for s in valid]
    counts = {r: regs.count(r) for r in ("TREND_UP", "TREND_DOWN", "RANGE", "TRANSITION")}
    positive = sum(1 for s in snaps if (_safe_float(s.get("momentum_pct_20_3m")) or 0) > 0)
    negative = sum(1 for s in snaps if (_safe_float(s.get("momentum_pct_20_3m")) or 0) < 0)
    measured = positive + negative
    positive_pct = positive / measured if measured else 0.5
    negative_pct = negative / measured if measured else 0.5

    if btc_regime == "TREND_UP" and positive_pct >= 0.55:
        regime, confidence = "TREND_UP", min(0.95, 0.68 + positive_pct * 0.27)
    elif btc_regime == "TREND_DOWN" and negative_pct >= 0.55:
        regime, confidence = "TREND_DOWN", min(0.95, 0.68 + negative_pct * 0.27)
    elif counts["TRANSITION"] >= max(2, len(valid) // 3):
        regime, confidence = "TRANSITION", min(0.92, 0.58 + counts["TRANSITION"] / max(1, len(valid)) * 0.30)
    elif counts["RANGE"] >= max(3, len(valid) // 2) and abs(positive_pct - 0.5) < 0.18:
        regime, confidence = "RANGE", min(0.90, 0.60 + counts["RANGE"] / max(1, len(valid)) * 0.25)
    elif positive_pct >= 0.62:
        regime, confidence = "TREND_UP", 0.72
    elif negative_pct >= 0.62:
        regime, confidence = "TREND_DOWN", 0.72
    elif counts["RANGE"] >= counts["TREND_UP"] and counts["RANGE"] >= counts["TREND_DOWN"]:
        regime, confidence = "RANGE", 0.62
    else:
        regime, confidence = "TRANSITION", 0.55

    details = {
        "btc_regime": btc_regime,
        "btc_momentum_20": btc_strength,
        "counts": counts,
        "positive_pct": round(positive_pct * 100, 1) if measured else None,
        "negative_pct": round(negative_pct * 100, 1) if measured else None,
        "measured": measured,
    }
    return regime, round(confidence, 3), details


def _apply_regime_hysteresis(candidate: str, confidence: float) -> Dict[str, Any]:
    state = _read_json(REGIME_FILE, {})
    current = str(state.get("regime") or "UNCLEAR")
    pending = str(state.get("pending_candidate") or "")
    count = int(state.get("pending_count") or 0)

    if current == candidate:
        pending, count = "", 0
    else:
        if pending == candidate:
            count += 1
        else:
            pending, count = candidate, 1
        if current == "UNCLEAR" or count >= REGIME_HYSTERESIS_CYCLES:
            current = candidate
            pending, count = "", 0

    out = {
        "regime": current,
        "candidate": candidate,
        "confidence": confidence,
        "pending_candidate": pending,
        "pending_count": count,
        "updated_at": _now().isoformat(),
    }
    _write_json(REGIME_FILE, out)
    return out


def _vol_liq_state(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    rvol_p = _safe_float(snapshot.get("rvol_percentile"))
    rvol = _safe_float(snapshot.get("rvol"))
    if rvol_p is not None:
        if rvol_p >= 95:
            volatility = "EXTREME"
        elif rvol_p >= 80:
            volatility = "HIGH"
        elif rvol_p <= 20:
            volatility = "LOW"
        else:
            volatility = "NORMAL"
    elif rvol is not None:
        if rvol >= 2.5:
            volatility = "EXTREME"
        elif rvol >= 1.5:
            volatility = "HIGH"
        elif rvol < 0.65:
            volatility = "LOW"
        else:
            volatility = "NORMAL"
    else:
        volatility = "UNKNOWN"

    if rvol is None:
        liquidity = "UNKNOWN"
    elif rvol < 0.55:
        liquidity = "THIN"
    elif rvol >= 1.5:
        liquidity = "ACTIVE"
    else:
        liquidity = "NORMAL"
    return {"volatility": volatility, "liquidity": liquidity}


def _trigger_for_playbook(snapshot: Dict[str, Any], playbook: str, direction: str | None) -> Tuple[str, float | None, bool]:
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    rng = ms.get("range") or {}
    latest = s3.get("latest_break") or {}
    direction = str(direction or "").lower()
    wanted = "bullish" if direction == "long" else "bearish"

    sweeps = s3.get("liquidity_sweeps") or []
    if playbook == "LIQUIDITY_REVERSAL":
        if direction == "long":
            for x in reversed(sweeps):
                if "low" in str(x.get("type", "")):
                    return "LOW_SWEEP_RECLAIM", _safe_float(x.get("level")), True
        if direction == "short":
            for x in reversed(sweeps):
                if "high" in str(x.get("type", "")):
                    return "HIGH_SWEEP_REJECTION", _safe_float(x.get("level")), True
    if "RANGE_LONG" == playbook or (playbook == "RANGE_REVERSAL" and direction == "long"):
        return "RANGE_LOW_RECLAIM", _safe_float(rng.get("range_low")), bool(rng.get("near_low"))
    if "RANGE_SHORT" == playbook or (playbook == "RANGE_REVERSAL" and direction == "short"):
        return "RANGE_HIGH_REJECTION", _safe_float(rng.get("range_high")), bool(rng.get("near_high"))
    if playbook in {"BREAKOUT_RETEST", "BREAKDOWN_RETEST", "LONG_BREAKOUT", "SHORT_BREAKDOWN", "RANGE_BREAKOUT_WATCH"}:
        return "STRUCTURAL_BREAK", _safe_float(latest.get("level")), str(latest.get("direction")) == wanted
    if playbook in {"LONG_PULLBACK", "SHORT_RALLY_REJECTION"}:
        level = _safe_float(s3.get("ema21"))
        return "EMA21_PULLBACK", level, False
    if playbook in {"LONG_CONTINUATION", "SHORT_CONTINUATION"}:
        return "CONTINUATION_RETEST", _safe_float(latest.get("level")), str(latest.get("direction")) == wanted
    return "WAIT_FOR_EVENT", _safe_float(latest.get("level")), False


def _setup_score(snapshot: Dict[str, Any], playbook: str) -> Tuple[int, list[str]]:
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    s15 = ms.get("15m") or {}
    rng = ms.get("range") or {}
    score, reasons = 0, []
    local = _normalize_regime(ms.get("market_regime"))
    wanted_long = "LONG" in playbook
    wanted_short = "SHORT" in playbook
    wanted = "bullish" if wanted_long else ("bearish" if wanted_short else None)

    if wanted and s3.get("structure_bias") == wanted:
        score += 2; reasons.append("3m_structure_aligned")
    if wanted and s15.get("structure_bias") in {wanted, "neutral", "unclear", None}:
        score += 1
    if local == "RANGE" and bool(rng.get("candidate")):
        score += 2; reasons.append("range_structure")
    latest = s3.get("latest_break") or {}
    if latest:
        score += 1; reasons.append("recent_structural_event")
    if (_safe_float(snapshot.get("rvol")) or 0) >= 1.0:
        score += 1; reasons.append("active_volume")
    cont = (snapshot.get("entry_quality_context") or {}).get("continuation") or {}
    freshness = str(cont.get("break_freshness") or "")
    exhaustion = _safe_float(s3.get("exhaustion_score_3m")) or 0
    if freshness in {"fresh", "mature"}:
        score += 1; reasons.append("fresh_or_mature_break")
    if exhaustion >= 3:
        score -= 2; reasons.append("heavy_exhaustion")
    if cont.get("continuation_quality") == "late_exhausted":
        score -= 2; reasons.append("late_exhausted")
    return max(0, min(5, score)), reasons


def _preferred_playbook(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    rng = ms.get("range") or {}
    regime = _normalize_regime(ms.get("market_regime"))
    latest = s3.get("latest_break") or {}
    sweeps = s3.get("liquidity_sweeps") or []

    if regime == "TREND_UP":
        if str(latest.get("direction")) == "bullish" and str(latest.get("type")) in {"BOS", "CHoCH"}:
            playbook = "LONG_BREAKOUT"
        elif bool(rng.get("candidate")) and bool(rng.get("near_low")):
            playbook = "LONG_PULLBACK"
        else:
            playbook = "LONG_CONTINUATION"
    elif regime == "TREND_DOWN":
        if str(latest.get("direction")) == "bearish" and str(latest.get("type")) in {"BOS", "CHoCH"}:
            playbook = "SHORT_BREAKDOWN"
        elif bool(rng.get("candidate")) and bool(rng.get("near_high")):
            playbook = "SHORT_RALLY_REJECTION"
        else:
            playbook = "SHORT_CONTINUATION"
    elif regime == "RANGE":
        high_sweep = any("high" in str(x.get("type", "")) for x in sweeps)
        low_sweep = any("low" in str(x.get("type", "")) for x in sweeps)
        if bool(rng.get("near_low")) or low_sweep:
            playbook = "RANGE_LONG" if not high_sweep else "RANGE_REVERSAL"
        elif bool(rng.get("near_high")) or high_sweep:
            playbook = "RANGE_SHORT" if not low_sweep else "RANGE_REVERSAL"
        else:
            playbook = "RANGE_BREAKOUT_WATCH"
    elif regime == "TRANSITION":
        if str(latest.get("direction")) == "bullish":
            playbook = "BREAKOUT_RETEST"
        elif str(latest.get("direction")) == "bearish":
            playbook = "BREAKDOWN_RETEST"
        elif sweeps:
            playbook = "LIQUIDITY_REVERSAL"
        else:
            playbook = "BREAKOUT_RETEST"
    else:
        playbook = "LIQUIDITY_REVERSAL" if sweeps else "RANGE_BREAKOUT_WATCH"

    allowed = {
        "TREND_UP": ["LONG_PULLBACK", "LONG_CONTINUATION", "LONG_BREAKOUT"],
        "TREND_DOWN": ["SHORT_RALLY_REJECTION", "SHORT_CONTINUATION", "SHORT_BREAKDOWN"],
        "RANGE": ["RANGE_LONG", "RANGE_SHORT", "RANGE_REVERSAL", "RANGE_BREAKOUT_WATCH"],
        "TRANSITION": ["BREAKOUT_RETEST", "BREAKDOWN_RETEST", "LIQUIDITY_REVERSAL"],
        "UNCLEAR": ["LIQUIDITY_REVERSAL", "RANGE_BREAKOUT_WATCH"],
    }[regime]
    return {"preferred": playbook, "allowed": allowed}


def _watch_state() -> Dict[str, Dict[str, Any]]:
    data = _read_json(WATCH_FILE, {})
    return data if isinstance(data, dict) else {}


def _prune_watch(now: datetime | None = None) -> Dict[str, Dict[str, Any]]:
    now = now or _now()
    state = _watch_state()
    keep: Dict[str, Dict[str, Any]] = {}
    for coin, item in state.items():
        created = _parse_time(item.get("created_at"))
        if created is None:
            continue
        age = (now - created).total_seconds() / 60.0
        if age <= WATCH_EXPIRY_MINUTES:
            keep[coin] = item
    if keep != state:
        _write_json(WATCH_FILE, keep)
    return keep


def _active_watch_context(coin: str, snapshot: Dict[str, Any]) -> Dict[str, Any] | None:
    state = _prune_watch()
    item = state.get(coin)
    if not item:
        return None
    close = _safe_float(snapshot.get("close")); atr = _safe_float((snapshot.get("market_structure") or {}).get("atr14_3m"))
    level = _safe_float(item.get("trigger_level"))
    armed = False
    if close is not None and atr and level is not None and atr > 0:
        armed = abs(close - level) / atr <= ARM_DISTANCE_ATR
    status = "ARMED" if armed else "WATCH"
    item = dict(item)
    item["status"] = status
    item["minutes_open"] = round((_now() - (_parse_time(item.get("created_at")) or _now())).total_seconds() / 60, 1)
    return item


def _regime_context(snapshots: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    candidate, confidence, details = _global_regime_candidate(snapshots)
    active = _apply_regime_hysteresis(candidate, confidence)
    active["details"] = details
    return active


def _enrich_snapshots(signals: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    snaps = list(signals)
    global_state = _regime_context(snaps)
    result = {}
    for snap in snaps:
        ms = snap.get("market_structure") or {}
        local_regime = _normalize_regime(ms.get("market_regime"))
        pb = _preferred_playbook(snap)
        vol_liq = _vol_liq_state(snap)
        setup_score, setup_reasons = _setup_score(snap, pb["preferred"])
        direction = "long" if "LONG" in pb["preferred"] else ("short" if "SHORT" in pb["preferred"] else None)
        trigger_name, trigger_level, trigger_confirmed = _trigger_for_playbook(snap, pb["preferred"], direction)
        watch = _active_watch_context(str(snap.get("coin")), snap)
        lifecycle = "ARMED" if watch and watch.get("status") == "ARMED" else ("WATCH" if watch else "CANDIDATE")
        snap.setdefault("entry_quality_context", {})
        snap["entry_quality_context"]["adaptive"] = {
            "global_regime": global_state["regime"],
            "global_regime_candidate": global_state["candidate"],
            "global_regime_confidence": global_state["confidence"],
            "global_regime_details": global_state.get("details", {}),
            "local_regime": local_regime,
            "local_regime_confidence": 0.80 if local_regime != "UNCLEAR" else 0.35,
            "volatility_state": vol_liq["volatility"],
            "liquidity_state": vol_liq["liquidity"],
            "preferred_playbook": pb["preferred"],
            "allowed_playbooks": pb["allowed"],
            "setup_quality": {
                "score": setup_score,
                "max": 5,
                "reasons": setup_reasons,
                "watch_eligible": setup_score >= 3,
            },
            "event": {
                "type": trigger_name,
                "level": trigger_level,
                "confirmed": trigger_confirmed,
            },
            "lifecycle": lifecycle,
            "active_watch": watch,
        }
        # Make the compact context visible to all downstream telemetry.
        snap["adaptive_playbook"] = pb["preferred"]
        snap["adaptive_allowed_playbooks"] = pb["allowed"]
        snap["adaptive_global_regime"] = global_state["regime"]
        snap["adaptive_local_regime"] = local_regime
        result[str(snap.get("coin"))] = snap
    return result


def _append_watch_or_clear(flagged: Dict[str, Dict[str, Any]], enriched: Dict[str, Dict[str, Any]]) -> None:
    state = _prune_watch()
    now = _now()
    for coin, signal in flagged.items():
        snap = enriched.get(coin)
        if not snap:
            continue
        adaptive = ((snap.get("entry_quality_context") or {}).get("adaptive") or {})
        setup = adaptive.get("setup_quality") or {}
        playbook = str(signal.get("adaptive_playbook") or adaptive.get("preferred_playbook") or "")
        take = bool(signal.get("take_trade"))
        direction = str(signal.get("direction") or "").lower()
        if take:
            signal["decision"] = "TAKE"
            signal["lifecycle"] = "TRIGGERED"
            signal["adaptive_playbook"] = playbook
            state.pop(coin, None)
            continue
        if setup.get("watch_eligible"):
            trigger = adaptive.get("event") or {}
            item = {
                "coin": coin,
                "direction": direction,
                "playbook": playbook,
                "created_at": state.get(coin, {}).get("created_at", now.isoformat()),
                "last_seen_at": now.isoformat(),
                "trigger_type": trigger.get("type"),
                "trigger_level": trigger.get("level"),
                "setup_score": setup.get("score"),
                "reasoning": signal.get("reasoning"),
            }
            existing = state.get(coin)
            if existing and str(existing.get("direction")) != direction:
                item["created_at"] = now.isoformat()
            state[coin] = item
            armed = _active_watch_context(coin, snap)
            signal["decision"] = "WATCH"
            signal["lifecycle"] = armed.get("status") if armed else "WATCH"
            signal["adaptive_playbook"] = playbook
        else:
            signal["decision"] = "SKIP"
            signal["lifecycle"] = "SKIP"
            signal["adaptive_playbook"] = playbook
    _write_json(WATCH_FILE, state)


def _adaptive_system_prompt() -> str:
    return """
ADAPTIVE PLAYBOOK ENGINE (V5):
Treat market regime as a routing state, not as a one-direction trade command.
For every coin, distinguish GLOBAL REGIME, LOCAL REGIME, VOLATILITY/LIQUIDITY,
PLAYBOOK, SETUP QUALITY, EVENT/TRIGGER, and ENTRY QUALITY.

PLAYBOOK ROUTING:
- TREND_UP: LONG_PULLBACK, LONG_CONTINUATION, LONG_BREAKOUT.
- TREND_DOWN: SHORT_RALLY_REJECTION, SHORT_CONTINUATION, SHORT_BREAKDOWN.
- RANGE: RANGE_LONG, RANGE_SHORT, RANGE_REVERSAL, RANGE_BREAKOUT_WATCH.
- TRANSITION: BREAKOUT_RETEST, BREAKDOWN_RETEST, LIQUIDITY_REVERSAL.
A local playbook can differ from the global regime. Counter-global direction is
allowed when the local structure and playbook explicitly support it; do not
force every coin to BTC's direction.

SETUP VS ENTRY:
A GOOD SETUP is not the same as a GOOD ENTRY. When setup quality is >=3/5 but
the trigger or entry location is not confirmed, return the candidate in
new_signals with take_trade=false. This is an explicit WATCH/ARMED state, not
an ordinary low-quality SKIP. Reuse the same playbook and explain the missing
trigger. Only set take_trade=true when the playbook event/trigger is actually
confirmed and the proposed entry is defensible.
When the setup itself is poor, omit it or return take_trade=false without
watch eligibility.

RANGE PLAYBOOK:
Use boundary interaction, sweep/rejection/reclaim, room to the opposite
boundary, and confirmation. Range-middle trades are not valid RANGE_LONG or
RANGE_SHORT entries. RANGE_BREAKOUT_WATCH is a watch state until a closed-candle
break plus acceptance/retest is visible.

TRANSITION PLAYBOOK:
Prefer break -> retest -> acceptance rather than chasing the first expansion.
LIQUIDITY_REVERSAL requires a real sweep/rejection/reclaim event and a clear
invalidation level.

LIFECYCLE:
CANDIDATE -> WATCH -> ARMED -> TRIGGERED. Python records the lifecycle.
Do not invent certainty: ARMED means price is approaching the deterministic
trigger level; TRIGGERED means the entry conditions are confirmed now.

TARGETS / INVALIDATION:
Choose targets from the playbook's natural objective (opposite range boundary,
next opposing swing, or measured continuation objective). Define invalidation
from the playbook, not a generic percentage. Keep minimum RR and fee geometry
valid.
"""


def adaptive_get_trade_suggestions_batch(signals, scorecard=None, open_positions=None):
    enriched_map = _enrich_snapshots(signals)
    enriched = list(enriched_map.values())
    ok, flagged, position_updates = _BASE_GEMINI_BATCH(enriched, scorecard, open_positions)
    if not ok:
        return ok, flagged, position_updates
    _append_watch_or_clear(flagged, enriched_map)
    for coin, signal in flagged.items():
        snap = enriched_map.get(coin, {})
        adaptive = ((snap.get("entry_quality_context") or {}).get("adaptive") or {})
        signal["adaptive_playbook"] = adaptive.get("preferred_playbook")
        signal["adaptive_allowed_playbooks"] = adaptive.get("allowed_playbooks", [])
        signal["adaptive_global_regime"] = adaptive.get("global_regime")
        signal["adaptive_local_regime"] = adaptive.get("local_regime")
        signal["adaptive_setup_quality"] = adaptive.get("setup_quality")
        signal["adaptive_event"] = adaptive.get("event")
        _append_jsonl(PLAYBOOK_TELEMETRY_FILE, {
            "time": _now().isoformat(),
            "coin": coin,
            "playbook": signal.get("adaptive_playbook"),
            "global_regime": signal.get("adaptive_global_regime"),
            "local_regime": signal.get("adaptive_local_regime"),
            "direction": signal.get("direction"),
            "take_trade": bool(signal.get("take_trade")),
            "decision": signal.get("decision"),
            "lifecycle": signal.get("lifecycle"),
            "setup_quality": signal.get("adaptive_setup_quality"),
            "event": signal.get("adaptive_event"),
            "conviction": signal.get("conviction"),
            "risk_validation_error": signal.get("risk_validation_error"),
        })
    return ok, flagged, position_updates


def _entry_metrics(signal: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    s15 = ms.get("15m") or {}
    rng = ms.get("range") or {}
    entry = _safe_float(signal.get("entry_price"))
    close = _safe_float(snapshot.get("close"))
    atr = _safe_float(ms.get("atr14_3m"))
    latest = s3.get("latest_break") or {}
    cont = (snapshot.get("entry_quality_context") or {}).get("continuation") or {}
    sweeps = s3.get("liquidity_sweeps") or []
    return {"ms": ms, "s3": s3, "s15": s15, "rng": rng, "entry": entry, "close": close, "atr": atr,
            "latest": latest, "cont": cont, "sweeps": sweeps}


def _playbook_gate(signal: Dict[str, Any], snapshot: Dict[str, Any]) -> Tuple[bool, str]:
    if not signal.get("take_trade"):
        return True, "non-TAKE lifecycle"
    d = str(signal.get("direction") or "").lower()
    if d not in {"long", "short"}:
        return False, "invalid direction"
    p = str(signal.get("adaptive_playbook") or "")
    m = _entry_metrics(signal, snapshot)
    ms, s3, s15, rng, entry, close, atr, latest, cont, sweeps = (
        m["ms"], m["s3"], m["s15"], m["rng"], m["entry"], m["close"], m["atr"],
        m["latest"], m["cont"], m["sweeps"]
    )
    local = _normalize_regime(ms.get("market_regime"))
    wanted = "bullish" if d == "long" else "bearish"

    if entry is None or close is None or atr is None or atr <= 0:
        return False, "adaptive gate missing entry/close/ATR"
    if abs(entry - close) / atr > ADAPTIVE_MAX_ENTRY_DISTANCE_ATR:
        return False, f"entry is {abs(entry-close)/atr:.2f} ATR from current price"

    if p in {"LONG_PULLBACK", "LONG_CONTINUATION", "LONG_BREAKOUT"}:
        if local != "TREND_UP" or d != "long":
            return False, "trend-up playbook requires local TREND_UP + LONG"
        if p == "LONG_PULLBACK":
            ema21 = _safe_float(s3.get("ema21"))
            if ema21 is None or abs(entry - ema21) / atr > 1.25:
                return False, "LONG_PULLBACK entry is not near EMA21"
        elif p == "LONG_BREAKOUT":
            if str(latest.get("direction")) != wanted:
                return False, "LONG_BREAKOUT lacks a confirmed bullish structural break"
        else:
            if str(latest.get("direction")) not in {"bullish", wanted} and not cont.get("has_break"):
                return False, "LONG_CONTINUATION lacks a usable bullish break/structure"
        if s3.get("structure_bias") not in {"bullish", "neutral"}:
            return False, "3m structure is not compatible with long trend playbook"
    elif p in {"SHORT_RALLY_REJECTION", "SHORT_CONTINUATION", "SHORT_BREAKDOWN"}:
        if local != "TREND_DOWN" or d != "short":
            return False, "trend-down playbook requires local TREND_DOWN + SHORT"
        if p == "SHORT_RALLY_REJECTION":
            ema21 = _safe_float(s3.get("ema21"))
            if ema21 is None or abs(entry - ema21) / atr > 1.25:
                return False, "SHORT_RALLY_REJECTION entry is not near EMA21"
        elif p == "SHORT_BREAKDOWN":
            if str(latest.get("direction")) != wanted:
                return False, "SHORT_BREAKDOWN lacks a confirmed bearish structural break"
        else:
            if str(latest.get("direction")) not in {"bearish", wanted} and not cont.get("has_break"):
                return False, "SHORT_CONTINUATION lacks a usable bearish break/structure"
        if s3.get("structure_bias") not in {"bearish", "neutral"}:
            return False, "3m structure is not compatible with short trend playbook"
    elif p in {"RANGE_LONG", "RANGE_SHORT", "RANGE_REVERSAL", "RANGE_BREAKOUT_WATCH"}:
        if local != "RANGE":
            return False, "range playbook requires local RANGE"
        position = _safe_float(rng.get("position_pct"))
        high = position >= 80 if position is not None else False
        low = position <= 20 if position is not None else False
        high_sweep = any("high" in str(x.get("type", "")) for x in sweeps)
        low_sweep = any("low" in str(x.get("type", "")) for x in sweeps)
        if p == "RANGE_LONG":
            if d != "long" or not (low or low_sweep):
                return False, "RANGE_LONG requires lower-boundary interaction or low sweep/reclaim"
        elif p == "RANGE_SHORT":
            if d != "short" or not (high or high_sweep):
                return False, "RANGE_SHORT requires upper-boundary interaction or high sweep/rejection"
        elif p == "RANGE_REVERSAL":
            if d == "long" and not low_sweep:
                return False, "RANGE_REVERSAL LONG requires a low sweep/reclaim"
            if d == "short" and not high_sweep:
                return False, "RANGE_REVERSAL SHORT requires a high sweep/rejection"
        else:
            if str(latest.get("direction")) not in {"bullish" if d == "long" else "bearish"}:
                return False, "RANGE_BREAKOUT_WATCH cannot become TAKE before a matching closed-candle break"
        rlo = _safe_float(rng.get("range_low")); rhi = _safe_float(rng.get("range_high")); target = _safe_float(signal.get("target_price"))
        if rlo is not None and rhi is not None and target is not None:
            if d == "long" and target <= entry:
                return False, "range LONG target has no upside room"
            if d == "short" and target >= entry:
                return False, "range SHORT target has no downside room"
            if d == "long" and target > rhi * 1.02:
                return False, "range LONG target is beyond the observed range objective"
            if d == "short" and target < rlo * 0.98:
                return False, "range SHORT target is beyond the observed range objective"
    elif p in {"BREAKOUT_RETEST", "BREAKDOWN_RETEST"}:
        required = "bullish" if p == "BREAKOUT_RETEST" else "bearish"
        if local != "TRANSITION" or d != ("long" if required == "bullish" else "short"):
            return False, "transition retest direction does not match playbook"
        if str(latest.get("direction")) != required:
            return False, "transition retest lacks the required fresh structural break"
        bars = _safe_float(cont.get("bars_since_break"))
        if bars is not None and bars > float(os.environ.get("BREAKOUT_MAX_BARS", "12")):
            return False, "transition retest break is stale"
        level = _safe_float(latest.get("level"))
        if level is not None and abs(entry - level) / atr > 1.5:
            return False, "transition retest entry is too far from break level"
    elif p == "LIQUIDITY_REVERSAL":
        if d == "long" and not any("low" in str(x.get("type", "")) for x in sweeps):
            return False, "long liquidity reversal requires low sweep/reclaim"
        if d == "short" and not any("high" in str(x.get("type", "")) for x in sweeps):
            return False, "short liquidity reversal requires high sweep/rejection"
    else:
        return False, f"unknown adaptive playbook '{p}'"

    # Playbook-specific invalidation sanity: stops must sit in a directionally
    # logical region, while Gemini's existing geometry gate remains authoritative
    # for fee-adjusted RR / minimum stop distance.
    stop = _safe_float(signal.get("stop_loss"))
    target = _safe_float(signal.get("target_price"))
    if stop is None or target is None:
        return False, "missing stop or target"
    if d == "long" and not (stop < entry < target):
        return False, "long geometry violates entry < target and stop < entry"
    if d == "short" and not (target < entry < stop):
        return False, "short geometry violates target < entry < stop"

    return True, f"adaptive playbook gate OK: {p}"


def _read_today_pnl() -> Tuple[float, int]:
    today = _now().date().isoformat()
    sources = []
    state = _read_json(ROOT / "trend_scanner_state.json", {})
    if isinstance(state, dict):
        sources.extend(state.get("ledger") or [])
    resolved = ROOT / os.environ.get("RESOLVED_TRADES_FILE", "resolved_trades.jsonl")
    if resolved.exists():
        try:
            for line in resolved.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    sources.append(json.loads(line))
        except (OSError, ValueError, TypeError):
            pass
    seen = set(); pnl = 0.0; count = 0
    for e in sources:
        status = e.get("status")
        if status in {"pending", "invalid", "abandoned"}:
            continue
        rt = _parse_time(e.get("resolved_time") or e.get("time"))
        if rt is None or rt.date().isoformat() != today:
            continue
        key = (e.get("coin"), e.get("time"), e.get("resolved_time"), status, e.get("resolved_pnl"))
        if key in seen:
            continue
        seen.add(key)
        pnl += _safe_float(e.get("resolved_pnl")) or 0.0
        count += 1
    return round(pnl, 2), count


def _refresh_daily_risk() -> Dict[str, Any]:
    pnl, count = _read_today_pnl()
    halted = pnl <= -abs(DAILY_MAX_LOSS_INR)
    state = {
        "utc_date": _now().date().isoformat(),
        "realized_pnl_inr": pnl,
        "resolved_trades": count,
        "max_daily_loss_inr": DAILY_MAX_LOSS_INR,
        "trading_halted_for_day": halted,
        "updated_at": _now().isoformat(),
    }
    _write_json(DAILY_RISK_FILE, state)
    return state


def adaptive_entry_quality_gate(flagged: Dict[str, Dict[str, Any]], snapshots: Iterable[Dict[str, Any]]) -> int:
    by_coin = {s.get("coin"): s for s in snapshots}
    rejected = 0
    daily = _refresh_daily_risk()
    for coin, signal in flagged.items():
        if not signal.get("take_trade"):
            continue
        snap = by_coin.get(coin)
        if not snap:
            signal["take_trade"] = False
            signal["risk_validation_error"] = "adaptive gate: missing snapshot"
            signal["lifecycle"] = "TRIGGERED_REJECTED"
            rejected += 1
            continue
        ok, reason = _playbook_gate(signal, snap)
        if ok:
            # Keep the original V4 geometry/fee guard for all non-range modes;
            # range/reversal/transition are deliberately validated by the
            # playbook-aware rules above because the old universal gate was
            # over-restrictive for those modes.
            playbook = str(signal.get("adaptive_playbook") or "")
            if playbook not in {"RANGE_LONG", "RANGE_SHORT", "RANGE_REVERSAL", "BREAKOUT_RETEST", "BREAKDOWN_RETEST", "LIQUIDITY_REVERSAL"}:
                ok, base_reason = _BASE_ENTRY_GATE({coin: signal.copy()}, [snap]) == 0, ""
                # _BASE_ENTRY_GATE mutates its supplied signal; re-run on the real
                # signal so its precise reason/telemetry is retained.
                if ok:
                    proxy = {coin: signal}
                    _BASE_ENTRY_GATE(proxy, [snap])
                    ok = bool(signal.get("take_trade"))
                    base_reason = signal.get("entry_quality_gate_reason") or signal.get("risk_validation_error") or "base V4 gate"
                    reason = base_reason
        if daily.get("trading_halted_for_day"):
            ok = False
            reason = f"calendar-day risk governor: realized P&L {daily['realized_pnl_inr']:.2f} <= -₹{DAILY_MAX_LOSS_INR:.0f}"
        signal["adaptive_gate_reason"] = reason
        signal["adaptive_gate_version"] = "2026-09-17-playbook-v1"
        if not ok:
            signal["take_trade"] = False
            signal["risk_validation_error"] = f"adaptive_gate: {reason}"
            signal["lifecycle"] = "TRIGGERED_REJECTED"
            signal["decision"] = "SKIP"
            rejected += 1
    return rejected


def _patch_message() -> None:
    original = live._scanner._build_message
    if getattr(original, "_adaptive_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        text = original(*args, **kwargs)
        watches = _prune_watch()
        if watches:
            counts: Dict[str, int] = {}
            for item in watches.values():
                status = _active_watch_context(str(item.get("coin")), {"close": None, "market_structure": {}})
                key = status.get("status") if status else "WATCH"
                counts[key] = counts.get(key, 0) + 1
            parts = [f"{k} {v}" for k, v in sorted(counts.items())]
            text += "\n\n🧭 Adaptive lifecycle: " + " | ".join(parts)
        daily = _read_json(DAILY_RISK_FILE, {})
        if daily:
            text += f"\n🛡 Daily risk: {_safe_float(daily.get('realized_pnl_inr')) or 0:.2f} INR realized"
            if daily.get("trading_halted_for_day"):
                text += " | HALTED"
        return text

    wrapped._adaptive_wrapped = True
    live._scanner._build_message = wrapped


def _patch_system_prompt() -> None:
    marker = "ADAPTIVE PLAYBOOK ENGINE (V5):"
    if marker not in gemini_advisor.SYSTEM_PROMPT:
        gemini_advisor.SYSTEM_PROMPT += "\n\n" + _adaptive_system_prompt()


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_system_prompt()
    live._original_get_trade_suggestions_batch = adaptive_get_trade_suggestions_batch
    live.apply_entry_quality_gate = adaptive_entry_quality_gate
    _patch_message()
    _PATCHED = True


install()
