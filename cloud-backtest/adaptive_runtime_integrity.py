"""Final production-path integrity overlay for AdvisorX V5.

This module is intentionally small and sits after adaptive_hardening. It does
not alter the research engine. It closes integration-level gaps that are safer
to fix as a final wrapper than by changing the mature V4/V5 modules in place:

* derive local-regime confidence from actual evidence instead of constants;
* reconcile RANGE playbooks strictly with the side of the range actually
  supporting the proposed direction;
* resynchronise WATCH playbooks after Gemini direction reconciliation;
* seed WATCH for sweep-only setups where the first-pass preferred-playbook
  heuristic has no LONG/SHORT token;
* persist WATCH -> ARMED status when price approaches the trigger;
* make Telegram display WATCH/ARMED instead of presenting WATCH as SKIP;
* keep the final lifecycle state deterministic across repeated imports.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

import adaptive_hardening as hardening
import adaptive_playbook_layer as adaptive
import trend_alignment_scanner_live as live


_PATCHED = False
_BASE_ENRICH = hardening._BASE_ENRICH
_BASE_GET = hardening.hardening_get_trade_suggestions_batch
_BASE_PLAYBOOK = hardening._directional_playbook


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if x == x and abs(x) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _local_regime_confidence(snapshot: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """Estimate local-regime confidence from observable deterministic evidence."""
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    s15 = ms.get("15m") or {}
    s1h = ms.get("1h") or {}
    rng = ms.get("range") or {}
    regime = adaptive._normalize_regime(ms.get("market_regime"))

    if regime == "UNCLEAR":
        return 0.25, {"base": 0.25, "evidence": ["unclear_regime"]}

    evidence: list[str] = []
    score = 0.50
    b3 = str(s3.get("structure_bias") or "").lower()
    b15 = str(s15.get("structure_bias") or "").lower()
    b1h = str(s1h.get("structure_bias") or "").lower()

    if regime == "TREND_UP":
        if b3 == "bullish": score += 0.15; evidence.append("3m_bullish")
        if b15 == "bullish": score += 0.10; evidence.append("15m_bullish")
        if b1h == "bullish": score += 0.10; evidence.append("1h_bullish")
        adx = _safe_float(s3.get("adx14"))
        if adx is not None and adx >= 25: score += 0.05; evidence.append("adx>=25")
        er = _safe_float(s3.get("efficiency_ratio_20"))
        if er is not None and er >= 0.35: score += 0.05; evidence.append("efficiency>=0.35")
    elif regime == "TREND_DOWN":
        if b3 == "bearish": score += 0.15; evidence.append("3m_bearish")
        if b15 == "bearish": score += 0.10; evidence.append("15m_bearish")
        if b1h == "bearish": score += 0.10; evidence.append("1h_bearish")
        adx = _safe_float(s3.get("adx14"))
        if adx is not None and adx >= 25: score += 0.05; evidence.append("adx>=25")
        er = _safe_float(s3.get("efficiency_ratio_20"))
        if er is not None and er >= 0.35: score += 0.05; evidence.append("efficiency>=0.35")
    elif regime == "RANGE":
        if rng.get("candidate"): score += 0.18; evidence.append("range_candidate")
        pos = _safe_float(rng.get("position_pct"))
        if pos is not None and 0 <= pos <= 100: score += 0.08; evidence.append("bounded_range_position")
        if b3 in {"neutral", "unclear", ""}: score += 0.08; evidence.append("3m_non_directional")
        if b15 in {"neutral", "unclear", ""}: score += 0.06; evidence.append("15m_non_directional")
    elif regime == "TRANSITION":
        latest = s3.get("latest_break") or {}
        if latest.get("direction") in {"bullish", "bearish"}: score += 0.16; evidence.append("structural_transition_event")
        bars = _safe_float(((snapshot.get("entry_quality_context") or {}).get("continuation") or {}).get("bars_since_break"))
        if bars is not None and 0 <= bars <= 12: score += 0.08; evidence.append("break_age_usable")
        if s3.get("liquidity_sweeps"): score += 0.05; evidence.append("liquidity_event")

    return round(min(0.95, max(0.20, score)), 3), {"base": 0.50, "evidence": evidence}


def _directional_playbook_final(snapshot: Dict[str, Any], direction: str | None) -> str | None:
    """Final direction/playbook reconciliation.

    RANGE is intentionally strict: a LONG must have lower-boundary support or
    a low sweep, while a SHORT must have upper-boundary support or a high
    sweep. A directional break away from that boundary is a WATCH opportunity,
    not permission to fade the opposite end of the range.
    """
    d = str(direction or "").lower()
    if d not in {"long", "short"}:
        return None
    ms = snapshot.get("market_structure") or {}
    regime = adaptive._normalize_regime(ms.get("market_regime"))
    if regime != "RANGE":
        return _BASE_PLAYBOOK(snapshot, d)
    s3 = ms.get("3m") or {}
    rng = ms.get("range") or {}
    latest = s3.get("latest_break") or {}
    latest_dir = str(latest.get("direction") or "").lower()
    sweeps = s3.get("liquidity_sweeps") or []
    low_sweep = any("low" in str(x.get("type") or "").lower() for x in sweeps)
    high_sweep = any("high" in str(x.get("type") or "").lower() for x in sweeps)
    near_low = bool(rng.get("near_low"))
    near_high = bool(rng.get("near_high"))

    if d == "long":
        if low_sweep and high_sweep:
            return "RANGE_REVERSAL"
        if low_sweep:
            return "RANGE_REVERSAL"
        if near_low:
            return "RANGE_LONG"
        if latest_dir == "bullish":
            return "RANGE_BREAKOUT_WATCH"
        return None

    if high_sweep and low_sweep:
        return "RANGE_REVERSAL"
    if high_sweep:
        return "RANGE_REVERSAL"
    if near_high:
        return "RANGE_SHORT"
    if latest_dir == "bearish":
        return "RANGE_BREAKOUT_WATCH"
    return None


def _enrich_final(signals: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result = _BASE_ENRICH(signals)
    for snap in result.values():
        confidence, details = _local_regime_confidence(snap)
        adaptive_ctx = ((snap.get("entry_quality_context") or {}).get("adaptive") or {})
        adaptive_ctx["local_regime_confidence"] = confidence
        adaptive_ctx["local_regime_confidence_evidence"] = details
        snap.setdefault("entry_quality_context", {})["adaptive"] = adaptive_ctx
    return result


def _direction_from_sweep(snapshot: Dict[str, Any]) -> str | None:
    sweeps = ((snapshot.get("market_structure") or {}).get("3m") or {}).get("liquidity_sweeps") or []
    for item in reversed(sweeps):
        typ = str(item.get("type") or "").lower()
        if "low" in typ:
            return "long"
        if "high" in typ:
            return "short"
    return None


def _persist_watch_status(snapshots: Iterable[Dict[str, Any]]) -> None:
    state = adaptive._prune_watch()
    changed = False
    now = datetime.now(timezone.utc).isoformat()
    for snap in snapshots:
        coin = str(snap.get("coin"))
        item = state.get(coin)
        if not item:
            continue
        close = _safe_float(snap.get("close"))
        atr = _safe_float((snap.get("market_structure") or {}).get("atr14_3m"))
        level = _safe_float(item.get("trigger_level"))
        armed = bool(close is not None and atr is not None and atr > 0 and level is not None and abs(close - level) / atr <= adaptive.ARM_DISTANCE_ATR)
        status = "ARMED" if armed else "WATCH"
        if item.get("status") != status:
            item["status"] = status
            changed = True
        item["last_seen_at"] = now
        changed = True
    if changed:
        adaptive._write_json(adaptive.WATCH_FILE, state)


def _resync_flagged_watches(flagged: Dict[str, Dict[str, Any]], snapshots: Iterable[Dict[str, Any]]) -> None:
    by_coin = {str(s.get("coin")): s for s in snapshots}
    for coin, signal in flagged.items():
        snap = by_coin.get(str(coin))
        if not snap or bool(signal.get("take_trade")):
            continue
        setup = (((snap.get("entry_quality_context") or {}).get("adaptive") or {}).get("setup_quality") or {})
        if int(setup.get("score") or 0) < 3 and str(signal.get("decision") or "").upper() != "WATCH":
            continue
        direction = str(signal.get("direction") or "").lower()
        if direction not in {"long", "short"}:
            direction = _direction_from_sweep(snap) or ""
            if direction:
                signal["direction"] = direction
        if direction not in {"long", "short"}:
            continue
        playbook = _directional_playbook_final(snap, direction)
        if not playbook:
            continue
        signal["adaptive_playbook"] = playbook
        signal["adaptive_allowed_playbooks"] = (((snap.get("entry_quality_context") or {}).get("adaptive") or {}).get("allowed_playbooks") or [])
        if str(signal.get("decision") or "").upper() == "WATCH" or int(setup.get("score") or 0) >= 3:
            signal["decision"] = "WATCH"
            signal["lifecycle"] = "WATCH"
            hardening._upsert_watch_from_signal(signal, snap, signal.get("adaptive_watch_reason") or signal.get("reasoning") or "adaptive setup watch")


def _seed_sweep_only_watches(snapshots: Iterable[Dict[str, Any]]) -> None:
    state = adaptive._prune_watch()
    for snap in snapshots:
        adaptive_ctx = ((snap.get("entry_quality_context") or {}).get("adaptive") or {})
        setup = adaptive_ctx.get("setup_quality") or {}
        if int(setup.get("score") or 0) < 3:
            continue
        preferred = str(adaptive_ctx.get("preferred_playbook") or "")
        direction = "long" if "LONG" in preferred else ("short" if "SHORT" in preferred else _direction_from_sweep(snap))
        if direction not in {"long", "short"}:
            continue
        playbook = _directional_playbook_final(snap, direction) or preferred
        if not playbook:
            continue
        coin = str(snap.get("coin"))
        if coin in state:
            continue
        signal = {"direction": direction, "adaptive_playbook": playbook}
        item = hardening._watch_item(signal, snap, "deterministic sweep/setup watch")
        if item:
            state[coin] = item
    adaptive._write_json(adaptive.WATCH_FILE, state)


def _final_get_trade_suggestions_batch(signals, scorecard=None, open_positions=None):
    ok, flagged, position_updates = _BASE_GET(signals, scorecard, open_positions)
    if ok:
        _resync_flagged_watches(flagged, signals)
        _seed_sweep_only_watches(signals)
        _persist_watch_status(signals)
    return ok, flagged, position_updates


def _final_message(*args, **kwargs):
    text = _BASE_MESSAGE(*args, **kwargs)
    flagged = kwargs.get("flagged") or {}
    for coin, signal in flagged.items():
        if str(signal.get("decision") or "").upper() != "WATCH":
            continue
        direction = str(signal.get("direction") or "?").upper()
        text = text.replace(f"{coin} {direction} — SKIP", f"{coin} {direction} — WATCH")
    text = re.sub(r"\n\n🧭 Adaptive lifecycle: [^\n]*", "", text)
    text = re.sub(r"\n🧭 WATCH / ARMED candidates:\n(?:• .*\n?)*", "", text)
    watches = adaptive._prune_watch()
    if watches:
        lines = ["\n🧭 WATCH / ARMED candidates:"]
        for coin in sorted(watches):
            item = watches[coin]
            lines.append(
                f"• {coin} {str(item.get('direction') or '?').upper()} | "
                f"{item.get('playbook') or '-'} | {item.get('status') or 'WATCH'} | "
                f"trigger {item.get('trigger_type') or '-'} @ {item.get('trigger_level')}"
            )
        text += "\n".join(lines)
    return text


def install() -> None:
    global _PATCHED, _BASE_MESSAGE
    if _PATCHED:
        return
    _BASE_MESSAGE = live._scanner._build_message
    hardening._BASE_ENRICH = _enrich_final
    adaptive._enrich_snapshots = _enrich_final
    hardening._directional_playbook = _directional_playbook_final
    live._original_get_trade_suggestions_batch = _final_get_trade_suggestions_batch
    live._scanner._build_message = _final_message
    _PATCHED = True


install()
