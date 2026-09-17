"""Second-pass hardening for the V5 adaptive AdvisorX execution layer.

This module is intentionally separate from the research engine. It closes the
remaining integration gaps found during a full production-path review:
- global regime breadth survives temporarily stale/missing coins;
- Gemini's direction is reconciled with the local playbook instead of blindly
  inheriting one preferred range direction;
- setup candidates can become durable WATCH records even when Gemini returns no
  proposal for them;
- TAKE is only allowed after a playbook-specific trigger/acceptance check;
- playbook-specific stop/target geometry is validated;
- a rejected TAKE can be converted into WATCH rather than being lost as a
  generic Python reject when the setup remains valid but the entry is early;
- daily risk uses an Indian calendar day (IST), matching the scanner's user
  reporting convention.

The research engine is not imported or modified here.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import adaptive_playbook_layer as adaptive
import trend_alignment_scanner_live as live


ROOT = Path(__file__).resolve().parent
GLOBAL_MEMORY_FILE = ROOT / "adaptive_global_regime_memory.json"
GLOBAL_MEMORY_MAX_AGE_MINUTES = 9.0
IST = timezone(timedelta(hours=5, minutes=30))

_BASE_GLOBAL_REGIME = adaptive._global_regime_candidate
_BASE_ADAPTIVE_GET = adaptive.adaptive_get_trade_suggestions_batch
_BASE_ADAPTIVE_GATE = adaptive.adaptive_entry_quality_gate
_BASE_ENRICH = adaptive._enrich_snapshots
_PATCHED = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
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
        print(f"  Adaptive hardening state WARNING ({path.name}): {exc}")


def _append_telemetry(record: Dict[str, Any]) -> None:
    adaptive._append_jsonl(adaptive.PLAYBOOK_TELEMETRY_FILE, record)


def _normalized_local_regime(snapshot: Dict[str, Any]) -> str:
    return adaptive._normalize_regime((snapshot.get("market_structure") or {}).get("market_regime"))


def _directional_playbook(snapshot: Dict[str, Any], direction: str | None) -> str | None:
    """Choose a playbook that is compatible with Gemini's actual direction.

    The first-pass V5 layer chooses a single preferred playbook before Gemini.
    That is useful for context but is not sufficient for RANGE because Gemini
    may legitimately select the opposite side of the same range. This helper
    reconciles the final direction with the local regime and current events.
    """
    d = str(direction or "").lower()
    if d not in {"long", "short"}:
        return None
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    rng = ms.get("range") or {}
    regime = _normalized_local_regime(snapshot)
    latest = s3.get("latest_break") or {}
    sweeps = s3.get("liquidity_sweeps") or []
    latest_dir = str(latest.get("direction") or "").lower()
    low_sweep = any("low" in str(x.get("type", "")).lower() for x in sweeps)
    high_sweep = any("high" in str(x.get("type", "")).lower() for x in sweeps)

    if regime == "TREND_UP" and d == "long":
        if latest_dir == "bullish" and str(latest.get("type", "")).upper() in {"BOS", "CHOCH"}:
            return "LONG_BREAKOUT"
        if bool(rng.get("near_low")):
            return "LONG_PULLBACK"
        return "LONG_CONTINUATION"
    if regime == "TREND_DOWN" and d == "short":
        if latest_dir == "bearish" and str(latest.get("type", "")).upper() in {"BOS", "CHOCH"}:
            return "SHORT_BREAKDOWN"
        if bool(rng.get("near_high")):
            return "SHORT_RALLY_REJECTION"
        return "SHORT_CONTINUATION"
    if regime == "RANGE":
        if d == "long":
            if low_sweep and high_sweep:
                return "RANGE_REVERSAL"
            if low_sweep or bool(rng.get("near_low")):
                return "RANGE_REVERSAL" if low_sweep else "RANGE_LONG"
            if latest_dir == "bullish":
                return "RANGE_BREAKOUT_WATCH"
            return "RANGE_LONG"
        if high_sweep and low_sweep:
            return "RANGE_REVERSAL"
        if high_sweep or bool(rng.get("near_high")):
            return "RANGE_REVERSAL" if high_sweep else "RANGE_SHORT"
        if latest_dir == "bearish":
            return "RANGE_BREAKOUT_WATCH"
        return "RANGE_SHORT"
    if regime == "TRANSITION":
        if d == "long" and latest_dir == "bullish":
            return "BREAKOUT_RETEST"
        if d == "short" and latest_dir == "bearish":
            return "BREAKDOWN_RETEST"
        if d == "long" and low_sweep:
            return "LIQUIDITY_REVERSAL"
        if d == "short" and high_sweep:
            return "LIQUIDITY_REVERSAL"
        return None
    if regime == "UNCLEAR":
        if d == "long" and low_sweep:
            return "LIQUIDITY_REVERSAL"
        if d == "short" and high_sweep:
            return "LIQUIDITY_REVERSAL"
        if d == "long" and latest_dir == "bullish":
            return "RANGE_BREAKOUT_WATCH"
        if d == "short" and latest_dir == "bearish":
            return "RANGE_BREAKOUT_WATCH"
    return None


def _trigger_for(snapshot: Dict[str, Any], playbook: str, direction: str | None) -> Tuple[str, float | None]:
    name, level, _ = adaptive._trigger_for_playbook(snapshot, playbook, direction)
    return name, level


def _has_sweep(snapshot: Dict[str, Any], side: str) -> bool:
    sweeps = ((snapshot.get("market_structure") or {}).get("3m") or {}).get("liquidity_sweeps") or []
    return any(side in str(x.get("type", "")).lower() for x in sweeps)


def _latest_break_metrics(snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], int | None]:
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    latest = s3.get("latest_break") or {}
    cont = (snapshot.get("entry_quality_context") or {}).get("continuation") or {}
    bars = _safe_float(cont.get("bars_since_break", latest.get("bars_since_break")))
    return latest, int(bars) if bars is not None else None


def _trigger_confirmed(snapshot: Dict[str, Any], playbook: str, direction: str, entry: float) -> Tuple[bool, str]:
    """Strict playbook-specific trigger check for converting TAKE to execution."""
    d = str(direction or "").lower()
    wanted = "bullish" if d == "long" else "bearish"
    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    rng = ms.get("range") or {}
    atr = _safe_float(ms.get("atr14_3m"))
    close = _safe_float(snapshot.get("close"))
    latest, bars = _latest_break_metrics(snapshot)
    latest_dir = str(latest.get("direction") or "").lower()
    level = _safe_float(latest.get("level"))
    bias = str(s3.get("structure_bias") or "").lower()
    cont = (snapshot.get("entry_quality_context") or {}).get("continuation") or {}
    freshness = str(cont.get("break_freshness") or "").lower()

    if atr is None or atr <= 0 or close is None:
        return False, "missing close/ATR for trigger confirmation"

    if playbook == "LONG_PULLBACK":
        ema = _safe_float(s3.get("ema21"))
        if ema is None or abs(entry - ema) / atr > 0.90 or bias != "bullish":
            return False, "LONG_PULLBACK requires EMA21 proximity + bullish 3m structure"
        return True, "EMA21 pullback confirmed"
    if playbook == "SHORT_RALLY_REJECTION":
        ema = _safe_float(s3.get("ema21"))
        if ema is None or abs(entry - ema) / atr > 0.90 or bias != "bearish":
            return False, "SHORT_RALLY_REJECTION requires EMA21 proximity + bearish 3m structure"
        return True, "EMA21 rally rejection confirmed"

    if playbook in {"LONG_BREAKOUT", "SHORT_BREAKDOWN"}:
        if latest_dir != wanted or level is None:
            return False, "breakout playbook lacks matching structural break"
        if bars is not None and bars > 12:
            return False, "structural break is stale"
        if d == "long" and close < level:
            return False, "close has not held above bullish break"
        if d == "short" and close > level:
            return False, "close has not held below bearish break"
        return True, "fresh structural breakout confirmed"

    if playbook in {"LONG_CONTINUATION", "SHORT_CONTINUATION"}:
        if latest_dir != wanted or level is None:
            return False, "continuation lacks a matching structural break"
        if bars is not None and bars > 12:
            return False, "continuation break is stale"
        if bias != wanted:
            return False, "3m structure is not aligned with continuation direction"
        return True, "continuation structure confirmed"

    if playbook in {"RANGE_LONG", "RANGE_SHORT", "RANGE_REVERSAL"}:
        position = _safe_float(rng.get("position_pct"))
        low = position is not None and position <= 20
        high = position is not None and position >= 80
        if d == "long":
            if not (low or _has_sweep(snapshot, "low")):
                return False, "range LONG needs lower-boundary interaction"
            if not (_has_sweep(snapshot, "low") or bias == "bullish"):
                return False, "range LONG needs low sweep/reclaim or bullish 3m confirmation"
        else:
            if not (high or _has_sweep(snapshot, "high")):
                return False, "range SHORT needs upper-boundary interaction"
            if not (_has_sweep(snapshot, "high") or bias == "bearish"):
                return False, "range SHORT needs high sweep/rejection or bearish 3m confirmation"
        if playbook == "RANGE_REVERSAL" and not _has_sweep(snapshot, "low" if d == "long" else "high"):
            return False, "range reversal requires a matching liquidity sweep"
        return True, "range boundary + directional confirmation present"

    if playbook in {"BREAKOUT_RETEST", "BREAKDOWN_RETEST"}:
        if latest_dir != wanted or level is None:
            return False, "transition playbook lacks matching break"
        if bars is not None and bars > 12:
            return False, "transition break is stale"
        if bars is not None and bars < 1:
            return False, "first expansion is not yet a retest"
        if abs(entry - level) / atr > 0.75:
            return False, "entry is not at the break retest zone"
        if bias != wanted:
            return False, "3m structure has not accepted the transition direction"
        return True, "break-retest-acceptance confirmed"

    if playbook == "LIQUIDITY_REVERSAL":
        side = "low" if d == "long" else "high"
        if not _has_sweep(snapshot, side):
            return False, f"{d} liquidity reversal requires a {side} sweep"
        if bias != wanted:
            return False, "liquidity reversal requires directional 3m reclaim/rejection"
        return True, "liquidity sweep + directional confirmation present"

    if playbook == "RANGE_BREAKOUT_WATCH":
        return False, "RANGE_BREAKOUT_WATCH remains watch-only until a break-retest playbook is established"

    return False, "no trigger rule for playbook"


def _playbook_geometry_ok(signal: Dict[str, Any], snapshot: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate natural invalidation/objective for the selected playbook."""
    d = str(signal.get("direction") or "").lower()
    entry = _safe_float(signal.get("entry_price"))
    stop = _safe_float(signal.get("stop_loss"))
    target = _safe_float(signal.get("target_price"))
    if d not in {"long", "short"} or entry is None or stop is None or target is None:
        return False, "missing direction/levels"
    if d == "long" and not (stop < entry < target):
        return False, "LONG geometry is invalid"
    if d == "short" and not (target < entry < stop):
        return False, "SHORT geometry is invalid"

    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    rng = ms.get("range") or {}
    latest = s3.get("latest_break") or {}
    p = str(signal.get("adaptive_playbook") or "")

    if p in {"RANGE_LONG", "RANGE_REVERSAL"} and d == "long":
        low = _safe_float(rng.get("range_low"))
        high = _safe_float(rng.get("range_high"))
        if low is not None and stop >= low:
            return False, "range LONG stop must invalidate below the range low"
        if high is not None and target > high * 1.02:
            return False, "range LONG target exceeds the observed range objective"
    if p in {"RANGE_SHORT", "RANGE_REVERSAL"} and d == "short":
        low = _safe_float(rng.get("range_low"))
        high = _safe_float(rng.get("range_high"))
        if high is not None and stop <= high:
            return False, "range SHORT stop must invalidate above the range high"
        if low is not None and target < low * 0.98:
            return False, "range SHORT target exceeds the observed range objective"

    if p in {"BREAKOUT_RETEST", "LONG_BREAKOUT"} and d == "long":
        level = _safe_float(latest.get("level"))
        if level is not None and stop >= level:
            return False, "LONG breakout stop should invalidate back below the break"
    if p in {"BREAKDOWN_RETEST", "SHORT_BREAKDOWN"} and d == "short":
        level = _safe_float(latest.get("level"))
        if level is not None and stop <= level:
            return False, "SHORT breakdown stop should invalidate back above the break"

    if p == "LONG_PULLBACK" and d == "long":
        ema = _safe_float(s3.get("ema21"))
        if ema is not None and stop >= ema:
            return False, "LONG_PULLBACK stop should sit below EMA21"
    if p == "SHORT_RALLY_REJECTION" and d == "short":
        ema = _safe_float(s3.get("ema21"))
        if ema is not None and stop <= ema:
            return False, "SHORT_RALLY_REJECTION stop should sit above EMA21"

    swings_h = s3.get("swing_highs") or []
    swings_l = s3.get("swing_lows") or []
    if d == "long" and swings_h:
        high_price = _safe_float((swings_h[-1] or {}).get("price"))
        if high_price is not None and p not in {"RANGE_LONG", "RANGE_REVERSAL"} and target < high_price:
            return False, "LONG target should clear the next confirmed swing high"
    if d == "short" and swings_l:
        low_price = _safe_float((swings_l[-1] or {}).get("price"))
        if low_price is not None and p not in {"RANGE_SHORT", "RANGE_REVERSAL"} and target > low_price:
            return False, "SHORT target should clear the next confirmed swing low"
    return True, "playbook geometry valid"


def _watch_item(signal: Dict[str, Any], snapshot: Dict[str, Any], reason: str) -> Dict[str, Any] | None:
    d = str(signal.get("direction") or "").lower()
    p = str(signal.get("adaptive_playbook") or "")
    if d not in {"long", "short"} or not p:
        return None
    trigger_type, trigger_level = _trigger_for(snapshot, p, d)
    state = adaptive._prune_watch()
    old = state.get(str(snapshot.get("coin"))) or {}
    created = old.get("created_at") if old and str(old.get("direction")) == d else _now().isoformat()
    return {
        "coin": snapshot.get("coin"),
        "direction": d,
        "playbook": p,
        "created_at": created,
        "last_seen_at": _now().isoformat(),
        "trigger_type": trigger_type,
        "trigger_level": trigger_level,
        "setup_score": ((snapshot.get("entry_quality_context") or {}).get("adaptive") or {}).get("setup_quality", {}).get("score"),
        "reasoning": reason,
        "status": "WATCH",
    }


def _seed_watch_candidates(snapshots: Iterable[Dict[str, Any]]) -> int:
    """Persist setup-quality WATCH records even when Gemini returns no signal."""
    state = adaptive._prune_watch()
    created = 0
    now = _now()
    for snapshot in snapshots:
        adaptive_ctx = ((snapshot.get("entry_quality_context") or {}).get("adaptive") or {})
        setup = adaptive_ctx.get("setup_quality") or {}
        if int(setup.get("score") or 0) < 3:
            continue
        playbook = str(adaptive_ctx.get("preferred_playbook") or "")
        direction = "long" if "LONG" in playbook else ("short" if "SHORT" in playbook else None)
        if direction is None:
            latest = (((snapshot.get("market_structure") or {}).get("3m") or {}).get("latest_break") or {})
            ld = str(latest.get("direction") or "").lower()
            if ld in {"bullish", "bearish"}:
                direction = "long" if ld == "bullish" else "short"
        if direction is None:
            continue
        remapped = _directional_playbook(snapshot, direction) or playbook
        if not remapped:
            continue
        signal = {"direction": direction, "adaptive_playbook": remapped}
        item = _watch_item(signal, snapshot, "deterministic setup-quality watch")
        if item is None:
            continue
        coin = str(snapshot.get("coin"))
        old = state.get(coin)
        if old and str(old.get("direction")) == direction and str(old.get("playbook")) == remapped:
            item["created_at"] = old.get("created_at", item["created_at"])
        state[coin] = item
        created += 1
        _append_telemetry({
            "time": now.isoformat(),
            "coin": coin,
            "playbook": remapped,
            "direction": direction,
            "decision": "WATCH",
            "lifecycle": "WATCH",
            "source": "deterministic_setup_quality_seed",
            "setup_quality": setup,
        })
    if state != adaptive._watch_state():
        adaptive._write_json(adaptive.WATCH_FILE, state)
    return created


def _global_regime_with_memory(snapshots: Iterable[Dict[str, Any]]) -> Tuple[str, float, Dict[str, Any]]:
    snaps = list(snapshots)
    now = _now()
    memory = _read_json(GLOBAL_MEMORY_FILE, {})
    keep: Dict[str, Any] = {}
    merged = list(snaps)
    present = {str(s.get("coin")) for s in snaps}
    for s in snaps:
        coin = str(s.get("coin"))
        if not coin:
            continue
        keep[coin] = {
            "timestamp": now.isoformat(),
            "momentum_pct_20_3m": _safe_float(s.get("momentum_pct_20_3m")),
            "market_regime": (s.get("market_structure") or {}).get("market_regime"),
        }
    for coin, item in memory.items() if isinstance(memory, dict) else []:
        if coin in present or not isinstance(item, dict):
            continue
        try:
            age = (now - adaptive._parse_time(item.get("timestamp"))).total_seconds() / 60.0
        except Exception:
            continue
        if age < 0 or age > GLOBAL_MEMORY_MAX_AGE_MINUTES:
            continue
        merged.append({
            "coin": coin,
            "momentum_pct_20_3m": item.get("momentum_pct_20_3m"),
            "market_structure": {"market_regime": item.get("market_regime")},
        })
    _write_json(GLOBAL_MEMORY_FILE, keep)
    return _BASE_GLOBAL_REGIME(merged)


def _reconcile_flagged_playbooks(flagged: Dict[str, Dict[str, Any]], snapshots: Iterable[Dict[str, Any]]) -> None:
    by_coin = {str(s.get("coin")): s for s in snapshots}
    for coin, signal in flagged.items():
        snap = by_coin.get(str(coin))
        if not snap:
            continue
        d = str(signal.get("direction") or "").lower()
        mapped = _directional_playbook(snap, d)
        if mapped:
            signal["adaptive_playbook"] = mapped
            signal["adaptive_allowed_playbooks"] = ((snap.get("entry_quality_context") or {}).get("adaptive") or {}).get("allowed_playbooks", [])
            signal["adaptive_playbook_reconciled"] = True


def _upsert_watch_from_signal(signal: Dict[str, Any], snapshot: Dict[str, Any], reason: str) -> None:
    item = _watch_item(signal, snapshot, reason)
    if item is None:
        return
    state = adaptive._prune_watch()
    state[str(snapshot.get("coin"))] = item
    adaptive._write_json(adaptive.WATCH_FILE, state)
    _append_telemetry({
        "time": _now().isoformat(),
        "coin": snapshot.get("coin"),
        "playbook": signal.get("adaptive_playbook"),
        "direction": signal.get("direction"),
        "decision": "WATCH",
        "lifecycle": "WATCH",
        "source": "take_to_watch_hardening",
        "reason": reason,
    })


def hardening_get_trade_suggestions_batch(signals, scorecard=None, open_positions=None):
    signals = list(signals)
    # Build deterministic adaptive context once for WATCH seeding. The base V5
    # wrapper enriches again before Gemini; this second pass is deliberate so
    # persisted state is available to the real Gemini payload.
    seed_context = _BASE_ENRICH(signals)
    _seed_watch_candidates(seed_context.values())
    ok, flagged, position_updates = _BASE_ADAPTIVE_GET(signals, scorecard, open_positions)
    if ok:
        _reconcile_flagged_playbooks(flagged, signals)
    return ok, flagged, position_updates


def hardening_entry_quality_gate(flagged: Dict[str, Dict[str, Any]], snapshots: Iterable[Dict[str, Any]]) -> int:
    snapshots = list(snapshots)
    original_take = {str(c): bool(s.get("take_trade")) for c, s in flagged.items()}
    by_coin = {str(s.get("coin")): s for s in snapshots}

    rejected = _BASE_ADAPTIVE_GATE(flagged, snapshots)

    for coin, signal in flagged.items():
        if not original_take.get(str(coin)):
            continue
        snap = by_coin.get(str(coin))
        if not snap:
            continue
        # Reconcile once more because a caller can construct flagged data
        # directly in tests/recovery paths.
        mapped = _directional_playbook(snap, signal.get("direction"))
        if mapped:
            signal["adaptive_playbook"] = mapped

        if signal.get("take_trade"):
            entry = _safe_float(signal.get("entry_price"))
            if entry is None:
                continue
            p = str(signal.get("adaptive_playbook") or "")
            trigger_ok, trigger_reason = _trigger_confirmed(snap, p, str(signal.get("direction") or ""), entry)
            geom_ok, geom_reason = _playbook_geometry_ok(signal, snap)
            if not trigger_ok or not geom_ok:
                reason = trigger_reason if not trigger_ok else geom_reason
                signal["take_trade"] = False
                signal["decision"] = "WATCH"
                signal["lifecycle"] = "WATCH"
                signal["adaptive_watch_reason"] = reason
                signal["adaptive_gate_reason"] = reason
                signal.pop("risk_validation_error", None)
                signal.pop("_entry_quality_reject_reason", None)
                _upsert_watch_from_signal(signal, snap, reason)
                rejected += 1
            continue

        # A Python V5 rejection is often an early-entry rejection rather than a
        # bad setup. Preserve it as WATCH when the deterministic setup is still
        # eligible. Hard daily-risk and missing-data failures remain hard rejects.
        reason = str(signal.get("adaptive_gate_reason") or signal.get("risk_validation_error") or "").lower()
        if reason.startswith("calendar-day risk governor") or "missing snapshot" in reason:
            continue
        adaptive_ctx = ((snap.get("entry_quality_context") or {}).get("adaptive") or {})
        setup = adaptive_ctx.get("setup_quality") or {}
        if int(setup.get("score") or 0) >= 3:
            p = str(signal.get("adaptive_playbook") or adaptive_ctx.get("preferred_playbook") or "")
            d = str(signal.get("direction") or "").lower()
            mapped = _directional_playbook(snap, d)
            if mapped:
                signal["adaptive_playbook"] = mapped
                p = mapped
            if d in {"long", "short"} and p:
                signal["decision"] = "WATCH"
                signal["lifecycle"] = "WATCH"
                signal["adaptive_watch_reason"] = signal.get("adaptive_gate_reason") or "entry not yet executable"
                signal.pop("risk_validation_error", None)
                signal.pop("_entry_quality_reject_reason", None)
                _upsert_watch_from_signal(signal, snap, signal["adaptive_watch_reason"])
    return rejected


def _refresh_daily_risk_ist() -> Dict[str, Any]:
    today = _now().astimezone(IST).date().isoformat()
    sources = []
    state = _read_json(ROOT / "trend_scanner_state.json", {})
    if isinstance(state, dict):
        sources.extend(state.get("ledger") or [])
    resolved = ROOT / "resolved_trades.jsonl"
    if resolved.exists():
        try:
            for line in resolved.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    sources.append(json.loads(line))
        except (OSError, ValueError, TypeError):
            pass
    seen = set()
    pnl = 0.0
    count = 0
    for entry in sources:
        if entry.get("status") in {"pending", "invalid", "abandoned"}:
            continue
        dt = adaptive._parse_time(entry.get("resolved_time") or entry.get("time"))
        if dt is None or dt.astimezone(IST).date().isoformat() != today:
            continue
        key = (entry.get("coin"), entry.get("time"), entry.get("resolved_time"), entry.get("status"), entry.get("resolved_pnl"))
        if key in seen:
            continue
        seen.add(key)
        pnl += _safe_float(entry.get("resolved_pnl")) or 0.0
        count += 1
    pnl = round(pnl, 2)
    halted = pnl <= -abs(adaptive.DAILY_MAX_LOSS_INR)
    payload = {
        "calendar_timezone": "Asia/Kolkata",
        "calendar_date": today,
        "realized_pnl_inr": pnl,
        "resolved_trades": count,
        "max_daily_loss_inr": adaptive.DAILY_MAX_LOSS_INR,
        "trading_halted_for_day": halted,
        "updated_at": _now().isoformat(),
    }
    adaptive._write_json(adaptive.DAILY_RISK_FILE, payload)
    return payload


def _patch_message() -> None:
    original = adaptive.live._scanner._build_message
    if getattr(original, "_adaptive_hardening_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        text = original(*args, **kwargs)
        watches = adaptive._prune_watch()
        if watches:
            lines = ["\n🧭 WATCH / ARMED candidates:"]
            for coin, item in sorted(watches.items()):
                status = str(item.get("status") or "WATCH")
                lines.append(
                    f"• {coin} {str(item.get('direction') or '?').upper()} | "
                    f"{item.get('playbook') or '-'} | {status} | "
                    f"trigger {item.get('trigger_type') or '-'} @ {item.get('trigger_level')}")
            text += "\n".join(lines)
        return text

    wrapped._adaptive_hardening_wrapped = True
    adaptive.live._scanner._build_message = wrapped


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    adaptive._global_regime_candidate = _global_regime_with_memory
    adaptive.live._original_get_trade_suggestions_batch = hardening_get_trade_suggestions_batch
    adaptive.live.apply_entry_quality_gate = hardening_entry_quality_gate
    adaptive._refresh_daily_risk = _refresh_daily_risk_ist
    _patch_message()
    _PATCHED = True


install()
