from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, Tuple

# Controlled loosening from the prior production gate:
# - entry distance: 1.0 -> 1.5 ATR
# - breakout freshness: 8 -> 12 closed 3m bars
# - breakout extension: 1.0 -> 1.5 ATR
#
# Risk geometry remains separate and stays at MIN_STOP_ATR_MULTIPLIER=2.0
# in the production workflow.
MAX_ENTRY_DISTANCE_ATR = float(os.environ.get("MAX_ENTRY_DISTANCE_ATR", "1.5"))
BREAKOUT_MAX_BARS = int(os.environ.get("BREAKOUT_MAX_BARS", "12"))
BREAKOUT_MAX_EXTENSION_ATR = float(
    os.environ.get("BREAKOUT_MAX_EXTENSION_ATR", "1.5")
)


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _opposite(direction: str, bias: str) -> bool:
    d = str(direction or "").lower()
    b = str(bias or "").lower()
    return (d == "long" and b == "bearish") or (
        d == "short" and b == "bullish"
    )


def evaluate_entry_quality(
    signal: Dict[str, Any], snapshot: Dict[str, Any]
) -> Tuple[bool, str]:
    d = str(signal.get("direction") or "").lower()
    if d not in {"long", "short"}:
        return False, "invalid direction"

    ms = snapshot.get("market_structure") or {}
    s3 = ms.get("3m") or {}
    s15 = ms.get("15m") or {}
    s1h = ms.get("1h") or {}
    regime = str(ms.get("market_regime") or "UNCLEAR").upper()
    cont = (snapshot.get("entry_quality_context") or {}).get(
        "continuation"
    ) or {}
    recent = snapshot.get("recent_signal_context") or {}

    if recent.get("same_direction_reentry_without_new_break"):
        return (
            False,
            "same-direction re-entry within cooldown without a new structural break",
        )

    # Hard structural veto stays on 3m/15m because these are the actual
    # execution timeframes. A contrary 1h bias is now cautionary rather than
    # an automatic veto; a strong 3m+15m setup may be developing inside a
    # slower higher-timeframe transition.
    for tf, st in (("3m", s3), ("15m", s15)):
        if _opposite(d, st.get("structure_bias")):
            return False, f"{tf} structure directly opposes proposed direction"

    # 1h directly opposing is only a veto when the trading regime is not a
    # confirmed transition/breakout. This allows early intraday continuation
    # while still blocking obvious counter-trend attempts in ordinary trend
    # regimes.
    if regime not in {
        "BREAKOUT_TRANSITION",
        "BREAKDOWN_TRANSITION",
        "EXHAUSTION_OR_TRANSITION",
    }:
        if _opposite(d, s1h.get("structure_bias")):
            return False, "1h structure directly opposes proposed direction"

    # In ordinary trend regimes 3m MUST align; 15m may be neutral while a
    # move is developing. Direct opposition was already rejected above.
    if regime in {"TREND_UP", "TREND_DOWN", "EXHAUSTION_OR_TRANSITION"}:
        wanted = "bullish" if d == "long" else "bearish"
        if s3.get("structure_bias") != wanted:
            return False, "3m structure is not aligned with proposed direction"
        if regime != "EXHAUSTION_OR_TRANSITION":
            s15_bias = str(s15.get("structure_bias") or "").lower()
            if s15_bias not in {wanted, "neutral", "unclear", ""}:
                return False, "15m structure is not aligned or neutral"

    if regime == "RANGE":
        rng = ms.get("range") or {}
        if d == "long" and not rng.get("near_low"):
            return False, "range LONG is not near the lower boundary"
        if d == "short" and not rng.get("near_high"):
            return False, "range SHORT is not near the upper boundary"

    if regime in {"BREAKOUT_TRANSITION", "BREAKDOWN_TRANSITION"}:
        latest = s3.get("latest_break") or {}
        expected = "bullish" if d == "long" else "bearish"
        latest_direction = str(latest.get("direction") or "").lower()

        if latest_direction != expected:
            return False, "latest 3m structural break does not support direction"

        bars = cont.get("bars_since_break")
        if _finite(bars) and float(bars) > BREAKOUT_MAX_BARS:
            return (
                False,
                f"breakout is stale (> {BREAKOUT_MAX_BARS} closed 3m bars)",
            )

        ext = cont.get("extension_atr_from_break")
        if _finite(ext) and abs(float(ext)) > BREAKOUT_MAX_EXTENSION_ATR:
            return (
                False,
                f"breakout entry is extended {float(ext):.2f} ATR",
            )

    cq = str(cont.get("continuation_quality") or "")
    if cq == "late_exhausted":
        return False, "continuation is late and shows exhaustion/failure evidence"

    if cq == "late_extended":
        ext = cont.get("extension_atr_from_break")
        if _finite(ext) and abs(float(ext)) > BREAKOUT_MAX_EXTENSION_ATR:
            return (
                False,
                "continuation is late and extended beyond the allowed ATR",
            )

    if regime in {
        "TREND_UP",
        "TREND_DOWN",
        "BREAKOUT_TRANSITION",
        "BREAKDOWN_TRANSITION",
    }:
        # Keep this veto only when both warning conditions are present.
        if (s3.get("exhaustion_flags") or []) and (
            s3.get("failed_breaks") or []
        ):
            return (
                False,
                "exhaustion and failed-break evidence conflict with continuation",
            )

    close = snapshot.get("close")
    atr = ms.get("atr14_3m")
    entry = signal.get("entry_price")
    if _finite(close) and _finite(atr) and _finite(entry) and float(atr) > 0:
        dist = abs(float(entry) - float(close)) / float(atr)
        if dist > MAX_ENTRY_DISTANCE_ATR:
            return False, f"proposed entry is {dist:.2f} ATR from current price"

    return True, "ok"


def apply_entry_quality_gate(
    flagged: Dict[str, Dict[str, Any]],
    snapshots: Iterable[Dict[str, Any]],
) -> int:
    by_coin = {s.get("coin"): s for s in snapshots}
    rejected = 0

    for coin, signal in flagged.items():
        if not signal.get("take_trade"):
            continue

        snap = by_coin.get(coin)
        if snap is None:
            signal["take_trade"] = False
            signal["risk_validation_error"] = (
                "entry_quality_gate: missing deterministic market snapshot"
            )
            rejected += 1
            continue

        ok, reason = evaluate_entry_quality(signal, snap)
        if not ok:
            signal["take_trade"] = False
            signal["risk_validation_error"] = (
                f"entry_quality_gate: {reason}"
            )
            signal["_entry_quality_reject_reason"] = reason
            rejected += 1

    return rejected
