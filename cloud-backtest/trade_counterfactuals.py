"""Append-only counterfactual research ledger for the live AdvisorX path.

This module is deliberately non-invasive: it records what Gemini/Python did
and the market snapshot at the decision point. It never changes live orders,
ledger status, stops, targets, or position sizing.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

FILE = os.environ.get("COUNTERFACTUAL_FILE", "trade_counterfactuals.jsonl")


def _f(value: Any):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def log(
    kind: str,
    coin: str,
    direction: str | None = None,
    price: Any = None,
    decision: Dict[str, Any] | None = None,
    reason: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row = {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "observation_only": True,
        "coin": coin,
        "opportunity_id": (decision or {}).get("_opportunity_id") if isinstance(decision, dict) else None,
        "direction": direction,
        "decision_price": _f(price),
        "reason": reason,
        "future_outcome_pending": True,
        "decision": decision or {},
    }
    if extra:
        row.update(extra)

    directory = os.path.dirname(os.path.abspath(FILE)) or "."
    os.makedirs(directory, exist_ok=True)
    with open(FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def log_skip(signal: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    return log(
        "GEMINI_SKIP",
        str(signal.get("coin")),
        decision.get("direction"),
        decision.get("entry_price"),
        decision,
        decision.get("reasoning"),
        {"candle_time": signal.get("candle_time"), "market_regime": (signal.get("market_structure") or {}).get("market_regime")},
    )


def log_python_reject(signal: Dict[str, Any], decision: Dict[str, Any], reason: str | None) -> Dict[str, Any]:
    return log(
        "PYTHON_REJECT",
        str(signal.get("coin")),
        decision.get("direction"),
        decision.get("entry_price"),
        decision,
        reason,
        {"candle_time": signal.get("candle_time"), "market_regime": (signal.get("market_structure") or {}).get("market_regime")},
    )


def log_take(signal: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    return log(
        "TAKE",
        str(signal.get("coin")),
        decision.get("direction"),
        decision.get("entry_price"),
        decision,
        decision.get("reasoning"),
        {"candle_time": signal.get("candle_time"), "market_regime": (signal.get("market_structure") or {}).get("market_regime")},
    )


def log_watch(signal: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    return log(
        "WATCH",
        str(signal.get("coin")),
        decision.get("direction"),
        decision.get("entry_price"),
        decision,
        decision.get("reasoning"),
        {"candle_time": signal.get("candle_time"), "market_regime": (signal.get("market_structure") or {}).get("market_regime")},
    )


def log_exit(position: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    return log(
        "GEMINI_EXIT",
        str(position.get("coin")),
        position.get("direction"),
        position.get("current_price"),
        update,
        update.get("reasoning"),
        {
            "entry_price": position.get("entry_price"),
            "stop_loss": position.get("stop_loss"),
            "target_price": position.get("target_price"),
            "minutes_open": position.get("minutes_open"),
            "mfe_r": position.get("mfe_r"),
            "mfe_pnl_inr": position.get("mfe_pnl_inr"),
            "mae_pnl_inr": position.get("mae_pnl_inr"),
        },
    )
