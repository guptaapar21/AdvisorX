"""AdvisorX production policy overlays justified by the existing live ledger.

This module does not add a new entry strategy. It only:
1) makes position-management decisions less trigger-happy;
2) caps MFE/MAE telemetry at the first actual target/stop event visible in
   the available 1m candles; and
3) records clear provenance for Gemini's proposed entry levels.

The hard risk/geometry gate remains in gemini_advisor.py.
"""
from __future__ import annotations

from typing import Any, Mapping

import os

import pandas as pd

POLICY_VERSION = "2026-09-07-trade-management-v1"
DEFAULT_MIN_TIGHTEN_R = 0.50


def _utc(value: Any):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def current_profit_r(position: Mapping[str, Any]) -> float | None:
    """Return current gross P&L divided by the original planned INR risk."""
    try:
        entry = float(position["entry_price"])
        current = float(position["current_price"])
        risk = float(position.get("max_loss_this_trade_inr") or position.get("risk_inr") or 0)
        amount = float(position.get("trade_amount_inr") or 0)
        direction = str(position.get("direction") or "").lower()
        fx = float(position.get("usdt_inr_rate") or os.environ.get("USDT_INR_RATE", "99.44"))
        if entry <= 0 or amount <= 0 or risk <= 0 or fx <= 0 or direction not in {"long", "short"}:
            return None
        qty = amount / (entry * fx)
        gross = qty * ((current - entry) if direction == "long" else (entry - current)) * fx
        return gross / risk
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def apply_management_policy(
    position_updates: Mapping[str, Mapping[str, Any]],
    open_positions: list[Mapping[str, Any]],
    min_tighten_r: float = DEFAULT_MIN_TIGHTEN_R,
) -> dict[str, dict[str, Any]]:
    """Keep genuine exits intact, but suppress premature stop tightening.

    A Gemini ``tighten_stop`` is only executable once the open position has
    reached at least +0.50R gross. This is deliberately narrower than blocking
    exits: a genuine thesis invalidation must still be able to close a trade.
    """
    by_coin = {str(p.get("coin")): p for p in open_positions if p.get("coin")}
    out: dict[str, dict[str, Any]] = {}
    for coin, raw in position_updates.items():
        update = dict(raw or {})
        action = str(update.get("action") or "hold").strip().lower()
        update["action"] = action
        position = by_coin.get(str(coin))
        if action == "tighten_stop" and position is not None:
            r = current_profit_r(position)
            if r is None or r < min_tighten_r:
                update["action"] = "hold"
                update["management_policy"] = "tighten_suppressed_before_profit_threshold"
                update["management_policy_r"] = round(r, 3) if r is not None else None
                update["original_action"] = "tighten_stop"
                reason = str(update.get("reasoning") or "")
                update["reasoning"] = (
                    "Management policy kept the position on HOLD because a stop tightening "
                    f"was requested before +{min_tighten_r:.2f}R was earned. "
                    + reason
                ).strip()
        out[str(coin)] = update
    return out


def add_signal_provenance(flagged: Mapping[str, Mapping[str, Any]]) -> None:
    """Add explicit raw-vs-active level provenance in-place."""
    for item in flagged.values():
        if not isinstance(item, dict):
            continue
        item.setdefault("gemini_proposed_entry", item.get("entry_price"))
        item.setdefault("gemini_proposed_stop", item.get("stop_loss"))
        item.setdefault("gemini_proposed_target", item.get("target_price"))
        item.setdefault("trade_management_policy_version", POLICY_VERSION)


def _active_levels_at(history: list[Mapping[str, Any]], candle_time):
    if not history:
        return None, None
    ct = _utc(candle_time)
    active = history[0]
    for level in sorted(history, key=lambda x: _utc(x["from"])):
        if _utc(level["from"]) <= ct:
            active = level
        else:
            break
    return active.get("stop"), active.get("target")


def update_mfe_mae_until_exit(ledger: list[dict[str, Any]], fetched: Mapping[str, Any]) -> None:
    """Update MFE/MAE from 1m data but stop at the first actual exit event.

    This replaces the production telemetry function that currently walks the
    whole fetched window before ``resolve_ledger`` determines the exit. Thus a
    target/stop reached at 14:27 cannot acquire an MFE timestamp from 14:32.
    """
    for entry in ledger:
        if entry.get("status") != "pending":
            continue
        data = fetched.get(entry.get("coin"))
        if not data or len(data) < 4:
            continue
        c1 = data[3]
        if c1 is None or getattr(c1, "empty", True):
            continue
        try:
            entry_price = float(entry.get("entry_price"))
            amount = float(entry.get("trade_amount_inr"))
            fx = float(os.environ.get("USDT_INR_RATE", "99.44"))
            direction = str(entry.get("direction") or "").lower()
            if entry_price <= 0 or amount <= 0 or direction not in {"long", "short"}:
                continue
            qty = amount / (entry_price * fx)
        except (TypeError, ValueError, ZeroDivisionError):
            continue

        entry.setdefault("peak_price", entry_price)
        entry.setdefault("trough_price", entry_price)
        entry.setdefault("mfe_pnl_inr", 0.0)
        entry.setdefault("mae_pnl_inr", 0.0)
        entry.setdefault("mfe_time", entry.get("time"))
        entry.setdefault("mae_time", entry.get("time"))

        call = _utc(entry.get("time")).tz_localize(None)
        since = c1[c1.index > call]
        if since.empty:
            continue

        history = entry.get("level_history") or [
            {"from": entry.get("time"), "stop": entry.get("stop_loss"), "target": entry.get("target_price")}
        ]

        for idx, row in since.iterrows():
            candle_time = _utc(idx)
            stop, target = _active_levels_at(history, candle_time)
            try:
                stop = float(stop) if stop is not None else None
                target = float(target) if target is not None else None
            except (TypeError, ValueError):
                stop = target = None

            high = float(row["high"])
            low = float(row["low"])

            if direction == "long":
                fav_price, adverse_price = high, low
                target_hit = target is not None and high >= target
                stop_hit = stop is not None and low <= stop
                if fav_price > float(entry["peak_price"]):
                    entry["peak_price"] = fav_price
                    entry["mfe_time"] = candle_time.isoformat()
                if adverse_price < float(entry["trough_price"]):
                    entry["trough_price"] = adverse_price
                    entry["mae_time"] = candle_time.isoformat()
                fav = qty * (fav_price - entry_price) * fx
                adverse = qty * (adverse_price - entry_price) * fx
            else:
                fav_price, adverse_price = low, high
                target_hit = target is not None and low <= target
                stop_hit = stop is not None and high >= stop
                if fav_price < float(entry["trough_price"]):
                    entry["trough_price"] = fav_price
                    entry["mfe_time"] = candle_time.isoformat()
                if adverse_price > float(entry["peak_price"]):
                    entry["peak_price"] = adverse_price
                    entry["mae_time"] = candle_time.isoformat()
                fav = qty * (entry_price - fav_price) * fx
                adverse = qty * (entry_price - adverse_price) * fx

            if fav > float(entry.get("mfe_pnl_inr") or 0.0):
                entry["mfe_pnl_inr"] = round(fav, 2)
            if adverse < float(entry.get("mae_pnl_inr") or 0.0):
                entry["mae_pnl_inr"] = round(adverse, 2)

            # Match resolve_ledger's deterministic precedence when both levels
            # are touched inside one 1m candle: stop wins.
            if target_hit and stop_hit:
                break
            if target_hit or stop_hit:
                break

        risk = float(entry.get("max_loss_this_trade_inr") or 0)
        mfe = float(entry.get("mfe_pnl_inr") or 0)
        entry["mfe_r"] = round(mfe / risk, 3) if risk > 0 else None
        entry["mfe_mae_policy_version"] = POLICY_VERSION
