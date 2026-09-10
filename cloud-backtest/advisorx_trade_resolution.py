"""Deterministic trade outcome resolution on the finest live timeframe.

The live scanner fetches 1m candles and already uses them for MFE/MAE. Using
that same 1m stream for target/stop resolution keeps realized outcomes aligned
with the telemetry used to judge giveback and trade health.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any, Mapping

import pandas as pd


def _utc(value: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return ts.tz_convert("UTC").to_pydatetime()


def _append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), separators=(",", ":"), default=str) + "\n")
        handle.flush()


def resolve_ledger_1m(
    ledger: list[dict[str, Any]],
    fetched: Mapping[str, Any],
    now: Any,
    *,
    usdt_inr_rate: float,
    taker_fee_rate: float,
    expiry_hours: float = 2.0,
    ledger_max_age_hours: float = 26.0,
    resolved_trades_file: str = "resolved_trades.jsonl",
    research_telemetry_version: str = "",
) -> list[dict[str, Any]]:
    """Resolve pending trades from closed 1m candles and prune old entries.

    A candle that crosses both target and stop is conservatively classified as
    a stop hit because OHLC data cannot establish the intrabar order of events.
    That convention is explicit and deterministic; it is also identical across
    back-to-back live runs because only closed 1m candles are considered.
    """
    now_utc = _utc(now)
    fx = float(usdt_inr_rate)
    fee_rate = float(taker_fee_rate)

    for entry in ledger:
        if entry.get("status") != "pending":
            continue

        try:
            entry_price = float(entry.get("entry_price"))
            trade_amount = float(entry.get("trade_amount_inr"))
        except (TypeError, ValueError):
            entry["status"] = "invalid"
            entry["resolved_pnl"] = 0.0
            entry["resolved_time"] = now_utc.isoformat()
            continue
        if entry_price <= 0 or trade_amount <= 0 or fx <= 0:
            entry["status"] = "invalid"
            entry["resolved_pnl"] = 0.0
            entry["resolved_time"] = now_utc.isoformat()
            continue

        coin = entry.get("coin")
        data = fetched.get(coin)
        if not data or len(data) < 4:
            continue
        candles_1m = data[3]
        if candles_1m is None or getattr(candles_1m, "empty", True):
            continue

        call_time = _utc(entry.get("time"))
        index = candles_1m.index
        compare_time = call_time.replace(tzinfo=None)
        since = candles_1m[index > compare_time]
        if since.empty:
            continue

        quantity = trade_amount / (entry_price * fx)
        fees = trade_amount * fee_rate * 2
        direction = str(entry.get("direction") or "").lower()
        if direction not in {"long", "short"}:
            entry["status"] = "invalid"
            entry["resolved_pnl"] = 0.0
            entry["resolved_time"] = now_utc.isoformat()
            continue

        history = entry.get("level_history") or [
            {"from": entry.get("time"), "stop": entry.get("stop_loss"), "target": entry.get("target_price")}
        ]
        history = sorted(history, key=lambda item: _utc(item["from"]))

        for _, row in since.iterrows():
            candle_time = _utc(row.name)
            active = history[0]
            for level in history:
                if _utc(level["from"]) <= candle_time:
                    active = level
                else:
                    break

            try:
                target = float(active.get("target")) if active.get("target") is not None else None
                stop = float(active.get("stop")) if active.get("stop") is not None else None
            except (TypeError, ValueError):
                continue
            if target is None or stop is None:
                continue

            high = float(row["high"])
            low = float(row["low"])
            if direction == "long":
                target_hit = high >= target
                stop_hit = low <= stop
            else:
                target_hit = low <= target
                stop_hit = high >= stop

            if target_hit and stop_hit:
                entry["status"] = "stop_hit"
                entry["resolved_pnl"] = round(-quantity * abs(stop - entry_price) * fx - fees, 2)
                entry["resolved_time"] = candle_time.isoformat()
                entry["resolution_timeframe"] = "1m"
                entry["resolution_policy"] = "conservative_stop_when_target_and_stop_share_candle"
                break
            if target_hit:
                entry["status"] = "target_hit"
                entry["resolved_pnl"] = round(quantity * abs(target - entry_price) * fx - fees, 2)
                entry["resolved_time"] = candle_time.isoformat()
                entry["resolution_timeframe"] = "1m"
                break
            if stop_hit:
                entry["status"] = "stop_hit"
                entry["resolved_pnl"] = round(-quantity * abs(stop - entry_price) * fx - fees, 2)
                entry["resolved_time"] = candle_time.isoformat()
                entry["resolution_timeframe"] = "1m"
                break
        else:
            age_hours = (now_utc - call_time).total_seconds() / 3600.0
            if age_hours >= float(expiry_hours):
                last = float(since["close"].iloc[-1])
                pnl = quantity * ((last - entry_price) if direction == "long" else (entry_price - last)) * fx - fees
                entry["status"] = "expired"
                entry["resolved_pnl"] = round(pnl, 2)
                entry["resolved_time"] = _utc(since.index[-1]).isoformat()
                entry["resolution_timeframe"] = "1m"

    cutoff = now_utc - timedelta(hours=float(ledger_max_age_hours))
    kept: list[dict[str, Any]] = []
    for entry in ledger:
        try:
            created = _utc(entry.get("time"))
        except Exception:
            created = now_utc
        if entry.get("status") == "pending" or created > cutoff:
            kept.append(entry)
            continue

        archive_record = dict(entry)
        archive_record.setdefault("archive_time", now_utc.isoformat())
        archive_record.setdefault("archive_reason", "operational_ledger_pruned")
        if research_telemetry_version:
            archive_record.setdefault("research_telemetry_version", research_telemetry_version)
        try:
            _append_jsonl(resolved_trades_file, archive_record)
        except OSError:
            # Preserve the existing scanner behavior: archive failure must not
            # resurrect an old ledger entry or crash the live scan.
            print(f"  Resolved-trade archive WARNING: unable to append {resolved_trades_file}")

    return kept
