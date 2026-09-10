#!/usr/bin/env python3
"""Capture compact market-behaviour events and their future outcomes.

This module deliberately does NOT discover or alter trading rules.  It records
observable 1m CoinDCX futures behaviour so future research can measure whether
support/rejection, resistance/rejection, absorption, failed breaks and long
wicks are followed by useful price behaviour.

Important: ``delta_proxy`` is a signed-volume proxy derived from the direction
of the 1m OHLC candle.  It is NOT exchange aggressor-side delta/CVD.  The field
names and schema keep that distinction explicit so this dataset cannot be
mistaken for true trade-level order-flow data.

Storage is append-only at the observation/outcome level and sharded by UTC
hour.  Pending observations live in a small state file.  No monolithic JSONL
file is ever rewritten, avoiding the GitHub 100 MB failure mode that killed the
former ResearchLab.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

BASE_URL = "https://public.coindcx.com/market_data/candles"
INTERVAL_MS = 60_000
DEFAULT_COINS = "BTC,ETH,BNB,SOL,XRP,DOGE,LTC,LINK,TRX,AVAX,HEI,BICO,HYPE,ZEC,ZBT,ADA,ACE,PAXG"
HORIZONS_MIN = (5, 15, 30, 60, 120, 180)
LOOKBACK_MIN = 240
RETENTION_DAYS = 90
COOLDOWN_MIN = 10
SCHEMA_VERSION = "market_behavior_v1"


@dataclass(frozen=True)
class Bar:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def ts(self) -> datetime:
        return datetime.fromtimestamp(self.ts_ms / 1000, tz=timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def parse_coin_list(value: str) -> list[str]:
    coins = [c.strip().upper() for c in value.split(",") if c.strip()]
    if not coins:
        raise ValueError("No coins supplied")
    return list(dict.fromkeys(coins))


def shard_for(ts_ms: int, root: Path, kind: str) -> Path:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return root / kind / dt.strftime("%Y-%m-%d") / f"{dt.strftime('%H')}.jsonl"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
        return value
    except (OSError, ValueError, TypeError):
        return default


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    rows = list(records)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    return len(rows)


def fetch_1m(symbol: str, lookback_min: int = LOOKBACK_MIN, timeout: int = 20) -> list[Bar]:
    """Fetch closed CoinDCX futures 1m candles, newest first or unsorted."""
    now_ms = int(utc_now().timestamp() * 1000)
    # Move the right edge to the last completed minute so the forming candle
    # can never leak into an observation or a future outcome.
    end_ms = (now_ms // INTERVAL_MS) * INTERVAL_MS
    start_ms = end_ms - int(lookback_min * INTERVAL_MS)
    params = {
        "pair": f"B-{symbol}_USDT",
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": min(1000, lookback_min + 5),
    }
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=timeout)
            if resp.status_code == 422:
                raise RuntimeError(f"CoinDCX rejected {symbol} 1m request (HTTP 422)")
            resp.raise_for_status()
            raw = resp.json()
            if not isinstance(raw, list):
                raise ValueError(f"Unexpected candle payload for {symbol}")
            bars: list[Bar] = []
            for row in raw:
                if not isinstance(row, dict):
                    continue
                ts = row.get("time")
                vals = [safe_float(row.get(k)) for k in ("open", "high", "low", "close", "volume")]
                if ts is None or any(v is None for v in vals):
                    continue
                try:
                    ts_ms = int(ts)
                except (TypeError, ValueError):
                    continue
                if ts_ms >= end_ms:
                    continue
                bars.append(Bar(ts_ms, vals[0], vals[1], vals[2], vals[3], max(vals[4], 0.0)))
            unique = {b.ts_ms: b for b in bars}
            return [unique[k] for k in sorted(unique)]
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_exc = exc
            if isinstance(exc, RuntimeError) and "422" in str(exc):
                break
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to fetch {symbol} 1m candles: {last_exc}")


def median(values: list[float], fallback: float = 0.0) -> float:
    values = [x for x in values if math.isfinite(x)]
    return statistics.median(values) if values else fallback


def percentile_rank(value: float, values: list[float]) -> float:
    vals = sorted(x for x in values if math.isfinite(x))
    if not vals:
        return 0.0
    return sum(1 for x in vals if x <= value) / len(vals) * 100.0


def _event_id(symbol: str, ts_ms: int, families: list[str]) -> str:
    raw = f"{symbol}|{ts_ms}|{'|'.join(families)}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def build_event(symbol: str, bars: list[Bar]) -> dict[str, Any] | None:
    if len(bars) < 70:
        return None
    cur = bars[-1]
    prior = bars[-61:-1]
    prior30 = bars[-31:-1]
    ranges = [max(b.high - b.low, 0.0) for b in prior]
    vols = [b.volume for b in prior]
    atr = median(ranges[-20:], fallback=max(cur.high - cur.low, cur.close * 0.001))
    vol_base = median(vols[-60:], fallback=max(cur.volume, 1e-12))
    rng = max(cur.high - cur.low, 1e-12)
    body = abs(cur.close - cur.open)
    upper_wick = max(cur.high - max(cur.open, cur.close), 0.0)
    lower_wick = max(min(cur.open, cur.close) - cur.low, 0.0)
    close_position = (cur.close - cur.low) / rng
    delta_proxy_qty = cur.volume if cur.close > cur.open else (-cur.volume if cur.close < cur.open else 0.0)
    delta_proxy_notional = delta_proxy_qty * cur.close
    delta_proxy_ratio = delta_proxy_qty / cur.volume if cur.volume > 0 else 0.0
    activity_ratio = cur.volume / vol_base if vol_base > 0 else 1.0
    atr_pct = atr / cur.close * 100.0 if cur.close else 0.0
    prior_low = min(b.low for b in prior30)
    prior_high = max(b.high for b in prior30)
    level_tol = max(atr * 0.35, cur.close * 0.0015)
    near_support = abs(cur.low - prior_low) <= level_tol
    near_resistance = abs(cur.high - prior_high) <= level_tol
    failed_break_support = cur.low < prior_low and cur.close >= prior_low
    failed_break_resistance = cur.high > prior_high and cur.close <= prior_high
    long_lower_wick = lower_wick / rng >= 0.45 and close_position >= 0.60
    long_upper_wick = upper_wick / rng >= 0.45 and close_position <= 0.40
    high_activity = activity_ratio >= 1.80
    bearish_flow = delta_proxy_ratio <= -0.45
    bullish_flow = delta_proxy_ratio >= 0.45
    body_small = body / rng <= 0.45

    families: list[str] = []
    direction: str | None = None
    level: float | None = None
    if failed_break_support:
        families.append("failed_break_support")
        direction = "bullish"
        level = prior_low
    if failed_break_resistance:
        families.append("failed_break_resistance")
        direction = "bearish"
        level = prior_high
    if near_support and long_lower_wick:
        families.append("long_wick_support_rejection")
        direction = "bullish"
        level = prior_low if level is None else level
    if near_resistance and long_upper_wick:
        families.append("long_wick_resistance_rejection")
        direction = "bearish"
        level = prior_high if level is None else level
    if near_support and high_activity and bearish_flow and body_small and close_position >= 0.50:
        families.append("negative_flow_absorption_support")
        direction = "bullish"
        level = prior_low if level is None else level
    if near_resistance and high_activity and bullish_flow and body_small and close_position <= 0.50:
        families.append("positive_flow_absorption_resistance")
        direction = "bearish"
        level = prior_high if level is None else level

    if not families:
        return None

    score = min(5, 1 + len(families) + int(high_activity) + int(abs(delta_proxy_ratio) >= 0.65))
    families = list(dict.fromkeys(families))
    event_direction = direction or ("bullish" if close_position >= 0.5 else "bearish")
    level_type = "support" if event_direction == "bullish" else "resistance"
    level = float(level if level is not None else cur.close)

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": _event_id(symbol, cur.ts_ms, families),
        "observed_at": iso(cur.ts_ms),
        "observed_at_ms": cur.ts_ms,
        "symbol": symbol,
        "event_families": families,
        "direction": event_direction,
        "level_type": level_type,
        "reference_level": level,
        "open": cur.open,
        "high": cur.high,
        "low": cur.low,
        "close": cur.close,
        "volume": cur.volume,
        "range": rng,
        "body": body,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "upper_wick_ratio": upper_wick / rng,
        "lower_wick_ratio": lower_wick / rng,
        "close_position": close_position,
        "activity_ratio_vs_60m_median": activity_ratio,
        "activity_percentile_60m": percentile_rank(cur.volume, vols),
        "atr_20_abs": atr,
        "atr_20_pct": atr_pct,
        "delta_proxy_qty": delta_proxy_qty,
        "delta_proxy_notional": delta_proxy_notional,
        "delta_proxy_ratio": delta_proxy_ratio,
        "delta_proxy_source": "candle_direction_signed_volume",
        "prior_30m_low": prior_low,
        "prior_30m_high": prior_high,
        "distance_to_support_pct": abs(cur.close - prior_low) / cur.close * 100.0 if cur.close else None,
        "distance_to_resistance_pct": abs(cur.close - prior_high) / cur.close * 100.0 if cur.close else None,
        "failed_break_support": failed_break_support,
        "failed_break_resistance": failed_break_resistance,
        "event_score": score,
    }


def _future_slice(bars: list[Bar], observed_ms: int, horizon_min: int) -> list[Bar]:
    end = observed_ms + horizon_min * INTERVAL_MS
    return [b for b in bars if observed_ms < b.ts_ms <= end]


def resolve_outcome(event: dict[str, Any], bars: list[Bar], horizon_min: int) -> dict[str, Any] | None:
    future = _future_slice(bars, int(event["observed_at_ms"]), horizon_min)
    if len(future) < horizon_min:
        return None
    entry = float(event["close"])
    bullish = event["direction"] == "bullish"
    final_close = future[-1].close
    raw_return = (final_close - entry) / entry * 100.0 if entry else 0.0
    signed_return = raw_return if bullish else -raw_return
    if bullish:
        mfe = (max(b.high for b in future) - entry) / entry * 100.0
        mae = (entry - min(b.low for b in future)) / entry * 100.0
        level_held = min(b.low for b in future) >= float(event["reference_level"]) * 0.9985
        level_reclaimed = any(b.close >= float(event["reference_level"]) for b in future)
    else:
        mfe = (entry - min(b.low for b in future)) / entry * 100.0
        mae = (max(b.high for b in future) - entry) / entry * 100.0
        level_held = max(b.high for b in future) <= float(event["reference_level"]) * 1.0015
        level_reclaimed = any(b.close <= float(event["reference_level"]) for b in future)
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event["event_id"],
        "observed_at": event["observed_at"],
        "symbol": event["symbol"],
        "event_families": event["event_families"],
        "direction": event["direction"],
        "horizon_min": horizon_min,
        "outcome_at": iso(future[-1].ts_ms),
        "forward_return_pct": raw_return,
        "signed_forward_return_pct": signed_return,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "level_held": level_held,
        "level_reclaimed": level_reclaimed,
        "future_bars": len(future),
    }


def prune_old(root: Path, now: datetime) -> None:
    cutoff = now - timedelta(days=RETENTION_DAYS)
    for kind in ("observations", "outcomes"):
        base = root / kind
        if not base.exists():
            continue
        for path in base.rglob("*.jsonl"):
            try:
                if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff:
                    path.unlink()
            except OSError:
                continue


def record_run(
    coins: list[str],
    root: Path,
    state_path: Path,
    lookback_min: int = LOOKBACK_MIN,
) -> dict[str, Any]:
    now = utc_now()
    state = load_json(state_path, {"schema_version": 1, "pending": {}, "last_event_ms": {}})
    pending: dict[str, dict[str, Any]] = dict(state.get("pending") or {})
    last_event_ms: dict[str, int] = {k: int(v) for k, v in (state.get("last_event_ms") or {}).items()}
    observed_count = 0
    outcome_count = 0
    fetch_failures: list[dict[str, str]] = []
    by_symbol: dict[str, list[Bar]] = {}

    for coin in coins:
        try:
            bars = fetch_1m(coin, lookback_min=lookback_min)
            by_symbol[coin] = bars
        except Exception as exc:  # research must never break production scanning
            fetch_failures.append({"symbol": coin, "error": str(exc)[:240]})
            continue

        event = build_event(coin, bars)
        if event is not None:
            last_ms = last_event_ms.get(coin, 0)
            if int(event["observed_at_ms"]) - last_ms >= COOLDOWN_MIN * INTERVAL_MS:
                obs_path = shard_for(int(event["observed_at_ms"]), root, "observations")
                observed_count += append_jsonl(obs_path, [event])
                pending[event["event_id"]] = event
                last_event_ms[coin] = int(event["observed_at_ms"])

    remaining: dict[str, dict[str, Any]] = {}
    outcome_groups: dict[Path, list[dict[str, Any]]] = {}
    for event_id, event in pending.items():
        bars = by_symbol.get(event["symbol"])
        if not bars:
            remaining[event_id] = event
            continue
        emitted_any = False
        for horizon in HORIZONS_MIN:
            outcome = resolve_outcome(event, bars, horizon)
            if outcome is None:
                continue
            outcome_groups.setdefault(shard_for(int(event["observed_at_ms"]), root, "outcomes"), []).append(outcome)
            emitted_any = True
        age_min = (now - datetime.fromtimestamp(int(event["observed_at_ms"]) / 1000, tz=timezone.utc)).total_seconds() / 60.0
        if age_min < max(HORIZONS_MIN) or not emitted_any:
            remaining[event_id] = event

    for path, rows in outcome_groups.items():
        # Idempotency within a shard: outcomes are keyed by event_id+horizon.
        existing = set()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            rec = json.loads(line)
                            existing.add(f"{rec.get('event_id')}|{rec.get('horizon_min')}")
            except (OSError, ValueError, TypeError):
                existing = set()
        fresh = [r for r in rows if f"{r['event_id']}|{r['horizon_min']}" not in existing]
        outcome_count += append_jsonl(path, fresh)

    state = {
        "schema_version": 1,
        "updated_at": now.isoformat(),
        "last_event_ms": last_event_ms,
        "pending": remaining,
        "stats": {
            "observations_written_this_run": observed_count,
            "outcomes_written_this_run": outcome_count,
            "pending_events": len(remaining),
            "fetch_failures": len(fetch_failures),
        },
    }
    write_json_atomic(state_path, state)
    prune_old(root, now)
    return {
        "observations_written": observed_count,
        "outcomes_written": outcome_count,
        "pending_events": len(remaining),
        "fetch_failures": fetch_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", default=DEFAULT_COINS)
    parser.add_argument("--root", default="market_behavior_research")
    parser.add_argument("--state", default="market_behavior_state.json")
    parser.add_argument("--lookback-min", type=int, default=LOOKBACK_MIN)
    args = parser.parse_args()
    result = record_run(parse_coin_list(args.coins), Path(args.root), Path(args.state), args.lookback_min)
    print(json.dumps(result, indent=2, sort_keys=True))
    # Fetch gaps are expected to be observable but must never break the scanner.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
