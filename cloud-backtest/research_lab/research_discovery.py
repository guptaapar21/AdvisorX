from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests

from .research_hypotheses import _group_rows, _split_grouped


# Conservative defaults for Gemini free-tier input quota.
# The full dataset remains available to Python validation/holdout; only a
# balanced discovery sample is sent to Gemini.
DISCOVERY_PASSES = int(os.getenv("RESEARCH_DISCOVERY_PASSES", "3"))
DISCOVERY_PASS_OBSERVATIONS = int(
    os.getenv("RESEARCH_DISCOVERY_PASS_OBSERVATIONS", "200")
)
DISCOVERY_INPUT_CHAR_BUDGET = int(
    os.getenv("RESEARCH_DISCOVERY_INPUT_CHAR_BUDGET", "120000")
)
MAX_HYPOTHESES_PER_PASS = int(
    os.getenv("RESEARCH_MAX_HYPOTHESES_PER_PASS", "10")
)
MAX_TOTAL_HYPOTHESES = int(
    os.getenv("RESEARCH_MAX_TOTAL_HYPOTHESES", "60")
)
REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("RESEARCH_GEMINI_TIMEOUT_SECONDS", "45")
)

SYSTEM = """You are a neutral market research scientist, not a trader.
Use ONLY the DISCOVERY sample supplied in this request.
Do not use validation or holdout rows; Python tests those later.
Do not recommend live changes. Consider LONG and SHORT symmetrically.
Search for simple, interpretable, repeatable patterns supported across multiple
coins and time periods. Prefer small, consistent NET returns after fees over
rare large winners. Return JSON only:
{"hypotheses":[
 {"name":"...","direction":"LONG|SHORT","horizon_min":5|10|15|30|60,
  "conditions":[{"feature":"...","op":">|>=|<|<=|between","value":...}],
  "rationale":"..."}
]}"""

FEATURE_KEYS = (
    "close", "return_1m", "return_3m", "return_5m", "return_10m",
    "return_15m", "return_30m", "return_60m",
    "volatility_5m", "volatility_15m", "volatility_30m",
    "volatility_60m", "rvol20", "rvol60", "atr14_pct",
    "candle_range_pct", "body_to_range", "upper_wick_to_range",
    "lower_wick_to_range", "close_location", "ema9_gap_pct",
    "ema20_gap_pct", "ema50_gap_pct", "ema9_20_gap_pct",
    "rsi14", "adx14", "plus_di14", "minus_di14",
    "adx_slope_5m", "vwap_gap_pct",
    "distance_from_30m_high", "distance_from_30m_low",
    "distance_from_60m_high", "distance_from_60m_low",
    "efficiency_20",
)
HORIZONS = (5, 10, 15, 30, 60)


def _num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _compact_observation(row):
    features = row.get("features") or {}
    outcomes = row.get("outcomes") or {}
    values = [_num(features.get(key)) for key in FEATURE_KEYS]

    for horizon in HORIZONS:
        outcome = outcomes.get(str(horizon), {}) or {}
        for side in ("long", "short"):
            values.extend(
                [
                    _num(outcome.get(f"net_return_{side}")),
                    _num(outcome.get(f"mfe_{side}")),
                    _num(outcome.get(f"mae_{side}")),
                ]
            )

    return [row.get("symbol", "?"), row.get("feature_time", ""), *values]


def _prompt(sample, memory):
    header = ["symbol", "feature_time", *FEATURE_KEYS]
    for horizon in HORIZONS:
        for side in ("long", "short"):
            for metric in ("net_return", "mfe", "mae"):
                header.append(f"{side}_{metric}_{horizon}")

    return "\n".join(
        [
            SYSTEM,
            "Column order:",
            json.dumps(header, separators=(",", ":")),
            "Rows are compact arrays in that exact order. Missing numbers are null.",
            json.dumps({"memory": memory or {}}, separators=(",", ":"), default=str),
            json.dumps(sample, separators=(",", ":"), default=str),
        ]
    )


def _evenly_spaced(rows, count):
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    if count == 1:
        return [rows[len(rows) // 2]]
    return [
        rows[int(i * (len(rows) - 1) / (count - 1))]
        for i in range(count)
    ]


def _sample_pass(rows, pass_index, pass_count, target):
    buckets = {}
    for row in rows:
        buckets.setdefault(str(row.get("symbol", "?")), []).append(row)

    for symbol_rows in buckets.values():
        symbol_rows.sort(key=lambda row: pd.Timestamp(row["feature_time"]))

    symbols = sorted(buckets)
    if not symbols:
        return []

    per_symbol = max(1, target // len(symbols))
    selected = []

    for symbol in symbols:
        symbol_rows = buckets[symbol]
        n = len(symbol_rows)
        start = (n * pass_index) // pass_count
        end = (n * (pass_index + 1)) // pass_count
        selected.extend(
            _evenly_spaced(symbol_rows[start:end], per_symbol)
        )

    return selected[:target]


def _parse_json(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {"hypotheses": []}

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {"hypotheses": []}
    except json.JSONDecodeError:
        return {"hypotheses": []}


def _candidate_signature(hypothesis):
    conditions = []
    for condition in hypothesis.get("conditions", []) or []:
        conditions.append(
            (
                str(condition.get("feature", "")),
                str(condition.get("op", "")),
                json.dumps(
                    condition.get("value"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )

    return json.dumps(
        (
            str(hypothesis.get("direction", "")).upper(),
            int(hypothesis.get("horizon_min", 0) or 0),
            sorted(conditions),
        ),
        separators=(",", ":"),
    )


def _build_payload(sample, memory):
    return _prompt(
        [_compact_observation(row) for row in sample],
        memory,
    )


def _find_budgeted_sample(rows, target, memory):
    requested = min(target, len(rows))
    while requested >= 20:
        sample = rows if requested == len(rows) else _sample_pass(
            rows, 0, 1, requested
        )
        prompt = _build_payload(sample, memory)
        if len(prompt) <= DISCOVERY_INPUT_CHAR_BUDGET:
            return sample, prompt

        requested = max(20, requested // 2)

    return [], ""


def _discover_one(key, sample, model, memory):
    prompt = _build_payload(sample, memory)

    if not prompt or len(prompt) > DISCOVERY_INPUT_CHAR_BUDGET:
        return {
            "status": "payload_too_large",
            "hypotheses": [],
            "prompt_chars": len(prompt),
            "sample_observations": len(sample),
        }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )

    try:
        response = requests.post(
            url,
            params={"key": key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return {
            "status": "request_error",
            "error": str(exc),
            "hypotheses": [],
        }

    if response.status_code == 429:
        retry_after = None
        match = re.search(
            r"retry in\s+([0-9.]+)s",
            response.text or "",
            re.IGNORECASE,
        )
        if match:
            retry_after = float(match.group(1))

        return {
            "status": "rate_limited",
            "retry_after_seconds": retry_after,
            "message": (response.text or "")[:500],
            "hypotheses": [],
        }

    if response.status_code >= 500:
        return {
            "status": f"server_{response.status_code}",
            "message": (response.text or "")[:500],
            "hypotheses": [],
        }

    try:
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {
            "status": "response_error",
            "error": str(exc),
            "hypotheses": [],
        }

    text = "".join(
        part.get("text", "")
        for candidate in data.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
    )

    clean = []
    for hypothesis in _parse_json(text).get("hypotheses", [])[:MAX_HYPOTHESES_PER_PASS]:
        if not isinstance(hypothesis, dict):
            continue

        direction = str(hypothesis.get("direction", "")).upper()
        if direction not in {"LONG", "SHORT"}:
            continue

        try:
            horizon = int(hypothesis.get("horizon_min"))
        except (TypeError, ValueError):
            continue

        if horizon not in HORIZONS:
            continue

        if not isinstance(hypothesis.get("conditions"), list):
            continue
        if not hypothesis["conditions"]:
            continue

        hypothesis["direction"] = direction
        hypothesis["horizon_min"] = horizon
        clean.append(hypothesis)

    return {
        "status": "ok",
        "hypotheses": clean,
        "prompt_chars": len(prompt),
        "sample_observations": len(sample),
    }


def discover_multi(api_key, rows, model, memory=None):
    keys = [key.strip() for key in str(api_key or "").split(",") if key.strip()]

    grouped = _group_rows(rows)
    discovery_rows, validation_rows, holdout_rows = _split_grouped(grouped)

    if (
        len(discovery_rows) < 300
        or len(validation_rows) < 100
        or len(holdout_rows) < 100
    ):
        return {
            "status": "insufficient_data",
            "discovery_observations": len(discovery_rows),
            "validation_observations": len(validation_rows),
            "holdout_observations": len(holdout_rows),
            "hypotheses": [],
        }

    if not keys:
        return {"status": "no_api_key", "hypotheses": []}

    pass_count = max(1, min(DISCOVERY_PASSES, len(keys)))
    results = []

    # Sequential passes are deliberate. Concurrent Gemini calls can burst the
    # project's shared free-tier input quota even when individual requests fit.
    for pass_index in range(pass_count):
        sample_seed = _sample_pass(
            discovery_rows,
            pass_index,
            pass_count,
            DISCOVERY_PASS_OBSERVATIONS,
        )
        sample, _ = _find_budgeted_sample(
            sample_seed,
            min(DISCOVERY_PASS_OBSERVATIONS, len(sample_seed)),
            memory or {},
        )

        if not sample:
            results.append({
                "status": "payload_too_large",
                "hypotheses": [],
                "sample_observations": 0,
            })
            continue

        try:
            results.append(
                _discover_one(
                    keys[pass_index],
                    sample,
                    model,
                    memory or {},
                )
            )
        except Exception as exc:
            results.append({
                "status": "error",
                "error": str(exc),
                "hypotheses": [],
            })

    merged = []
    seen = set()

    for result in results:
        for hypothesis in result.get("hypotheses", []):
            signature = _candidate_signature(hypothesis)
            if signature in seen:
                continue

            seen.add(signature)
            merged.append(hypothesis)

            if len(merged) >= MAX_TOTAL_HYPOTHESES:
                break

        if len(merged) >= MAX_TOTAL_HYPOTHESES:
            break

    successful_passes = sum(
        1
        for result in results
        if result.get("status") == "ok"
    )
    rate_limited_passes = sum(
        1
        for result in results
        if result.get("status") == "rate_limited"
    )

    pass_statuses = [
        result.get("status", "missing")
        for result in results
    ]

    if successful_passes == 0 and rate_limited_passes:
        return {
            "status": "quota_deferred",
            "discovery_observations": len(discovery_rows),
            "validation_observations": len(validation_rows),
            "holdout_observations": len(holdout_rows),
            "discovery_passes": pass_count,
            "successful_passes": 0,
            "rate_limited_passes": rate_limited_passes,
            "pass_statuses": pass_statuses,
            "hypotheses": [],
        }

    if successful_passes == 0:
        return {
            "status": "discovery_failed",
            "discovery_observations": len(discovery_rows),
            "validation_observations": len(validation_rows),
            "holdout_observations": len(holdout_rows),
            "discovery_passes": pass_count,
            "successful_passes": 0,
            "rate_limited_passes": rate_limited_passes,
            "pass_statuses": pass_statuses,
            "hypotheses": [],
        }

    return {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "hypotheses": merged,
        "discovery_observations": len(discovery_rows),
        "discovery_sample_observations": sum(
            result.get("sample_observations", 0)
            for result in results
        ),
        "discovery_passes": pass_count,
        "successful_passes": successful_passes,
        "rate_limited_passes": rate_limited_passes,
        "pass_statuses": pass_statuses,
        "discovery_input_char_budget": DISCOVERY_INPUT_CHAR_BUDGET,
        "per_pass_observations": DISCOVERY_PASS_OBSERVATIONS,
        "max_hypotheses_per_pass": MAX_HYPOTHESES_PER_PASS,
        "discovery_only": True,
    }
