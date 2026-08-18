
from __future__ import annotations
import json, statistics
from collections import defaultdict
import numpy as np
import pandas as pd
from .research_config import *
from .research_walkforward import split_purged

def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

def _safe_num(value):
    try:
        x = float(value)
        return x if pd.notna(x) else None
    except (TypeError, ValueError):
        return None

def _augment_grouped_features(grouped):
    grouped = sorted(
        grouped,
        key=lambda r: (_ts(r["feature_time"]), str(r.get("symbol", "")))
    )
    history = defaultdict(list)

    for row in grouped:
        symbol = str(row.get("symbol", "?"))
        feature = dict(row.get("features") or {})
        t = _ts(row["feature_time"])

        feature["hour_utc"] = int(t.hour)
        feature["minute_of_day"] = int(t.hour * 60 + t.minute)
        feature["day_of_week"] = int(t.dayofweek)
        feature["is_weekend"] = int(t.dayofweek >= 5)
        feature["session_asia_utc"] = int(0 <= t.hour < 9)
        feature["session_europe_utc"] = int(7 <= t.hour < 17)
        feature["session_us_utc"] = int(13 <= t.hour < 22)

        previous = history[symbol]
        if len(previous) >= 5:
            p = previous[-5]
            for name in (
                "rsi14","adx14","rvol20","ema9_gap_pct",
                "ema20_gap_pct","vwap_gap_pct",
                "return_5m","return_15m",
            ):
                now_v = _safe_num(feature.get(name))
                prev_v = _safe_num(p.get(name))
                feature[f"{name}_delta_5m"] = (
                    None if now_v is None or prev_v is None else now_v - prev_v
                )

            r5 = _safe_num(feature.get("return_5m"))
            r5_prev = _safe_num(p.get("return_5m"))
            feature["return_5m_accel"] = (
                None if r5 is None or r5_prev is None else r5 - r5_prev
            )

        history[symbol].append(feature)
        if len(history[symbol]) > 12:
            history[symbol].pop(0)
        row["features"] = feature

    # Same-timestamp context only; coverage is stored explicitly.
    by_time = defaultdict(list)
    for row in grouped:
        by_time[_ts(row["feature_time"])].append(row)

    for rows_at_t in by_time.values():
        total = len(rows_at_t)
        required = max(1, int(np.ceil(total * MIN_XSEC_COVERAGE)))
        # Require sufficient contemporaneous coverage before calculating market breadth.
        valid_rows = [
            row for row in rows_at_t
            if float(row["features"].get("bar_coverage_60m", 0.0) or 0.0) >= 0.95
        ]
        coverage = len(valid_rows) / total if total else 0.0

        def values(name):
            out = []
            for row in valid_rows:
                value = _safe_num(row["features"].get(name))
                if value is not None:
                    out.append(value)
            return out

        r5 = values("return_5m")
        r15 = values("return_15m")
        rvol = values("rvol20")
        adx = values("adx14")
        ema20 = values("ema20_gap_pct")

        for row in rows_at_t:
            f = row["features"]
            f["xsec_symbol_count"] = total
            f["xsec_coverage_fraction"] = coverage

            if len(valid_rows) >= required:
                f["xsec_breadth_up_5m"] = sum(x > 0 for x in r5) / len(r5) if r5 else None
                f["xsec_breadth_up_15m"] = sum(x > 0 for x in r15) / len(r15) if r15 else None
                f["xsec_trend_breadth"] = sum(x > 0 for x in ema20) / len(ema20) if ema20 else None
                f["xsec_median_return_15m"] = statistics.median(r15) if r15 else None
                f["xsec_median_rvol20"] = statistics.median(rvol) if rvol else None
                f["xsec_median_adx14"] = statistics.median(adx) if adx else None
                f["xsec_dispersion_return_15m"] = statistics.pstdev(r15) if len(r15) > 1 else None
            else:
                for key in (
                    "xsec_breadth_up_5m","xsec_breadth_up_15m","xsec_trend_breadth",
                    "xsec_median_return_15m","xsec_median_rvol20","xsec_median_adx14",
                    "xsec_dispersion_return_15m",
                ):
                    f[key] = None

    return grouped

def _group_rows(rows):
    grouped = {}
    for r in rows:
        oid = r.get("observation_id")
        if oid is None:
            raise AssertionError("Missing observation_id")
        horizon = int(r["outcome"]["horizon_min"])
        key = (oid, horizon)

        if oid not in grouped:
            grouped[oid] = {
                "observation_id": oid,
                "symbol": r.get("symbol"),
                "feature_time": r["feature_time"],
                "features": dict(r.get("feature") or {}),
                "outcomes": {},
            }
        else:
            if str(grouped[oid]["symbol"]) != str(r.get("symbol")):
                raise AssertionError(f"Observation {oid} has inconsistent symbols")
            if _ts(grouped[oid]["feature_time"]) != _ts(r["feature_time"]):
                raise AssertionError(f"Observation {oid} has inconsistent feature_time")

        if str(horizon) in grouped[oid]["outcomes"]:
            raise AssertionError(
                f"Duplicate outcome for observation {oid}, horizon {horizon}"
            )

        grouped[oid]["outcomes"][str(horizon)] = r["outcome"]

    return _augment_grouped_features(list(grouped.values()))

def _bounds(grouped):
    ts = sorted({_ts(x["feature_time"]) for x in grouped})
    if len(ts) < 60:
        return None
    return (
        ts[int(len(ts)*0.60)-1],
        ts[int(len(ts)*0.80)-1],
        ts[-1],
    )

def _split_grouped(grouped):
    bounds = _bounds(grouped)
    if not bounds:
        return [], [], []
    d, v, h = bounds
    discovery = [r for r in grouped if _ts(r["feature_time"]) < d]
    validation = [
        r for r in grouped
        if _ts(r["feature_time"]) >= d + pd.Timedelta(minutes=PURGE_MIN)
        and _ts(r["feature_time"]) < v
    ]
    holdout = [
        r for r in grouped
        if _ts(r["feature_time"]) >= v + pd.Timedelta(minutes=PURGE_MIN)
        and _ts(r["feature_time"]) < h
    ]
    for partition in (discovery, validation, holdout):
        seen = set()
        for row in partition:
            if row["observation_id"] in seen:
                raise AssertionError("Duplicate observation in partition")
            seen.add(row["observation_id"])
    return discovery, validation, holdout

def _match(feature, condition):
    value = feature.get(condition.get("feature"))
    op = condition.get("op")
    threshold = condition.get("value")
    if value is None:
        return False
    try:
        if op == "between":
            return float(threshold[0]) <= float(value) <= float(threshold[1])
        return {
            ">": float(value) > float(threshold),
            ">=": float(value) >= float(threshold),
            "<": float(value) < float(threshold),
            "<=": float(value) <= float(threshold),
        }[op]
    except (TypeError, ValueError, KeyError, IndexError):
        return False

def _metrics(values, mfes, maes):
    if not values:
        return {
            "n":0,"positive_rate":0.0,"avg_net_return":0.0,
            "median_net_return":0.0,"profit_factor":0.0,"worst_return":0.0,
            "avg_mfe":0.0,"avg_mae":0.0,"max_drawdown":0.0,"return_std":0.0,
        }
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v < 0]
    gp = sum(wins)
    gl = -sum(losses)
    pf = gp / gl if gl else (999.0 if gp else 0.0)
    eq = peak = dd = 0.0
    for v in values:
        eq += v
        peak = max(peak, eq)
        dd = max(dd, peak-eq)
    return {
        "n":len(values),
        "positive_rate":len(wins)/len(values),
        "avg_net_return":sum(values)/len(values),
        "median_net_return":statistics.median(values),
        "profit_factor":pf,
        "worst_return":min(values),
        "avg_mfe":sum(mfes)/len(mfes) if mfes else 0.0,
        "avg_mae":sum(maes)/len(maes) if maes else 0.0,
        "max_drawdown":dd,
        "return_std":statistics.pstdev(values) if len(values)>1 else 0.0,
    }

def _robustness(rule, grouped):
    horizon = int(rule.get("horizon_min", 15))
    direction = str(rule.get("direction", "long")).lower()
    return_key = "net_return_short" if direction == "short" else "net_return_long"

    matches = []
    for row in grouped:
        outcome = row["outcomes"].get(str(horizon))
        if not outcome:
            continue
        if all(_match(row["features"], c) for c in rule.get("conditions", [])):
            matches.append((
                _ts(row["feature_time"]),
                str(row.get("symbol","?")),
                float(outcome.get(return_key, 0.0)),
            ))

    if not matches:
        return {
            "raw_n":0,"coin_count":0,"positive_coin_fraction":0.0,
            "time_block_count":0,"positive_block_fraction":0.0,
            "block_mean":0.0,"block_std":0.0,"block_ci95_low":0.0,
            "stress_avg_net_return":{},
            "effective_sample_note":"raw observations are overlapping; block breadth is diagnostic",
        }

    by_coin = defaultdict(list)
    by_block = defaultdict(list)
    for t, symbol, value in matches:
        by_coin[symbol].append(value)
        by_block[t.floor(f"{RESEARCH_BLOCK_MINUTES}min")].append(value)

    coin_means = [sum(v)/len(v) for v in by_coin.values()]
    block_means = [sum(v)/len(v) for v in by_block.values()]
    block_mean = sum(block_means)/len(block_means)
    block_std = statistics.pstdev(block_means) if len(block_means)>1 else 0.0
    block_se = block_std / np.sqrt(len(block_means)) if len(block_means)>1 else 0.0
    ci_low = block_mean - 1.96 * block_se
    stress = {}
    for multiplier in STRESS_COST_MULTIPLIERS:
        extra = max(0.0, (multiplier - 1.0) * TAKER_FEE_RATE * 2.0)
        stress[str(multiplier)] = sum(v-extra for _,_,v in matches) / len(matches)

    return {
        "raw_n":len(matches),
        "coin_count":len(by_coin),
        "positive_coin_fraction":sum(x>0 for x in coin_means)/len(coin_means),
        "time_block_count":len(block_means),
        "positive_block_fraction":sum(x>0 for x in block_means)/len(block_means),
        "block_mean":block_mean,
        "block_std":block_std,
        "block_ci95_low":ci_low,
        "stress_avg_net_return":stress,
        "effective_sample_note":"raw observations are overlapping; block statistics are used for robustness, not independence",
    }

def evaluate_one(rule, grouped):
    horizon = int(rule.get("horizon_min", 15))
    direction = str(rule.get("direction", "long")).lower()
    key = "net_return_short" if direction == "short" else "net_return_long"
    mk = "mfe_short" if direction == "short" else "mfe_long"
    ak = "mae_short" if direction == "short" else "mae_long"

    values=[]; mfes=[]; maes=[]
    for row in sorted(grouped,key=lambda x:_ts(x["feature_time"])):
        outcome = row["outcomes"].get(str(horizon))
        if not outcome:
            continue
        if all(_match(row["features"], c) for c in rule.get("conditions", [])):
            values.append(float(outcome.get(key,0.0)))
            mfes.append(float(outcome.get(mk,0.0)))
            maes.append(float(outcome.get(ak,0.0)))

    result = _metrics(values, mfes, maes)
    result["robustness"] = _robustness(rule, grouped)
    return result

def _validation_rank(item):
    v = item["validation"]
    r = v.get("robustness", {}) or {}
    if v["n"] <= 0:
        return (
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
        )

    def _rank_float(value, default=float("-inf")):
        try:
            number = float(value)
            return number if np.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    return (
        _rank_float(r.get("block_ci95_low")),
        _rank_float(v.get("profit_factor")),
        _rank_float(v.get("avg_net_return")),
        _rank_float(r.get("positive_block_fraction"), 0.0),
        _rank_float(r.get("positive_coin_fraction"), 0.0),
    )

def evaluate(hypotheses, rows, holdout_budget=None):
    grouped = _group_rows(rows)
    _, validation, holdout = _split_grouped(grouped)
    if holdout_budget is None:
        holdout_budget = HOLDOUT_TEST_BUDGET_PER_EPOCH
    holdout_budget = max(0, int(holdout_budget))

    all_validation = []
    for h in hypotheses:
        all_validation.append({
            "hypothesis": h,
            "validation": evaluate_one(h, validation),
        })

    finalists = sorted(
        all_validation,
        key=_validation_rank,
        reverse=True,
    )[:min(HOLDOUT_FINALISTS, holdout_budget)]

    finalist_ids = {
        json.dumps(x["hypothesis"], sort_keys=True, default=str)
        for x in finalists
    }

    result = []
    for item in all_validation:
        key = json.dumps(item["hypothesis"], sort_keys=True, default=str)
        if key in finalist_ids:
            hold_metrics = evaluate_one(item["hypothesis"], holdout)
        else:
            hold_metrics = {
                "n":0,
                "positive_rate":0.0,
                "avg_net_return":0.0,
                "median_net_return":0.0,
                "profit_factor":0.0,
                "worst_return":0.0,
                "avg_mfe":0.0,
                "avg_mae":0.0,
                "max_drawdown":0.0,
                "return_std":0.0,
                "robustness": {
                    "not_tested": True
                },
            }
        result.append({
            "hypothesis": item["hypothesis"],
            "validation": item["validation"],
            "holdout": hold_metrics,
            "holdout_tested": key in finalist_ids,
        })
    return result

def classify(result):
    v = result["validation"]
    h = result["holdout"]
    vr = v.get("robustness", {})
    hr = h.get("robustness", {})

    validation_robust = (
        vr.get("time_block_count",0) >= MIN_VALIDATION_TIME_BLOCKS
        and vr.get("positive_block_fraction",0.0) >= MIN_POSITIVE_BLOCK_FRACTION
        and vr.get("positive_coin_fraction",0.0) >= MIN_POSITIVE_COIN_FRACTION
        and vr.get("block_ci95_low", -1e99) > 0.0
    )
    holdout_robust = (
        result.get("holdout_tested", False)
        and hr.get("time_block_count",0) >= MIN_HOLDOUT_TIME_BLOCKS
        and hr.get("positive_block_fraction",0.0) >= MIN_POSITIVE_BLOCK_FRACTION
        and hr.get("positive_coin_fraction",0.0) >= MIN_POSITIVE_COIN_FRACTION
        and hr.get("block_ci95_low", -1e99) > 0.0
        and hr.get("stress_avg_net_return",{}).get(
            str(max(STRESS_COST_MULTIPLIERS)),
            -1e99,
        ) > 0.0
    )

    if (
        v["n"] >= PROMOTE_MIN_VALID_N
        and h["n"] >= PROMOTE_MIN_HOLDOUT_N
        and v["profit_factor"] >= PROMOTE_MIN_VALID_PF
        and h["profit_factor"] >= PROMOTE_MIN_HOLDOUT_PF
        and v["avg_net_return"] >= PROMOTE_MIN_VALID_AVG
        and h["avg_net_return"] >= PROMOTE_MIN_HOLDOUT_AVG
        and h["positive_rate"] >= PROMOTE_MIN_HOLDOUT_WINRATE
        and h["max_drawdown"] <= PROMOTE_MAX_HOLDOUT_DD
        and validation_robust
        and holdout_robust
    ):
        return "HOLDOUT_PASSED"

    if (
        v["n"] >= PROMOTE_MIN_VALID_N
        and v["profit_factor"] >= PROMOTE_MIN_VALID_PF
        and v["avg_net_return"] >= PROMOTE_MIN_VALID_AVG
        and validation_robust
    ):
        return "VALIDATION_PASSED"

    return "VALIDATION_FAILED"

def discover(api_key, rows, model, memory=None):
    from .research_discovery import discover_multi
    return discover_multi(api_key, rows, model, memory)
