from __future__ import annotations
import json, os
from datetime import datetime, timezone
import pandas as pd
from .research_config import *
from .research_time import utc_ts, completed_minute, canonical_minute
from .research_fetcher import fetch_1m
from .research_features import feature_snapshot
from .outcomes import label_observation
from .research_store import append_jsonl, read_jsonl, read_state, write_state, load_memory, write_memory
from .research_discovery import discover_multi
from .research_hypotheses import evaluate, classify

def _observation_id(symbol,t):
    return f"{symbol}:{canonical_minute(t).strftime('%Y%m%dT%H%M%SZ')}"

def _candidate_signature(h):
    conditions=[]
    for c in h.get("conditions",[]) or []:
        conditions.append((
            str(c.get("feature","")),
            str(c.get("op","")),
            json.dumps(c.get("value"),sort_keys=True,separators=(",",":")),
        ))
    return json.dumps(
        (str(h.get("direction","")).upper(),int(h.get("horizon_min",0) or 0),sorted(conditions)),
        separators=(",",":")
    )

def collect_once(now=None, fetch_fn=fetch_1m):
    now=utc_ts(now)
    cutoff=completed_minute(now)
    state=read_state()
    observations=read_jsonl(OBS_FILE)
    outcomes=read_jsonl(OUTCOME_FILE)
    known={x.get("observation_id") for x in observations}
    outcome_keys={(x.get("observation_id"),int(x.get("horizon_min",-1))) for x in outcomes}
    candles_by_symbol={}

    for symbol in COINS:
        try:
            df=fetch_fn(
                symbol,
                cutoff-pd.Timedelta(minutes=OBSERVATION_LOOKBACK_MIN),
                cutoff,
            )
            if df.empty:
                continue
            df.index=pd.to_datetime(df.index,utc=True)
            df=df[df.index<=cutoff].sort_index()
            candles_by_symbol[symbol]=df

            for t in df.index:
                if t>cutoff:
                    continue
                try:
                    f=feature_snapshot(df,t)
                except ValueError:
                    continue
                if float(f.get("bar_coverage_60m",0.0) or 0.0)<0.95:
                    continue
                oid=_observation_id(symbol,t)
                if oid in known:
                    continue
                append_jsonl(
                    OBS_FILE,
                    {"observation_id":oid,"symbol":symbol,"features":f}
                )
                known.add(oid)

        except Exception as exc:
            append_jsonl(
                ERROR_FILE,
                {
                    "time":now.isoformat(),
                    "symbol":symbol,
                    "stage":"collect",
                    "error":str(exc),
                }
            )

    all_obs=read_jsonl(OBS_FILE)

    for obs in all_obs:
        oid=obs["observation_id"]
        symbol=obs["symbol"]
        t=canonical_minute(obs["features"]["feature_time"])

        if t+pd.Timedelta(minutes=MAX_HORIZON_MIN)>cutoff:
            continue

        if all(
            (oid,int(h)) in outcome_keys
            for h in HORIZONS_MIN
        ):
            continue

        c=candles_by_symbol.get(symbol)
        if c is None or c.empty:
            continue

        for y in label_observation(
            c,
            t,
            now=cutoff,
            fee_rate=TAKER_FEE_RATE,
        ):
            key=(oid,int(y["horizon_min"]))
            if key in outcome_keys:
                continue

            append_jsonl(
                OUTCOME_FILE,
                {"observation_id":oid,"symbol":symbol,**y}
            )
            outcome_keys.add(key)

    state["last_run"]=now.isoformat()
    state["last_closed_candle"]=cutoff.isoformat()
    state["last_observation_count"]=len(all_obs)
    state["last_outcome_count"]=len(outcome_keys)
    state["research_only"]=True
    write_state(state)

def _analysis_due(
    now,
    force=False,
    outcome_count=0,
):
    """
    Research eligibility depends ONLY on:
      1. elapsed time since the previous analysis attempt
      2. number of newly matured outcomes

    It deliberately does not depend on:
      - cron wall-clock phase
      - PERSIST_EVERY_MINUTES
      - Git persistence timing
    """
    if force:
        return True

    state=read_state()

    last=state.get(
        "last_analysis_attempt_at"
    )

    if last:
        try:
            ts=datetime.fromisoformat(
                str(last).replace(
                    "Z",
                    "+00:00",
                )
            )

            if ts.tzinfo is None:
                ts=ts.replace(
                    tzinfo=timezone.utc
                )

            elapsed=(
                now
                - ts.astimezone(
                    timezone.utc
                )
            ).total_seconds()/60.0

            if elapsed < ANALYSIS_INTERVAL_MIN:
                return False

        except (TypeError,ValueError):
            # Corrupt/legacy timestamp must not permanently disable research.
            pass

    previous=int(
        state.get(
            "last_analysis_outcome_count",
            0,
        ) or 0
    )

    return (
        outcome_count-previous
        >= MIN_NEW_OUTCOMES_FOR_ANALYSIS
    )

def _mark_analysis(
    now,
    status,
    rows,
    outcomes,
    successful_at=None,
):
    state=read_state()

    state["last_analysis_attempt_at"]=now.isoformat()
    state["last_analysis_attempt_status"]=status
    state["last_analysis_attempt_rows"]=int(rows)
    state["last_analysis_outcome_count"]=int(outcomes)

    if successful_at is not None:
        state["last_successful_analysis_at"]=successful_at.isoformat()
        state["last_successful_analysis_outcome_count"]=int(
            outcomes
        )

    write_state(state)

def analyze_if_due(
    now=None,
    force=False,
    discover_fn=discover_multi,
    evaluate_fn=evaluate,
):
    now=utc_ts(now)

    observations=read_jsonl(OBS_FILE)
    outcomes=read_jsonl(OUTCOME_FILE)

    # IMPORTANT:
    # No wall-clock modulo gate here. A cron sequence such as
    # 06:44, 06:49, 06:54... still reaches the hourly research timer.
    if not _analysis_due(
        now,
        force,
        len(outcomes),
    ):
        return False

    byid={
        o["observation_id"]:o
        for o in observations
    }

    rows=[]

    for y in outcomes:
        o=byid.get(
            y.get("observation_id")
        )

        if o:
            rows.append({
                "observation_id":y[
                    "observation_id"
                ],
                "symbol":y.get(
                    "symbol",
                    o.get("symbol"),
                ),
                "feature_time":o[
                    "features"
                ]["feature_time"],
                "feature":o[
                    "features"
                ],
                "outcome":y,
            })

    _mark_analysis(
        now,
        "started",
        len(rows),
        len(outcomes),
    )

    memory=load_memory()
    raw_key_pool=os.getenv(
        "RESEARCH_GEMINI_KEY",
        "",
    )

    try:
        d=discover_fn(
            raw_key_pool,
            rows,
            GEMINI_MODEL,
            memory,
        )

    except Exception as exc:
        _mark_analysis(
            now,
            "discovery_error",
            len(rows),
            len(outcomes),
        )

        append_jsonl(
            ANALYSIS_FILE,
            {
                "time":now.isoformat(),
                "status":"discovery_error",
                "rows":len(rows),
                "outcomes":len(outcomes),
                "error":str(exc),
            }
        )
        return False

    status=str(
        d.get(
            "status",
            "unknown",
        )
    )

    pool=[]
    seen=set()

    for h in (
        d.get("quant_candidates",[])
        + d.get("hypotheses",[])
    ):
        sig=_candidate_signature(h)

        if sig in seen:
            continue

        seen.add(sig)
        pool.append(h)

    if not pool:
        _mark_analysis(
            now,
            status,
            len(rows),
            len(outcomes),
        )

        append_jsonl(
            ANALYSIS_FILE,
            {
                "time":now.isoformat(),
                "status":status,
                "rows":len(rows),
                "outcomes":len(outcomes),
                "quant_candidate_count":len(
                    d.get(
                        "quant_candidates",
                        [],
                    )
                ),
                "gemini_hypothesis_count":len(
                    d.get(
                        "hypotheses",
                        [],
                    )
                ),
                "gemini_statuses":d.get(
                    "gemini_statuses",
                    [],
                ),
            }
        )
        return False

    latest_feature=max(
        (
            pd.Timestamp(
                x["feature_time"]
            )
            for x in rows
        ),
        default=now,
    )

    holdout_epoch=(
        latest_feature
        .tz_convert("UTC")
        .floor("D")
        .isoformat()
    )

    state=read_state()

    if state.get(
        "holdout_epoch"
    ) != holdout_epoch:
        state["holdout_epoch"]=holdout_epoch
        state["holdout_tests_used"]=0

    budget_used=int(
        state.get(
            "holdout_tests_used",
            0,
        ) or 0
    )

    holdout_budget=max(
        0,
        HOLDOUT_TEST_BUDGET_PER_EPOCH
        - budget_used,
    )

    evaluated=evaluate_fn(
        pool,
        rows,
        holdout_budget=holdout_budget,
    )

    state["holdout_tests_used"]=(
        budget_used
        + sum(
            bool(
                x.get(
                    "holdout_tested"
                )
            )
            for x in evaluated
        )
    )

    write_state(state)

    for item in evaluated:
        item["status"]=classify(item)

    state=read_state()

    experiment_id=(
        f"{now.strftime('%Y%m%dT%H%M%SZ')}:"
        f"{len(evaluated)}"
    )

    state.setdefault(
        "research_experiments",
        [],
    )

    state["research_experiments"].append({
        "experiment_id":experiment_id,
        "time":now.isoformat(),
        "discovery_observations":d.get(
            "discovery_observations",
            0,
        ),
        "candidate_count":len(pool),
        "quant_candidate_count":len(
            d.get(
                "quant_candidates",
                [],
            )
        ),
        "gemini_hypothesis_count":len(
            d.get(
                "hypotheses",
                [],
            )
        ),
        "holdout_tested_count":sum(
            bool(
                x.get(
                    "holdout_tested"
                )
            )
            for x in evaluated
        ),
    })

    state["research_experiments"]=(
        state["research_experiments"]
    )[-200:]

    write_state(state)

    validated=[
        {
            "name":x["hypothesis"].get(
                "name"
            ),
            "direction":x["hypothesis"].get(
                "direction"
            ),
            "horizon_min":x["hypothesis"].get(
                "horizon_min"
            ),
            "status":x["status"],
            "validation":x["validation"],
            "holdout":x["holdout"],
            "holdout_tested":x.get(
                "holdout_tested",
                False,
            ),
        }
        for x in evaluated
        if x["status"]=="HOLDOUT_PASSED"
    ]

    rejected=[
        {
            "name":x["hypothesis"].get(
                "name"
            ),
            "direction":x["hypothesis"].get(
                "direction"
            ),
            "horizon_min":x["hypothesis"].get(
                "horizon_min"
            ),
            "status":x["status"],
            "validation":x["validation"],
            "holdout":x["holdout"],
            "holdout_tested":x.get(
                "holdout_tested",
                False,
            ),
        }
        for x in evaluated
        if x["status"]!="HOLDOUT_PASSED"
    ]

    memory["validated"]=(
        memory.get(
            "validated",
            [],
        )
        + validated
    )[-100:]

    memory["rejected"]=(
        memory.get(
            "rejected",
            [],
        )
        + rejected
    )[-100:]

    write_memory(memory)

    append_jsonl(
        HYPOTHESIS_FILE,
        {
            "generated_at":d.get(
                "generated_at",
                now.isoformat(),
            ),
            "research_only":True,
            "research_rule_type":RESEARCH_RULE_TYPE,
            "discovery":d,
            "candidate_count":len(pool),
            "holdout_tested_count":sum(
                bool(
                    x.get(
                        "holdout_tested"
                    )
                )
                for x in evaluated
            ),
            "evaluated":evaluated,
        }
    )

    _mark_analysis(
        now,
        "ok",
        len(rows),
        len(outcomes),
        successful_at=now,
    )

    append_jsonl(
        ANALYSIS_FILE,
        {
            "time":now.isoformat(),
            "status":"ok",
            "rows":len(rows),
            "outcomes":len(outcomes),
            "candidate_count":len(pool),
            "quant_candidate_count":len(
                d.get(
                    "quant_candidates",
                    [],
                )
            ),
            "gemini_hypothesis_count":len(
                d.get(
                    "hypotheses",
                    [],
                )
            ),
            "holdout_tested_count":sum(
                bool(
                    x.get(
                        "holdout_tested"
                    )
                )
                for x in evaluated
            ),
            "holdout_passed_count":sum(
                x["status"]=="HOLDOUT_PASSED"
                for x in evaluated
            ),
            "validation_passed_count":sum(
                x["status"]=="VALIDATION_PASSED"
                for x in evaluated
            ),
            "validation_failed_count":sum(
                x["status"]=="VALIDATION_FAILED"
                for x in evaluated
            ),
            "gemini_statuses":d.get(
                "gemini_statuses",
                [],
            ),
        }
    )

    return True

def main():
    now=utc_ts()
    collect_once(now)
    analyze_if_due(now)

if __name__=="__main__":
    main()
