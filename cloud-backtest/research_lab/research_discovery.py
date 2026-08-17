
from __future__ import annotations
import json, os, re
from datetime import datetime, timezone
import requests
from .research_config import (
    DISCOVERY_INPUT_CHAR_BUDGET, DISCOVERY_PASSES,
    DISCOVERY_PASS_CANDIDATES, MAX_TOTAL_HYPOTHESES,
    MAX_HYPOTHESES_PER_PASS, REQUEST_TIMEOUT_SECONDS,
    MIN_DISCOVERY_OBSERVATIONS, MIN_VALIDATION_OBSERVATIONS,
    MIN_HOLDOUT_OBSERVATIONS, RESEARCH_RULE_TYPE,
)
from .research_hypotheses import _group_rows, _split_grouped
from .research_miner import mine_discovery

SYSTEM = """You are a neutral market research scientist, not a trader.
Review ONLY the NUMERICALLY MINED DISCOVERY candidates supplied below.
Validation and holdout are performed independently by Python.
Do not invent data. Prefer simple, repeatable candidates supported across
multiple coins and time blocks. Treat LONG and SHORT symmetrically.
Return JSON only with hypotheses using only operators >, >=, <, <=.
For a between range, emit two conditions instead of a 'between' operator."""

def _parse_json(text):
    try:
        value = json.loads(text or "")
        return value if isinstance(value, dict) else {"hypotheses":[]}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            return {"hypotheses":[]}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"hypotheses":[]}

def _signature(h):
    cond = []
    for c in h.get("conditions",[]) or []:
        cond.append((
            str(c.get("feature","")),
            str(c.get("op","")),
            json.dumps(c.get("value"), sort_keys=True, separators=(",",":")),
        ))
    return json.dumps(
        (str(h.get("direction","")).upper(), int(h.get("horizon_min",0) or 0), sorted(cond)),
        separators=(",",":")
    )

def _compact(candidates):
    out=[]
    for c in candidates:
        d=c.get("discovery",{})
        out.append({
            "name":c.get("name"),
            "direction":c.get("direction"),
            "horizon_min":c.get("horizon_min"),
            "conditions":c.get("conditions",[]),
            "n":d.get("n"),
            "avg":d.get("avg_net_return"),
            "median":d.get("median_net_return"),
            "pf":d.get("profit_factor"),
            "win":d.get("positive_rate"),
            "coin_count":d.get("coin_count"),
            "coin_positive_fraction":d.get("coin_positive_fraction"),
            "score":d.get("discovery_score"),
        })
    return out

def _prompt(candidates, memory):
    return "\n".join([
        SYSTEM,
        "Historical memory is context only; do not use it as new market evidence.",
        json.dumps({
            "validated":(memory or {}).get("validated",[])[-10:],
            "rejected":(memory or {}).get("rejected",[])[-10:],
        }, separators=(",",":"), default=str),
        "Mined discovery candidates:",
        json.dumps(_compact(candidates), separators=(",",":"), default=str),
    ])

def _sample(candidates, index, count, target):
    if len(candidates) <= target:
        return list(candidates)
    start=len(candidates)*index//count
    end=len(candidates)*(index+1)//count
    chunk=candidates[start:end]
    if len(chunk)<=target:
        return chunk
    if target<=1:
        return [chunk[len(chunk)//2]]
    return [chunk[int(i*(len(chunk)-1)/(target-1))] for i in range(target)]

def _discover_one(key, candidates, model, memory):
    prompt=_prompt(candidates,memory)
    if len(prompt)>DISCOVERY_INPUT_CHAR_BUDGET:
        return {"status":"payload_too_large","hypotheses":[],"prompt_chars":len(prompt)}

    schema = {
        "type":"object",
        "properties":{
            "hypotheses":{
                "type":"array",
                "items":{
                    "type":"object",
                    "properties":{
                        "name":{"type":"string"},
                        "direction":{"type":"string","enum":["LONG","SHORT"]},
                        "horizon_min":{"type":"integer","enum":[5,10,15,30,60]},
                        "conditions":{
                            "type":"array",
                            "items":{
                                "type":"object",
                                "properties":{
                                    "feature":{"type":"string"},
                                    "op":{"type":"string","enum":[">",">=","<","<="]},
                                    "value":{"type":"number"},
                                },
                                "required":["feature","op","value"],
                            },
                        },
                        "rationale":{"type":"string"},
                    },
                    "required":["name","direction","horizon_min","conditions","rationale"],
                },
            }
        },
        "required":["hypotheses"],
    }

    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        r=requests.post(
            url,
            params={"key":key},
            json={
                "contents":[{"parts":[{"text":prompt}]}],
                "generationConfig":{
                    "temperature":0.2,
                    "responseMimeType":"application/json",
                    "responseSchema":schema,
                },
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return {"status":"request_error","error":str(exc),"hypotheses":[]}

    if r.status_code==429:
        m=re.search(r"retry in\s+([0-9.]+)s",r.text or "",re.I)
        return {
            "status":"rate_limited",
            "retry_after_seconds":float(m.group(1)) if m else None,
            "message":(r.text or "")[:500],
            "hypotheses":[],
        }

    try:
        r.raise_for_status()
        data=r.json()
    except (requests.RequestException,ValueError) as exc:
        return {"status":"response_error","error":str(exc),"hypotheses":[]}

    text="".join(
        part.get("text","")
        for candidate in data.get("candidates",[])
        for part in candidate.get("content",{}).get("parts",[])
    )

    clean=[]
    for h in _parse_json(text).get("hypotheses",[])[:MAX_HYPOTHESES_PER_PASS]:
        if not isinstance(h,dict): continue
        direction=str(h.get("direction","")).upper()
        if direction not in {"LONG","SHORT"}: continue
        try: horizon=int(h.get("horizon_min"))
        except (TypeError,ValueError): continue
        if horizon not in {5,10,15,30,60}: continue
        conditions=h.get("conditions")
        if not isinstance(conditions,list) or not conditions: continue
        if any(
            c.get("op") not in {">",">=","<","<="}
            or not isinstance(c.get("feature"),str)
            for c in conditions if isinstance(c,dict)
        ):
            continue
        clean.append({**h,"direction":direction,"horizon_min":horizon,"source":"gemini_reviewer"})
    return {"status":"ok","hypotheses":clean,"prompt_chars":len(prompt),"sample_candidates":len(candidates)}

def discover_multi(api_key, rows, model, memory=None):
    keys=[k.strip() for k in str(api_key or "").split(",") if k.strip()]
    grouped=_group_rows(rows)
    discovery, validation, holdout=_split_grouped(grouped)

    if len(discovery)<MIN_DISCOVERY_OBSERVATIONS or len(validation)<MIN_VALIDATION_OBSERVATIONS or len(holdout)<MIN_HOLDOUT_OBSERVATIONS:
        return {
            "status":"insufficient_data",
            "discovery_observations":len(discovery),
            "validation_observations":len(validation),
            "holdout_observations":len(holdout),
            "quant_candidate_count":0,
            "quant_candidates":[],
            "hypotheses":[],
        }

    quant=mine_discovery(discovery)
    gemini_results=[]
    gemini_hyp=[]

    # Gemini is optional. Project-level quota means API keys are not additive quota.
    if keys and quant:
        passes=max(1,min(DISCOVERY_PASSES,len(keys)))
        for i in range(passes):
            sample=_sample(
                quant,
                i,
                passes,
                min(DISCOVERY_PASS_CANDIDATES,len(quant)),
            )
            gemini_results.append(
                _discover_one(keys[i],sample,model,memory or {})
            )

        seen=set()
        for result in gemini_results:
            for h in result.get("hypotheses",[]):
                sig=_signature(h)
                if sig in seen: continue
                seen.add(sig)
                gemini_hyp.append(h)
                if len(gemini_hyp)>=MAX_TOTAL_HYPOTHESES:
                    break
            if len(gemini_hyp)>=MAX_TOTAL_HYPOTHESES:
                break

    return {
        "status":"ok" if quant else "no_candidates",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "model":model,
        "research_rule_type":RESEARCH_RULE_TYPE,
        "discovery_observations":len(discovery),
        "validation_observations":len(validation),
        "holdout_observations":len(holdout),
        "quant_candidate_count":len(quant),
        "quant_candidates":quant[:250],
        "gemini_passes":len(gemini_results),
        "gemini_successful_passes":sum(r.get("status")=="ok" for r in gemini_results),
        "gemini_rate_limited_passes":sum(r.get("status")=="rate_limited" for r in gemini_results),
        "gemini_statuses":[r.get("status","unknown") for r in gemini_results],
        "hypotheses":gemini_hyp,
        "discovery_only":True,
        "lookahead_safe":True,
    }
