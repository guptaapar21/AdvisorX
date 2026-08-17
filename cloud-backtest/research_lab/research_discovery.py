from __future__ import annotations
import json, os, re, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import pandas as pd
import requests
from .research_hypotheses import _group_rows, _split_grouped

DISCOVERY_PASSES = int(os.getenv("RESEARCH_DISCOVERY_PASSES", "12"))
DISCOVERY_PASS_OBSERVATIONS = int(os.getenv("RESEARCH_DISCOVERY_PASS_OBSERVATIONS", "600"))
DISCOVERY_INPUT_CHAR_BUDGET = int(os.getenv("RESEARCH_DISCOVERY_INPUT_CHAR_BUDGET", "450000"))
MAX_HYPOTHESES_PER_PASS = int(os.getenv("RESEARCH_MAX_HYPOTHESES_PER_PASS", "10"))
MAX_TOTAL_HYPOTHESES = int(os.getenv("RESEARCH_MAX_TOTAL_HYPOTHESES", "120"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("RESEARCH_GEMINI_TIMEOUT_SECONDS", "45"))

SYSTEM = """You are a neutral market research scientist, not a trader.
Use ONLY the DISCOVERY sample in this request to generate research hypotheses.
Do not use validation or holdout rows; Python tests those later. Do not recommend live changes.
Consider LONG and SHORT symmetrically. Search for simple, interpretable, repeatable patterns
supported across multiple coins and time periods. Return JSON only with 5-10 materially
different candidate hypotheses when evidence allows.
Format: {"hypotheses":[{"name":"...","direction":"LONG|SHORT","horizon_min":5|10|15|30|60,
"conditions":[{"feature":"...","op":">|>=|<|<=|between","value":...}],"rationale":"..."}]}"""

FEATURE_KEYS = (
    "close","return_1m","return_3m","return_5m","return_10m","return_15m","return_30m","return_60m",
    "volatility_5m","volatility_15m","volatility_30m","volatility_60m","rvol20","rvol60","atr14_pct",
    "candle_range_pct","body_to_range","upper_wick_to_range","lower_wick_to_range","close_location",
    "ema9_gap_pct","ema20_gap_pct","ema50_gap_pct","ema9_20_gap_pct","rsi14","adx14","plus_di14",
    "minus_di14","adx_slope_5m","vwap_gap_pct","distance_from_30m_high","distance_from_30m_low",
    "distance_from_60m_high","distance_from_60m_low","efficiency_20",
)
HORIZONS = (5,10,15,30,60)

def _num(v):
    if v is None: return None
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def _compact_observation(row):
    f=row.get("features") or {}
    o=row.get("outcomes") or {}
    vals=[_num(f.get(k)) for k in FEATURE_KEYS]
    for h in HORIZONS:
        oh=o.get(str(h),{}) or {}
        for side in ("long","short"):
            vals.extend([_num(oh.get(f"net_return_{side}")),
                         _num(oh.get(f"mfe_{side}")),
                         _num(oh.get(f"mae_{side}"))])
    return [row.get("symbol","?"),row.get("feature_time",""),*vals]

def _prompt(sample,memory):
    header=list(("symbol","feature_time",*FEATURE_KEYS))
    for h in HORIZONS:
        for side in ("long","short"):
            for metric in ("net_return","mfe","mae"):
                header.append(f"{side}_{metric}_{h}")
    return "\n".join([
        SYSTEM,
        "Column order:",
        json.dumps(header,separators=(",",":")),
        "Rows are compact arrays in that exact order. Missing numbers are null.",
        json.dumps({"memory":memory or {}},separators=(",",":"),default=str),
        json.dumps(sample,separators=(",",":"),default=str),
    ])

def _sample_pass(disc,pass_index,pass_count,target):
    buckets={}
    for r in disc: buckets.setdefault(str(r.get("symbol","?")),[]).append(r)
    for rows in buckets.values(): rows.sort(key=lambda x:pd.Timestamp(x["feature_time"]))
    symbols=sorted(buckets)
    if not symbols:return []

    # Core coverage: partition each symbol's chronological history into disjoint
    # chunks across passes, so every discovery observation is seen at least once.
    selected=[]
    for symbol in symbols:
        rows=buckets[symbol]; n=len(rows)
        start_i=(n*pass_index)//pass_count
        end_i=(n*(pass_index+1))//pass_count
        selected.extend(rows[start_i:end_i])

    # Use the remaining prompt budget for a second, evenly-spaced sample. This
    # gives each pass both fresh chronology and broader interaction coverage.
    need=max(0,target-len(selected))
    if need:
        step=max(1,len(disc)//need)
        offset=pass_index%step
        extras=disc[offset::step]
        seen={r.get("observation_id") for r in selected}
        for r in extras:
            if r.get("observation_id") in seen: continue
            selected.append(r)
            seen.add(r.get("observation_id"))
            if len(selected)>=target: break

    selected.sort(key=lambda x:(str(x.get("symbol","?")),pd.Timestamp(x["feature_time"])))
    return selected[:target]

def _parse_json(text):
    m=re.search(r"\{.*\}",text or "",re.S)
    if not m:return {"hypotheses":[]}
    try:
        d=json.loads(m.group(0))
        return d if isinstance(d,dict) else {"hypotheses":[]}
    except json.JSONDecodeError:
        return {"hypotheses":[]}

def _candidate_signature(h):
    conds=[]
    for c in h.get("conditions",[]) or []:
        conds.append((str(c.get("feature","")),str(c.get("op","")),
                      json.dumps(c.get("value"),sort_keys=True,separators=(",",":"))))
    return json.dumps((str(h.get("direction","")).upper(),int(h.get("horizon_min",0) or 0),
                       sorted(conds)),separators=(",",":"))

def _discover_one(key,sample,model,memory):
    prompt=_prompt([_compact_observation(r) for r in sample],memory)
    if len(prompt)>DISCOVERY_INPUT_CHAR_BUDGET:
        return {"status":"payload_too_large","hypotheses":[],"prompt_chars":len(prompt)}
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r=requests.post(url,params={"key":key},
                    json={"contents":[{"parts":[{"text":prompt}]}],
                          "generationConfig":{"temperature":0.2}},
                    timeout=REQUEST_TIMEOUT_SECONDS)
    if r.status_code==429:return {"status":"rate_limited","hypotheses":[]}
    if r.status_code>=500:return {"status":f"server_{r.status_code}","hypotheses":[]}
    r.raise_for_status()
    data=r.json()
    text="".join(p.get("text","") for c in data.get("candidates",[])
                  for p in c.get("content",{}).get("parts",[]))
    hs=_parse_json(text).get("hypotheses",[])
    clean=[]
    for h in hs[:MAX_HYPOTHESES_PER_PASS]:
        if not isinstance(h,dict):continue
        if str(h.get("direction","")).upper() not in {"LONG","SHORT"}:continue
        try:h["horizon_min"]=int(h.get("horizon_min"))
        except (TypeError,ValueError):continue
        if h["horizon_min"] not in HORIZONS:continue
        if not isinstance(h.get("conditions"),list) or not h["conditions"]:continue
        clean.append(h)
    return {"status":"ok","hypotheses":clean,"prompt_chars":len(prompt),"sample_observations":len(sample)}

def discover_multi(api_key,rows,model,memory=None):
    keys=[k.strip() for k in str(api_key or "").split(",") if k.strip()]
    grouped=_group_rows(rows)
    disc,val,hold=_split_grouped(grouped)
    if len(disc)<300 or len(val)<100 or len(hold)<100:
        return {"status":"insufficient_data","discovery_observations":len(disc),
                "validation_observations":len(val),"holdout_observations":len(hold),"hypotheses":[]}
    if not keys:
        return {"status":"no_api_key","hypotheses":[]}
    pass_count=min(DISCOVERY_PASSES,len(keys))
    samples=[_sample_pass(disc,i,pass_count,DISCOVERY_PASS_OBSERVATIONS) for i in range(pass_count)]
    results=[None]*pass_count
    with ThreadPoolExecutor(max_workers=pass_count) as pool:
        futures={pool.submit(_discover_one,keys[i],samples[i],model,memory or {}):i for i in range(pass_count)}
        for fut in as_completed(futures):
            i=futures[fut]
            try:results[i]=fut.result()
            except Exception as exc:results[i]={"status":"error","error":str(exc),"hypotheses":[]}
    merged=[];seen=set()
    for result in results:
        if not result:continue
        for h in result.get("hypotheses",[]):
            sig=_candidate_signature(h)
            if sig in seen:continue
            seen.add(sig); merged.append(h)
            if len(merged)>=MAX_TOTAL_HYPOTHESES:break
        if len(merged)>=MAX_TOTAL_HYPOTHESES:break
    ok=sum(1 for r in results if r and r.get("status")=="ok")
    if ok==0:
        return {"status":"discovery_failed","discovery_observations":len(disc),
                "discovery_passes":pass_count,"successful_passes":0,"hypotheses":[]}
    return {"status":"ok","generated_at":datetime.now(timezone.utc).isoformat(),"model":model,
            "hypotheses":merged,"discovery_observations":len(disc),
            "discovery_sample_observations":sum(map(len,samples)),
            "discovery_passes":pass_count,"successful_passes":ok,
            "discovery_input_char_budget":DISCOVERY_INPUT_CHAR_BUDGET,
            "per_pass_observations":DISCOVERY_PASS_OBSERVATIONS,
            "max_hypotheses_per_pass":MAX_HYPOTHESES_PER_PASS,"discovery_only":True}
