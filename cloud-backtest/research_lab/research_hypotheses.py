from __future__ import annotations
import json, re, statistics, os
from datetime import datetime, timezone
import pandas as pd
import requests
from .research_config import *
from .research_walkforward import split_purged

SYSTEM = """You are a neutral market research scientist, not a trader. Use ONLY the DISCOVERY dataset supplied in this request. Do not use validation or holdout rows; those are tested later by Python. You have no authority over AdvisorX. Search for simple, interpretable, repeatable conditions associated with small consistent NET returns after fees, reasonable MFE/MAE and low drawdown. Do not assume AdvisorX's current strategy is correct. Consider LONG and SHORT symmetrically. Prefer broad, modest, repeatable effects over rare huge winners. Do not invent data. Return JSON only: {hypotheses:[{name,direction,horizon_min,conditions:[{feature,op,value}],rationale}]}. Allowed operators: >, >=, <, <=, between. Hypotheses are research candidates only."""

# Gemini free-tier input-token quota is currently the limiting resource for
# discovery. Keep the actual request comfortably below the 250k-token ceiling.
# This is a character budget, deliberately conservative (~150k tokens worst-case
# for typical JSON/text tokenization), while the complete ResearchLab dataset
# remains on disk for Python validation/holdout testing.
DISCOVERY_INPUT_CHAR_BUDGET = int(
    os.getenv("RESEARCH_DISCOVERY_INPUT_CHAR_BUDGET", "600000")
)
MIN_DISCOVERY_PER_SYMBOL = int(
    os.getenv("RESEARCH_MIN_DISCOVERY_PER_SYMBOL", "8")
)

def _parse(text):
    m=re.search(r'\{.*\}',text,re.S)
    if not m:return {'hypotheses':[]}
    try:return json.loads(m.group(0))
    except json.JSONDecodeError:return {'hypotheses':[]}

def _group_rows(rows):
    grouped={}
    for r in rows:
        oid=r.get('observation_id') or f"{r['feature_time']}"
        g=grouped.setdefault(oid,{'observation_id':oid,'symbol':r.get('symbol'), 'feature_time':r['feature_time'],'features':r['feature'],'outcomes':{}})
        g['outcomes'][str(int(r['outcome']['horizon_min']))]=r['outcome']
    return list(grouped.values())

def _bounds(grouped):
    ts=sorted({_ts(x['feature_time']) for x in grouped})
    if len(ts)<60:return None
    return ts[int(len(ts)*.60)-1],ts[int(len(ts)*.80)-1],ts[-1]

def _ts(x):
    t=pd.Timestamp(x); return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')

def _split_grouped(grouped):
    b=_bounds(grouped)
    if not b:return [],[],[]
    d,v,h=b
    disc=[r for r in grouped if _ts(r['feature_time'])<d]
    val=[r for r in grouped if _ts(r['feature_time'])>=d+pd.Timedelta(minutes=PURGE_MIN) and _ts(r['feature_time'])<v]
    hold=[r for r in grouped if _ts(r['feature_time'])>=v+pd.Timedelta(minutes=PURGE_MIN) and _ts(r['feature_time'])<h]
    _assert_grouped_boundaries(disc,val,hold)
    return disc,val,hold

def _assert_grouped_boundaries(a,b,c):
    sa={_ts(r['feature_time']) for r in a}; sb={_ts(r['feature_time']) for r in b}; sc={_ts(r['feature_time']) for r in c}
    assert sa.isdisjoint(sb) and sb.isdisjoint(sc) and sa.isdisjoint(sc)
    if sa and sb: assert min(sb)-max(sa)>=pd.Timedelta(minutes=PURGE_MIN)
    if sb and sc: assert min(sc)-max(sb)>=pd.Timedelta(minutes=PURGE_MIN)

def _memory_prompt(memory):
    validated=memory.get('validated',[])[-12:]; rejected=memory.get('rejected',[])[-12:]
    return json.dumps({'previous_research_memory':{'validated':validated,'rejected':rejected}},separators=(',',':'),default=str)

def _sample_by_symbol(disc, per):
    buckets={}
    for r in disc:buckets.setdefault(r.get('symbol','?'),[]).append(r)
    sample=[]
    for sym,vals in buckets.items():
        vals=sorted(vals,key=lambda x:_ts(x['feature_time']))
        if len(vals)<=per:
            sample.extend(vals)
        else:
            idx=[int(i*(len(vals)-1)/(per-1)) for i in range(per)] if per>1 else [len(vals)-1]
            sample.extend(vals[i] for i in sorted(set(idx)))
    return sample

def _build_prompt(sample, memory):
    payload={'discovery_only':True,'observations':sample,'memory':memory or {}}
    return (
        SYSTEM
        + '\nDo not use previous memory to override evidence; it is historical context only.\n'
        + _memory_prompt(memory or {})
        + '\n'
        + json.dumps(payload,separators=(',',':'),default=str)
    )

def _budgeted_discovery_prompt(disc, memory):
    buckets=max(1,len({r.get('symbol','?') for r in disc}))
    requested=max(1,MAX_DISCOVERY_OBSERVATIONS//buckets)
    per=min(requested,max(1,min(len(disc),requested)))

    # Binary-search the largest uniform per-symbol sample that stays inside
    # the conservative request-size budget.
    lo=MIN_DISCOVERY_PER_SYMBOL
    hi=per
    best=None
    while lo<=hi:
        mid=(lo+hi)//2
        sample=_sample_by_symbol(disc,mid)
        prompt=_build_prompt(sample,memory)
        if len(prompt)<=DISCOVERY_INPUT_CHAR_BUDGET:
            best=(mid,prompt,len(sample))
            lo=mid+1
        else:
            hi=mid-1

    if best is None:
        # Absolute fallback: one compact representative row per symbol (or
        # MIN_DISCOVERY_PER_SYMBOL where possible). Never send an over-budget
        # request; return a deterministic status instead.
        per_fallback=max(1,MIN_DISCOVERY_PER_SYMBOL)
        sample=_sample_by_symbol(disc,per_fallback)
        prompt=_build_prompt(sample,memory)
        if len(prompt)>DISCOVERY_INPUT_CHAR_BUDGET:
            return None,0,len(prompt)
        return prompt,len(sample),len(prompt)

    _,prompt,count=best
    return prompt,count,len(prompt)

def discover(api_key, rows, model, memory=None):
    api_keys=[k.strip() for k in str(api_key or '').split(',') if k.strip()]
    grouped=_group_rows(rows); disc,val,hold=_split_grouped(grouped)
    if len(disc)<MIN_DISCOVERY_OBSERVATIONS or len(val)<MIN_VALIDATION_OBSERVATIONS or len(hold)<MIN_HOLDOUT_OBSERVATIONS:
        return {'status':'insufficient_data','discovery_observations':len(disc),'validation_observations':len(val),'holdout_observations':len(hold)}
    if not api_keys:return {'status':'no_api_key','discovery_observations':len(disc),'validation_observations':len(val),'holdout_observations':len(hold)}

    prompt,sample_count,prompt_chars=_budgeted_discovery_prompt(disc,memory or {})
    if not prompt:
        return {
            'status':'discovery_payload_too_large',
            'discovery_observations':len(disc),
            'validation_observations':len(val),
            'holdout_observations':len(hold),
            'prompt_chars':prompt_chars,
            'budget_chars':DISCOVERY_INPUT_CHAR_BUDGET,
        }

    url=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    last=None; data=None
    # ResearchLab's run.py supplies the full pool in groups; this function
    # also accepts a comma-separated pool directly and rotates across the
    # available keys it receives.
    for attempt in range(min(3, max(1, len(api_keys)))):
        key=api_keys[attempt % len(api_keys)]
        try:
            r=requests.post(url,params={'key':key},json={'contents':[{'parts':[{'text':prompt}]}]},timeout=60)
            if r.status_code==429:
                last=RuntimeError(f'Gemini 429: {r.text[:500]}'); continue
            if r.status_code>=500:
                last=RuntimeError(f'Gemini {r.status_code}: {r.text[:500]}'); continue
            r.raise_for_status(); data=r.json(); last=None; break
        except requests.RequestException as exc:
            last=exc
    if last or data is None: raise last or RuntimeError('Gemini request failed without a response')
    text=''.join(p.get('text','') for c in data.get('candidates',[]) for p in c.get('content',{}).get('parts',[]))
    parsed=_parse(text)
    return {
        'status':'ok',
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'model':model,
        'hypotheses':parsed.get('hypotheses',[]),
        'discovery_observations':len(disc),
        'discovery_sample_observations':sample_count,
        'discovery_prompt_chars':prompt_chars,
        'discovery_input_char_budget':DISCOVERY_INPUT_CHAR_BUDGET,
        'validation_observations':len(val),
        'holdout_observations':len(hold),
        'discovery_only':True
    }

def _match(feature,cond):
    v=feature.get(cond.get('feature')); op=cond.get('op'); x=cond.get('value')
    if v is None:return False
    try:
        if op=='between':return float(x[0])<=float(v)<=float(x[1])
        return {'>':float(v)>float(x),'>=':float(v)>=float(x),'<':float(v)<float(x),'<=':float(v)<=float(x)}[op]
    except (TypeError,ValueError,KeyError,IndexError):return False

def _metrics(vals,mfes,maes):
    if not vals:return {'n':0,'positive_rate':0.0,'avg_net_return':0.0,'median_net_return':0.0,'profit_factor':0.0,'worst_return':0.0,'avg_mfe':0.0,'avg_mae':0.0,'max_drawdown':0.0,'return_std':0.0}
    wins=[v for v in vals if v>0]; losses=[v for v in vals if v<0]; gl=sum(wins); ll=-sum(losses)
    pf=gl/ll if ll else (999.0 if gl else 0.0)
    eq=peak=0.0; dd=0.0
    for v in vals:
        eq+=v; peak=max(peak,eq); dd=max(dd,peak-eq)
    return {'n':len(vals),'positive_rate':len(wins)/len(vals),'avg_net_return':sum(vals)/len(vals),'median_net_return':statistics.median(vals),'profit_factor':pf,'worst_return':min(vals),'avg_mfe':sum(mfes)/len(mfes) if mfes else 0.0,'avg_mae':sum(maes)/len(maes) if maes else 0.0,'max_drawdown':dd,'return_std':statistics.pstdev(vals) if len(vals)>1 else 0.0}

def evaluate_one(rule, grouped):
    h=int(rule.get('horizon_min',15)); direction=str(rule.get('direction','long')).lower(); key='net_return_short' if direction=='short' else 'net_return_long'; mk='mfe_short' if direction=='short' else 'mfe_long'; ak='mae_short' if direction=='short' else 'mae_long'
    vals=[]; mfes=[]; maes=[]
    rows=sorted(grouped,key=lambda r:_ts(r['feature_time']))
    for r in rows:
        o=r['outcomes'].get(str(h))
        if not o:continue
        if all(_match(r['features'],c) for c in rule.get('conditions',[])):
            vals.append(float(o.get(key,0))); mfes.append(float(o.get(mk,0))); maes.append(float(o.get(ak,0)))
    return _metrics(vals,mfes,maes)

def evaluate(hypotheses, rows):
    grouped=_group_rows(rows); disc,val,hold=_split_grouped(grouped)
    result=[]
    for h in hypotheses:
        result.append({'hypothesis':h,'validation':evaluate_one(h,val),'holdout':evaluate_one(h,hold)})
    return result

def classify(e):
    v=e['validation']; h=e['holdout']
    return 'HOLDOUT_PASSED' if (
        v['n']>=PROMOTE_MIN_VALID_N and h['n']>=PROMOTE_MIN_HOLDOUT_N and
        v['profit_factor']>=PROMOTE_MIN_VALID_PF and h['profit_factor']>=PROMOTE_MIN_HOLDOUT_PF and
        v['avg_net_return']>=PROMOTE_MIN_VALID_AVG and h['avg_net_return']>=PROMOTE_MIN_HOLDOUT_AVG and
        h['positive_rate']>=PROMOTE_MIN_HOLDOUT_WINRATE and h['max_drawdown']<=PROMOTE_MAX_HOLDOUT_DD
    ) else ('VALIDATION_PASSED' if v['n']>=PROMOTE_MIN_VALID_N and v['profit_factor']>=PROMOTE_MIN_VALID_PF and v['avg_net_return']>=PROMOTE_MIN_VALID_AVG else 'VALIDATION_FAILED')
