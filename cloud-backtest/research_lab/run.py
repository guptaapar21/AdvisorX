from __future__ import annotations
import os, json
from datetime import datetime, timezone
import pandas as pd
from .research_config import *
from .research_time import utc_ts, completed_minute, canonical_minute
from .research_fetcher import fetch_1m, store_candles, load_candles
from .outcomes import label_observation
from .research_store import append_jsonl, read_jsonl, read_state, write_state, load_memory, write_memory
from .research_discovery import discover_multi
from .research_hypotheses import evaluate, classify


def _observation_id(symbol,t):return f'{symbol}:{canonical_minute(t).strftime("%Y%m%dT%H%M%SZ")}'


def _gemini_key_pool(raw):
    """Return the complete configured Gemini key pool without exposing keys."""
    return [k.strip() for k in str(raw or '').split(',') if k.strip()]


def _discover_with_key_pool(raw_key_pool, rows, model, memory, discover_fn=discover_multi):
    """Run the new multi-pass discovery engine with the complete configured key pool."""
    keys = _gemini_key_pool(raw_key_pool)
    if not keys:
        return discover_fn('', rows, model, memory)
    return discover_fn(','.join(keys), rows, model, memory)


def collect_once(now=None, fetch_fn=fetch_1m):
    now=utc_ts(now); cutoff=completed_minute(now); state=read_state(); state.setdefault('last_observation_minute',None)
    observations=read_jsonl(OBS_FILE); outcomes=read_jsonl(OUTCOME_FILE)
    known={x.get('observation_id') for x in observations}; outcome_keys={(x.get('observation_id'),int(x.get('horizon_min',-1))) for x in outcomes}
    data_by_symbol={}
    for symbol in COINS:
        try:
            df=fetch_fn(symbol,cutoff-pd.Timedelta(minutes=OBSERVATION_LOOKBACK_MIN),cutoff)
            if df.empty:continue
            df.index=pd.to_datetime(df.index,utc=True); df=df[df.index<=cutoff].sort_index()
            store_candles(df,symbol,CANDLE_DIR,CANDLE_RETENTION_DAYS); data_by_symbol[symbol]=load_candles(symbol,CANDLE_DIR)
            for t in df.index:
                if t>cutoff or len(data_by_symbol[symbol].loc[data_by_symbol[symbol].index<=t])<65:continue
                oid=_observation_id(symbol,t)
                if oid in known:continue
                append_jsonl(OBS_FILE,{'observation_id':oid,'symbol':symbol,'features':feature_snapshot(data_by_symbol[symbol],t)})
                known.add(oid)
        except Exception as exc:
            append_jsonl(ERROR_FILE,{'time':now.isoformat(),'symbol':symbol,'stage':'collect','error':str(exc)})

    all_obs=read_jsonl(OBS_FILE)
    by_symbol={s:load_candles(s,CANDLE_DIR) for s in COINS}
    for obs in all_obs:
        oid=obs['observation_id']; symbol=obs['symbol']; t=canonical_minute(obs['features']['feature_time']); max_end=t+pd.Timedelta(minutes=MAX_HORIZON_MIN)
        if max_end>cutoff:continue
        if all((oid,int(h)) in outcome_keys for h in HORIZONS_MIN):
            continue
        c=by_symbol.get(symbol)
        if c is None or c.empty:continue
        for y in label_observation(c,t,now=cutoff,fee_rate=TAKER_FEE_RATE):
            key=(obs['observation_id'],int(y['horizon_min']))
            if key in outcome_keys:continue
            append_jsonl(OUTCOME_FILE,{'observation_id':obs['observation_id'],'symbol':symbol,**y}); outcome_keys.add(key)
    state['last_run']=now.isoformat(); state['last_closed_candle']=cutoff.isoformat(); state['research_only']=True
    write_state(state)


def _analysis_due(now,force=False):
    if force:return True
    state=read_state(); today=now.date().isoformat()
    if now.hour<ANALYSIS_UTC_HOUR:
        return False
    return state.get('last_analysis_attempt_date')!=today


def _mark_analysis_attempt(now, status, rows=None):
    state=read_state()
    state['last_analysis_attempt_date']=now.date().isoformat()
    state['last_analysis_attempt_at']=now.isoformat()
    state['last_analysis_attempt_status']=status
    if rows is not None:
        state['last_analysis_attempt_rows']=int(rows)
    write_state(state)


def analyze_if_due(now=None, force=False, discover_fn=discover_multi, evaluate_fn=evaluate):
    now=utc_ts(now)
    if not _analysis_due(now,force):return False

    observations=read_jsonl(OBS_FILE); outcomes=read_jsonl(OUTCOME_FILE); byid={o['observation_id']:o for o in observations}
    rows=[]
    for y in outcomes:
        o=byid.get(y.get('observation_id'))
        if o:rows.append({'observation_id':y['observation_id'],'symbol':y.get('symbol',o.get('symbol')),'feature_time':o['features']['feature_time'],'feature':o['features'],'outcome':y})

    _mark_analysis_attempt(now,'started',len(rows))

    raw_key_pool=os.getenv('RESEARCH_GEMINI_KEY','')
    if not _gemini_key_pool(raw_key_pool):
        _mark_analysis_attempt(now,'no_api_key',len(rows))
        append_jsonl(ANALYSIS_FILE,{'time':now.isoformat(),'status':'no_api_key','rows':len(rows)})
        return False

    memory=load_memory()
    d=_discover_with_key_pool(raw_key_pool,rows,GEMINI_MODEL,memory,discover_fn=discover_fn)
    if d.get('status')!='ok':
        status=str(d.get('status') or 'discovery_failed')
        _mark_analysis_attempt(now,status,len(rows))
        append_jsonl(ANALYSIS_FILE,{'time':now.isoformat(),'status':status,'rows':len(rows),'discovery':d})
        return False

    evaluated=evaluate_fn(d['hypotheses'],rows)
    for item in evaluated:item['status']=classify(item)
    validated=[{'name':x['hypothesis'].get('name'),'direction':x['hypothesis'].get('direction'),'horizon_min':x['hypothesis'].get('horizon_min'),'status':x['status'],'validation':x['validation'],'holdout':x['holdout']} for x in evaluated if x['status']=='HOLDOUT_PASSED']
    rejected=[{'name':x['hypothesis'].get('name'),'direction':x['hypothesis'].get('direction'),'horizon_min':x['hypothesis'].get('horizon_min'),'status':x['status'],'validation':x['validation'],'holdout':x['holdout']} for x in evaluated if x['status']!='HOLDOUT_PASSED']
    memory['validated']=(memory.get('validated',[])+validated)[-100:]; memory['rejected']=(memory.get('rejected',[])+rejected)[-100:]; write_memory(memory)
    append_jsonl(HYPOTHESIS_FILE,{'generated_at':d['generated_at'],'research_only':True,'discovery':d,'evaluated':evaluated})
    state=read_state(); state['last_analysis_date']=now.date().isoformat(); state['last_analysis_at']=now.isoformat(); state['last_analysis_status']='ok'; write_state(state)
    return True


def main():
    now=utc_ts(); collect_once(now); analyze_if_due(now)


if __name__=='__main__':main()
