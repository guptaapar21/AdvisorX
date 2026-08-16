import json
from pathlib import Path
import pandas as pd
from .research_features import feature_snapshot
from .outcomes import label_observation
from .research_walkforward import split_purged, assert_clean_boundaries
from .research_hypotheses import evaluate_one, _group_rows, discover
from . import run as runmod
from . import research_store


def candles(n=240, start='2026-01-01T00:00:00Z'):
    i=pd.date_range(start,periods=n,freq='1min')
    close=pd.Series(range(n),index=i,dtype=float)+100
    return pd.DataFrame({'open':close,'high':close+1,'low':close-1,'close':close,'volume':10.0},index=i)


def test_feature_cutoff_ignores_future_mutation():
    c=candles(); t=c.index[120]; a=feature_snapshot(c,t); c.loc[c.index>t,['close','volume']]=9999; b=feature_snapshot(c,t); assert a==b


def test_outcome_strict_future_and_maturity():
    c=candles(); t=c.index[100]
    assert label_observation(c,t,now=t+pd.Timedelta(minutes=4))==[]
    o=label_observation(c,t,now=t+pd.Timedelta(minutes=61)); assert {x['horizon_min'] for x in o}=={5,10,15,30,60}
    assert all(pd.Timestamp(x['outcome_time'])>t for x in o)


def test_current_forming_endpoint_is_rejected():
    c=candles(); t=c.index[100]; now=t+pd.Timedelta(minutes=5,seconds=10)
    assert label_observation(c,t,now=now)==[]


def test_closed_endpoint_is_accepted():
    c=candles(); t=c.index[100]; now=t+pd.Timedelta(minutes=6)
    o=label_observation(c,t,now=now); assert 5 in {x['horizon_min'] for x in o}


def test_multihorizon_purge_real_shape():
    rows=[]
    for i in range(500):
        t=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(minutes=i)
        for h in (5,10,15,30,60): rows.append({'feature_time':t.isoformat(),'horizon_min':h})
    d,v,h=split_purged(rows,'2026-01-01T05:00Z','2026-01-01T10:00Z','2026-01-01T12:00Z')
    assert_clean_boundaries(d,v,h,pd.Timestamp('2026-01-01T05:00Z'),pd.Timestamp('2026-01-01T10:00Z'),pd.Timestamp('2026-01-01T12:00Z'))


def test_grouped_rows_are_one_observation_for_discovery():
    rows=[]
    for i in range(20):
        t=(pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(minutes=i)).isoformat()
        for h in (5,10,15,30,60):
            rows.append({'observation_id':f'O{i}','symbol':'BTC','feature_time':t,'feature':{'rvol20':1.5},'outcome':{'horizon_min':h,'net_return_long':.01,'net_return_short':-.01}})
    grouped=_group_rows(rows); assert len(grouped)==20; assert all(len(x['outcomes'])==5 for x in grouped)


def _production_rows(n=800):
    rows=[]
    for i in range(n):
        t=(pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(minutes=i)).isoformat()
        for h in (5,10,15,30,60):
            rows.append({'observation_id':f'BTC:{i:04d}','symbol':'BTC','feature_time':t,'feature':{'rvol20':1.5,'return_5m':0.001},'outcome':{'horizon_min':h,'net_return_long':.001,'net_return_short':-.001,'mfe_long':.002,'mae_long':.0005,'mfe_short':0.0,'mae_short':.003}})
    return rows


def test_discovery_actual_path_handles_production_multihorizon_shape(monkeypatch):
    rows=_production_rows()
    class R:
        status_code=200
        def raise_for_status(self): pass
        def json(self): return {'candidates':[{'content':{'parts':[{'text':'{"hypotheses": [{"name":"rvol","direction":"long","horizon_min":15,"conditions":[{"feature":"rvol20","op":">=","value":1.0}],"rationale":"test"}]}' }]}}]}
    monkeypatch.setattr('requests.post',lambda *a,**k:R())
    result=discover('key',rows,'test-model')
    assert result['status']=='ok'; assert result['hypotheses']


def test_evaluation_production_shape():
    rule={'name':'test','direction':'long','horizon_min':15,'conditions':[{'feature':'rvol20','op':'>=','value':1.0}]}
    m=evaluate_one(rule,_group_rows(_production_rows(800))); assert m['n']==800 and m['avg_net_return']>0 and m['profit_factor']>1


def test_integration_maturity_backfill(tmp_path,monkeypatch):
    # Patch both run.py imported paths AND store module constants so the test stays isolated.
    obs=tmp_path/'observations.jsonl'; out=tmp_path/'outcomes.jsonl'; errors=tmp_path/'errors.jsonl'; state=tmp_path/'state.json'; data=tmp_path/'candles'
    monkeypatch.setattr(runmod,'OBS_FILE',obs); monkeypatch.setattr(runmod,'OUTCOME_FILE',out); monkeypatch.setattr(runmod,'ERROR_FILE',errors); monkeypatch.setattr(runmod,'CANDLE_DIR',data); monkeypatch.setattr(runmod,'COINS',('BTC',)); monkeypatch.setattr(runmod,'TAKER_FEE_RATE',0.00075)
    monkeypatch.setattr(research_store,'STATE_FILE',state); monkeypatch.setattr(research_store,'ROOT',tmp_path)
    c=candles(250,'2026-01-01T00:00:00Z')
    mem={'BTC':pd.DataFrame()}
    def fake_store(df,symbol,root,retention_days):
        mem[symbol]=pd.concat([mem.get(symbol,pd.DataFrame()),df]).sort_index()
        mem[symbol]=mem[symbol][~mem[symbol].index.duplicated(keep='last')]
    monkeypatch.setattr(runmod,'store_candles',fake_store)
    monkeypatch.setattr(runmod,'load_candles',lambda symbol,root: mem.get(symbol,pd.DataFrame()).copy())
    def fake_fetch(symbol,s,e): return c.loc[(c.index>=s)&(c.index<=e)]
    runmod.collect_once(pd.Timestamp('2026-01-01T00:50:10Z'),fetch_fn=fake_fetch)
    runmod.collect_once(pd.Timestamp('2026-01-01T02:10:10Z'),fetch_fn=fake_fetch)
    outrows=runmod.read_jsonl(out); assert outrows


def test_delayed_run_backfills_old_unresolved_observation(tmp_path,monkeypatch):
    obs=tmp_path/'observations.jsonl'; out=tmp_path/'outcomes.jsonl'; errors=tmp_path/'errors.jsonl'; state=tmp_path/'state.json'; data=tmp_path/'candles'
    monkeypatch.setattr(runmod,'OBS_FILE',obs); monkeypatch.setattr(runmod,'OUTCOME_FILE',out); monkeypatch.setattr(runmod,'ERROR_FILE',errors); monkeypatch.setattr(runmod,'CANDLE_DIR',data); monkeypatch.setattr(runmod,'COINS',('BTC',)); monkeypatch.setattr(research_store,'STATE_FILE',state); monkeypatch.setattr(research_store,'ROOT',tmp_path)
    c=candles(300,'2026-01-01T00:00:00Z')
    # Avoid requiring the optional parquet engine in the local unit test: simulate the
    # persistent candle store while exercising the real run.py backfill path.
    store={'BTC':c.copy()}
    monkeypatch.setattr(runmod,'store_candles',lambda df,symbol,root,retention_days: store.__setitem__(symbol, pd.concat([store.get(symbol,pd.DataFrame()),df]).sort_index()))
    monkeypatch.setattr(runmod,'load_candles',lambda symbol,root: store.get(symbol,pd.DataFrame()).copy())
    t=c.index[100]
    obs.parent.mkdir(parents=True,exist_ok=True); obs.write_text(json.dumps({'observation_id':'BTC:test','symbol':'BTC','features':feature_snapshot(c,t)})+'\n')
    runmod.collect_once(c.index[260]+pd.Timedelta(seconds=10),fetch_fn=lambda *args,**kwargs: c.iloc[:1])
    outrows=runmod.read_jsonl(out); assert outrows
