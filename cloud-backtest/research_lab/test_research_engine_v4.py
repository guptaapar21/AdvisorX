
import json
import pandas as pd
import pytest

from .research_features import feature_snapshot
from .outcomes import label_observation
from .research_hypotheses import _group_rows, _split_grouped, evaluate_one
from .research_walkforward import split_purged, assert_clean_boundaries
from .research_discovery import discover_multi
from .research_miner import mine_discovery
from .research_config import FEATURE_SCHEMA_VERSION

def candles(n=400,start="2026-01-01T00:00:00Z"):
    idx=pd.date_range(start,periods=n,freq="1min",tz="UTC")
    close=pd.Series(range(n),index=idx,dtype=float)+100
    return pd.DataFrame({
        "open":close,
        "high":close+1,
        "low":close-1,
        "close":close,
        "volume":10.0,
    },index=idx)

def rows(n=1100):
    out=[]
    for i in range(n):
        t=(pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(minutes=i)).isoformat()
        for symbol in ("BTC","ETH","SOL","ADA","LINK"):
            for h in (5,10,15,30,60):
                edge=0.001 if (i<int(n*.60) and h==15 and i%5==0) else -0.0002
                out.append({
                    "observation_id":f"{symbol}:{i}",
                    "symbol":symbol,
                    "feature_time":t,
                    "feature":{"rvol20":1.8 if i%5==0 else .8,"return_5m":.002 if i%5==0 else -.001},
                    "outcome":{"horizon_min":h,"net_return_long":edge,"net_return_short":-edge,
                               "mfe_long":max(edge,0),"mae_long":.001,"mfe_short":max(-edge,0),"mae_short:.001":0.001},
                })
    # fix test typo deterministically
    for r in out:
        r["outcome"]["mae_short"]=r["outcome"].pop("mae_short:.001")
    return out

def test_feature_is_causal_and_versioned():
    c=candles()
    t=c.index[200]
    a=feature_snapshot(c,t)
    c2=c.copy()
    c2.loc[c2.index>t,["close","volume"]]=9999
    b=feature_snapshot(c2,t)
    assert a==b
    assert a["feature_schema_version"]==FEATURE_SCHEMA_VERSION

def test_outcome_requires_complete_future_path():
    c=candles()
    t=c.index[100]
    assert label_observation(c,t,now=t+pd.Timedelta(minutes=5,seconds=1))==[]
    c_gap=c.drop(c.index[105])
    assert 5 not in {x["horizon_min"] for x in label_observation(c_gap,t,now=t+pd.Timedelta(minutes=7))}

def test_outcome_uses_only_strict_future():
    c=candles()
    t=c.index[100]
    o=label_observation(c,t,now=t+pd.Timedelta(minutes=61))
    assert {5,10,15,30,60}=={x["horizon_min"] for x in o}
    assert all(pd.Timestamp(x["outcome_time"])>t for x in o)

def test_group_rows_rejects_duplicate_horizon():
    r=rows(80)
    r.append(dict(r[0]))
    with pytest.raises(AssertionError):
        _group_rows(r)

def test_walkforward_keys_include_symbol_observation():
    rr=[]
    for i in range(500):
        t=(pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(minutes=i)).isoformat()
        for sym in ("BTC","ETH","SOL"):
            for h in (5,10,15,30,60):
                rr.append({"observation_id":f"{sym}:{i}","symbol":sym,"feature_time":t,"horizon_min":h})
    d,v,h=split_purged(rr)
    assert d and v and h
    assert_clean_boundaries(d,v,h,pd.Timestamp(rr[0]["feature_time"]),pd.Timestamp("2026-01-01T05:00Z"),pd.Timestamp("2026-01-01T08:20Z"))

def test_miner_finds_both_directions():
    grouped=_group_rows(rows(1100))
    discovery,_,_= _split_grouped(grouped)
    c=mine_discovery(discovery)
    assert c
    assert {"LONG","SHORT"} <= {x["direction"] for x in c}

def test_future_rows_cannot_change_past_feature_context():
    base=rows(1100)
    extended=rows(1400)
    base_ids={r["observation_id"] for r in base}
    past_extended=[r for r in extended if r["observation_id"] in base_ids]
    assert _group_rows(base)[0]["features"] == _group_rows(past_extended)[0]["features"]

def test_mined_candidates_work_without_gemini():
    result=discover_multi("",rows(1100),"test-model",{"validated":[],"rejected":[]})
    assert result["status"]=="ok"
    assert result["quant_candidate_count"]>0

def test_evaluation_reports_block_robustness():
    grouped=_group_rows(rows(1100))
    discovery,_,_= _split_grouped(grouped)
    rule={"name":"rvol","direction":"LONG","horizon_min":15,"conditions":[{"feature":"rvol20","op":">=","value":0.5}]}
    m=evaluate_one(rule,discovery)
    assert "robustness" in m
    assert m["robustness"]["time_block_count"]>0


def test_holdout_budget_is_bounded():
    from .research_hypotheses import evaluate
    grouped = _group_rows(rows(1100))
    rules = [
        {
            "name": f"r{i}",
            "direction": "LONG",
            "horizon_min": 15,
            "conditions": [
                {"feature": "rvol20", "op": ">=", "value": 0.5 + (i % 3) * 0.1}
            ],
        }
        for i in range(20)
    ]
    evaluated = evaluate(rules, rows(1100), holdout_budget=3)
    assert sum(bool(x.get("holdout_tested")) for x in evaluated) == 3


def test_validation_rank_mixed_empty_and_nonempty_candidates():
    from .research_hypotheses import _validation_rank
    empty = {"validation": {"n": 0, "robustness": {}}}
    populated = {
        "validation": {
            "n": 100,
            "profit_factor": 1.2,
            "avg_net_return": 0.001,
            "robustness": {
                "block_ci95_low": 0.0001,
                "positive_block_fraction": 0.6,
                "positive_coin_fraction": 0.8,
            },
        }
    }
    assert isinstance(_validation_rank(empty), tuple)
    assert isinstance(_validation_rank(populated), tuple)
    assert sorted([empty, populated], key=_validation_rank, reverse=True)[0] is populated
