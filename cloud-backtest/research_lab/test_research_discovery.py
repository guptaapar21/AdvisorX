
import pandas as pd
from .research_discovery import (
    DISCOVERY_INPUT_CHAR_BUDGET,
    DISCOVERY_PASS_OBSERVATIONS,
    _compact_observation,
    _prompt,
    _sample_pass,
    discover_multi,
)

def _rows(n=800):
    rows=[]
    base=pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(n):
        t=base+pd.Timedelta(minutes=i)
        f={"rsi14":35.0+i%30, "adx14":25.0+i%15, "rvol20":1.0+i%5}
        for h in (5,10,15,30,60):
            rows.append({
                "observation_id":f"O{i}","symbol":f"S{i%15}","feature_time":t.isoformat(),
                "feature":f,
                "outcome":{"horizon_min":h,"net_return_long":.001,"net_return_short":-.001,
                           "mfe_long":.002,"mae_long":.001,"mfe_short":.001,"mae_short":.002},
            })
    return rows

def test_prompt_budget():
    rows=_rows()
    from .research_discovery import _group_rows, _split_grouped
    disc,_,_=_split_grouped(_group_rows(rows))
    sample=_sample_pass(disc,0,12,DISCOVERY_PASS_OBSERVATIONS)
    assert len(sample)<=DISCOVERY_PASS_OBSERVATIONS
    assert len(_prompt([_compact_observation(r) for r in sample],{})) <= DISCOVERY_INPUT_CHAR_BUDGET

def test_twelve_key_multi_pass(monkeypatch):
    calls=[]
    def fake(key,sample,model,memory):
        calls.append((key,len(sample)))
        return {
            "status":"ok",
            "hypotheses":[{
                "name":f"idea_{key}",
                "direction":"LONG" if len(calls)%2 else "SHORT",
                "horizon_min":15,
                "conditions":[{"feature":"rsi14","op":">","value":30+len(calls)}],
                "rationale":"test"
            }]
        }
    monkeypatch.setattr("research_lab.research_discovery._discover_one", fake)
    result=discover_multi(",".join(f"k{i}" for i in range(12)),_rows(),"test-model",{})
    assert result["status"]=="ok"
    assert result["discovery_passes"]==12
    assert result["successful_passes"]==12
    assert len(calls)==12
    assert sum(n for _,n in calls)==12*DISCOVERY_PASS_OBSERVATIONS
    assert len(result["hypotheses"])==12


def test_all_discovery_rows_get_coverage_across_passes():
    rows=_rows()
    from .research_discovery import _group_rows, _split_grouped
    from collections import Counter
    disc,_,_=_split_grouped(_group_rows(rows))
    counts=Counter()
    for i in range(12):
        for row in _sample_pass(disc,i,12,DISCOVERY_PASS_OBSERVATIONS):
            counts[row["observation_id"]]+=1
    assert len(counts)==len(disc)
