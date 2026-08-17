from __future__ import annotations
import statistics
import pandas as pd
from .research_config import *
from .research_walkforward import split_purged


def _group_rows(rows):
    grouped={}
    for r in rows:
        oid=r.get('observation_id') or f"{r['feature_time']}"
        g=grouped.setdefault(
            oid,
            {
                'observation_id':oid,
                'symbol':r.get('symbol'),
                'feature_time':r['feature_time'],
                'features':r['feature'],
                'outcomes':{},
            },
        )
        g['outcomes'][str(int(r['outcome']['horizon_min']))]=r['outcome']
    return list(grouped.values())


def _bounds(grouped):
    ts=sorted({_ts(x['feature_time']) for x in grouped})
    if len(ts)<60:
        return None
    return ts[int(len(ts)*.60)-1],ts[int(len(ts)*.80)-1],ts[-1]


def _ts(x):
    t=pd.Timestamp(x)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def _split_grouped(grouped):
    b=_bounds(grouped)
    if not b:
        return [],[],[]
    d,v,h=b
    disc=[r for r in grouped if _ts(r['feature_time'])<d]
    val=[
        r for r in grouped
        if _ts(r['feature_time'])>=d+pd.Timedelta(minutes=PURGE_MIN)
        and _ts(r['feature_time'])<v
    ]
    hold=[
        r for r in grouped
        if _ts(r['feature_time'])>=v+pd.Timedelta(minutes=PURGE_MIN)
        and _ts(r['feature_time'])<h
    ]
    _assert_grouped_boundaries(disc,val,hold)
    return disc,val,hold


def _assert_grouped_boundaries(a,b,c):
    sa={_ts(r['feature_time']) for r in a}
    sb={_ts(r['feature_time']) for r in b}
    sc={_ts(r['feature_time']) for r in c}
    assert sa.isdisjoint(sb) and sb.isdisjoint(sc) and sa.isdisjoint(sc)
    if sa and sb:
        assert min(sb)-max(sa)>=pd.Timedelta(minutes=PURGE_MIN)
    if sb and sc:
        assert min(sc)-max(sb)>=pd.Timedelta(minutes=PURGE_MIN)


def _match(feature,cond):
    v=feature.get(cond.get('feature'))
    op=cond.get('op')
    x=cond.get('value')
    if v is None:
        return False
    try:
        if op=='between':
            return float(x[0])<=float(v)<=float(x[1])
        return {
            '>':float(v)>float(x),
            '>=':float(v)>=float(x),
            '<':float(v)<float(x),
            '<=':float(v)<=float(x),
        }[op]
    except (TypeError,ValueError,KeyError,IndexError):
        return False


def _metrics(vals,mfes,maes):
    if not vals:
        return {
            'n':0,'positive_rate':0.0,'avg_net_return':0.0,
            'median_net_return':0.0,'profit_factor':0.0,'worst_return':0.0,
            'avg_mfe':0.0,'avg_mae':0.0,'max_drawdown':0.0,'return_std':0.0,
        }
    wins=[v for v in vals if v>0]
    losses=[v for v in vals if v<0]
    gl=sum(wins)
    ll=-sum(losses)
    pf=gl/ll if ll else (999.0 if gl else 0.0)
    eq=peak=0.0
    dd=0.0
    for v in vals:
        eq+=v
        peak=max(peak,eq)
        dd=max(dd,peak-eq)
    return {
        'n':len(vals),
        'positive_rate':len(wins)/len(vals),
        'avg_net_return':sum(vals)/len(vals),
        'median_net_return':statistics.median(vals),
        'profit_factor':pf,
        'worst_return':min(vals),
        'avg_mfe':sum(mfes)/len(mfes) if mfes else 0.0,
        'avg_mae':sum(maes)/len(maes) if maes else 0.0,
        'max_drawdown':dd,
        'return_std':statistics.pstdev(vals) if len(vals)>1 else 0.0,
    }


def evaluate_one(rule,grouped):
    h=int(rule.get('horizon_min',15))
    direction=str(rule.get('direction','long')).lower()
    key='net_return_short' if direction=='short' else 'net_return_long'
    mk='mfe_short' if direction=='short' else 'mfe_long'
    ak='mae_short' if direction=='short' else 'mae_long'
    vals=[]
    mfes=[]
    maes=[]
    rows=sorted(grouped,key=lambda r:_ts(r['feature_time']))
    for r in rows:
        o=r['outcomes'].get(str(h))
        if not o:
            continue
        if all(_match(r['features'],c) for c in rule.get('conditions',[])):
            vals.append(float(o.get(key,0)))
            mfes.append(float(o.get(mk,0)))
            maes.append(float(o.get(ak,0)))
    return _metrics(vals,mfes,maes)


def evaluate(hypotheses,rows):
    grouped=_group_rows(rows)
    disc,val,hold=_split_grouped(grouped)
    return [
        {
            'hypothesis':h,
            'validation':evaluate_one(h,val),
            'holdout':evaluate_one(h,hold),
        }
        for h in hypotheses
    ]


def classify(e):
    v=e['validation']
    h=e['holdout']
    return 'HOLDOUT_PASSED' if (
        v['n']>=PROMOTE_MIN_VALID_N
        and h['n']>=PROMOTE_MIN_HOLDOUT_N
        and v['profit_factor']>=PROMOTE_MIN_VALID_PF
        and h['profit_factor']>=PROMOTE_MIN_HOLDOUT_PF
        and v['avg_net_return']>=PROMOTE_MIN_VALID_AVG
        and h['avg_net_return']>=PROMOTE_MIN_HOLDOUT_AVG
        and h['positive_rate']>=PROMOTE_MIN_HOLDOUT_WINRATE
        and h['max_drawdown']<=PROMOTE_MAX_HOLDOUT_DD
    ) else (
        'VALIDATION_PASSED'
        if (
            v['n']>=PROMOTE_MIN_VALID_N
            and v['profit_factor']>=PROMOTE_MIN_VALID_PF
            and v['avg_net_return']>=PROMOTE_MIN_VALID_AVG
        )
        else 'VALIDATION_FAILED'
    )
