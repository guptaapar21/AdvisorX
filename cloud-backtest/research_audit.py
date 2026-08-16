from __future__ import annotations
import json, math, os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, pstdev
from .research_config import DATA_DIR, OBS_FILE, OUTCOME_FILE, HYPOTHESIS_FILE, ANALYSIS_FILE, STATE_FILE, MEMORY_FILE

def utc_now(): return datetime.now(timezone.utc).replace(microsecond=0)
def parse_ts(v):
    if v is None: return None
    try:
        x=datetime.fromisoformat(str(v).replace('Z','+00:00'))
        if x.tzinfo is None: x=x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc)
    except (TypeError,ValueError): return None

def read_jsonl(p):
    if not Path(p).exists(): return []
    out=[]
    for line in Path(p).read_text(encoding='utf-8').splitlines():
        try:
            if line.strip(): out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return [x for x in out if isinstance(x,dict)]

def read_json(p, default):
    try: return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception: return default

def ff(v):
    try:
        x=float(v); return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None

def build_audit(cutoff=None):
    cutoff=cutoff or utc_now()
    raw_obs=read_jsonl(OBS_FILE); raw_out=read_jsonl(OUTCOME_FILE)
    raw_h=read_jsonl(HYPOTHESIS_FILE); raw_a=read_jsonl(ANALYSIS_FILE)
    state=read_json(STATE_FILE,{}); memory=read_json(MEMORY_FILE,{"validated":[],"rejected":[]})
    obs=[]
    for r in raw_obs:
        f=r.get('features') or {}; t=parse_ts(r.get('feature_time') or f.get('feature_time'))
        if t and t<=cutoff: obs.append(r)
    ids={r.get('observation_id') for r in obs}
    outcomes=[]
    for r in raw_out:
        start=parse_ts(r.get('feature_time') or r.get('outcome_start') or r.get('observation_time'))
        end=parse_ts(r.get('outcome_end') or r.get('end_time') or r.get('horizon_end'))
        if r.get('observation_id') in ids and start and start<=cutoff and end and end<=cutoff: outcomes.append(r)
    hypotheses=[r for r in raw_h if (parse_ts(r.get('generated_at') or r.get('analysis_time')) is None or parse_ts(r.get('generated_at') or r.get('analysis_time'))<=cutoff)]
    analyses=[r for r in raw_a if (parse_ts(r.get('time') or r.get('generated_at')) is None or parse_ts(r.get('time') or r.get('generated_at'))<=cutoff)]
    obs_sym=Counter(str(x.get('symbol','UNKNOWN')) for x in obs); out_sym=Counter(str(x.get('symbol','UNKNOWN')) for x in outcomes)
    out_h=Counter()
    for x in outcomes:
        try: out_h[str(int(x.get('horizon_min')))]+=1
        except: pass
    dret=defaultdict(list)
    for x in outcomes:
        for d,k in [('LONG','net_return_long'),('SHORT','net_return_short')]:
            v=ff(x.get(k))
            if v is not None: dret[d].append(v)
    def summ(vals):
        if not vals:return {'n':0,'positive_rate':0.0,'avg':0.0,'median':0.0,'std':0.0}
        return {'n':len(vals),'positive_rate':sum(v>0 for v in vals)/len(vals),'avg':sum(vals)/len(vals),'median':median(vals),'std':pstdev(vals) if len(vals)>1 else 0.0}
    statuses=Counter(); survivors=[]; rejected=[]
    for batch in hypotheses:
        for item in batch.get('evaluated',[]):
            h=item.get('hypothesis') or {}; status=str(item.get('status','UNKNOWN')); statuses[status]+=1
            rec={'name':h.get('name'),'direction':h.get('direction'),'horizon_min':h.get('horizon_min'),'status':status,'conditions':h.get('conditions',[]),'validation':item.get('validation',{}),'holdout':item.get('holdout',{})}
            (survivors if status=='HOLDOUT_PASSED' else rejected).append(rec)
    survivors.sort(key=lambda r: ((r.get('holdout') or {}).get('profit_factor') or -1,(r.get('holdout') or {}).get('avg_net_return') or -1), reverse=True)
    last_run=parse_ts(state.get('last_run')); last_candle=parse_ts(state.get('last_closed_candle'))
    warnings=[]
    if not obs:warnings.append('No observations available at audit cutoff.')
    if not outcomes:warnings.append('No matured outcomes available at audit cutoff.')
    if last_run and (cutoff-last_run).total_seconds()>600:warnings.append('ResearchLab last_run is more than 10 minutes before audit cutoff.')
    if state.get('research_only',True) is not True:warnings.append('State does not advertise research_only=true.')
    report={'audit_cutoff':cutoff.isoformat(),'research_only':state.get('research_only',True),'state':{'last_run':last_run.isoformat() if last_run else None,'last_closed_candle':last_candle.isoformat() if last_candle else None},'data_health':{'observations':len(obs),'matured_outcomes':len(outcomes),'observations_by_symbol':dict(sorted(obs_sym.items())),'outcomes_by_symbol':dict(sorted(out_sym.items())),'outcomes_by_horizon':dict(sorted(out_h.items(),key=lambda x:int(x[0]))),'hypothesis_batches':len(hypotheses),'analysis_runs':len(analyses),'validated_memory':len(memory.get('validated',[])),'rejected_memory':len(memory.get('rejected',[]))},'research_performance':{'hypothesis_status_counts':dict(sorted(statuses.items())),'holdout_survivors':survivors[:10],'recent_rejections':list(reversed(rejected[-10:])),'direction_outcomes':{k:summ(v) for k,v in sorted(dret.items())}},'warnings':warnings}
    return report,render_markdown(report)

def render_markdown(r):
    d=r['data_health']; p=r['research_performance']; L=['# AdvisorX ResearchLab Audit','',f"**Audit cutoff (UTC):** `{r['audit_cutoff']}`",f"**Research-only:** `{r['research_only']}`",'','## Executive summary','',f"- Observations: **{d['observations']:,}**",f"- Matured outcomes: **{d['matured_outcomes']:,}**",f"- Research analyses: **{d['analysis_runs']:,}**",f"- Hypothesis batches: **{d['hypothesis_batches']:,}**",f"- Validated memory: **{d['validated_memory']:,}**",f"- Rejected memory: **{d['rejected_memory']:,}**",'','## Hypothesis status','','| Status | Count |','|---|---:|']
    for s,n in (p['hypothesis_status_counts'] or {'NONE':0}).items(): L.append(f'| {s} | {n} |')
    L += ['','## Outcomes by horizon','','| Horizon | Matured |','|---:|---:|']
    for h,n in d['outcomes_by_horizon'].items(): L.append(f'| {h} min | {n:,} |')
    L += ['','## LONG / SHORT research outcome summary','','| Direction | N | Positive rate | Avg net return | Median | Std |','|---|---:|---:|---:|---:|---:|']
    for direction,s in p['direction_outcomes'].items(): L.append(f"| {direction} | {s['n']:,} | {s['positive_rate']:.1%} | {s['avg']:.6f} | {s['median']:.6f} | {s['std']:.6f} |")
    L += ['','## Holdout-passed hypotheses','']
    if not p['holdout_survivors']: L.append('No hypothesis has passed the current validation + holdout gates yet.')
    for i,x in enumerate(p['holdout_survivors'],1):
        h=x['holdout']; L += [f"### {i}. {x.get('name') or 'Unnamed hypothesis'}",f"- Direction: `{x.get('direction')}` | Horizon: `{x.get('horizon_min')}m`",f"- Holdout N: `{h.get('n',0)}` | PF: `{h.get('profit_factor',0):.3f}` | Avg net return: `{h.get('avg_net_return',0):.6f}`",f"- Win rate: `{h.get('positive_rate',0):.1%}` | Max drawdown: `{h.get('max_drawdown',0):.6f}` | Avg MFE: `{h.get('avg_mfe',0):.6f}` | Avg MAE: `{h.get('avg_mae',0):.6f}`",f"- Conditions: `{json.dumps(x.get('conditions',[]),separators=(',',':'))}`",'']
    L += ['## Recent rejected hypotheses','']
    if not p['recent_rejections']: L.append('No rejected hypotheses recorded yet.')
    else:
        L += ['| Name | Direction | Horizon | Status | Validation PF | Holdout PF |','|---|---|---:|---|---:|---:|']
        for x in p['recent_rejections']:
            v=x.get('validation') or {}; h=x.get('holdout') or {}; L.append(f"| {x.get('name') or '-'} | {x.get('direction') or '-'} | {x.get('horizon_min') or '-'} | {x.get('status')} | {v.get('profit_factor',0):.3f} | {h.get('profit_factor',0):.3f} |")
    L += ['','## Warnings','']; L.extend(f'- {w}' for w in r['warnings']) if r['warnings'] else L.append('None.')
    L += ['','## Safety','','This report is observational research only. It does not modify AdvisorX decisions, risk, positions, prompts, ledger, or Telegram output.','']
    return '\n'.join(L)

def main():
    cutoff=parse_ts(os.getenv('RESEARCH_AUDIT_CUTOFF')) or utc_now(); report,md=build_audit(cutoff)
    out=Path(os.getenv('RESEARCH_AUDIT_OUTPUT_DIR','research_audit_output')); out.mkdir(parents=True,exist_ok=True)
    (out/'research_audit.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8'); (out/'research_audit.md').write_text(md,encoding='utf-8')
    if os.getenv('GITHUB_STEP_SUMMARY'): Path(os.environ['GITHUB_STEP_SUMMARY']).open('a',encoding='utf-8').write(md)
    print(md)
if __name__=='__main__': main()
