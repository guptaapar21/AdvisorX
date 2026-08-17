
from __future__ import annotations
import json, math, os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, pstdev
import pandas as pd
from .research_config import OBS_FILE, OUTCOME_FILE, HYPOTHESIS_FILE, ANALYSIS_FILE, STATE_FILE, MEMORY_FILE

def parse_ts(value):
    if value is None: return None
    try:
        t=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (TypeError,ValueError):
        return None

def read_jsonl(path):
    p=Path(path)
    if not p.exists(): return []
    rows=[]
    for line_no,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt JSONL {p} line {line_no}") from exc
    return rows

def read_json(path, default):
    p=Path(path)
    if not p.exists(): return default
    try: return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise ValueError(f"Corrupt JSON {p}") from exc

def _num(v):
    try:
        x=float(v); return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None

def build_audit(cutoff=None):
    cutoff=cutoff or datetime.now(timezone.utc)
    obs=read_jsonl(OBS_FILE)
    outcomes=read_jsonl(OUTCOME_FILE)
    batches=read_jsonl(HYPOTHESIS_FILE)
    analysis=read_jsonl(ANALYSIS_FILE)
    state=read_json(STATE_FILE,{})
    memory=read_json(MEMORY_FILE,{"validated":[],"rejected":[]})

    valid_obs=[x for x in obs if parse_ts((x.get("features") or {}).get("feature_time")) and parse_ts((x.get("features") or {}).get("feature_time"))<=cutoff]
    valid_ids={x.get("observation_id") for x in valid_obs}
    valid_out=[x for x in outcomes if x.get("observation_id") in valid_ids and parse_ts(x.get("outcome_time")) and parse_ts(x.get("outcome_time"))<=cutoff]

    status_counts=Counter(str(x.get("status","unknown")) for x in analysis)
    evaluated=[]
    quant_leads=0
    candidate_count=0
    holdout_tested=0
    for batch in batches:
        if parse_ts(batch.get("generated_at")) and parse_ts(batch["generated_at"])>cutoff:
            continue
        d=batch.get("discovery") or {}
        quant_leads+=int(d.get("quant_candidate_count",0) or 0)
        candidate_count+=int(batch.get("candidate_count",0) or 0)
        holdout_tested+=int(batch.get("holdout_tested_count",0) or 0)
        evaluated.extend(batch.get("evaluated",[]))

    hstatus=Counter(str(x.get("status","unknown")) for x in evaluated)
    holdout_passed=[x for x in evaluated if x.get("status")=="HOLDOUT_PASSED"]
    raw_effective_blocks=set()
    symbols=set()
    for x in valid_out:
        try:
            symbols.add(x.get("symbol"))
            raw_effective_blocks.add(
                (pd.Timestamp(x.get("feature_time")).floor("30min").isoformat())
            )
        except Exception:
            pass

    direction=defaultdict(list)
    for x in valid_out:
        for name,key in (("LONG","net_return_long"),("SHORT","net_return_short")):
            v=_num(x.get(key))
            if v is not None: direction[name].append(v)

    def summarize(values):
        if not values:
            return {"n":0,"positive_rate":0.0,"avg":0.0,"median":0.0,"std":0.0}
        return {
            "n":len(values),
            "positive_rate":sum(v>0 for v in values)/len(values),
            "avg":sum(values)/len(values),
            "median":median(values),
            "std":pstdev(values) if len(values)>1 else 0.0,
        }

    last_run=parse_ts(state.get("last_run"))
    warnings=[]
    if last_run and (cutoff-last_run).total_seconds()>600:
        warnings.append("last_run is more than 10 minutes old")

    report={
        "audit_cutoff":cutoff.isoformat(),
        "research_only":state.get("research_only",True),
        "state":{
            "last_run":last_run.isoformat() if last_run else None,
            "last_analysis_at":state.get("last_analysis_attempt_at"),
            "last_analysis_status":state.get("last_analysis_attempt_status"),
            "last_outcome_count":state.get("last_outcome_count",0),
            "experiments":len(state.get("research_experiments",[]) or []),
        },
        "data_health":{
            "observations":len(valid_obs),
            "matured_outcomes":len(valid_out),
            "raw_outcome_rows":len(outcomes),
            "unique_symbols":len(symbols),
            "30m_time_blocks":len(raw_effective_blocks),
            "analysis_attempts":len(analysis),
            "analysis_status_counts":dict(status_counts),
            "candidate_leads_mined":quant_leads,
            "candidate_evaluations":candidate_count,
            "holdout_tests":holdout_tested,
            "holdout_passed":len(holdout_passed),
            "memory_validated":len(memory.get("validated",[])),
            "memory_rejected":len(memory.get("rejected",[])),
        },
        "hypothesis_status":dict(hstatus),
        "direction_baseline":{k:summarize(v) for k,v in direction.items()},
        "holdout_passed":holdout_passed[:10],
        "recent_rejected":list(reversed(evaluated[-10:])),
        "warnings":warnings,
    }
    return report, render_markdown(report)

def render_markdown(report):
    d=report["data_health"]
    lines=[
        "# AdvisorX ResearchLab Audit","",
        f"**Audit cutoff:** `{report['audit_cutoff']}`",
        f"**Research-only:** `{report['research_only']}`","",
        "## Research activity","",
        f"- Observations: **{d['observations']:,}**",
        f"- Matured outcomes: **{d['matured_outcomes']:,}**",
        f"- Unique symbols: **{d['unique_symbols']:,}**",
        f"- 30m time blocks: **{d['30m_time_blocks']:,}**",
        f"- Analysis attempts: **{d['analysis_attempts']:,}**",
        f"- Numerical candidates mined: **{d['candidate_leads_mined']:,}**",
        f"- Candidate evaluations: **{d['candidate_evaluations']:,}**",
        f"- Holdout tests: **{d['holdout_tests']:,}**",
        f"- Holdout passed: **{d['holdout_passed']:,}**","",
        "## Analysis status","",
        "|Status|Count|","|---|---:|",
    ]
    for k,v in d["analysis_status_counts"].items():
        lines.append(f"|{k}|{v}|")
    lines += ["","## Hypothesis status","","|Status|Count|","|---|---:|"]
    for k,v in report["hypothesis_status"].items():
        lines.append(f"|{k}|{v}|")
    lines += ["","## LONG / SHORT baseline","","|Direction|N|Positive|Avg|Median|Std|","|---|---:|---:|---:|---:|---:|"]
    for k,v in report["direction_baseline"].items():
        lines.append(f"|{k}|{v['n']}|{v['positive_rate']:.1%}|{v['avg']:.6f}|{v['median']:.6f}|{v['std']:.6f}|")
    lines += ["","## Holdout passed",""]
    if not report["holdout_passed"]:
        lines.append("None. No candidate has passed validation + robustness + holdout yet.")
    else:
        for x in report["holdout_passed"][:10]:
            h=x.get("holdout",{}); hp=x.get("hypothesis",{})
            lines.append(f"- **{hp.get('name','unnamed')}** {hp.get('direction')} {hp.get('horizon_min')}m: PF {h.get('profit_factor',0):.3f}, avg {h.get('avg_net_return',0):.6f}")
    lines += ["","## Warnings",""]
    lines.extend(f"- {w}" for w in report["warnings"]) or lines.append("None.")
    lines += ["","## Safety","","Research-only. This report never modifies AdvisorX decisions, risk, positions, prompts, ledger, or Telegram output."]
    return "\n".join(lines)

def main():
    cutoff=parse_ts(os.getenv("RESEARCH_AUDIT_CUTOFF"))
    report,markdown=build_audit(cutoff)
    output=Path(os.getenv("RESEARCH_AUDIT_OUTPUT_DIR","research_audit_output"))
    output.mkdir(parents=True,exist_ok=True)
    (output/"research_audit.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    (output/"research_audit.md").write_text(markdown,encoding="utf-8")
    summary=os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary,"a",encoding="utf-8") as f: f.write(markdown)
    print(markdown)

if __name__=="__main__":
    main()
