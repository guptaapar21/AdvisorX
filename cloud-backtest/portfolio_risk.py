"""Deterministic portfolio concentration guard for AdvisorX V4.

The market is highly correlated across the tracked altcoin futures. V4 keeps
diagnostics but adds a modest execution guard so one BTC-beta impulse cannot
silently become many full-risk positions.
"""
from __future__ import annotations
import os
from typing import Any, Mapping

BETA_CLUSTERS = {
    "BTC_BETA": {"BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "LTC", "LINK", "TRX", "AVAX", "HEI", "BICO", "HYPE", "ZEC", "ZBT", "ADA", "ACE"},
    "GOLD": {"PAXG"},
}
MAX_SAME_DIRECTION = int(os.environ.get("V4_MAX_CORRELATED_SAME_DIRECTION", "2"))
MAX_CLUSTER_TOTAL = int(os.environ.get("V4_MAX_CORRELATED_CLUSTER_POSITIONS", "3"))

def cluster_for_coin(coin: str) -> str:
    value = str(coin).upper()
    for name, coins in BETA_CLUSTERS.items():
        if value in coins: return name
    return "OTHER"

def summarize(open_positions):
    counts={}
    for position in open_positions or []:
        key=(cluster_for_coin(position.get("coin")),str(position.get("direction","")).upper()); counts[key]=counts.get(key,0)+1
    return [{"cluster":k[0],"direction":k[1],"positions":v} for k,v in sorted(counts.items())]

def apply_execution_gate(flagged: Mapping[str, Mapping[str, Any]], open_positions: list[Mapping[str, Any]]) -> int:
    total={}; by_dir={}
    for p in open_positions or []:
        c=cluster_for_coin(p.get("coin")); d=str(p.get("direction") or "").lower(); total[c]=total.get(c,0)+1; by_dir[(c,d)]=by_dir.get((c,d),0)+1
    candidates=[]
    for coin,d in flagged.items():
        if isinstance(d,dict) and d.get("take_trade"):
            candidates.append((int(d.get("conviction") or 0),str(coin),cluster_for_coin(coin),str(d.get("direction") or "").lower(),d))
    candidates.sort(key=lambda x:(-x[0],x[1])); rejected=0
    for _,coin,c,d,decision in candidates:
        same=by_dir.get((c,d),0); all_count=total.get(c,0)
        if same>=MAX_SAME_DIRECTION:
            reason=f"correlated {c} {d} exposure already at {same}/{MAX_SAME_DIRECTION}"
        elif all_count>=MAX_CLUSTER_TOTAL:
            reason=f"correlated {c} exposure already at {all_count}/{MAX_CLUSTER_TOTAL}"
        else:
            by_dir[(c,d)]=same+1; total[c]=all_count+1; continue
        decision["take_trade"]=False; decision["risk_validation_error"]=f"portfolio_risk_gate: {reason}"; decision["_entry_quality_reject_reason"]=reason; decision["portfolio_risk_gate_version"]="2026-09-10-v4"; rejected+=1
    return rejected
