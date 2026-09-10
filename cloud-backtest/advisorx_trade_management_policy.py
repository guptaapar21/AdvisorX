"""AdvisorX V4 deterministic trade-management policy.

This module complements Gemini rather than replacing it:
- Gemini decides whether/why to enter and whether a thesis is invalidated.
- Python progressively protects already-earned profit.
- MFE/MAE measurement stops at the first target/stop event visible in 1m data.
- Original and active levels remain distinguishable through level_history.
"""
from __future__ import annotations
from typing import Any, Mapping
import os
import pandas as pd

POLICY_VERSION = "2026-09-10-trade-management-v4"
DEFAULT_MIN_TIGHTEN_R = 0.50
PROFIT_LADDER = ((0.75, 0.00), (1.00, 0.25), (1.50, 0.60), (2.00, 1.00))
SEVERE_GIVEBACK_PCT = float(os.environ.get("V4_SEVERE_GIVEBACK_PCT", "75"))
SEVERE_GIVEBACK_MFE_R = float(os.environ.get("V4_SEVERE_GIVEBACK_MFE_R", "1.0"))
SEVERE_HEALTH_SCORE = int(os.environ.get("V4_SEVERE_HEALTH_SCORE", "3"))

def _utc(value: Any):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

def current_profit_r(position: Mapping[str, Any]) -> float | None:
    try:
        entry=float(position["entry_price"]); current=float(position["current_price"]); risk=float(position.get("max_loss_this_trade_inr") or position.get("risk_inr") or 0); amount=float(position.get("trade_amount_inr") or 0); direction=str(position.get("direction") or "").lower(); fx=float(position.get("usdt_inr_rate") or os.environ.get("USDT_INR_RATE","99.44"))
        if entry<=0 or amount<=0 or risk<=0 or fx<=0 or direction not in {"long","short"}: return None
        qty=amount/(entry*fx); gross=qty*((current-entry) if direction=="long" else (entry-current))*fx
        return gross/risk
    except (KeyError,TypeError,ValueError,ZeroDivisionError): return None

def _gross_at_price(entry_price: float, price: float, amount: float, direction: str, fx: float) -> float:
    qty=amount/(entry_price*fx)
    return qty*((price-entry_price) if direction=="long" else (entry_price-price))*fx

def apply_management_policy(position_updates: Mapping[str, Mapping[str, Any]], open_positions: list[Mapping[str, Any]], min_tighten_r: float=DEFAULT_MIN_TIGHTEN_R) -> dict[str,dict[str,Any]]:
    by_coin={str(p.get("coin")):p for p in open_positions if p.get("coin")}; out={}
    for coin,raw in position_updates.items():
        update=dict(raw or {}); action=str(update.get("action") or "hold").strip().lower(); update["action"]=action; position=by_coin.get(str(coin))
        if action=="tighten_stop" and position is not None:
            r=current_profit_r(position); proposed=update.get("updated_stop_loss"); current_stop=position.get("stop_loss"); direction=str(position.get("direction") or "").lower(); bad=False
            try:
                if proposed is not None and current_stop is not None:
                    bad=(direction=="long" and float(proposed)<=float(current_stop)) or (direction=="short" and float(proposed)>=float(current_stop))
            except (TypeError,ValueError): bad=True
            if r is None or r<min_tighten_r:
                update["action"]="hold"; update["management_policy"]="tighten_suppressed_before_profit_threshold"; update["management_policy_r"]=round(r,3) if r is not None else None; update["original_action"]="tighten_stop"; update["reasoning"]=("Management policy kept HOLD because tighten_stop was requested before "+f"+{min_tighten_r:.2f}R. "+str(update.get("reasoning") or "")).strip()
            elif bad:
                update["action"]="hold"; update["management_policy"]="non_monotonic_stop_move_suppressed"; update["original_action"]="tighten_stop"; update["reasoning"]=("Management policy kept HOLD because the proposed stop was not a strict tightening. "+str(update.get("reasoning") or "")).strip()
        elif action=="move_target" and position is not None:
            proposed=update.get("updated_target_price"); current_target=position.get("target_price"); direction=str(position.get("direction") or "").lower()
            try: bad=proposed is not None and current_target is not None and ((direction=="long" and float(proposed)<float(current_target)) or (direction=="short" and float(proposed)>float(current_target)))
            except (TypeError,ValueError): bad=True
            if bad:
                update["action"]="hold"; update["management_policy"]="inward_target_move_suppressed"; update["original_action"]="move_target"; update["reasoning"]=("Management policy kept HOLD because the proposed target moved inward. "+str(update.get("reasoning") or "")).strip()
        out[str(coin)]=update
    return out

def add_signal_provenance(flagged: Mapping[str, Mapping[str, Any]]) -> None:
    for item in flagged.values():
        if not isinstance(item,dict): continue
        item.setdefault("gemini_proposed_entry",item.get("entry_price")); item.setdefault("gemini_proposed_stop",item.get("stop_loss")); item.setdefault("gemini_proposed_target",item.get("target_price")); item.setdefault("trade_management_policy_version",POLICY_VERSION)

def _active_levels_at(history:list[Mapping[str,Any]], candle_time):
    if not history: return None,None
    ct=_utc(candle_time); active=history[0]
    for level in sorted(history,key=lambda x:_utc(x["from"])):
        if _utc(level["from"])<=ct: active=level
        else: break
    return active.get("stop"),active.get("target")

def update_mfe_mae_until_exit(ledger:list[dict[str,Any]], fetched:Mapping[str,Any]) -> None:
    for entry in ledger:
        if entry.get("status")!="pending": continue
        data=fetched.get(entry.get("coin"))
        if not data or len(data)<4: continue
        c1=data[3]
        if c1 is None or getattr(c1,"empty",True): continue
        try:
            entry_price=float(entry.get("entry_price")); amount=float(entry.get("trade_amount_inr")); fx=float(os.environ.get("USDT_INR_RATE","99.44")); direction=str(entry.get("direction") or "").lower(); qty=amount/(entry_price*fx)
            if entry_price<=0 or amount<=0 or direction not in {"long","short"}: continue
        except (TypeError,ValueError,ZeroDivisionError): continue
        entry.setdefault("peak_price",entry_price); entry.setdefault("trough_price",entry_price); entry.setdefault("mfe_pnl_inr",0.0); entry.setdefault("mae_pnl_inr",0.0); entry.setdefault("mfe_time",entry.get("time")); entry.setdefault("mae_time",entry.get("time"))
        since=c1[c1.index>_utc(entry.get("time")).tz_localize(None)]
        if since.empty: continue
        history=entry.get("level_history") or [{"from":entry.get("time"),"stop":entry.get("stop_loss"),"target":entry.get("target_price")}]
        for idx,row in since.iterrows():
            ct=_utc(idx); stop,target=_active_levels_at(history,ct)
            try: stop=float(stop) if stop is not None else None; target=float(target) if target is not None else None
            except (TypeError,ValueError): stop=target=None
            high=float(row["high"]); low=float(row["low"])
            if direction=="long":
                fav,adv=high,low; target_hit=target is not None and high>=target; stop_hit=stop is not None and low<=stop
                if fav>float(entry["peak_price"]): entry["peak_price"]=fav; entry["mfe_time"]=ct.isoformat()
                if adv<float(entry["trough_price"]): entry["trough_price"]=adv; entry["mae_time"]=ct.isoformat()
                fav_pnl=qty*(fav-entry_price)*fx; adv_pnl=qty*(adv-entry_price)*fx
            else:
                fav,adv=low,high; target_hit=target is not None and low<=target; stop_hit=stop is not None and high>=stop
                if fav<float(entry["trough_price"]): entry["trough_price"]=fav; entry["mfe_time"]=ct.isoformat()
                if adv>float(entry["peak_price"]): entry["peak_price"]=adv; entry["mae_time"]=ct.isoformat()
                fav_pnl=qty*(entry_price-fav)*fx; adv_pnl=qty*(entry_price-adv)*fx
            if fav_pnl>float(entry.get("mfe_pnl_inr") or 0): entry["mfe_pnl_inr"]=round(fav_pnl,2)
            if adv_pnl<float(entry.get("mae_pnl_inr") or 0): entry["mae_pnl_inr"]=round(adv_pnl,2)
            if target_hit or stop_hit: break
        risk=float(entry.get("max_loss_this_trade_inr") or 0); mfe=float(entry.get("mfe_pnl_inr") or 0); entry["mfe_r"]=round(mfe/risk,3) if risk>0 else None; entry["mfe_mae_policy_version"]=POLICY_VERSION

def apply_profit_ladder(ledger:list[dict[str,Any]], current_prices:Mapping[str,float], now:Any, fetched=None)->list[dict[str,Any]]:
    summaries=[]
    for entry in ledger:
        if entry.get("status")!="pending": continue
        coin=entry.get("coin"); cp=current_prices.get(coin)
        if cp is None: continue
        try:
            ep=float(entry["entry_price"]); amount=float(entry.get("trade_amount_inr") or 0); risk=float(entry.get("max_loss_this_trade_inr") or 0); fx=float(os.environ.get("USDT_INR_RATE","99.44")); direction=str(entry.get("direction") or "").lower(); current_gross=_gross_at_price(ep,float(cp),amount,direction,fx); mfe=float(entry.get("mfe_pnl_inr") or 0); mfe_r=mfe/risk; current_r=current_gross/risk
            if ep<=0 or amount<=0 or risk<=0 or direction not in {"long","short"}: continue
        except (TypeError,ValueError,ZeroDivisionError): continue
        entry["current_r"]=round(current_r,3); entry["mfe_r"]=round(mfe_r,3)
        if mfe_r<0.75: continue
        lock_r=0.0; trigger=0.75
        for t,l in PROFIT_LADDER:
            if mfe_r>=t: trigger=t; lock_r=l
            else: break
        health=int(entry.get("profit_health_score") or 0); giveback=((mfe-current_gross)/mfe*100) if mfe>0 else 0; entry["mfe_giveback_pct"]=round(max(0,giveback),1)
        if mfe_r>=SEVERE_GIVEBACK_MFE_R and giveback>=SEVERE_GIVEBACK_PCT and health>=SEVERE_HEALTH_SCORE and current_r<=0.25:
            fees=amount*float(os.environ.get("TAKER_FEE_RATE","0.00075"))*2; entry["status"]="python_profit_exit"; entry["resolved_pnl"]=round(current_gross-fees,2); entry["resolved_time"]=_utc(now).isoformat(); entry["exit_reasoning"]=f"V4 profit protection: {giveback:.0f}% MFE giveback after {mfe_r:.2f}R MFE with health score {health}."; summaries.append({"coin":coin,"action":"profit_exit","pnl":entry["resolved_pnl"],"mfe_r":mfe_r,"giveback_pct":giveback}); continue
        desired_move=(lock_r*risk)/(amount/ep); desired=ep+desired_move if direction=="long" else ep-desired_move; old=float(entry.get("stop_loss") or 0)
        can=((direction=="long" and old<desired<float(cp) and current_r>lock_r+0.03) or (direction=="short" and float(cp)<desired<old and current_r>lock_r+0.03))
        if not can: continue
        entry["stop_loss"]=desired; entry["revised_at"]=_utc(now).isoformat(); entry.setdefault("level_history",[]).append({"from":_utc(now).isoformat(),"stop":desired,"target":entry.get("target_price"),"source":"python_v4_profit_ladder","lock_r":lock_r,"trigger_mfe_r":trigger}); entry["profit_lock_r"]=lock_r
        summaries.append({"coin":coin,"direction":direction,"action":"profit_ladder","new_stop_loss":desired,"lock_r":lock_r,"mfe_r":mfe_r,"current_r":current_r})
    return summaries
