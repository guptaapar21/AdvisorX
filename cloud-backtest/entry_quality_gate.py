from __future__ import annotations
import math, os
from typing import Any, Dict, Iterable, Tuple

MAX_NOTIONAL_INR = float(os.environ.get('MAX_NOTIONAL_INR','100000'))
MAX_ENTRY_DISTANCE_ATR = float(os.environ.get('MAX_ENTRY_DISTANCE_ATR','1.0'))
BREAKOUT_MAX_BARS = int(os.environ.get('BREAKOUT_MAX_BARS','5'))
BREAKOUT_MAX_EXTENSION_ATR = float(os.environ.get('BREAKOUT_MAX_EXTENSION_ATR','1.0'))

def _finite(v: Any) -> bool:
    try: return math.isfinite(float(v))
    except (TypeError, ValueError): return False

def _opposite(direction: str, bias: str) -> bool:
    d,b=str(direction or '').lower(),str(bias or '').lower()
    return (d=='long' and b=='bearish') or (d=='short' and b=='bullish')

def evaluate_entry_quality(signal: Dict[str,Any], snapshot: Dict[str,Any]) -> Tuple[bool,str]:
    d=str(signal.get('direction') or '').lower()
    if d not in {'long','short'}: return False,'invalid direction'
    ms=snapshot.get('market_structure') or {}; s3=ms.get('3m') or {}; s15=ms.get('15m') or {}; s1h=ms.get('1h') or {}
    regime=str(ms.get('market_regime') or 'UNCLEAR').upper()
    q=snapshot.get('entry_quality_context') or {}; cont=q.get('continuation') or {}; recent=snapshot.get('recent_signal_context') or {}
    if recent.get('same_direction_reentry_without_new_break'):
        return False,'same-direction re-entry within cooldown without a new structural break'
    for tf,st in (('3m',s3),('15m',s15),('1h',s1h)):
        if _opposite(d,st.get('structure_bias')): return False,f'{tf} structure directly opposes proposed direction'
    if regime in {'TREND_UP','TREND_DOWN','EXHAUSTION_OR_TRANSITION'}:
        wanted='bullish' if d=='long' else 'bearish'
        if s3.get('structure_bias')!=wanted: return False,'3m structure is not aligned with proposed direction'
        if s15.get('structure_bias')!=wanted: return False,'15m structure is not aligned with proposed direction'
    if regime=='RANGE':
        rng=ms.get('range') or {}
        if d=='long' and not rng.get('near_low'): return False,'range LONG is not near the lower boundary'
        if d=='short' and not rng.get('near_high'): return False,'range SHORT is not near the upper boundary'
    if regime in {'BREAKOUT_TRANSITION','BREAKDOWN_TRANSITION'}:
        latest=s3.get('latest_break') or {}; expected='bullish' if d=='long' else 'bearish'
        if str(latest.get('direction') or '').lower()!=expected: return False,'latest 3m structural break does not support direction'
        bars=cont.get('bars_since_break')
        if _finite(bars) and float(bars)>BREAKOUT_MAX_BARS: return False,f'breakout is stale (> {BREAKOUT_MAX_BARS} closed 3m bars)'
        ext=cont.get('extension_atr_from_break')
        if _finite(ext) and abs(float(ext))>BREAKOUT_MAX_EXTENSION_ATR: return False,f'breakout entry is extended {float(ext):.2f} ATR'
    cq=str(cont.get('continuation_quality') or '')
    if cq=='late_exhausted': return False,'continuation is late and shows exhaustion/failure evidence'
    if cq=='late_extended':
        ext=cont.get('extension_atr_from_break')
        if _finite(ext) and abs(float(ext))>BREAKOUT_MAX_EXTENSION_ATR: return False,'continuation is late and extended beyond the allowed ATR'
    if regime in {'TREND_UP','TREND_DOWN','BREAKOUT_TRANSITION','BREAKDOWN_TRANSITION'}:
        if (s3.get('exhaustion_flags') or []) and (s3.get('failed_breaks') or []): return False,'exhaustion and failed-break evidence conflict with continuation'
    close,atr,entry=snapshot.get('close'),ms.get('atr14_3m'),signal.get('entry_price')
    if _finite(close) and _finite(atr) and _finite(entry) and float(atr)>0:
        dist=abs(float(entry)-float(close))/float(atr)
        if dist>MAX_ENTRY_DISTANCE_ATR: return False,f'proposed entry is {dist:.2f} ATR from current price'
    return True,'ok'

def apply_entry_quality_gate(flagged: Dict[str,Dict[str,Any]], snapshots: Iterable[Dict[str,Any]]) -> int:
    by_coin={s.get('coin'):s for s in snapshots}; rejected=0
    for coin,signal in flagged.items():
        if not signal.get('take_trade'): continue
        amt=signal.get('trade_amount_inr')
        if _finite(amt) and float(amt)>MAX_NOTIONAL_INR:
            signal['take_trade']=False
            signal['risk_validation_error']=f'entry_quality_gate: notional ₹{float(amt):.2f} exceeds MAX_NOTIONAL_INR ₹{MAX_NOTIONAL_INR:.2f}'
            rejected+=1; continue
        snap=by_coin.get(coin)
        if snap is None:
            signal['take_trade']=False; signal['risk_validation_error']='entry_quality_gate: missing deterministic market snapshot'; rejected+=1; continue
        ok,reason=evaluate_entry_quality(signal,snap)
        if not ok:
            signal['take_trade']=False; signal['risk_validation_error']=f'entry_quality_gate: {reason}'; rejected+=1
    return rejected
