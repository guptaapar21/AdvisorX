import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-backtest'))
from entry_quality_gate import evaluate_entry_quality

def snap(regime='TREND_UP',b3='bullish',b15='bullish',b1h='bullish',**kw):
    return {'coin':'TEST','close':100.0,'market_structure':{'market_regime':regime,'atr14_3m':1.0,'3m':{'structure_bias':b3,'latest_break':{'direction':'bullish'},'failed_breaks':[],'exhaustion_flags':[]},'15m':{'structure_bias':b15},'1h':{'structure_bias':b1h},'range':{'near_low':True,'near_high':True}},'entry_quality_context':{'continuation':{'bars_since_break':kw.get('bars',2),'extension_atr_from_break':kw.get('ext',0.2),'continuation_quality':kw.get('cq','fresh_continuation')}},'recent_signal_context':{'same_direction_reentry_without_new_break':kw.get('reentry',False)}}

def run():
    assert evaluate_entry_quality({'direction':'long','entry_price':100.1},snap())[0]
    assert not evaluate_entry_quality({'direction':'long','entry_price':100.1},snap(b15='bearish'))[0]
    assert not evaluate_entry_quality({'direction':'long','entry_price':100.1},snap(regime='BREAKOUT_TRANSITION',bars=7,ext=1.4))[0]
    s=snap(regime='RANGE'); s['market_structure']['range']['near_low']=False; assert not evaluate_entry_quality({'direction':'long','entry_price':100.0},s)[0]
    assert not evaluate_entry_quality({'direction':'long','entry_price':100.1},snap(reentry=True))[0]
    assert not evaluate_entry_quality({'direction':'long','entry_price':102.1},snap())[0]
    print('entry_quality_gate tests: PASS')
if __name__=='__main__': run()
