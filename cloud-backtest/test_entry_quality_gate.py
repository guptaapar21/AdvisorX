import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cloud-backtest"))
from entry_quality_gate import evaluate_entry_quality

def s(regime="TREND_UP", b3="bullish", b15="bullish", b1h="bullish", bars=2, ext=.2, cq="fresh_continuation", reentry=False):
    return {"close":100.0,"market_structure":{
        "market_regime":regime,"atr14_3m":1.0,
        "3m":{"structure_bias":b3,"latest_break":{"direction":"bullish"},"failed_breaks":[],"exhaustion_flags":[]},
        "15m":{"structure_bias":b15},"1h":{"structure_bias":b1h},
        "range":{"near_low":True,"near_high":True}},
        "entry_quality_context":{"continuation":{"bars_since_break":bars,"extension_atr_from_break":ext,"continuation_quality":cq}},
        "recent_signal_context":{"same_direction_reentry_without_new_break":reentry}}

def test_good_long(): assert evaluate_entry_quality({"direction":"long","entry_price":100.1}, s())[0]
def test_15m_contradiction():
    ok,_=evaluate_entry_quality({"direction":"long","entry_price":100.1}, s(b15="bearish")); assert not ok
def test_breakout_8_allowed():
    ok,_=evaluate_entry_quality({"direction":"long","entry_price":100.1}, s(regime="BREAKOUT_TRANSITION",bars=8)); assert ok
def test_breakout_9_rejected():
    ok,reason=evaluate_entry_quality({"direction":"long","entry_price":100.1}, s(regime="BREAKOUT_TRANSITION",bars=9)); assert not ok and "stale" in reason
def test_range_middle_rejected():
    x=s(regime="RANGE"); x["market_structure"]["range"]["near_low"]=False
    ok,_=evaluate_entry_quality({"direction":"long","entry_price":100},x); assert not ok
def test_reentry_rejected():
    ok,_=evaluate_entry_quality({"direction":"long","entry_price":100.1},s(reentry=True)); assert not ok
def test_far_entry_rejected():
    ok,_=evaluate_entry_quality({"direction":"long","entry_price":102.1},s()); assert not ok

if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and fn.__name__.startswith("test_"): fn()
    print("entry_quality_gate tests: PASS")
