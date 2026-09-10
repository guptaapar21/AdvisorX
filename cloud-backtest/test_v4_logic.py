from __future__ import annotations
import unittest
from entry_quality_gate import evaluate_entry_quality
from portfolio_risk import apply_execution_gate


def _snapshot(**loc):
    return {
        "close": 100.0,
        "market_structure": {
            "market_regime": "TREND_UP", "atr14_3m": 1.0,
            "3m": {"structure_bias": "bullish", "latest_break": {"direction": "bullish"}},
            "15m": {"structure_bias": "bullish"}, "1h": {"structure_bias": "bullish"},
            "range": {"position_pct": 65, "near_high": False, "near_low": False},
        },
        "entry_quality_context": {"continuation": {}}, "entry_location_telemetry": dict(loc),
    }


def _signal(**extra):
    value={"coin":"TEST","direction":"long","take_trade":True,"entry_price":100.0}; value.update(extra); return value


class V4LogicTests(unittest.TestCase):
    def test_clean_continuation_accepts(self):
        ok, reason=evaluate_entry_quality(_signal(), _snapshot(range_position_pct=65, extension_atr_from_break=0.5, bars_since_break=2, exhaustion_score_3m=0, failed_break_count_3m=0, liquidity_sweep_count_3m=0, break_freshness="fresh"))
        self.assertTrue(ok, reason)

    def test_extreme_exhausted_long_rejects(self):
        ok, _=evaluate_entry_quality(_signal(), _snapshot(range_position_pct=98, extension_atr_from_break=1.4, bars_since_break=2, exhaustion_score_3m=3, failed_break_count_3m=4, liquidity_sweep_count_3m=4, break_freshness="fresh"))
        self.assertFalse(ok)

    def test_huge_extension_rejects(self):
        ok, reason=evaluate_entry_quality(_signal(), _snapshot(range_position_pct=60, extension_atr_from_break=2.3, bars_since_break=3))
        self.assertFalse(ok); self.assertIn("atr", reason.lower())

    def test_portfolio_caps_same_direction(self):
        flagged={"ETH":{"take_trade":True,"direction":"long","conviction":8},"SOL":{"take_trade":True,"direction":"long","conviction":7}}
        opened=[{"coin":"BTC","direction":"long"},{"coin":"ADA","direction":"long"}]
        self.assertEqual(apply_execution_gate(flagged,opened),2)
        self.assertFalse(flagged["ETH"]["take_trade"]); self.assertFalse(flagged["SOL"]["take_trade"])

if __name__ == "__main__":
    unittest.main()
