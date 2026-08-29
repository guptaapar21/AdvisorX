"""Regression tests for V2's observational contract."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
import candidate_funnel
import portfolio_risk
import trade_counterfactuals
import v2_live_integration

class V2ObservationalInvariantTests(unittest.TestCase):
    def test_record_cycle_does_not_mutate_signal_or_decision(self):
        signal = {
            "coin": "BTC", "candle_time": "2026-08-29T06:30:00Z",
            "market_structure": {"market_regime": "TREND_UP", "3m": {
                "latest_break": {"direction": "bullish", "type": "BOS"},
                "structure_bias": "bullish",}},
            "entry_quality_context": {"continuation": {"bars_since_break": 1}},}
        decision = {"direction": "long", "take_trade": True, "conviction": 7,
                    "entry_price": 100.0, "stop_loss": 98.0, "target_price": 104.0,
                    "entry_location_telemetry": {"entry_location_class": "near_break"}}
        before_signal = copy.deepcopy(signal); before_decision = copy.deepcopy(decision)
        with tempfile.TemporaryDirectory() as td:
            old_funnel = candidate_funnel.FUNNEL_FILE
            old_cf = trade_counterfactuals.FILE
            candidate_funnel.FUNNEL_FILE = str(Path(td) / "funnel.jsonl")
            trade_counterfactuals.FILE = str(Path(td) / "cf.jsonl")
            try:
                v2_live_integration.record_cycle([signal], {"BTC": decision}, [])
            finally:
                candidate_funnel.FUNNEL_FILE = old_funnel
                trade_counterfactuals.FILE = old_cf
        self.assertEqual(signal, before_signal)
        self.assertEqual(decision, before_decision)

    def test_portfolio_summary_does_not_mutate_positions(self):
        positions = [{"coin": "BTC", "direction": "long"}]
        before = copy.deepcopy(positions)
        portfolio_risk.summarize(positions)
        self.assertEqual(positions, before)

    def test_classification_never_turns_skip_into_take(self):
        original = {"take_trade": False, "reasoning": "skip"}
        self.assertEqual(candidate_funnel.classify_decision(original), "GEMINI_SKIP")
        self.assertFalse(original["take_trade"])

if __name__ == "__main__":
    unittest.main()
