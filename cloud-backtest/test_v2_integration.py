import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import candidate_funnel
import v2_live_integration


class V2IntegrationTests(unittest.TestCase):
    def test_gemini_skip(self):
        self.assertEqual(candidate_funnel.classify_decision({"take_trade": False}), "GEMINI_SKIP")

    def test_real_python_entry_quality_reject(self):
        self.assertEqual(
            candidate_funnel.classify_decision(
                {"take_trade": False, "_entry_quality_reject_reason": "entry too far"}
            ),
            "PYTHON_REJECT",
        )

    def test_real_python_risk_reject(self):
        self.assertEqual(
            candidate_funnel.classify_decision(
                {"take_trade": False, "risk_validation_error": "risk/reward 1.25 below MIN_RR 1.50"}
            ),
            "PYTHON_REJECT",
        )

    def test_take(self):
        self.assertEqual(candidate_funnel.classify_decision({"take_trade": True}), "TAKE")

    def test_watch_only_when_explicit(self):
        self.assertEqual(
            candidate_funnel.classify_decision({"decision": "WATCH", "take_trade": False}),
            "WATCH",
        )

    def test_unknown_break_direction_is_none(self):
        signal = {"market_structure": {"3m": {"latest_break": {"foo": "bar"}}}}
        self.assertEqual(candidate_funnel.classify_event(signal), "NONE")

    def test_cycle_records_real_python_risk_reject(self):
        signal = {
            "coin": "BTC",
            "candle_time": "2026-08-29T06:30:00Z",
            "close": 100000,
            "market_structure": {"3m": {"latest_break": {"direction": "bullish", "type": "BOS"}}},
            "entry_quality_context": {"continuation": {}},
        }
        decision = {
            "take_trade": False,
            "direction": "long",
            "risk_validation_error": "risk/reward 1.25 below MIN_RR 1.50",
            "_entry_location_telemetry": {"entry_location_class": "near_range_high"},
        }
        with tempfile.TemporaryDirectory() as td:
            funnel_path = str(Path(td) / "funnel.jsonl")
            cf_path = str(Path(td) / "cf.jsonl")
            old_funnel = candidate_funnel.FUNNEL_FILE
            old_cf = v2_live_integration.trade_counterfactuals.FILE
            candidate_funnel.FUNNEL_FILE = funnel_path
            v2_live_integration.trade_counterfactuals.FILE = cf_path
            try:
                summary = v2_live_integration.record_cycle([signal], {"BTC": decision}, [])
                self.assertEqual(summary["buckets"], {"PYTHON_REJECT": 1})
                with open(funnel_path, encoding="utf-8") as f:
                    row = f.readline()
                self.assertIn('"decision_bucket":"PYTHON_REJECT"', row)
                self.assertIn("entry_location_class", row)
                with open(cf_path, encoding="utf-8") as f:
                    cf = f.readline()
                self.assertIn('"kind":"PYTHON_REJECT"', cf)
            finally:
                candidate_funnel.FUNNEL_FILE = old_funnel
                v2_live_integration.trade_counterfactuals.FILE = old_cf


    def test_failed_break_precedes_generic_breakout(self):
        signal = {
            "market_structure": {
                "3m": {
                    "latest_break": {"direction": "bullish", "type": "BOS"},
                    "failed_breaks": [{"type": "rejection"}],
                    "phase": "bull_continuation",
                    "structure_bias": "bullish",
                }
            }
        }
        self.assertEqual(candidate_funnel.classify_event(signal), "FAILED_BREAK")

    def test_pullback_precedes_generic_breakout(self):
        signal = {
            "market_structure": {
                "3m": {
                    "latest_break": {"direction": "bullish", "type": "BOS"},
                    "phase": "bull_continuation",
                    "structure_bias": "bullish",
                    "ema21": 100.0,
                    "close": 100.3,
                }
            }
        }
        self.assertEqual(candidate_funnel.classify_event(signal), "TREND_PULLBACK")

    def test_opportunity_id_is_deterministic(self):
        signal = {"coin": "BTC", "candle_time": "2026-08-29T06:30:00Z", "market_structure": {}}
        a = candidate_funnel.opportunity_id(signal, "NONE")
        b = candidate_funnel.opportunity_id(signal, "NONE")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("opp_"))

    def test_portfolio_is_observational(self):
        summary = v2_live_integration.record_cycle([], {}, [{"coin": "BTC", "direction": "long"}])
        self.assertEqual(summary["portfolio"], [{"cluster": "BTC_BETA", "direction": "LONG", "positions": 1}])

    def test_counterfactual_schema_is_observation_only(self):
        record = v2_live_integration.trade_counterfactuals.log(
            "TEST", "BTC", "long", 100000, {"take_trade": False}
        )
        self.assertTrue(record["observation_only"])
        self.assertTrue(record["future_outcome_pending"])


if __name__ == "__main__":
    unittest.main()
