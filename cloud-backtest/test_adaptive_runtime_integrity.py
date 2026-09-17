import unittest

import adaptive_runtime_integrity as r


class AdaptiveRuntimeIntegrityTests(unittest.TestCase):
    def _snap(self, regime="TREND_UP", b3="bullish", b15="bullish", b1h="bullish"):
        return {
            "coin": "TEST",
            "close": 100.0,
            "market_structure": {
                "market_regime": regime,
                "3m": {
                    "structure_bias": b3,
                    "adx14": 30,
                    "efficiency_ratio_20": 0.50,
                    "latest_break": {"direction": "bullish", "type": "BOS", "level": 99.0},
                    "liquidity_sweeps": [],
                },
                "15m": {"structure_bias": b15},
                "1h": {"structure_bias": b1h},
                "range": {
                    "candidate": regime == "RANGE",
                    "position_pct": 15,
                    "range_low": 95,
                    "range_high": 110,
                },
                "atr14_3m": 2.0,
            },
            "entry_quality_context": {
                "continuation": {"bars_since_break": 2},
            },
        }

    def test_local_confidence_is_derived_not_constant(self):
        strong, _ = r._local_regime_confidence(self._snap())
        weak, _ = r._local_regime_confidence(self._snap(b15="bearish", b1h="bearish"))
        self.assertGreater(strong, weak)
        self.assertNotEqual(strong, 0.80)

    def test_unclear_regime_has_low_confidence(self):
        confidence, evidence = r._local_regime_confidence(self._snap(regime="UNCLEAR"))
        self.assertEqual(confidence, 0.25)
        self.assertIn("unclear_regime", evidence["evidence"])

    def test_sweep_direction_is_latest_directional_evidence(self):
        snap = self._snap(regime="TRANSITION")
        snap["market_structure"]["3m"]["liquidity_sweeps"] = [
            {"type": "high_sweep"},
            {"type": "low_sweep"},
        ]
        self.assertEqual(r._direction_from_sweep(snap), "long")

    def test_armed_status_uses_trigger_distance(self):
        snap = self._snap()
        original_prune = r.adaptive._prune_watch
        original_write = r.adaptive._write_json
        fake_state = {
            "TEST": {
                "coin": "TEST",
                "direction": "long",
                "playbook": "LONG_BREAKOUT",
                "created_at": "2026-09-17T13:00:00+00:00",
                "trigger_type": "STRUCTURAL_BREAK",
                "trigger_level": 100.2,
                "status": "WATCH",
            }
        }
        try:
            r.adaptive._prune_watch = lambda: fake_state
            writes = []
            r.adaptive._write_json = lambda path, payload: writes.append((path, payload))
            r._persist_watch_status([snap])
            self.assertTrue(writes)
            payload = writes[-1][1]
            self.assertIn("TEST", payload)
            self.assertEqual(payload["TEST"]["status"], "ARMED")
        finally:
            r.adaptive._prune_watch = original_prune
            r.adaptive._write_json = original_write


if __name__ == "__main__":
    unittest.main()
