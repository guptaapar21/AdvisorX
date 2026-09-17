import unittest

import adaptive_playbook_layer as a


class AdaptivePlaybookTests(unittest.TestCase):
    def _snap(self, regime="TREND_UP", bias="bullish", position=15, latest_direction="bullish", sweep=None):
        s3 = {
            "structure_bias": bias,
            "latest_break": {"direction": latest_direction, "type": "BOS", "level": 100.0},
            "ema21": 100.0,
            "liquidity_sweeps": sweep or [],
            "exhaustion_score_3m": 0,
        }
        s15 = {"structure_bias": bias}
        rng = {"candidate": regime == "RANGE", "position_pct": position, "near_low": position <= 20, "near_high": position >= 80, "range_low": 95.0, "range_high": 110.0}
        return {
            "coin": "TEST",
            "close": 100.0,
            "rvol": 1.2,
            "rvol_percentile": 55,
            "momentum_pct_20_3m": 1.0,
            "market_structure": {"market_regime": regime, "atr14_3m": 2.0, "3m": s3, "15m": s15, "1h": {"structure_bias": bias}, "range": rng},
            "entry_quality_context": {"continuation": {"has_break": True, "break_freshness": "fresh", "bars_since_break": 1, "continuation_quality": "fresh_continuation"}},
        }

    def test_global_regime_candidate_uses_breadth_and_btc(self):
        snaps = [self._snap("TREND_UP") for _ in range(4)]
        regime, confidence, details = a._global_regime_candidate(snaps)
        self.assertEqual(regime, "TREND_UP")
        self.assertGreaterEqual(confidence, 0.68)
        self.assertEqual(details["counts"]["TREND_UP"], 4)

    def test_range_long_playbook_accepts_lower_boundary(self):
        snap = self._snap("RANGE", bias="neutral", position=10, latest_direction="bullish")
        pb = a._preferred_playbook(snap)
        self.assertIn(pb["preferred"], {"RANGE_LONG", "RANGE_REVERSAL"})
        signal = {"take_trade": True, "direction": "long", "adaptive_playbook": "RANGE_LONG", "entry_price": 100.0, "stop_loss": 96.0, "target_price": 108.0}
        ok, _ = a._playbook_gate(signal, snap)
        self.assertTrue(ok)

    def test_transition_retest_requires_matching_break(self):
        snap = self._snap("TRANSITION", bias="bearish", latest_direction="bullish")
        signal = {"take_trade": True, "direction": "long", "adaptive_playbook": "BREAKOUT_RETEST", "entry_price": 100.5, "stop_loss": 97.0, "target_price": 107.0}
        ok, reason = a._playbook_gate(signal, snap)
        self.assertTrue(ok, reason)

    def test_range_middle_not_take(self):
        snap = self._snap("RANGE", bias="neutral", position=50, latest_direction="bullish")
        signal = {"take_trade": True, "direction": "long", "adaptive_playbook": "RANGE_LONG", "entry_price": 100.0, "stop_loss": 96.0, "target_price": 108.0}
        ok, reason = a._playbook_gate(signal, snap)
        self.assertFalse(ok)
        self.assertIn("boundary", reason)


if __name__ == "__main__":
    unittest.main()
