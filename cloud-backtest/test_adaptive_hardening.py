import unittest

import adaptive_hardening as h
import adaptive_runtime_integrity  # noqa: F401 - activate final routing overlay


class AdaptiveHardeningTests(unittest.TestCase):
    def _snap(self, regime="RANGE", bias="neutral", position=10, latest_direction="bullish", sweeps=None):
        sweeps = sweeps or []
        s3 = {
            "structure_bias": bias,
            "latest_break": {"direction": latest_direction, "type": "BOS", "level": 100.0},
            "ema21": 100.0,
            "liquidity_sweeps": sweeps,
            "exhaustion_score_3m": 0,
        }
        return {
            "coin": "TEST",
            "close": 100.5,
            "rvol": 1.2,
            "rvol_percentile": 55,
            "momentum_pct_20_3m": 1.0,
            "market_structure": {
                "market_regime": regime,
                "atr14_3m": 2.0,
                "3m": s3,
                "15m": {"structure_bias": bias},
                "1h": {"structure_bias": bias},
                "range": {
                    "candidate": regime == "RANGE",
                    "position_pct": position,
                    "near_low": position <= 20,
                    "near_high": position >= 80,
                    "range_low": 95.0,
                    "range_high": 110.0,
                },
            },
            "entry_quality_context": {
                "continuation": {
                    "has_break": True,
                    "break_freshness": "fresh",
                    "bars_since_break": 2,
                    "continuation_quality": "fresh_continuation",
                },
            },
        }

    def test_range_direction_reconciles_to_gemini_direction(self):
        snap = self._snap(regime="RANGE", bias="neutral", position=90, latest_direction="bearish")
        self.assertEqual(h._directional_playbook(snap, "short"), "RANGE_SHORT")
        self.assertIsNone(h._directional_playbook(snap, "long"))

    def test_range_long_requires_confirmation_when_not_swept(self):
        snap = self._snap(regime="RANGE", bias="neutral", position=10)
        ok, reason = h._trigger_confirmed(snap, "RANGE_LONG", "long", 100.0)
        self.assertFalse(ok)
        self.assertIn("confirmation", reason)

    def test_range_long_accepts_boundary_plus_bullish_confirmation(self):
        snap = self._snap(regime="RANGE", bias="bullish", position=10)
        ok, reason = h._trigger_confirmed(snap, "RANGE_LONG", "long", 100.0)
        self.assertTrue(ok, reason)

    def test_transition_requires_retest_not_first_break(self):
        snap = self._snap(regime="TRANSITION", bias="bullish", latest_direction="bullish")
        snap["entry_quality_context"]["continuation"]["bars_since_break"] = 0
        ok, reason = h._trigger_confirmed(snap, "BREAKOUT_RETEST", "long", 100.0)
        self.assertFalse(ok)
        self.assertIn("retest", reason)

    def test_playbook_geometry_requires_range_invalidation(self):
        snap = self._snap(regime="RANGE", bias="bullish", position=10)
        bad = {"direction": "long", "adaptive_playbook": "RANGE_LONG", "entry_price": 100.0, "stop_loss": 97.0, "target_price": 108.0}
        ok, reason = h._playbook_geometry_ok(bad, snap)
        self.assertFalse(ok)
        self.assertIn("range low", reason)

    def test_short_trend_playbook_is_supported(self):
        snap = self._snap(regime="TREND_DOWN", bias="bearish", position=85, latest_direction="bearish")
        self.assertIn(h._directional_playbook(snap, "short"), {"SHORT_BREAKDOWN", "SHORT_RALLY_REJECTION", "SHORT_CONTINUATION"})

    def test_persistent_regime_memory_does_not_replace_pure_candidate(self):
        snaps = [self._snap(regime="TREND_UP") for _ in range(4)]
        regime, _, details = h.adaptive._global_regime_candidate(snaps)
        self.assertEqual(regime, "TREND_UP")
        self.assertEqual(details["counts"]["TREND_UP"], 4)


if __name__ == "__main__":
    unittest.main()
