from __future__ import annotations

from datetime import datetime, timezone
import tempfile
import unittest

import pandas as pd

from advisorx_trade_management_policy import apply_management_policy, apply_profit_ladder
from advisorx_trade_resolution import resolve_ledger_1m
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
    value = {"coin": "TEST", "direction": "long", "take_trade": True, "entry_price": 100.0}
    value.update(extra)
    return value


def _position(direction="long"):
    return {
        "coin": "TEST",
        "direction": direction,
        "entry_price": 100.0,
        "trade_amount_inr": 10000.0,
        "max_loss_this_trade_inr": 500.0,
        "stop_loss": 95.0 if direction == "long" else 105.0,
        "target_price": 110.0 if direction == "long" else 90.0,
        "mfe_pnl_inr": 500.0,
    }


def _candles(rows):
    index = pd.date_range("2026-09-10 10:00:00", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


class V4LogicTests(unittest.TestCase):
    def test_clean_continuation_accepts(self):
        ok, reason = evaluate_entry_quality(
            _signal(),
            _snapshot(
                range_position_pct=65,
                extension_atr_from_break=0.5,
                bars_since_break=2,
                exhaustion_score_3m=0,
                failed_break_count_3m=0,
                liquidity_sweep_count_3m=0,
                break_freshness="fresh",
            ),
        )
        self.assertTrue(ok, reason)

    def test_extreme_exhausted_long_rejects(self):
        ok, _ = evaluate_entry_quality(
            _signal(),
            _snapshot(
                range_position_pct=98,
                extension_atr_from_break=1.4,
                bars_since_break=2,
                exhaustion_score_3m=3,
                failed_break_count_3m=4,
                liquidity_sweep_count_3m=4,
                break_freshness="fresh",
            ),
        )
        self.assertFalse(ok)

    def test_huge_extension_rejects(self):
        ok, reason = evaluate_entry_quality(
            _signal(), _snapshot(range_position_pct=60, extension_atr_from_break=2.3, bars_since_break=3)
        )
        self.assertFalse(ok)
        self.assertIn("atr", reason.lower())

    def test_portfolio_caps_same_direction(self):
        flagged = {
            "ETH": {"take_trade": True, "direction": "long", "conviction": 8},
            "SOL": {"take_trade": True, "direction": "long", "conviction": 7},
        }
        opened = [{"coin": "BTC", "direction": "long"}, {"coin": "ADA", "direction": "long"}]
        self.assertEqual(apply_execution_gate(flagged, opened), 2)
        self.assertFalse(flagged["ETH"]["take_trade"])
        self.assertFalse(flagged["SOL"]["take_trade"])

    def test_portfolio_applies_cluster_total_after_higher_conviction_candidate(self):
        flagged = {
            "ETH": {"take_trade": True, "direction": "short", "conviction": 8},
            "SOL": {"take_trade": True, "direction": "long", "conviction": 7},
        }
        opened = [
            {"coin": "BTC", "direction": "long"},
            {"coin": "ADA", "direction": "short"},
        ]
        self.assertEqual(apply_execution_gate(flagged, opened), 1)
        self.assertTrue(flagged["ETH"]["take_trade"])
        self.assertFalse(flagged["SOL"]["take_trade"])

    def test_tighten_stop_is_suppressed_before_profit_threshold(self):
        position = _position("long")
        position["current_price"] = 100.4
        position["mfe_pnl_inr"] = 100.0
        updates = {"TEST": {"action": "tighten_stop", "updated_stop_loss": 99.5, "reasoning": "early"}}
        result = apply_management_policy(updates, [position], min_tighten_r=0.5)
        self.assertEqual(result["TEST"]["action"], "hold")
        self.assertEqual(result["TEST"]["management_policy"], "tighten_suppressed_before_profit_threshold")

    def test_non_monotonic_short_stop_is_suppressed(self):
        position = _position("short")
        position["current_price"] = 98.0
        position["mfe_pnl_inr"] = 500.0
        updates = {"TEST": {"action": "tighten_stop", "updated_stop_loss": 106.0}}
        result = apply_management_policy(updates, [position], min_tighten_r=0.5)
        self.assertEqual(result["TEST"]["action"], "hold")
        self.assertEqual(result["TEST"]["management_policy"], "non_monotonic_stop_move_suppressed")

    def test_profit_ladder_moves_long_stop_at_1r_mfe(self):
        position = _position("long")
        position["mfe_pnl_inr"] = 500.0
        current_prices = {"TEST": 102.5}
        actions = apply_profit_ladder([position], current_prices, datetime.now(timezone.utc))
        self.assertEqual(actions[0]["action"], "profit_ladder")
        self.assertAlmostEqual(position["stop_loss"], 101.25, places=6)
        self.assertEqual(position["profit_lock_r"], 0.25)
        self.assertEqual(position["level_history"][-1]["source"], "python_v4_profit_ladder")

    def test_profit_ladder_is_symmetric_for_short(self):
        position = _position("short")
        position["mfe_pnl_inr"] = 500.0
        current_prices = {"TEST": 97.5}
        actions = apply_profit_ladder([position], current_prices, datetime.now(timezone.utc))
        self.assertEqual(actions[0]["action"], "profit_ladder")
        self.assertAlmostEqual(position["stop_loss"], 98.75, places=6)

    def test_1m_resolution_prefers_target_when_only_target_is_touched(self):
        ledger = [_position("long")]
        ledger[0]["time"] = "2026-09-10T09:59:00+00:00"
        candles = _candles([
            [100, 101, 99.5, 100.5, 10],
            [100.5, 110.1, 100.0, 109.0, 10],
        ])
        with tempfile.NamedTemporaryFile() as archive:
            result = resolve_ledger_1m(
                ledger, {"TEST": (None, None, None, candles)}, "2026-09-10T10:03:00+00:00",
                usdt_inr_rate=99.44, taker_fee_rate=0.00075, resolved_trades_file=archive.name,
            )
        self.assertEqual(result, [])
        self.assertEqual(ledger[0]["status"], "target_hit")
        self.assertEqual(ledger[0]["resolution_timeframe"], "1m")

    def test_1m_resolution_conservatively_marks_both_sides_as_stop(self):
        ledger = [_position("long")]
        ledger[0]["time"] = "2026-09-10T09:59:00+00:00"
        candles = _candles([
            [100, 110.5, 94.5, 100.0, 10],
        ])
        with tempfile.NamedTemporaryFile() as archive:
            result = resolve_ledger_1m(
                ledger, {"TEST": (None, None, None, candles)}, "2026-09-10T10:02:00+00:00",
                usdt_inr_rate=99.44, taker_fee_rate=0.00075, resolved_trades_file=archive.name,
            )
        self.assertEqual(result, [])
        self.assertEqual(ledger[0]["status"], "stop_hit")
        self.assertEqual(
            ledger[0]["resolution_policy"],
            "conservative_stop_when_target_and_stop_share_candle",
        )


if __name__ == "__main__":
    unittest.main()
