import unittest
from datetime import datetime, timezone

from market_behavior_recorder import Bar, build_event, resolve_outcome


class MarketBehaviorRecorderTests(unittest.TestCase):
    def _bars(self, n=80, base=100.0):
        bars = []
        start = int(datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
        for i in range(n):
            close = base + i * 0.01
            bars.append(Bar(start + i * 60_000, close - 0.02, close + 0.04, close - 0.04, close, 100.0))
        return bars

    def test_support_long_wick_creates_event(self):
        bars = self._bars()
        for i in range(49, 79):
            b = bars[i]
            bars[i] = Bar(b.ts_ms, b.open, b.high, 99.80, b.close, b.volume)
        last = bars[-1]
        bars[-1] = Bar(last.ts_ms, 100.72, 100.76, 99.80, 100.70, 250.0)
        event = build_event("TEST", bars)
        self.assertIsNotNone(event)
        self.assertIn("long_wick_support_rejection", event["event_families"])
        self.assertEqual(event["direction"], "bullish")
        self.assertGreater(event["lower_wick_ratio"], 0.45)
        self.assertEqual(event["delta_proxy_source"], "candle_direction_signed_volume")

    def test_failed_resistance_break_is_bearish(self):
        bars = self._bars()
        for i in range(49, 79):
            b = bars[i]
            bars[i] = Bar(b.ts_ms, b.open, 100.45, b.low, b.close, b.volume)
        last = bars[-1]
        bars[-1] = Bar(last.ts_ms, 100.45, 101.0, 100.40, 100.40, 220.0)
        event = build_event("TEST", bars)
        self.assertIsNotNone(event)
        self.assertIn("failed_break_resistance", event["event_families"])
        self.assertEqual(event["direction"], "bearish")

    def test_outcome_uses_future_only_and_signed_direction(self):
        bars = self._bars(100)
        event = {
            "schema_version": "market_behavior_v1",
            "event_id": "abc123",
            "observed_at": datetime.fromtimestamp(bars[60].ts_ms / 1000, tz=timezone.utc).isoformat(),
            "observed_at_ms": bars[60].ts_ms,
            "symbol": "TEST",
            "event_families": ["long_wick_support_rejection"],
            "direction": "bullish",
            "reference_level": 100.0,
            "close": bars[60].close,
        }
        result = resolve_outcome(event, bars, 5)
        self.assertIsNotNone(result)
        self.assertEqual(result["future_bars"], 5)
        self.assertGreaterEqual(result["mfe_pct"], 0.0)
        self.assertGreaterEqual(result["mae_pct"], 0.0)

    def test_outcome_requires_full_horizon(self):
        bars = self._bars(65)
        event = {
            "event_id": "abc123",
            "observed_at": datetime.fromtimestamp(bars[60].ts_ms / 1000, tz=timezone.utc).isoformat(),
            "observed_at_ms": bars[60].ts_ms,
            "symbol": "TEST",
            "event_families": ["x"],
            "direction": "bullish",
            "reference_level": 100.0,
            "close": bars[60].close,
        }
        self.assertIsNone(resolve_outcome(event, bars, 10))


if __name__ == "__main__":
    unittest.main()
