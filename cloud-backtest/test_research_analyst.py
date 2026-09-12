import json
import unittest
from datetime import datetime, timedelta, timezone

from research_analyst import Row, analyse


class ResearchAnalystTests(unittest.TestCase):
    def _rows(self, direction="bullish", family="failed_break_support"):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(90):
            # Development half has edge; holdout also has a smaller but still
            # positive edge, with several hourly blocks and coins represented.
            edge = 0.28 if i < 63 else 0.24
            ret = edge if i % 5 else -0.05
            rows.append(Row(
                observed_at=base + timedelta(minutes=60 * i),
                symbol=("BTC", "ETH", "SOL", "BNB")[i % 4],
                direction=direction,
                horizon_min=15,
                event_families=(family,),
                activity_ratio=2.2,
                activity_percentile=92,
                delta_abs=0.9,
                event_score=4,
                wick_ratio=0.55,
                signed_forward_return_pct=ret + 0.15,
            ))
        return rows

    def test_research_does_not_promote_without_holdout(self):
        rows = self._rows()
        result = analyse(rows, 0.15)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(result["discovery_leads"]), 0)
        for item in result["validated"]:
            self.assertEqual(item["status"], "HOLDOUT_PASSED")
            self.assertGreater(item["holdout"]["n"], 0)

    def test_bad_pattern_is_rejected(self):
        rows = self._rows()
        for i, row in enumerate(rows):
            rows[i] = Row(**{**row.__dict__, "signed_forward_return_pct": -0.20 + 0.15})
        result = analyse(rows, 0.15)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["validated"]), 0)

    def test_output_is_bounded_and_directional(self):
        result = analyse(self._rows(direction="bearish", family="failed_break_resistance"), 0.15)
        payload = {
            "validated": result["validated"],
            "rejected": result["rejected"],
            "discovery_leads": result["discovery_leads"],
        }
        json.dumps(payload, sort_keys=True)
        self.assertTrue(all(x["hypothesis"]["direction"] == "bearish" for x in result["validated"]))


if __name__ == "__main__":
    unittest.main()
