import json
import unittest
from datetime import datetime, timedelta, timezone

from research_analyst import Row, analyse, generate_candidates


class ResearchAnalystTests(unittest.TestCase):
    def _rows(self, direction="bullish", family="failed_break_support"):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(90):
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
        result = analyse(self._rows(), 0.15)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(result["discovery_leads"]), 0)
        self.assertIn("tested_candidates", result)
        self.assertIn("deduplicated_candidates", result)
        self.assertIn("fdr_discovery_candidates", result)
        for item in result["validated"]:
            self.assertEqual(item["status"], "HOLDOUT_PASSED")
            self.assertGreater(item["holdout"]["n"], 0)
            self.assertLessEqual(item["fdr_q"], 0.05)

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

    def test_statistical_controls_are_visible(self):
        result = analyse(self._rows(), 0.15)
        self.assertEqual(result["fdr_q_threshold"], 0.05)
        self.assertEqual(result["bootstrap_reps"], 1000)
        self.assertGreaterEqual(result["tested_candidates"], result["deduplicated_candidates"])
        self.assertGreaterEqual(result["deduplicated_candidates"], result["fdr_discovery_candidates"])
        self.assertTrue(all("bootstrap_block_ci95_low" in item["validation"] for item in result["rejected"]))

    def test_percentile_activity_thresholds_are_deduplicated(self):
        candidates = generate_candidates(self._rows())
        activity = [c for c in candidates if set(c["conditions"]) == {"activity_percentile_min"}]
        self.assertEqual(len(activity), 12)  # 1 family x 2 directions x 6 horizons x 2 thresholds
        # The analyser evaluates at most one of the nested 85/95 percentile pair
        # per family/direction/horizon while leaving the separate 1.8x ratio rule intact.
        result = analyse(self._rows(), 0.15)
        percentile_names = [x["hypothesis"]["name"] for x in result["discovery_leads"]]
        for prefix in ("failed_break_support |",):
            per_group = [name for name in percentile_names if name.startswith(prefix) and ("extreme_activity" in name or "very_extreme_activity" in name)]
            self.assertLessEqual(len(per_group), 6)


if __name__ == "__main__":
    unittest.main()
