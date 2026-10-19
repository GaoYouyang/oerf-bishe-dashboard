"""Public aggregate consistency and evidence-boundary regression only."""

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SLUG = "pinhole_native12_loto_2026-10-19"


class Native12Milestone(unittest.TestCase):
    def setUp(self):
        self.raw = (ROOT / f"docs/{SLUG}_public_summary.json").read_text()
        self.data = json.loads(self.raw)

    def test_population_and_nonclaims(self):
        p = self.data
        self.assertEqual(p["native12_queries"], 909)
        self.assertEqual(p["native12_complete_trajectories"], 9)
        self.assertEqual(p["unique_development_frames"], 909)
        self.assertEqual(
            p["inherited_first_queries"] + p["newly_scored_remaining_queries"], 909
        )
        self.assertEqual(p["neural_train_camera_count"], 5)
        self.assertFalse(p["neural_retraining_for_native12"])
        self.assertTrue(p["current_result_is_retained_9_plus_3_layout_only"])
        self.assertTrue(p["previous_new_ring_failure_not_overturned"])
        self.assertFalse(p["previous_new_twelve_ring"]["conditional_value_pass"])
        for key in (
            "resource_speedup",
            "actual_error_equivalence",
            "external_generalization",
            "real_bost",
            "algorithm_breakthrough",
            "paper_success",
            "full_goal_complete",
        ):
            self.assertFalse(p[key], key)

    def test_complete_metrics_and_strong_controls(self):
        p = self.data
        self.assertEqual(len(p["group_four_mse"]), 9)
        for group in p["group_four_mse"]:
            self.assertEqual(len(group), 14)
            self.assertTrue(all(len(row) == 4 for row in group))
        self.assertEqual(p["all_method_matched_counts"][0], 909)
        self.assertEqual(p["all_method_absolute_trajectory_counts"][0], 9)
        for arm in (
            "GammaGMM0",
            "BetaGMM0",
            "OptimalStatic0",
            "Range0",
            "Zero",
            "BP-LS1",
            "CGLS5",
            "Jacobi5",
            "DualRidge5",
            "Reference128",
        ):
            self.assertIn(arm, p["arms"])
        cost = p["cost"]["formal_neural_per_query"]
        self.assertEqual(cost["full_inverse"], 1)
        self.assertEqual(cost["camera_small_RHS"], 24)

    def test_public_privacy_and_bilingual_entrypoints(self):
        self.assertIsNone(re.search(r"\b[a-f0-9]{64}\b", self.raw))
        for token in (
            "/Users/",
            "/Volumes/",
            "private_results/",
            "private_notes/",
            "checkpoint",
        ):
            self.assertNotIn(token, self.raw)
        for name in (
            "index.html",
            "operator-learning/index.html",
            "operator-learning/daily-progress.html",
            "learning_log.html",
        ):
            html = (ROOT / name).read_text()
            self.assertEqual(html.count('id="pinhole-native12-loto-20261019"'), 1)
            self.assertIn(
                "Nine-fold validation when expanding the retained camera layout", html
            )
            self.assertIn(f"{SLUG}_public_summary.json", html)
        self.assertTrue((ROOT / f"assets/{SLUG}.png").stat().st_size > 20000)


if __name__ == "__main__":
    unittest.main()
