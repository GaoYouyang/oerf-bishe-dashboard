import json
from pathlib import Path
import unittest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


class RegularizedComplementNoteTest(unittest.TestCase):
    def test_four_bilingual_surfaces(self):
        for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
            soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
            p = soup.select('#regularized-complement-target')
            self.assertEqual(len(p), 1)
            self.assertIn('99/99', p[0]['data-i18n-zh'])
            self.assertIn('93/99', p[0]['data-i18n-en'])
            self.assertIn('FAIL', p[0]['data-i18n-en'])
            self.assertIn('7 harmed', p[0]['data-i18n-en'])
            self.assertIn('nonfree', p[0]['data-i18n-en'])
            self.assertEqual(p[0].get_text(), p[0]['data-i18n-zh'])

    def test_consistent_aggregates(self):
        a = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())['latest_signed_cross_ray']['regularized_complement_target']
        b = json.loads((ROOT/'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())['regularized_complement_target']
        self.assertEqual(a, b)
        self.assertEqual(a['decision'], 'FAIL_REGULARIZED_COMPLEMENT_TARGET')
        self.assertEqual(a['matched_queries'], 99)
        self.assertEqual(a['nonharm_vs_mean'], 93)
        self.assertEqual(a['nonharm_vs_zero'], 93)
        self.assertEqual(a['harmed_query_union'], 7)
        self.assertEqual(a['harmed_camera_counts'], [9])
        self.assertEqual(a['independent_checks'], 22)
        self.assertFalse(a['full_sequence'])
        self.assertTrue(a['fixed_recipe_closed'])
        for field in ('algorithm_breakthrough', 'resource_speedup', 'paper_success', 'real_bost', 'external_generalization', 'source_target_is_exact_null', 'main_goal_complete'):
            self.assertFalse(a[field])

    def test_cost_and_privacy(self):
        p = ROOT/'docs/regularized_complement_target_2026-10-11.md'
        text = p.read_text()
        self.assertIn('## 中文', text)
        self.assertIn('## English', text)
        self.assertIn('93/99', text)
        self.assertIn('full classical factors', text.lower())
        for term in ('/Users/', '/Volumes/', 'private_results', 'FROZEN.json', 'checkpoint', 'OPENED.json'):
            self.assertNotIn(term, text)
        self.assertTrue((ROOT/'assets/regularized_complement_target_2026-10-11.png').is_file())


if __name__ == '__main__':
    unittest.main()
