import json
from pathlib import Path
import unittest

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


class ConditionalPriorDeploymentCostTest(unittest.TestCase):
    def test_four_bilingual_surfaces(self):
        for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
            soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
            nodes = soup.select('#conditional-prior-deployment-cost')
            self.assertEqual(len(nodes), 1)
            p = nodes[0]
            self.assertEqual(p.get_text(), p['data-i18n-zh'])
            for text in (p['data-i18n-zh'], p['data-i18n-en']):
                for number in ('101', '18', '1.110/1.181', '3.034/3.018', 'FAIL'):
                    self.assertIn(number, text)
            self.assertIn('nonfree', p['data-i18n-en'])
            self.assertIn('not learned', p['data-i18n-en'])

    def test_consistent_resource_failure_scope(self):
        a = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())['latest_signed_cross_ray']['conditional_prior_deployment_cost']
        b = json.loads((ROOT/'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())['conditional_prior_deployment_cost']
        self.assertEqual(a, b)
        self.assertEqual(a['decision'], 'FAIL_PREPARED_DEPLOYMENT_RESOURCE_DOMINANCE')
        self.assertEqual(a['fresh_processes'], 18)
        self.assertEqual(a['held_trajectory_count'], 1)
        self.assertEqual(a['complete_frames_per_process'], 101)
        self.assertEqual(a['independent_replay_checks'], 60)
        for mode in ('formal', 'independent'):
            self.assertGreater(a['costs'][mode]['ratios']['wall_to_CGLS128'], 1)
            self.assertGreater(a['costs'][mode]['ratios']['rss_to_CGLS128'], 3)
        for key in ('preparation_training_included', 'all_folds_cost_test', 'learned_call_reduction', 'algorithm_breakthrough', 'resource_speedup', 'paper_success', 'real_bost', 'external_generalization', 'main_goal_complete'):
            self.assertFalse(a[key])

    def test_note_and_privacy(self):
        text = (ROOT/'docs/conditional_prior_deployment_cost_2026-10-11.md').read_text()
        self.assertIn('## 中文', text)
        self.assertIn('## English', text)
        self.assertIn('nonfree', text)
        self.assertIn('not', text)
        for term in ('/Users/', '/Volumes/', 'private_results', 'OPENED.json', 'FROZEN.json', 'checkpoint'):
            self.assertNotIn(term, text)
        self.assertTrue((ROOT/'assets/conditional_prior_deployment_cost_2026-10-11.png').exists())


if __name__ == '__main__':
    unittest.main()
