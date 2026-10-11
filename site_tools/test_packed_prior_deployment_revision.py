import json
from pathlib import Path
import unittest

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


class PackedPriorDeploymentRevisionTest(unittest.TestCase):
    def test_four_bilingual_surfaces_preserve_predecessor(self):
        for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
            soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
            nodes = soup.select('#packed-prior-deployment-revision')
            self.assertEqual(len(nodes), 1)
            p = nodes[0]
            self.assertEqual(p.get_text(), p['data-i18n-zh'])
            for text in (p['data-i18n-zh'], p['data-i18n-en']):
                for number in ('101', '18', '0.244/0.247', '1.725/1.738', 'FAIL'):
                    self.assertIn(number, text)
            self.assertIn('not learned', p['data-i18n-en'])
            self.assertIn('nonfree', p['data-i18n-en'])
            self.assertEqual(len(soup.select('#conditional-prior-deployment-cost')), 1)

    def test_time_gain_does_not_promote_joint_gate_or_learning(self):
        a = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())['latest_signed_cross_ray']['packed_prior_deployment_revision']
        b = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())['packed_prior_deployment_revision']
        self.assertEqual(a, b)
        self.assertEqual(a['decision'], 'FAIL_PREPARED_DEPLOYMENT_RESOURCE_DOMINANCE')
        self.assertTrue(a['prepared_deployment_wall_advantage'])
        self.assertTrue(a['previous_deployment_failure_retained'])
        self.assertEqual(a['held_trajectory_count'], 1)
        self.assertEqual(a['fresh_processes'], 18)
        self.assertEqual(a['independent_replay_checks'], 61)
        self.assertEqual(a['packed_factor_bytes'], 285373448)
        for mode in ('formal', 'independent'):
            self.assertLess(a['costs'][mode]['ratios']['wall_to_CGLS128'], .95)
            self.assertGreater(a['costs'][mode]['ratios']['rss_to_CGLS128'], 1)
        for key in ('joint_resource_gate_pass', 'preparation_training_included', 'offline_packing_included', 'all_folds_cost_test', 'learned_call_reduction', 'algorithm_breakthrough', 'resource_speedup', 'paper_success', 'real_bost', 'external_generalization', 'main_goal_complete'):
            self.assertFalse(a[key])

    def test_bilingual_note_privacy_and_visual(self):
        text = (ROOT / 'docs/packed_prior_deployment_revision_2026-10-11.md').read_text()
        self.assertIn('## 中文', text)
        self.assertIn('## English', text)
        self.assertIn('nonfree', text)
        self.assertIn('share standard BLAS/LAPACK', text)
        for term in ('/Users/', '/Volumes/', 'private_results', 'OPENED.json', 'FROZEN.json', 'checkpoint'):
            self.assertNotIn(term, text)
        self.assertTrue((ROOT / 'assets/packed_prior_deployment_revision_2026-10-11.png').exists())


if __name__ == '__main__':
    unittest.main()
