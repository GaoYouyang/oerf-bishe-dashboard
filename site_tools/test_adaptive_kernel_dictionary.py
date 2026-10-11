import json
from pathlib import Path
import unittest

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


class AdaptiveKernelDictionaryTest(unittest.TestCase):
    def test_four_bilingual_surfaces_preserve_previous_cost_evidence(self):
        for filename in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
            soup = BeautifulSoup((ROOT / filename).read_text(), 'html.parser')
            nodes = soup.select('#adaptive-kernel-dictionary')
            self.assertEqual(len(nodes), 1)
            p = nodes[0]
            self.assertEqual(p.get_text(), p['data-i18n-zh'])
            for text in (p['data-i18n-zh'], p['data-i18n-en']):
                for number in ('33/33', '11/11', '10.64%', '52.88%', '9.05%', 'FAIL'):
                    self.assertIn(number, text)
            self.assertIn('sampled', p['data-i18n-en'])
            self.assertIn('nonfree', p['data-i18n-en'])
            self.assertEqual(len(soup.select('#packed-prior-deployment-revision')), 1)

    def test_sampling_and_component_cost_do_not_promote_joint_gate(self):
        a = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())['latest_signed_cross_ray']['adaptive_kernel_dictionary']
        b = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())['adaptive_kernel_dictionary']
        self.assertEqual(a, b)
        self.assertEqual(a['decision'], 'FAIL_ADAPTIVE_KERNEL_DICTIONARY')
        self.assertEqual(a['primary_matched'], 33)
        self.assertEqual(a['primary_comparisons']['MeanNull-RidgeWarm4']['nonharm'], 32)
        self.assertGreater(a['primary_comparisons']['ConditionalNullRBF-RidgeWarm4']['median_field_ratio'], 1.05)
        self.assertLess(a['predictor_payload_ratio'], .2)
        self.assertLess(a['cached_batch_ratio'], .95)
        self.assertEqual(a['query_actions'], dict(A=5, AT=5))
        for key in ('complete_sequence', 'whole_pipeline_RSS_test', 'learned_call_reduction', 'resource_speedup', 'algorithm_breakthrough', 'paper_success', 'external_generalization', 'real_bost', 'main_goal_complete', 'neural_training_authorized'):
            self.assertFalse(a[key])

    def test_bilingual_note_and_privacy(self):
        text = (ROOT / 'docs/adaptive_kernel_dictionary_2026-10-11.md').read_text()
        self.assertIn('## 中文', text)
        self.assertIn('## English', text)
        for term in ('/Users/', '/Volumes/', 'private_results', 'FROZEN.json', 'OPENED.json', 'checkpoint'):
            self.assertNotIn(term, text)
        self.assertTrue((ROOT / 'assets/adaptive_kernel_dictionary_2026-10-11.png').exists())


if __name__ == '__main__':
    unittest.main()
