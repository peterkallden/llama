import copy
import unittest
from analysis import lock, describe, correlation, lag_curve, summarize


class AnalysisTests(unittest.TestCase):
    def selection(self):
        rows = []
        for p in range(3):
            for dose in (.01, .03, .1, .2):
                rows.append(dict(source_layer_input=3+p, destination_layer_input=1, alpha=dose, beta=1., mean_margin_change=-p-dose))
        return {'candidates': rows, 'selected': rows[0], 'geometry_ranked_pair': rows[4]}

    def test_locked_choices_negative_and_ties(self):
        source = self.selection()
        result = lock(copy.deepcopy(source))
        self.assertEqual(len(result['locked_comparisons']), 6)
        self.assertEqual(result['low_dose_response']['ranking'][0]['source_layer_input'], 3)
        self.assertEqual(result['locked_comparisons'][1]['configuration']['alpha'], .01)
        source['candidates'][4]['mean_margin_change'] = source['candidates'][0]['mean_margin_change']
        self.assertEqual(lock(source)['low_dose_response']['ranking'][0]['source_layer_input'], 3)

    def test_bootstrap_is_paired_reproducible(self):
        self.assertEqual(describe([1., -2., 3.]), describe([1., -2., 3.]))
        self.assertEqual(describe([0., 0.])['paired_bootstrap_95_percentile'], [0., 0.])
        self.assertEqual(describe([-1., -2.])['regressed'], 2)
        self.assertIsNone(correlation([1, 1, 1], [2, 3, 4]))
        self.assertEqual(correlation([1, 2, 3], [3, 2, 1]), -1)

    def test_report_keeps_negative_and_undefined_ratios(self):
        events = []
        for method in ('cosine', 'margin', 'low_dose_response'):
            for comparison in ('common_alpha', 'selected_dose'):
                arms = {}
                for arm, margin in [('baseline', 1.), ('targeted_recirculation', .5), ('targeted_norm_matched_control', .75)]:
                    arms[arm] = {'answer_margin': margin, 'expected_path_future_logprobs': [-1.], 'incorrect_path_future_logprobs': [-2.]}
                events.append({'event': 'locked_comparison', 'method': method, 'comparison': comparison,
                    'report': {'configuration': {}, 'summary': {}, 'cases': [{'case_id': 'held-1', 'task_type': 'object_location',
                    'arms': arms, 'targeted_margin_change_vs_baseline': -.5, 'update_token_position': 0, 'prompt_token_count': 1}]}})
        report = summarize(events)
        self.assertIsNone(report['low_dose_retention'][0]['ratio'])
        self.assertEqual(report['comparisons'][0]['by_task_type']['object_location']['regression_fraction'], 1.)
        self.assertEqual(report['between_methods'][0]['difference']['mean'], 0.)
        events[1]['report']['cases'][0]['case_id'] = 'unpaired'
        with self.assertRaises(ValueError):
            summarize(events)

    def test_lag_contract_and_separate_cumulative(self):
        case = {'update_token_position': 2, 'prompt_token_count': 4, 'arms': {}}
        for name, vals in [('baseline', [-1., -1.]), ('targeted_recirculation', [-1., -.5]), ('targeted_norm_matched_control', [-1., -.8])]:
            case['arms'][name] = {'expected_path_future_logprobs': vals}
        result = lag_curve(case, 'expected')
        self.assertEqual(result['first_effect_lag'], 2)
        self.assertEqual(result['effects'][1]['continuation_cumulative_gain_vs_baseline'], .5)
        case['arms']['targeted_recirculation']['expected_path_future_logprobs'][0] = 0
        with self.assertRaises(ValueError):
            lag_curve(case, 'expected')


if __name__ == '__main__':
    unittest.main()
