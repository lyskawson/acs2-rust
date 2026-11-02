import copy
import unittest

from analysis import statistics_for
from post_hoc import modal_statistics, uniform_pairs, winner_attributes
from test_analysis import record


class PostHocTests(unittest.TestCase):
    def test_uniform_weights_preserve_pairs_and_change_weight_sensitive_statistics(self):
        records = [record(0, 0, [0], 0, weight=3), record(0, 1, [1], 0), record(1, 0, [1], 1)]
        original = copy.deepcopy(records)
        uniform = uniform_pairs(records)
        self.assertEqual([item['weight'] for item in uniform], [1 / 3] * 3)
        self.assertAlmostEqual(statistics_for(uniform)['F'], 2 / 3)
        self.assertEqual(statistics_for(records)['F'], 4 / 5)
        self.assertEqual(records, original)
        with self.assertRaises(ValueError):
            uniform_pairs(records + [records[0]])

    def test_modal_gain_measures_useful_goal_dependent_deviations(self):
        records = [record(0, 0, [0], 0, weight=2), record(0, 1, [1], 1), record(0, 2, [2], 2)]
        result = modal_statistics(records)
        self.assertEqual(result['F_mod'], .5)
        self.assertEqual(result['F_minus_F_mod'], .5)
        self.assertEqual(result['nonmodal_mass'], .5)
        self.assertEqual(result['nonmodal_optimal'], 1)
        self.assertEqual(result['nonmodal_modal_optimal'], 0)
        self.assertEqual(result['F_minus_F_mod'], result['nonmodal_share_eligible'] * (result['nonmodal_optimal'] - result['nonmodal_modal_optimal']))
        self.assertTrue(result['post_hoc'])

    def test_modal_ties_use_lowest_action_and_fallbacks_have_no_modal_vote(self):
        records = [record(0, 0, [0], 0), record(0, 1, [0], 1), record(0, 2, [7], 7, True, 10)]
        result = modal_statistics(records)
        self.assertEqual(result['F_mod'], 1)
        self.assertEqual(result['F_minus_F_mod'], -.5)
        self.assertEqual(result['nonmodal_mass'], 1 / 12)
        self.assertEqual(result['nonmodal_share_eligible'], .5)
        self.assertEqual(result['nonmodal_optimal'], 0)
        self.assertEqual(result['nonmodal_modal_optimal'], 1)

    def test_constant_or_empty_policy_keeps_undefined_conditional_rates(self):
        result = modal_statistics([record(0, 0, [1], 0)])
        self.assertEqual(result['F_minus_F_mod'], 0)
        self.assertEqual(result['nonmodal_mass'], 0)
        self.assertIsNone(result['nonmodal_optimal'])
        empty = modal_statistics([record(0, 0, [1], 0, True)])
        self.assertIsNone(empty['F_mod'])
        self.assertIsNone(empty['nonmodal_share_eligible'])

    def test_goal_attribute_credit_splits_ties_and_retains_unconditional_denominator(self):
        good = record(0, 0, [0], 0)
        good['winners'] = [dict(specified_goal_attributes=value) for value in (0, 1, 2, 2)]
        bad = record(0, 1, [1], 0, True, 3)
        result = winner_attributes([good, bad])
        self.assertEqual(result['all']['mass'], [1 / 16, 1 / 16, 1 / 8])
        self.assertEqual(result['all']['conditional_share'], [.25, .25, .5])
        self.assertEqual(sum(result['all']['mass']), .25)
        self.assertEqual(result['failed']['mass'], [0, 0, 0])
        self.assertEqual(result['failed']['conditional_share'], [None, None, None])
        self.assertEqual(result['all']['mass'][1] + result['all']['mass'][2], statistics_for([good, bad])['values']['all']['winner_goal_mass'])


if __name__ == '__main__':
    unittest.main()
