import math
import unittest

from decision_metrics import decision_metrics
from test_analysis import record


class DecisionMechanismTests(unittest.TestCase):
    def test_within_start_demeaning_handles_unequal_weights_and_between_start_offsets(self):
        records = [dict(record(start, goal, [goal], goal, weight=weight), q_star=q,
                        estimated_first_action_value=2*q + offset)
                   for start, offset in [('a', 1000), ('b', -500)]
                   for goal, q, weight in [(0, 10, 1), (1, 14, 3)]]
        values = decision_metrics(records)
        self.assertAlmostEqual(values['within_start_slope'], 2)
        self.assertAlmostEqual(values['within_start_sd_q'], math.sqrt(3))
        self.assertAlmostEqual(values['within_start_sd_estimate'], 2 * math.sqrt(3))
        self.assertAlmostEqual(values['within_start_correlation'], 1)
        self.assertAlmostEqual(values['estimate_above_v'], .5)
        for item in records:
            item['q_star'] = 10
        self.assertIsNone(decision_metrics(records)['within_start_slope'])

    def test_overestimate_is_above_best_action_and_conditions_on_nonfallbacks(self):
        records = [dict(record(0, i, [i], i, fallback, weight), q_star=100, v_star=200,
                        estimated_first_action_value=estimate)
                   for i, estimate, fallback, weight in [(0, 150, False, 1), (1, 201, False, 3), (2, 1000, True, 20)]]
        result = decision_metrics(records)
        self.assertEqual(result['estimate_above_v'], .75)
        self.assertEqual(result['estimate_above_v_weight'], 4)
        self.assertAlmostEqual(result['estimate_minus_q'], (50 + 3*101 + 20*900) / 24)
        records[0]['estimated_first_action_value'] = math.nextafter(200.0, math.inf)
        self.assertEqual(decision_metrics(records)['estimate_above_v'], 1)
        records[0]['estimated_first_action_value'] = 200.0
        self.assertEqual(decision_metrics(records)['estimate_above_v'], .75)

    def test_modal_decomposition_and_cross_action_tie_use_separate_masses(self):
        records = [record(0, 0, [0], 0, weight=3), record(0, 1, [1], 1, weight=2),
                   record(0, 2, [2], 0, weight=1)]
        records[2]['winners'] = [dict(specified_goal_attributes=1)]
        result = decision_metrics(records)
        self.assertAlmostEqual(result['F_minus_C'], result['F_mod_minus_C'] + result['F_minus_F_mod'])
        self.assertAlmostEqual(result['F_mod'], .5)
        self.assertAlmostEqual(result['F'], 5/6)
        self.assertAlmostEqual(result['D_tie_implied'], 6/11)
        self.assertAlmostEqual(result['D_tie_share'], 6/8)
        self.assertEqual(result['D_tie_weight'], 6)
        self.assertEqual(result['D_different_weight'], 8)
        for item in records:
            item['first_action'] = 0
        self.assertIsNone(decision_metrics(records)['D_tie_share'])
        empty = decision_metrics([])
        self.assertIsNone(empty['within_start_sd_estimate'])
        self.assertIsNone(empty['estimate_above_v'])
