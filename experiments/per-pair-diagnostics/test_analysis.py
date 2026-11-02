import copy
import random
import unittest

from analysis import attach, ceiling, consistency, paired, pool_blocks, statistics_for, summarize, truth_index, weighted_mean


def record(start, goal, optimal, action, fallback=False, weight=1.0):
    return dict(start=start, goal=goal, optimal=set(optimal), first_action=action, random_fallback=fallback,
        weight=weight, success=action in optimal, estimated_first_action_value=970.0, q_star=950.0, v_star=1000.0,
        winners=[] if fallback else [dict(specified_goal_attributes=1), dict(specified_goal_attributes=0)])


class AnalysisTests(unittest.TestCase):
    def test_optimal_and_goal_independent_policies_have_separated_scores(self):
        optimal = [record(start, goal, [goal], goal) for start in range(3) for goal in range(2)]
        blind = [dict(item, first_action=0) for item in optimal]
        best, fixed = statistics_for(optimal), statistics_for(blind)
        self.assertEqual((best['F'], best['D'], best['D_both_optimal']), (1, 1, 1))
        self.assertEqual((fixed['M'], fixed['D'], fixed['F'], fixed['C']), (1, 0, .5, .5))
        self.assertLessEqual(fixed['F'], fixed['C'])
        self.assertLess(best['M'], fixed['M'])

    def test_ceiling_is_per_start_and_keeps_tied_optima_and_weights(self):
        records = [record('a', 0, [0], 0, weight=3), record('a', 1, [0, 1], 1),
                   record('b', 0, [1], 1, weight=6)]
        self.assertEqual(ceiling(records), 1)
        self.assertEqual(statistics_for(records, 2)['random_reference'], .55)
        self.assertNotEqual(ceiling(records), .7)

    def test_mode_is_the_agents_action_even_when_it_is_always_suboptimal(self):
        records = [record(0, goal, [goal % 2], 7) for goal in range(4)]
        measured = statistics_for(records)
        self.assertEqual((measured['M'], measured['D'], measured['F']), (1, 0, 0))

    def test_uniform_policy_matches_the_random_reference_without_looking_optimal(self):
        rng = random.Random(813)
        records = [record(start, goal, [goal % 8], rng.randrange(8)) for start in range(1000) for goal in range(8)]
        result = statistics_for(records)
        self.assertLess(abs(result['F'] - result['random_reference']), .015)
        self.assertLess(result['F'], .16)
        self.assertLess(result['D_both_optimal'], .04)
        self.assertGreater(result['D'], .8)

    def test_random_fallbacks_are_excluded_from_all_three_behavior_denominators(self):
        records = [record(0, 0, [0], 0), record(0, 1, [1], 1), record(0, 2, [2], 0, True, 10)]
        result = statistics_for(records)
        self.assertEqual((result['F'], result['M'], result['D'], result['D_both_optimal']), (1, .5, 1, 1))
        self.assertEqual(result['eligible_count'], 2)
        self.assertEqual(result['random_fallback_share'], 10 / 12)
        self.assertEqual(result['values']['all']['pair_count'], 3)
        self.assertEqual(result['values']['all']['estimate_minus_q'], 20)
        empty = statistics_for([records[-1]])
        self.assertEqual((empty['F'], empty['M'], empty['D']), (None, None, None))

    def test_discrimination_conditions_on_disjoint_sets_and_uses_pair_weights(self):
        records = [record(0, 0, [0, 1], 0, weight=2), record(0, 1, [1, 2], 2, weight=3),
                   record(0, 2, [3], 0, weight=4)]
        result = statistics_for(records)
        self.assertEqual(result['D_weight'], 20)
        self.assertEqual(result['D'], .6)
        self.assertEqual(result['D_both_optimal'], 0)

    def test_value_errors_and_tied_winner_credit_have_separate_denominators(self):
        records = [record(0, 0, [0], 0), record(0, 1, [1], 0, True, 3)]
        result = statistics_for(records)['values']
        self.assertEqual(result['all']['estimate_minus_q'], 20)
        self.assertEqual(result['all']['estimate_minus_v'], -30)
        self.assertEqual(result['all']['winner_goal_share'], .5)
        self.assertEqual(result['all']['winner_goal_mass'], .125)
        self.assertIsNone(result['failed']['winner_goal_share'])
        self.assertEqual(result['failed']['pair_count'], 1)

    def test_intervals_pair_before_computing_standard_errors_and_require_five_seeds(self):
        left = {seed: seed * 100.0 for seed in range(42, 47)}
        right = {seed: value - 3 for seed, value in left.items()}
        result = paired(left, right)
        self.assertEqual((result['mean'], result['se'], result['ci_low'], result['ci_high']), (3, 0, 3, 3))
        right[42] = None
        self.assertEqual(paired(left, right)['n'], 4)
        self.assertIsNone(paired(left, right)['mean'])
        with self.assertRaises(ValueError):
            paired({42: 1}, {42: 0})

    def test_consistency_is_pairwise_plugin_descriptive_and_refuses_missing_seeds(self):
        seeds = {seed: [dict(record(0, 0, [0], 0), success=seed < 44)] for seed in range(42, 47)}
        result = consistency(seeds)
        self.assertEqual(result['observed'][2], 1)
        self.assertAlmostEqual(result['binomial_plugin'][2], 10 * .4**2 * .6**3)
        self.assertTrue(result['post_hoc'])
        self.assertEqual(result['pairs'][0]['solved_seeds'], 2)
        with self.assertRaises(ValueError):
            consistency({42: seeds[42]})

    def test_recorded_distance_mismatch_stops_attachment(self):
        model = dict(gamma=.95, reward=1000, actions=8, cap=3, pairs=[dict(start=[1, 1], goal=[1, 3],
            shortest_distance=2, value=950.0, action_values=[950.0] + [902.5] * 7, optimal_actions=[0])])
        row = dict(start='(1, 1)', goal='Goal { symbols: [Token(1), Token(3)] }', shortest_distance=2,
                   first_action=0, first_decision=dict(random_fallback=False, winners=[]))
        self.assertEqual(attach([row], truth_index(model))[0]['q_star'], 950)
        changed = dict(row, shortest_distance=1)
        with self.assertRaisesRegex(ValueError, 'STOP'):
            attach([changed], truth_index(model))
        model['pairs'][0]['value'] = 951
        with self.assertRaises(ValueError):
            attach([row], truth_index(model))


    def test_attach_uses_suboptimal_first_action_value_instead_of_state_value(self):
        model = dict(gamma=.95, reward=1000, actions=8, cap=3, pairs=[dict(start=[1, 1], goal=[1, 3],
            shortest_distance=2, value=950.0, action_values=[950.0, 902.5] + [0.0] * 6, optimal_actions=[0])])
        raw = dict(start='(1, 1)', goal='Goal { symbols: [Token(1), Token(3)] }', shortest_distance=2,
                   first_action=1, first_decision=dict(random_fallback=False, winners=[]))
        measured = attach([raw], truth_index(model))[0]
        self.assertEqual(measured['q_star'], 902.5)
        self.assertEqual(measured['v_star'], 950.0)
        self.assertNotIn(measured['first_action'], measured['optimal'])

    def test_nonzero_spread_uses_sample_deviation_and_t4_interval(self):
        measured = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        se = 1 / 2**.5
        margin = 2.7764451051977987 / 2**.5
        self.assertEqual(measured['n'], 5)
        self.assertEqual(measured['mean'], 3)
        self.assertAlmostEqual(measured['se'], se, places=14)
        self.assertAlmostEqual(measured['ci_low'], 3 - margin, places=14)
        self.assertAlmostEqual(measured['ci_high'], 3 + margin, places=14)

    def test_full_set_ceiling_and_random_reference_keep_goal_selective_fallback(self):
        records = [record(0, 0, [0], 0), record(0, 1, [0], 0),
                   record(0, 2, [1, 2, 3, 4], 7, True, 5)]
        measured = statistics_for(records)
        self.assertEqual(measured['C'], 5 / 7)
        self.assertEqual(measured['C_eligible'], 1)
        self.assertEqual(measured['random_reference'], 11 / 28)
        self.assertEqual(measured['random_reference_eligible'], 1 / 8)
        self.assertEqual(measured['F'], 1)

    def test_primary_pool_concatenates_three_blocks_without_changing_record_weights(self):
        blocks = dict(training=[record(0, 0, [0], 0, weight=.5), record(0, 1, [0], 0, weight=.5)],
                      validation=[record(0, 2, [1], 0, weight=1)], test=[record(0, 3, [1], 0, weight=1)])
        original = copy.deepcopy(blocks)
        pooled = pool_blocks(blocks)
        self.assertEqual(len(pooled), 4)
        self.assertEqual([item['goal'] for item in pooled], [0, 1, 2, 3])
        self.assertEqual([item['weight'] for item in pooled], [.5, .5, 1, 1])
        self.assertEqual(weighted_mean(pooled, lambda item: item['success']), 1 / 3)
        self.assertEqual(blocks, original)
        with self.assertRaises(ValueError):
            pool_blocks(dict(test=blocks['test']))


if __name__ == '__main__':
    unittest.main()
