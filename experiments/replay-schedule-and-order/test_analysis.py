import copy
from fractions import Fraction
import math
import statistics
import unittest

from protocol import ER, SHORTENED, label
from analysis import CONTRASTS, analyze_maze, diagnostic_summaries, intervals, metrics
from paired_statistics import SEEDS, T19, contrast, crossing, indexed, scores, success


def row(seed, point, count=20, gap=8.0, successful=2.0):
    block = dict(evaluation_pairs=100, success=count / 100, evaluation_distribution='exhaustive_exact_weights',
                 value_diagnostics=dict(all=dict(mean_estimate_minus_return=gap),
                     successful=dict(mean_estimate_minus_return=successful) if count else None))
    return dict(seed=seed, nominal_step=point, actual_steps=point, online_updates=0, replay_updates=3 * point,
        knowledge=point / 10000, population_classifiers=10, population_numerosity=12,
        population_known_bytes_lower_bound=100, replay_logical_bytes=200, trajectory_logical_bytes=300,
        wall_seconds_train=point / 1000, wall_seconds_eval=0.5, wall_seconds_total=point / 1000 + 0.5,
        replay_diagnostics=None, evaluation_blocks={name: copy.deepcopy(block) for name in ('training', 'validation', 'test')})


def rows(counts=(20, 40)):
    return [row(seed, point, count) for seed in SEEDS for point, count in zip((1000, 3000), counts)]


class AnalysisTests(unittest.TestCase):
    def test_primary_is_the_within_seed_point_mean_and_final_is_the_last_point(self):
        data = rows((30, 20))
        self.assertEqual(scores(data, [1000, 3000], 'checkpoint_mean')[42], Fraction(1, 4))
        self.assertEqual(scores(data, [1000, 3000], 'final_point')[42], Fraction(1, 5))

    def test_paired_interaction_uses_covariance_and_t19_on_seed_differences(self):
        values = {seed: Fraction(seed - 42, 100) for seed in SEEDS}
        zero = {seed: Fraction() for seed in SEEDS}
        arms = dict(S_X=values, S=zero, E_X=values, E=zero)
        cancelled = contrast(arms, CONTRASTS['(S_X - S) - (E_X - E)'])
        self.assertEqual((cancelled['exact_mean'], cancelled['se'], cancelled['ci_low'], cancelled['ci_high']), ('0', 0, 0, 0))
        result = contrast(arms, CONTRASTS['S_X - S'])
        expected_se = statistics.stdev(values.values()) / math.sqrt(20)
        self.assertEqual(result['exact_mean'], '19/200')
        self.assertAlmostEqual(result['se'], expected_se)
        self.assertAlmostEqual(result['ci_low'], 0.095 - T19 * expected_se)
        self.assertAlmostEqual(result['ci_high'], 0.095 + T19 * expected_se)

    def test_exact_threshold_crossing_includes_a_tie_and_has_no_interpolation(self):
        data = rows((20, 40))
        self.assertEqual(crossing(data, [1000, 3000], '1/5'), 1000)
        self.assertEqual(crossing(data, [1000, 3000], '3/10'), 3000)
        self.assertIsNone(crossing(data, [1000, 3000], '1/2'))

    def test_missing_duplicate_noninteger_and_nonexhaustive_rows_are_refused(self):
        for kind in ('missing', 'duplicate', 'noninteger', 'nonexhaustive'):
            data = rows()
            if kind == 'missing': data.pop()
            elif kind == 'duplicate': data.append(data[0])
            elif kind == 'noninteger': data[0]['evaluation_blocks']['test']['success'] = 0.20001
            else: data[0]['evaluation_blocks']['test']['evaluation_distribution'] = 'sampled'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                indexed(data, [1000, 3000])
        with self.assertRaises(ValueError):
            contrast(dict(a={42: Fraction(1, 2)}), [('a', 1)])
        with self.assertRaises(ValueError):
            success(dict(evaluation_pairs=100, success=float('nan')))

    def test_failed_value_gap_uses_failed_pair_mass_and_empty_groups_stay_null(self):
        value = metrics(row(42, 1000, 20))
        self.assertAlmostEqual(value['test_value_gap_failed'], 9.5)
        self.assertEqual(value['test_value_gap_successful'], 2)
        self.assertIsNone(metrics(row(42, 1000, 100))['test_value_gap_failed'])
        empty = metrics(row(42, 1000, 0))
        self.assertIsNone(empty['test_value_gap_successful'])
        self.assertEqual(empty['test_value_gap_failed'], 8)

    def test_final_diagnostics_use_only_the_last_point(self):
        data = rows()
        result = diagnostic_summaries(data, [1000, 3000], 'final_point')
        self.assertEqual(result['knowledge']['mean'], 0.3)
        self.assertEqual(result['knowledge']['contributing_points_by_seed']['42'], 1)
        primary = diagnostic_summaries(data, [1000, 3000], 'checkpoint_mean')
        self.assertEqual(primary['knowledge']['mean'], 0.2)

    def test_interval_update_rates_are_differences_and_include_online_learning(self):
        data = [row(42, 1000), row(42, 3000)]
        data[-1]['replay_updates'] = 5000
        self.assertEqual([value['updates_per_interaction_exact'] for value in intervals(data)], ['3', '1'])
        for value in data:
            value.update(online_updates=value['actual_steps'], replay_updates=0)
        self.assertEqual([value['updates_per_interaction_exact'] for value in intervals(data)], ['1', '1'])

    def test_registered_contrasts_and_order_control_use_their_own_arms_and_skip_base_axes(self):
        config = dict(id='maze4_c5_pairs', task='maze4', cap=5, targets=[1000, 3000])
        s = label(config, SHORTENED, 'base')
        selected = dict(B=dict(setting='3to1at30000'), C=dict(setting='backward'))
        dataset = {s: rows((20, 20)), ER: rows((10, 10)), 'acs2': rows((40, 40)), 'acs2er': rows((20, 20))}
        for agent, mode, counts in [(SHORTENED, '3to1at30000', (30, 30)), (ER, '3to1at30000', (15, 15)),
                                    (SHORTENED, 'backward', (30, 50)), (ER, 'backward', (15, 25)),
                                    (SHORTENED, 'random-pass', (10, 10))]:
            dataset[label(config, agent, mode)] = rows(counts)
        plan = dict(analysis=dict(comparisons=list(CONTRASTS),
            thresholds={config['id']: dict(acs2_final='2/5', acs2er_final='1/5')},
            baseline_threshold_crossing_checks={config['id']: {}}))
        result = analyze_maze(dataset, config, plan, selected)
        self.assertEqual(len(result['comparisons']), 18)
        order = next(value for value in result['comparisons'] if value['comparison'] == 'S_backward - S_order_control' and value['view'] == 'checkpoint_mean')
        self.assertEqual(order['difference']['exact_mean'], '3/10')
        self.assertEqual(result['threshold_crossings']['C']['acs2_final'], 3000)
        self.assertEqual(result['threshold_crossings']['C']['acs2er_final'], 1000)
        selected['C']['setting'] = 'base'
        for agent, mode in [(SHORTENED, 'backward'), (ER, 'backward'), (SHORTENED, 'random-pass')]:
            del dataset[label(config, agent, mode)]
        result = analyze_maze(dataset, config, plan, selected)
        self.assertEqual(len(result['comparisons']), 8)
        self.assertNotIn('C', result['threshold_crossings'])


if __name__ == '__main__':
    unittest.main()
