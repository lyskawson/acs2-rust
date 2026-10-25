import copy
from fractions import Fraction
import math
import statistics
import unittest

from analysis import SEEDS, conditional_summary, diagnostic_summaries, exact_success, indexed, metrics, paired, seed_scores, success_summary
from thresholds import crossings


def block(successes, pairs=100):
    probability = successes / pairs
    return dict(evaluation_pairs=pairs, success=probability, evaluation_distribution='exhaustive_exact_weights',
        value_diagnostics=dict(all=dict(mean_first_action_estimate=400.0, mean_discounted_return=1000 * probability,
                                       mean_estimate_minus_return=400 - 1000 * probability),
            successful=dict(mean_first_action_estimate=1000.0, mean_discounted_return=1000.0, mean_estimate_minus_return=0.0)
            if successes else None))


def row(seed, point, successes):
    return dict(seed=seed, nominal_step=point, actual_steps=point, online_updates=point, replay_updates=0,
        population_classifiers=10, population_numerosity=12, knowledge=0.5,
        population_known_bytes_lower_bound=1000, replay_logical_bytes=0, trajectory_logical_bytes=0,
        wall_seconds_train=point / 100, wall_seconds_eval=point / 1000, wall_seconds_total=point * 0.011,
        evaluation_blocks={name: block(successes) for name in ['training', 'validation', 'test']})


class AnalysisTests(unittest.TestCase):
    def rows(self):
        return [row(seed, point, (seed - 42) if point == 100 else 2 * (seed - 42))
                for seed in SEEDS for point in [100, 200]]

    def test_primary_averages_points_within_each_seed_and_keeps_final_separate(self):
        data = self.rows()
        primary, final = seed_scores(data, [100, 200], 'checkpoint_mean'), seed_scores(data, [100, 200], 'final_point')
        self.assertEqual(primary[43], Fraction(3, 200))
        self.assertEqual(final[43], Fraction(1, 50))
        self.assertEqual(success_summary(primary)['mean_exact'], '57/400')
        self.assertEqual(success_summary(final)['mean_exact'], '19/100')

    def test_paired_intervals_use_seed_differences_instead_of_independent_arm_errors(self):
        left = {seed: Fraction(seed - 42, 100) for seed in SEEDS}
        right = {seed: left[seed] - Fraction(1, 10) for seed in SEEDS}
        result = paired(left, right)
        self.assertEqual(result['mean'], 0.1)
        self.assertEqual(result['se'], 0)
        self.assertEqual(result['ci_low'], 0.1)
        self.assertEqual(result['ci_high'], 0.1)
        varied = paired(left, {seed: Fraction() for seed in SEEDS})
        expected_se = statistics.stdev([i / 100 for i in range(20)]) / math.sqrt(20)
        self.assertAlmostEqual(varied['se'], expected_se)
        self.assertAlmostEqual(varied['ci_high'], 0.095 + 2.093024054408263 * expected_se)

    def test_threshold_crossing_includes_exact_equality_without_float_interpolation(self):
        rows = [row(seed, point, successes) for seed in SEEDS for point, successes in [(100, 10), (200, 20)]]
        result = crossings(rows, [100, 200], dict(equal='1/10', later='1/5', absent='1/4'))
        self.assertEqual(result, dict(equal=100, later=200, absent=None))

    def test_duplicate_missing_and_foreign_seed_rows_are_refused(self):
        data = self.rows()
        for changed in [data[:-1], data + [data[0]], data + [row(62, 100, 1)]]:
            with self.assertRaises(ValueError):
                indexed(changed, [100, 200])
        with self.assertRaises(ValueError):
            seed_scores(data, [100, 200], 'pooled_points')

    def test_inexact_or_sampled_success_is_refused(self):
        for changed in [dict(block(1), success=0.015), dict(block(1), evaluation_distribution='sampled')]:
            with self.assertRaises(ValueError):
                exact_success(changed)

    def test_unmeasured_wall_counters_stay_absent_in_archived_references(self):
        measured = metrics(row(42, 100, 25))
        self.assertIsNone(measured['wall_copy_share'])
        self.assertIsNone(measured['terminal_wall_copy_share'])
        self.assertEqual(measured['test_value_gap_failed'], 200)

    def test_empty_value_groups_and_contributing_seeds_are_explicit(self):
        self.assertEqual(conditional_summary([None] * 20), dict(n=0, mean=None, se=None))
        data = [row(seed, point, 0 if point == 100 else 10) for seed in SEEDS for point in [100, 200]]
        result = diagnostic_summaries(data, [100, 200], 'checkpoint_mean')
        successful = result['test_value_gap_successful']
        self.assertEqual(successful['n'], 20)
        self.assertEqual(successful['mean'], 0)
        self.assertEqual(set(successful['contributing_points_by_seed'].values()), {1})

    def test_copy_wall_fallback_and_selection_shares_use_their_registered_denominators(self):
        sample = row(42, 100, 25)
        sample['replay_diagnostics'] = dict(draws=10, fallbacks=3, routes=dict(relabeled=dict(
            draws=2, already_reached=1, after_counterfactual_end=1, done=1, reward_sum=1000)),
            wall_bumps=dict(unchanged_state=1, unchanged_state_terminal=0),
            objective_evaluations_by_purpose=dict(selection=dict(reach_evaluations=37, reward_evaluations=0, objective_evaluations=37)))
        result = metrics(sample)
        self.assertEqual(result['trivial_copy_share'], 0.5)
        self.assertEqual(result['after_end_copy_share'], 0.5)
        self.assertEqual(result['fallback_share'], 0.3)
        self.assertEqual(result['rewarded_copy_share'], 0.5)
        self.assertEqual(result['wall_copy_share'], 0.5)
        self.assertEqual(result['terminal_given_wall_copy_share'], 0)
        self.assertEqual(result['objective_selection_reach_evaluations_per_draw'], 3.7)

    def test_conditional_standard_error_counts_only_observed_seed_values(self):
        values = list(range(19))
        result = conditional_summary(values + [None])
        self.assertEqual(result['n'], 19)
        self.assertAlmostEqual(result['se'], statistics.stdev(values) / math.sqrt(19))


if __name__ == '__main__':
    unittest.main()
