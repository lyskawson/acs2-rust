from fractions import Fraction
import math
import unittest

from analysis import exact_success, head_spread, paired, seed_scores, threshold_crossing


def block(successes):
    return dict(success=sum(successes) / len(successes) / 10, evaluation_pairs=10,
                heads=[dict(success=count / 10) for count in successes])


class AnalysisTests(unittest.TestCase):
    def test_exact_success_keeps_half_counts_from_two_heads_and_rejects_inconsistent_means(self):
        self.assertEqual(exact_success(block([1, 2])), Fraction(3, 20))
        value = block([1, 2])
        value['success'] = 0.2
        with self.assertRaises(ValueError):
            exact_success(value)

    def test_primary_averages_points_within_seeds_and_final_view_is_separate(self):
        rows = [dict(seed=seed, nominal_step=point, evaluation_blocks=dict(test=block(counts)))
                for seed in range(42, 62) for point, counts in [(100, [0, 1]), (1000, [2, 3])]]
        scores = seed_scores(rows, [100, 1000])
        self.assertEqual(scores[42], dict(checkpoint_mean_within_seed=Fraction(3, 20), final_point=Fraction(1, 4)))
        self.assertEqual(threshold_crossing(rows, [100, 1000], '1/4'), 1000)
        self.assertIsNone(threshold_crossing(rows, [100, 1000], '250000000000000001/1000000000000000000'))
        with self.assertRaises(ValueError):
            seed_scores(rows + rows[:1], [100, 1000])

    def test_paired_intervals_use_twenty_seed_differences_in_both_views(self):
        left = {seed: dict(checkpoint_mean_within_seed=Fraction(seed - 42, 100), final_point=Fraction(1, 2)) for seed in range(42, 62)}
        right = {seed: dict(checkpoint_mean_within_seed=Fraction(), final_point=Fraction(1, 4)) for seed in range(42, 62)}
        result = paired(left, right)
        self.assertAlmostEqual(result['checkpoint_mean_within_seed']['mean'], 0.095)
        self.assertAlmostEqual(result['final_point']['mean'], 0.25)
        self.assertEqual(result['final_point']['se'], 0)
        expected_se = math.sqrt(sum((i / 100 - 0.095) ** 2 for i in range(20)) / 19 / 20)
        self.assertAlmostEqual(result['checkpoint_mean_within_seed']['se'], expected_se)
        self.assertAlmostEqual(result['checkpoint_mean_within_seed']['ci_low'], 0.095 - 2.093024054408263 * expected_se)

    def test_head_spread_uses_heads_within_a_seed_without_creating_replicates(self):
        self.assertAlmostEqual(head_spread(block([0, 10])), 0.5)


if __name__ == '__main__':
    unittest.main()
