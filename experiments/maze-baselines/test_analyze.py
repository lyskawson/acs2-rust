import statistics
import unittest

from analyze import analyze_values


class MazeAnalysisTests(unittest.TestCase):
    def series(self, values):
        return {(agent, seed): list(values) for agent in ('acs2', 'acs2er') for seed in range(42, 62)}

    def test_primary_metric_weights_points_equally_within_each_seed(self):
        result = analyze_values(self.series([0, 1, 1]), [1, 2, 100], 0.1, 1)
        self.assertEqual(result['primary_score']['acs2']['mean'], 2 / 3)
        self.assertEqual(result['paired_primary_difference']['mean'], 0)
        self.assertEqual(result['first_nominal_interactions_at_threshold']['acs2'], 2)

    def test_paired_interval_uses_twenty_differences_and_t19(self):
        series = self.series([0.2, 0.4])
        differences = []
        for seed in range(42, 62):
            difference = (seed - 42) / 1000
            differences.append(difference)
            series['acs2er', seed] = [value + difference for value in series['acs2', seed]]
        paired = analyze_values(series, [1000, 2000], 0.1, 1)['paired_primary_difference']
        expected_se = statistics.stdev(differences) / (20 ** 0.5)
        self.assertAlmostEqual(paired['se'], expected_se)
        self.assertAlmostEqual(paired['ci_low'], statistics.mean(differences) - 2.093024054408263 * expected_se)

    def test_registered_rejection_and_unreached_threshold_are_preserved(self):
        result = analyze_values(self.series([0.05, 0.08]), [1000, 2000], 0.1, 1)
        self.assertEqual(result['verdict'], 'floor_baselines_retain_for_successors')
        self.assertIsNone(result['first_nominal_interactions_at_threshold']['acs2'])
        result = analyze_values(self.series([0.95, 0.98]), [1000, 2000], 0.1, 1)
        self.assertEqual(result['verdict'], 'too_easy_no_speed_ranking')

    def test_missing_seed_point_or_nonfinite_value_is_refused(self):
        series = self.series([0.2, 0.4])
        del series['acs2', 43]
        with self.assertRaises(ValueError):
            analyze_values(series, [1000, 2000], 0.1, 1)
        series = self.series([0.2, float('nan')])
        with self.assertRaises(ValueError):
            analyze_values(series, [1000, 2000], 0.1, 1)


if __name__ == '__main__':
    unittest.main()
