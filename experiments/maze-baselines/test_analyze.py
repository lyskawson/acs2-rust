import json
import math
from pathlib import Path
import statistics
import unittest

from analyze import analyze_knowledge, analyze_values
from grid_protocol import require_budget


class MazeAnalysisTests(unittest.TestCase):
    def series(self, values):
        return {(agent, seed): list(values) for agent in ('acs2', 'acs2er') for seed in range(42, 62)}

    def test_primary_metric_weights_points_equally_within_each_seed(self):
        result = analyze_values(self.series([0, 1, 1]), [1, 2, 100], 0.1, 1)
        self.assertEqual(result['primary_score']['acs2']['mean'], 2 / 3)
        self.assertEqual(result['paired_primary_difference']['mean'], 0)
        self.assertEqual(result['first_nominal_interactions_at_threshold']['acs2'], 2)

    def test_knowledge_is_secondary_and_summarized_with_paired_seed_differences(self):
        series = self.series([0, 1, 1])
        series['acs2er', 42] = [0.3, 1, 1]
        result = analyze_knowledge(series, [1, 2, 100])
        self.assertEqual(result['checkpoint_mean']['acs2']['mean'], 2 / 3)
        self.assertAlmostEqual(result['paired_checkpoint_mean_difference']['mean'], 0.005)
        self.assertNotIn('verdict', result)
        self.assertNotIn('first_nominal_interactions_at_threshold', result)

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
        series = self.series([0.2])
        with self.assertRaises(ValueError):
            analyze_values(series, [1000, 2000], 0.1, 1)
        series = self.series([0.2, float('nan')])
        with self.assertRaises(ValueError):
            analyze_values(series, [1000, 2000], 0.1, 1)

    def test_proposed_budgets_use_geometry_and_keep_registered_distributions(self):
        root = Path(__file__).resolve().parent
        plan = json.loads((root / 'grid-plan.json').read_text())
        registration = json.loads((root / 'pair-protocol.json').read_text())
        self.assertEqual(plan['stage'], 'grid_pending_approval')
        self.assertEqual(plan['seeds'], list(range(42, 62)))
        self.assertEqual(plan['agents'], ['acs2', 'acs2er'])
        self.assertEqual(len(plan['configurations']), 6)
        for config, registered in zip(plan['configurations'], registration['configurations']):
            self.assertEqual(config['protocol'], 'maze-pairs-v1')
            self.assertEqual(config['row_schema'], 5)
            self.assertEqual(config['row_pool'], 'full')
            self.assertEqual(config['registration'], registered)
            self.assertEqual(config['budget_steps'], 2_000_000 * math.ceil(registered['diameter'] / 10))
            self.assertEqual(config['targets'][-1], config['budget_steps'])
            self.assertEqual(config['targets'][:8], [1000, 3000, 10000, 30000, 100000, 300000, 1000000, 2000000])
        self.assertEqual(plan['pilot_allocation_cpu_seconds'], 5488)
        self.assertEqual(plan['seed42_learning_reference']['mode'], 'new_protocol_pilot')
        self.assertEqual(json.loads((root / 'goal-grid-plan.json').read_text())['stage'], 'superseded_goal_grid')

    def test_mixed_array_limits_include_pilot_within_forty_cpu_hours(self):
        plan = json.loads(Path(__file__).with_name('grid-plan.json').read_text())
        reserved = sum(max(config['limits_minutes'].values()) * 60 * 40 for config in plan['configurations'])
        bound = require_budget(plan['pilot_allocation_cpu_seconds'], {'spent_seconds': 0, 'reserved_seconds': 0}, reserved)
        self.assertAlmostEqual(bound / 3600, plan['cost_rule']['worst_limits_including_pilots_cpu_hours'])
        self.assertLess(bound, 40 * 3600)
        for config in plan['configurations']:
            for agent, calibration in config['cost_calibration'].items():
                self.assertEqual(config['limits_minutes'][agent], math.ceil((2.5 * calibration['budget_body_wall_seconds'] + 90) / 60))


if __name__ == '__main__':
    unittest.main()
