import copy
import unittest

from protocol import ARMS, BLOCKS, analysis_registration, configurations, pilot
from prepare_plan import forecast
from execution import replication, scope
from unittest.mock import patch


class ProtocolTests(unittest.TestCase):
    def test_pilot_is_exactly_thirty_unchanged_runs_and_four_hours_of_reservations(self):
        manifest = pilot(dict(measurement_commit='a' * 40, binary_sha256='b' * 64, preset={}))
        self.assertEqual(manifest['agents'], list(ARMS))
        self.assertEqual(manifest['seeds'], [42])
        self.assertEqual(len(manifest['configurations']) * len(ARMS), 30)
        self.assertEqual(sum(max(config['limits_minutes'].values()) * 60 * 5 for config in manifest['configurations']), 4 * 3600)
        self.assertEqual(manifest['phase_cpu_hour_limit'], 4)
        for config, old in zip(manifest['configurations'], configurations()):
            self.assertEqual(config['registration'], old['registration'])
            self.assertEqual(config['targets'], old['targets'])
            self.assertEqual(config['evaluation_blocks'], BLOCKS)
            self.assertEqual(config['agent_settings'], {})
            self.assertTrue(config['per_start_diagnostics'])

    def test_registration_fixes_the_requested_comparisons_and_labels_other_views(self):
        analysis = analysis_registration()
        self.assertEqual(analysis['points'], [30000, 100000, 'final'])
        self.assertEqual(len(analysis['final_policy_contrasts']), 5)
        self.assertEqual(len(analysis['final_value_contrasts']), 3)
        self.assertEqual(analysis['paired_interval']['critical_value'], 2.7764451051977987)
        self.assertEqual(analysis['paired_interval']['degrees_of_freedom'], 4)
        self.assertEqual(analysis['final_policy_metrics'], ['F', 'M', 'D'])
        self.assertIn('consistency over seeds', analysis['post_hoc'])

    def test_forecast_doubles_allocations_and_reserves_actual_array_maxima(self):
        runs = [dict(configuration_id=config['id'], agent=arm, cost=dict(body_wall_seconds=30 + 50 * index,
                    allocation_cpu_seconds=40 + 50 * index, peak_rss_kib=1024))
                for config in configurations() for index, arm in enumerate(ARMS)]
        summary = dict(runs=runs, allocation_cpu_seconds=1200)
        measured = forecast(summary)
        expected = sum(4 * run['cost']['allocation_cpu_seconds'] for run in runs)
        self.assertEqual(measured['doubled_forecast_cpu_hours'], (1200 + 2 * expected) / 3600)
        self.assertEqual(measured['hard_reservations_cpu_hours'], (1200 + 6 * 20 * 9 * 60) / 3600)
        with patch('execution.measurement_identity', return_value={}):
            manifest = replication(dict(forecast=measured), 30)
        self.assertEqual(manifest['seeds'], [43, 44, 45, 46])
        self.assertTrue(all(set(config['limits_minutes'].values()) == {9} for config in manifest['configurations']))
        self.assertEqual(len(manifest['configurations']) * len(manifest['agents']) * len(manifest['seeds']), 120)
        self.assertEqual(scope('a', '/study')['pilot_allocation_cpu_seconds'], 0)
        self.assertEqual(scope('a', '/study')['batch_roots'], ['/study/pilot', '/study/replication'])

    def test_forecast_refuses_missing_cells(self):
        with self.assertRaises(ValueError):
            forecast(dict(runs=[], allocation_cpu_seconds=0))


if __name__ == '__main__':
    unittest.main()
