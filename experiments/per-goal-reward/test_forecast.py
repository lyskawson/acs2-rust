import math
import unittest

from protocol import NEW_ARMS, configurations
from prepare_plan import forecast


class ForecastTests(unittest.TestCase):
    def fixture(self):
        runs = []
        for m, config in enumerate(configurations()):
            for a, arm in enumerate(NEW_ARMS):
                runs.append(dict(configuration_id=config['id'], agent=arm, cost=dict(
                    body_wall_seconds=41 + 31 * m + 13 * a, allocation_cpu_seconds=47 + 37 * m + 19 * a,
                    peak_rss_kib=1000 + 1100 * m + 700 * a)))
        return dict(runs=runs, allocation_cpu_seconds=2000)

    def test_forecast_keeps_each_arm_maze_cost_and_counts_actual_array_padding(self):
        summary = self.fixture()
        actual = forecast(summary)
        expected = 19 * sum(run['cost']['allocation_cpu_seconds'] for run in summary['runs'])
        self.assertEqual(actual['future_runs'], 228)
        self.assertEqual(actual['future_expected_cpu_hours'], expected / 3600)
        self.assertEqual(actual['expected_total_cpu_hours'], (2000 + expected) / 3600)
        self.assertEqual(sum(array['runs'] for array in actual['arrays']), 228)
        self.assertEqual([array['runs'] for array in actual['arrays']], [24, 30, 30, 30] * 2)
        self.assertEqual({array['agent'] for array in actual['arrays']}, set(NEW_ARMS))
        limits = {(cell['configuration_id'], cell['agent']): cell['limit_minutes'] for cell in actual['cells']}
        for cell, run in zip(actual['cells'], summary['runs']):
            self.assertEqual(cell['expected_cpu_seconds'], 19 * run['cost']['allocation_cpu_seconds'])
            self.assertEqual(cell['limit_minutes'], max(3, math.ceil((2 * run['cost']['body_wall_seconds'] + 60) / 60)))
        hard = 0
        for array in actual['arrays']:
            limit = max(limits[identifier, array['agent']] for identifier in array['configurations'])
            self.assertEqual(array['limit_minutes'], limit)
            self.assertEqual(array['reservation_cpu_seconds'], 60 * limit * array['runs'])
            hard += array['reservation_cpu_seconds']
        self.assertEqual(actual['hard_reservations_cpu_hours'], (2000 + hard) / 3600)
        self.assertEqual(actual['requested_cpu_hour_limit'], math.ceil(max(2000 + 2 * expected, 2000 + hard) / 3600))

    def test_forecast_refuses_missing_duplicated_or_unmeasured_cells(self):
        for change in ['missing', 'duplicate', 'zero']:
            summary = self.fixture()
            if change == 'missing': summary['runs'].pop()
            elif change == 'duplicate': summary['runs'][-1] = summary['runs'][0]
            else: summary['runs'][0]['cost']['body_wall_seconds'] = 0
            with self.assertRaises(ValueError): forecast(summary)
