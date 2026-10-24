import copy
import unittest

from protocol import AGENTS, configurations, forecast, pilot, study
from submit import run_batches


class ProtocolTests(unittest.TestCase):
    def cells(self):
        return {(config['id'], agent): dict(body_wall_seconds=8, process_cpu_seconds=8,
                allocation_cpu_seconds=10, peak_rss_kib=1024) for config in configurations() for agent in AGENTS}

    def test_pilot_uses_registered_budgets_and_exactly_two_allocated_hours(self):
        plan = pilot()
        runs = run_batches(plan)
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0]), 12)
        self.assertEqual(max(config['limits_minutes'][agent] for config, agent, _ in runs[0]) * 60 * len(runs[0]), 7200)
        self.assertEqual(plan['phase_cpu_hour_limit'], 2)
        self.assertEqual(plan['seeds'], [42])
        self.assertEqual(len(plan['configurations']), 6)
        self.assertEqual([config['budget_steps'] for config in plan['configurations']], [2000000] * 4 + [4000000, 2000000])

    def test_forecast_charges_actual_twenty_job_reservations(self):
        result = forecast(self.cells(), 120)
        self.assertAlmostEqual(result['grid_expected_cpu_hours'], 2400 / 3600)
        self.assertAlmostEqual(result['doubled_forecast_with_pilot_cpu_hours'], 4920 / 3600)
        self.assertEqual(result['grid_reservation_cpu_seconds'], 240 * 180)
        self.assertEqual(result['requested_cpu_hour_limit'], 15)

    def test_forecast_rejects_missing_or_unmeasured_costs(self):
        cells = self.cells()
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError):
            forecast(cells, 120)
        for key, value in [('body_wall_seconds', 0), ('allocation_cpu_seconds', float('nan')), ('peak_rss_kib', -1)]:
            cells = self.cells()
            cells[next(iter(cells))][key] = value
            with self.assertRaises(ValueError):
                forecast(cells, 120)
        with self.assertRaises(ValueError):
            forecast(self.cells(), 7201)

    def test_registration_fixes_both_views_five_contrasts_and_archived_thresholds(self):
        plan = study(self.cells(), 120, {}, {})
        self.assertEqual(plan['runs'], 240)
        self.assertEqual(plan['seeds'], list(range(42, 62)))
        self.assertEqual(plan['analysis']['views'], ['checkpoint_mean', 'final_point'])
        self.assertEqual(plan['analysis']['comparisons'], [
            ['acs2her_nontrivial', 'acs2her'], ['acs2her_nontrivial', 'acs2er_episodic'],
            ['acs2her_shortened', 'acs2her'], ['acs2her_shortened', 'acs2er_episodic'],
            ['acs2her_shortened', 'acs2her_nontrivial']])
        self.assertEqual(len(plan['analysis']['thresholds']), 6)
        self.assertEqual(plan['analysis']['thresholds']['maze4_c5_pairs']['acs2_final'], '527/1620')
        for config in plan['configurations']:
            batch = dict(stage='approved_grid', configurations=[copy.deepcopy(config)], agents=[AGENTS[0]], seeds=plan['seeds'], batch_size=20)
            self.assertEqual(len(run_batches(batch)[0]), 20)


if __name__ == '__main__':
    unittest.main()
