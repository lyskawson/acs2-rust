import copy
import unittest

from protocol import CANDIDATES, HER, SHORTENED, PILOT_SETTINGS, configurations, forecast, label, pilot, setting, study
from submit import run_batches
from grid_authorization import require_derivation
from execution import derive


class ProtocolTests(unittest.TestCase):
    def cells(self):
        return {(config['id'], label(agent, p)): dict(body_wall_seconds=80 + 100 * p,
            process_cpu_seconds=70 + 100 * p, allocation_cpu_seconds=90 + 100 * p,
            peak_rss_kib=10000) for config in configurations() for agent, p, _ in PILOT_SETTINGS}

    def test_pilot_reserves_thirty_ten_minute_allocations_with_shared_limit_five(self):
        batches = [run_batches(pilot(index)) for index in range(5)]
        self.assertEqual([len(batch[0]) for batch in batches], [6] * 5)
        self.assertEqual(sum(len(batch[0]) * 600 for batch in batches), 5 * 3600)
        for index in range(5):
            value = pilot(index)
            self.assertEqual(value['phase_cpu_hour_limit'], 5)
            self.assertEqual(value['seeds'], [42])
            self.assertEqual([config['budget_steps'] for config in value['configurations']], [2000000] * 4 + [4000000, 2000000])
            self.assertEqual(value['configurations'][0]['evaluation_blocks'], ['validation'] if index == 4 else ['training', 'validation', 'test'])

    def test_parameters_are_free_in_unit_interval_with_unambiguous_labels(self):
        for p in [0.0, 0.5, 0.731, 1.0, 1 / 3]:
            config = setting(configurations()[0], SHORTENED, p, ['validation'])
            self.assertEqual(config['agent_settings'], {SHORTENED: dict(relabeled_proportion=p)})
        self.assertEqual(label(SHORTENED, 0.5), 'acs2her_shortened_p1of2')
        self.assertEqual(label(HER, 2 / 3), 'acs2her_p2of3')
        for p in [-0.01, 1.01, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                label(HER, p)
        with self.assertRaises(ValueError):
            label('acs2her_nontrivial', 0.5)

    def test_registration_derives_tuning_control_and_conditional_test(self):
        plan = study(self.cells(), 1000, {}, {})
        tuning = derive(plan, 'tuning')
        self.assertEqual(sum(map(len, run_batches(tuning))), 120)
        control = derive(plan, 'control')
        self.assertEqual(sum(map(len, run_batches(control))), 120)
        self.assertEqual(control['agents'], [HER])
        for config in control['configurations']:
            self.assertEqual(config['agent_settings'], {HER: dict(relabeled_proportion=0.5)})
            self.assertEqual(config['row_arm'], 'acs2her_p1of2')
        config_id = configurations()[0]['id']
        selection = dict(selections={config_id: dict(proportion=0.5)})
        test = derive(plan, 'test', selection, config_id)
        self.assertEqual(len(run_batches(test)[0]), 20)
        self.assertEqual(test['configurations'][0]['agent_settings'][SHORTENED]['relabeled_proportion'], 0.5)
        for p in [2 / 3, 0.7]:
            with self.assertRaises(ValueError):
                derive(plan, 'test', dict(selections={config_id: dict(proportion=p)}), config_id)
        altered = copy.deepcopy(control)
        altered['configurations'][0]['agent_settings'][HER]['relabeled_proportion'] = 2 / 3
        with self.assertRaises(ValueError):
            require_derivation(altered, control)
        altered = copy.deepcopy(tuning)
        altered['configurations'][0]['agent_settings'][SHORTENED]['relabeled_proportion'] = 0.7
        with self.assertRaises(ValueError):
            require_derivation(altered, tuning)

    def test_forecasts_include_both_candidate_sets_and_actual_array_reservations(self):
        plan = study(self.cells(), 1000, {}, {})
        full = plan['forecasts']['with_four_fifths']
        cut = plan['forecasts']['without_four_fifths']
        self.assertEqual(full['candidates'], CANDIDATES)
        self.assertEqual(cut['candidates'], CANDIDATES[:-1])
        self.assertEqual(sum(cell['runs'] for cell in full['cells'] if cell['purpose'] == 'tuning'), 120)
        self.assertEqual(sum(cell['runs'] for cell in cut['cells'] if cell['purpose'] == 'tuning'), 90)
        self.assertEqual(sum(cell['runs'] for cell in full['cells'] if cell['purpose'] == 'test'), 120)
        self.assertGreater(full['doubled_forecast_cpu_hours'], cut['doubled_forecast_cpu_hours'])
        cut_plan = study(self.cells(), 1000, {}, {}, CANDIDATES[:-1])
        tuning_batches = run_batches(derive(cut_plan, 'tuning'))
        self.assertEqual(sum(map(len, tuning_batches)), 90)
        self.assertTrue(all(20 <= len(batch) <= 40 for batch in tuning_batches))
        actual = sum(len(batch) * 60 * max(config['limits_minutes'][agent] for config, agent, _ in batch) for batch in tuning_batches)
        self.assertEqual(actual, sum(cell['reservation_cpu_seconds'] for cell in cut['cells'] if cell['purpose'] == 'tuning'))
        reserved = sum(cell['reservation_cpu_seconds'] for cell in full['cells'])
        self.assertAlmostEqual(full['hard_reservations_cpu_hours'], (1000 + reserved) / 3600)
        for cell in full['cells']:
            self.assertEqual(cell['reservation_cpu_seconds'], cell['runs'] * cell['limit_minutes'] * 60)
        for kwargs in [dict(pilot_seconds=18001), dict(proportions=[1 / 3, 0.5])]:
            with self.assertRaises(ValueError):
                forecast(self.cells(), **(dict(pilot_seconds=1000) | kwargs))
        cells = self.cells()
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError):
            forecast(cells, 1000)
        plan = study(self.cells(), 1000, {}, {}, [0.0, 2 / 3, 0.731, 1.0])
        self.assertEqual(plan['tuning']['proportions'], [0.0, 2 / 3, 0.731, 1.0])

    def test_analysis_registration_contains_exact_dose_decomposition_and_thresholds(self):
        plan = study(self.cells(), 1000, {}, {})
        self.assertEqual(plan['analysis']['comparisons'][-2:], [['S@2/3', 'H@1/2'], ['H@1/2', 'H@2/3']])
        self.assertEqual(plan['analysis']['thresholds']['maze4_c5_pairs']['acs2_final'], '527/1620')
        self.assertEqual(plan['analysis']['paired_interval']['critical_value'], 2.093024054408263)
        self.assertIn('post hoc', plan['analysis']['post_hoc'])


if __name__ == '__main__':
    unittest.main()
