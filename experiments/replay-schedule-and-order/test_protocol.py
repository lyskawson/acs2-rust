import copy
import hashlib
import json
import unittest

from protocol import (BASE_PROPORTIONS, BLOCKS, ER, PILOT_SETTINGS, PROPORTION_SHA256, ROOT, SCHEDULES,
                      SCREEN_SETTINGS, SHORTENED, configurations, forecast, label, pilot, pilot_roots, scope_roots, setting, study)
from execution import derive, scope
from grid_authorization import require_derivation
from grid_protocol import require_budget


def costs():
    return {(config['id'], label(config, agent, mode)): dict(body_wall_seconds=100,
        allocation_cpu_seconds=110, process_cpu_seconds=95, peak_rss_kib=12000)
        for config in configurations() for agent, mode in PILOT_SETTINGS}


def plan():
    return study(costs(), 6600, dict(measurement_commit='a' * 40, binary_sha256='b' * 64, preset={}), {})


class ProtocolTests(unittest.TestCase):
    def test_prior_registration_is_immutable_and_selected_proportions_match_the_scope(self):
        self.assertEqual(hashlib.sha256((ROOT / 'experiments/hindsight-proportion/plan.json').read_bytes()).hexdigest(), PROPORTION_SHA256)
        self.assertEqual(BASE_PROPORTIONS, dict(maze4=0.5, maze5=0.8, maze6=0.8, maze7=0.8, woods14=0.8, mazema=1 / 3))
        self.assertEqual(len(configurations()), 6)

    def test_exact_sixty_run_pilot_has_all_blocks_points_and_ten_settings(self):
        self.assertEqual(len(PILOT_SETTINGS), 10)
        self.assertEqual(len(pilot_roots()), 10)
        for index, (agent, mode) in enumerate(PILOT_SETTINGS):
            manifest = pilot(index)
            self.assertEqual(manifest['seeds'], [42])
            self.assertEqual(manifest['phase_cpu_hour_limit'], 12)
            self.assertEqual(manifest['agents'], [agent])
            for base, config in zip(configurations(), manifest['configurations']):
                self.assertEqual(config, setting(base, agent, mode, BLOCKS))
                self.assertEqual(config['evaluation_blocks'], BLOCKS)
        for index in (-1, 10, True):
            with self.assertRaises(ValueError):
                pilot(index)

    def test_scope_has_every_root_once_and_pilot_offset_zero(self):
        value = scope('a' * 64, '/study')
        self.assertEqual(len(scope_roots()), 41)
        self.assertEqual(len(set(value['batch_roots'])), 41)
        self.assertEqual(value['pilot_allocation_cpu_seconds'], 0)
        self.assertEqual(sum(6 * 10 * 60 for _ in PILOT_SETTINGS), 36000)
        require_budget(0, dict(spent_seconds=0, reserved_seconds=36000), 0, 12)

    def test_screening_contains_only_registered_shortened_settings_and_validation(self):
        manifest = derive(plan(), 'screening')
        self.assertEqual(len(manifest['configurations']) * len(manifest['seeds']), 120)
        self.assertEqual(manifest['agents'], [SHORTENED])
        self.assertEqual(manifest['seeds'], list(range(42, 47)))
        for config in manifest['configurations']:
            self.assertEqual(config['evaluation_blocks'], ['validation'])
            self.assertNotIn('pass_random', config['row_arm'])
        altered = copy.deepcopy(manifest)
        altered['configurations'][0]['agent_settings'][SHORTENED]['replay_schedule'] = '3to1at40000'
        with self.assertRaises(ValueError):
            require_derivation(altered, manifest)

    def test_test_manifest_is_conditional_per_axis_and_contains_the_matched_er_setting(self):
        registered = plan()
        config = configurations()[0]
        selection = dict(selections={base['id']: dict(B=dict(setting='base'), C=dict(setting='base')) for base in configurations()})
        for axis in ('B', 'C'):
            with self.assertRaises(ValueError):
                derive(registered, 'test', selection, config['id'], axis, 'shortened')
        for mode in SCHEDULES:
            selection['selections'][config['id']]['B']['setting'] = mode
            for slot, agent in [('shortened', SHORTENED), ('er', ER)]:
                manifest = derive(registered, 'test', selection, config['id'], 'B', slot)
                self.assertEqual(manifest['agents'], [agent])
                self.assertEqual(manifest['seeds'], list(range(42, 62)))
                self.assertEqual(manifest['configurations'][0]['agent_settings'][agent]['replay_schedule'], mode)
        selection['selections'][config['id']]['C']['setting'] = 'backward'
        for slot, agent, order in [('shortened', SHORTENED, 'backward'), ('er', ER, 'backward'), ('order-control', SHORTENED, 'random-pass')]:
            manifest = derive(registered, 'test', selection, config['id'], 'C', slot)
            settings = manifest['configurations'][0]['agent_settings'][agent]
            self.assertEqual(settings['replay_schedule'], 'constant')
            self.assertEqual(settings['replay_order'], order)

    def test_forecast_doubles_cost_and_reserves_the_actual_array_maxima(self):
        value = forecast(costs(), 6600)
        self.assertEqual(value['screening_runs'], 120)
        self.assertEqual(value['maximum_test_runs'], 600)
        self.assertAlmostEqual(value['expected_total_cpu_hours'], (6600 + 720 * 110) / 3600)
        self.assertAlmostEqual(value['doubled_forecast_cpu_hours'], (6600 + 2 * 720 * 110) / 3600)
        self.assertGreaterEqual(value['requested_cpu_hour_limit'], value['hard_reservations_cpu_hours'])
        self.assertGreaterEqual(value['requested_cpu_hour_limit'], value['doubled_forecast_cpu_hours'])
        for group in {cell['batch_group'] for cell in value['cells'] if cell['purpose'] == 'screening'}:
            members = [cell for cell in value['cells'] if cell.get('batch_group') == group]
            self.assertEqual(len({cell['limit_minutes'] for cell in members}), 1)
            self.assertEqual(sum(cell['runs'] for cell in members), 20)
        for invalid in (0, 43201):
            with self.assertRaises(ValueError):
                forecast(costs(), invalid)


if __name__ == '__main__':
    unittest.main()
