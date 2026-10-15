import copy
import sys
import unittest
from unittest.mock import patch

from execution import authorization_manifest, pilot_measurement_identity
from protocol import AGENTS, PRIORITIZED, PILOT_PARTS, candidates_from_priorities, configurations, forecast, pilot
from protocol import ROOT, RECOVERY_PARTS, SECOND_TEMPERATURE_ORDER, draft, recovery, reservation_batches

sys.path.append(str(ROOT / 'experiments/grid'))
from submit import run_batches
from grid_authorization import require_derivation


class ProtocolTests(unittest.TestCase):
    def test_recovery_reuses_completed_runs_and_reserves_small_independent_batches(self):
        plans = list(map(recovery, RECOVERY_PARTS))
        first = [plan for plan in plans if plan['pilot_part'].startswith('recovery-t1-')]
        self.assertEqual(sum(len(plan['agents']) for plan in first), 22)
        self.assertEqual(len(first), 11)
        self.assertNotIn('recovery-t1-maze4-k2', RECOVERY_PARTS)
        self.assertEqual(set(SECOND_TEMPERATURE_ORDER), {config['task'] for config in configurations()})
        for plan in plans:
            batches = run_batches(plan)
            self.assertEqual(len(batches), 1)
            self.assertLessEqual(len(batches[0]), 2)
            self.assertEqual(plan['phase_cpu_hour_limit'], 5)
            self.assertEqual(plan['seeds'], [42])
            self.assertEqual(authorization_manifest(plan), plan | pilot_measurement_identity())
        changed = copy.deepcopy(recovery(RECOVERY_PARTS[0]))
        changed['configurations'][0]['limits_minutes'][changed['agents'][0]] += 1
        with self.assertRaises(ValueError):
            require_derivation(changed, recovery(RECOVERY_PARTS[0]))

    def test_cost_pilot_reserves_each_maze_limit_instead_of_the_slowest_limit_for_every_job(self):
        plan = pilot(PILOT_PARTS[0])
        batches = run_batches(plan)
        self.assertEqual(len(batches), 6)
        self.assertEqual(sum(map(len, batches)), 24)
        reservation = sum(max(config['limits_minutes'][agent] for config, agent, _ in batch) * 60 * len(batch) for batch in batches)
        self.assertEqual(reservation, 4 * 3600)
        self.assertTrue(all(len({config['id'] for config, _, _ in batch}) == 1 for batch in batches))

    def test_pilot_scope_has_all_mazes_heads_temperatures_and_controls_with_a_shared_five_hour_limit(self):
        first, second, reduced = map(pilot, PILOT_PARTS)
        self.assertEqual(len(first['configurations']), 6)
        self.assertEqual(first['agents'], AGENTS)
        self.assertEqual(second['agents'], PRIORITIZED)
        self.assertEqual(len(second['configurations']), 6)
        self.assertEqual([config['task'] for config in reduced['configurations']], ['maze4', 'maze7', 'woods14'])
        for plan in [first, second, reduced]:
            self.assertEqual(plan['seeds'], [42])
            self.assertEqual(plan['phase_cpu_hour_limit'], 5)
            self.assertEqual(plan['pilot_allocation_cpu_seconds'], 0)
            for config in plan['configurations']:
                self.assertEqual(config['evaluation_blocks'], ['validation'])
                self.assertEqual(config['targets'][-1], config['budget_steps'])
                self.assertEqual(config['cap'], config['registration']['cap'])

    def test_pilot_authorization_freezes_binary_and_source_and_refuses_stage_two(self):
        identity = dict(measurement_commit='a' * 40, binary_sha256='b' * 64, preset={})
        with patch('execution.pilot_measurement_identity', return_value=identity):
            plan = authorization_manifest(pilot(PILOT_PARTS[0]))
            self.assertEqual(plan['measurement_commit'], identity['measurement_commit'])
            self.assertEqual(plan['binary_sha256'], identity['binary_sha256'])
            changed = dict(plan, stage='tuning_pending_approval')
            with self.assertRaisesRegex(ValueError, 'stage 2'):
                authorization_manifest(changed)

    def test_cost_pilot_cannot_substitute_measurement_identity_or_a_temperature(self):
        identity = dict(measurement_commit='a' * 40, binary_sha256='b' * 64, preset={})
        with patch('execution.pilot_measurement_identity', return_value=identity):
            expected = authorization_manifest(pilot(PILOT_PARTS[0]))
            for field, value in [('measurement_commit', 'c' * 40), ('binary_sha256', 'd' * 64), ('seeds', [43])]:
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        require_derivation(dict(expected, **{field: value}), expected)
            changed = copy.deepcopy(expected)
            changed['configurations'][0]['agent_settings'][AGENTS[0]]['temperature'] = 2
            with self.assertRaises(ValueError):
                require_derivation(changed, expected)

    def test_candidate_temperatures_read_only_priority_statistics(self):
        rows = []
        for config in configurations():
            for agent in PRIORITIZED:
                for temperature in [1, 24]:
                    rows.append(dict(task=config['task'], agent=agent, nominal_step=config['budget_steps'],
                        agent_parameters=dict(temperature=temperature),
                        replay_diagnostics=dict(priority=dict(proposals=1000,
                            candidate_temperatures=list(range(1, 25)), candidate_acceptance_sums=[1000 * 0.95 ** t for t in range(1, 25)]))))
        before = copy.deepcopy(rows)
        expected = candidates_from_priorities(rows)
        for row in rows:
            row['evaluation_blocks'] = {'validation': {'success': 1.0}, 'test': {'success': 0.0}}
        self.assertEqual(candidates_from_priorities(rows), expected)
        for choices in expected.values():
            for value in choices.values():
                self.assertEqual(value['temperatures'], [6, 14, 24])
        self.assertNotIn('evaluation_blocks', before[0])

    def forecast_inputs(self, body=60):
        runs = []
        calibration = dict(cells=[])
        for config in configurations():
            for agent in AGENTS:
                temperatures = [0] if agent.endswith('_uniform') else ([1, 24] if config['task'] != 'maze6' else [1])
                for temperature in temperatures:
                    runs.append(dict(configuration_id=config['id'], agent=agent, temperature=temperature,
                        cost=dict(body_wall_seconds=body, allocation_cpu_seconds=body + 5,
                            process_cpu_seconds=body, peak_rss_kib=8192,
                            pilot_population_sum_over_points=10, training_wall_seconds=body,
                            classifier_perception_tests=100 * body)))
                if agent.endswith('_uniform'):
                    calibration['cells'].append(dict(configuration_id=config['id'], agent=agent,
                        pilot_classifier_tests=1000, revised_classifier_tests=500, avoided_classifier_tests=500,
                        all_block_disagreement_transitions=4, additional_policy_pairs=0))
        return runs, calibration

    def test_forecast_doubles_fixed_grid_and_reserves_every_arm_and_twenty_seeds_without_tuning(self):
        runs, calibration = self.forecast_inputs()
        value = forecast(runs, calibration, 3600)
        self.assertEqual(value['grid_reservation_cpu_seconds'], 12 * 20 * (180 + 120))
        self.assertEqual(value['grid_runs'], 480)
        self.assertAlmostEqual(value['grid_expected_cpu_hours'], 240 * (65.4 + 35.4) / 3600)
        self.assertAlmostEqual(value['doubled_forecast_with_pilots_cpu_hours'], 1 + 2 * 240 * (65.4 + 35.4) / 3600)
        self.assertEqual(value['hard_bound_with_pilots_cpu_hours'], 21)
        self.assertNotIn('tuning_runs', value)
        self.assertTrue(all(cell['limit_minutes'] >= cell['pilot_derived_limit_minutes']
                            for cell in value['cells'] if cell['temperature'] == 0))
        for missing in [calibration['cells'][:-1], []]:
            with self.assertRaises(ValueError):
                forecast(runs, dict(cells=missing), 3600)

    def test_maze6_uses_the_largest_same_head_second_temperature_cost_ratio_and_sizes_its_limit(self):
        runs, calibration = self.forecast_inputs()
        for run in runs:
            if run['temperature'] == 24:
                ratio = (2 if run['configuration_id'].startswith('mazema') else 1.5) if run['agent'] == AGENTS[0] else (3 if run['configuration_id'].startswith('woods14') else 1.2)
                run['cost']['allocation_cpu_seconds'] = ratio * 65
        value = forecast(runs, calibration, 3600)
        estimates = {cell['agent']: cell for cell in value['cells'] if cell['task'] == 'maze6' and cell['temperature'] == 24}
        for agent, ratio, source in [(AGENTS[0], 2, 'mazema'), (AGENTS[1], 3, 'woods14')]:
            cell = estimates[agent]
            self.assertEqual(cell['scale'], ratio)
            self.assertTrue(cell['ratio_source_configuration'].startswith(source))
            self.assertAlmostEqual(cell['expected_body_seconds'], 60.4 * ratio)
            self.assertGreaterEqual(cell['limit_minutes'] * 60, 1.5 * cell['expected_body_seconds'] + 30)

    def test_hard_reservations_charge_the_shared_limit_of_each_actual_batch(self):
        jobs = [dict(limit_minutes=limit) for limit in [1] * 15 + [5] * 15]
        batches = reservation_batches(jobs)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]['jobs']), 30)
        self.assertEqual(batches[0]['reservation_cpu_seconds'], 30 * 5 * 60)
        self.assertGreater(batches[0]['reservation_cpu_seconds'], sum(job['limit_minutes'] * 60 for job in jobs))

    def test_draft_fixes_four_arms_and_nine_comparisons_and_requests_the_actual_large_forecast(self):
        runs, calibration = self.forecast_inputs(body=600)
        rows = [dict(task=config['task'], agent=agent, nominal_step=config['budget_steps'],
                     agent_parameters=dict(temperature=1), replay_diagnostics=dict(priority=dict(proposals=1000,
                     candidate_temperatures=list(range(1, 25)), candidate_acceptance_sums=[999 - t for t in range(24)])))
                for config in configurations() for agent in PRIORITIZED]
        plan = draft(runs, rows, calibration, 3600, dict(measurement_commit='a' * 40, binary_sha256='b' * 64, preset={}))
        self.assertEqual(plan['agents'], AGENTS)
        self.assertEqual(plan['final_grid']['runs'], 480)
        self.assertEqual(plan['temperature']['prioritized'], 24)
        self.assertEqual(plan['temperature']['controls'], 0)
        self.assertIs(plan['temperature']['tuning'], False)
        self.assertNotIn('tuning', plan)
        self.assertNotIn('cost_reduction', plan)
        self.assertEqual(len(plan['analysis']['comparisons']), 9)
        self.assertIn([AGENTS[1], AGENTS[0]], plan['analysis']['comparisons'])
        for config in plan['configurations']:
            self.assertEqual(config['agent_settings'], {agent: dict(temperature=24) for agent in PRIORITIZED})
            self.assertEqual(config['evaluation_blocks'], ['training', 'validation', 'test'])
        forecast_value = plan['forecast']
        maximum = max(forecast_value['doubled_forecast_with_pilots_cpu_hours'], forecast_value['hard_bound_with_pilots_cpu_hours'])
        self.assertGreater(maximum, 60)
        self.assertGreaterEqual(plan['requested_cpu_hour_limit'], maximum)
        self.assertLess(plan['requested_cpu_hour_limit'] - maximum, 5)
        self.assertEqual(plan['requested_cpu_hour_limit'] % 5, 0)
        self.assertIs(plan['stage2_authorized'], False)
        self.assertTrue(plan['hindsight']['confirmation_required_before_registration'])
        with self.assertRaisesRegex(ValueError, 'stage 2'):
            authorization_manifest(plan)


if __name__ == '__main__':
    unittest.main()
