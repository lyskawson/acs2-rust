import copy
import sys
import unittest
from unittest.mock import patch

from execution import authorization_manifest, measurement_identity
from protocol import AGENTS, PRIORITIZED, PILOT_PARTS, candidates_from_priorities, configurations, forecast, pilot, select_temperature
from protocol import ROOT, RECOVERY_PARTS, SECOND_TEMPERATURE_ORDER, cost_reduction, recovery, reservation_batches

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
            self.assertEqual(authorization_manifest(plan), plan | measurement_identity())
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
        with patch('execution.measurement_identity', return_value=identity):
            plan = authorization_manifest(pilot(PILOT_PARTS[0]))
            self.assertEqual(plan['measurement_commit'], identity['measurement_commit'])
            self.assertEqual(plan['binary_sha256'], identity['binary_sha256'])
            changed = dict(plan, stage='tuning_pending_approval')
            with self.assertRaisesRegex(ValueError, 'stage 2'):
                authorization_manifest(changed)

    def test_cost_pilot_cannot_substitute_measurement_identity_or_a_temperature(self):
        identity = dict(measurement_commit='a' * 40, binary_sha256='b' * 64, preset={})
        with patch('execution.measurement_identity', return_value=identity):
            expected = authorization_manifest(pilot(PILOT_PARTS[0]))
            for field, value in [('measurement_commit', 'c' * 40), ('binary_sha256', 'd' * 64), ('seeds', [43])]:
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        require_derivation(dict(expected, **{field: value}), expected)
            changed = copy.deepcopy(expected)
            changed['configurations'][0]['agent_settings'][AGENTS[0]]['temperature'] = 2
            with self.assertRaises(ValueError):
                require_derivation(changed, expected)

    def test_temperature_selection_is_validation_only_and_ties_choose_smaller(self):
        self.assertEqual(select_temperature({1: 0.4, 8: 0.4, 24: 0.3}, [1, 8, 24]), 1)
        self.assertEqual(select_temperature({1: 0.3, 8: 0.4}, [1, 8]), 8)
        with self.assertRaises(ValueError):
            select_temperature({1: 0.3}, [1, 8])

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

    def test_forecast_doubles_expected_work_and_reserves_every_arm_seed_and_candidate(self):
        cells = {(config['id'], agent): dict(body_wall_seconds=60, allocation_cpu_seconds=65,
            process_cpu_seconds=60, peak_rss_kib=8192) for config in configurations() for agent in AGENTS}
        candidates = {config['id']: {agent: dict(temperatures=[1, 8, 24]) for agent in PRIORITIZED}
                      for config in configurations()}
        value = forecast(cells, candidates, 3600)
        self.assertEqual(value['grid_reservation_cpu_seconds'], 6 * 4 * 20 * 120)
        self.assertEqual(value['tuning_reservation_cpu_seconds'], 6 * 2 * 5 * 3 * 120)
        self.assertAlmostEqual(value['doubled_forecast_with_pilots_cpu_hours'], 1 + 2 * 65 * (480 + 180) / 3600)

    def test_hard_reservations_charge_the_shared_limit_of_each_actual_batch(self):
        jobs = [dict(limit_minutes=limit) for limit in [1] * 15 + [5] * 15]
        batches = reservation_batches(jobs)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]['jobs']), 30)
        self.assertEqual(batches[0]['reservation_cpu_seconds'], 30 * 5 * 60)
        self.assertGreater(batches[0]['reservation_cpu_seconds'], sum(job['limit_minutes'] * 60 for job in jobs))

    def test_large_forecast_cuts_candidates_before_prioritized_k2_and_preserves_both_controls(self):
        cells = {(config['id'], agent): dict(body_wall_seconds=600, allocation_cpu_seconds=605,
            process_cpu_seconds=600, peak_rss_kib=8192) for config in configurations() for agent in AGENTS}
        candidates = {config['id']: {agent: dict(temperatures=[1, 8, 24],
            acceptance_curve=[0.95 ** t for t in range(1, 25)]) for agent in PRIORITIZED} for config in configurations()}
        agents, retained, prediction, changes = cost_reduction(cells, candidates, 3600)
        self.assertEqual(agents, ['acs2vcp_k4', 'acs2vcp_k2_uniform', 'acs2vcp_k4_uniform'])
        self.assertEqual(len(changes['applied']), 2)
        self.assertEqual(changes['trigger_cpu_hours'], 60)
        self.assertEqual(len(prediction['limits_minutes']), 6)
        self.assertEqual(sum(len(batch['jobs']) for batch in prediction['grid_batches']), 360)
        self.assertEqual(sum(len(batch['jobs']) for batch in prediction['tuning_batches']), 30)
        self.assertTrue(all(choice['temperatures'] == [8] for choices in retained.values() for choice in choices.values()))
        self.assertTrue(all(choice['temperatures'] == [1, 8, 24] for choices in candidates.values() for choice in choices.values()))


if __name__ == '__main__':
    unittest.main()
