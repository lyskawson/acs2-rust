import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from forecast import MENU, expected_refreshes, known_ratio, projected_utility
from grid_limits import (PILOT_SHA256, PLAN_SHA256, approved_registration, batches as grid_batches,
                         conditional_limit, derive)
from pilot import BUILD_LIMIT, batches, registered_plan
from recovery import grouped_runs, pilot_accounting, retry_runs
from registration import ARMS, REPOSITORY, S1, TRIAL, pilot_registration
from validate import validate_row


class StudyTest(unittest.TestCase):
    def test_pilot_reservations_and_throttles_cover_exactly_the_registered_cells(self):
        settings, _, configs, runs = pilot_registration()
        reservations = sum(run['limit_seconds'] for run in runs)
        self.assertEqual(len(runs), 84)
        self.assertEqual(len(configs), 28)
        self.assertEqual(sum(cfg['targets'][-1] > 300000 for cfg in configs.values()), 16)
        self.assertEqual({run['seed'] for run in runs}, {42})
        self.assertEqual({run['agent'] for run in runs}, set(ARMS))
        self.assertEqual(reservations, settings['reserved_seconds'])
        self.assertLessEqual(reservations + BUILD_LIMIT, 72000)
        grouped = batches(registered_plan('test', 'test'))
        self.assertEqual({run['attempt_id'] for batch, _, _ in grouped for run in batch},
                         {run['attempt_id'] for run in runs})
        concurrent = sum(throttle for _, throttle, short in grouped if not short)
        concurrent += max(throttle for _, throttle, short in grouped if short)
        self.assertLessEqual(concurrent, 40)
        for batch, _, _ in grouped:
            self.assertEqual(len({run['limit_seconds'] for run in batch}), 1)

    def test_retry_preserves_cells_and_counts_unstarted_dependency_cancellations(self):
        plan = registered_plan('test', 'test')
        first = next(run for run in plan['runs'] if run['attempt_id'].startswith('woods102_alcs') and run['agent'] == TRIAL)
        other = [run for run in plan['runs'] if run['attempt_id'].startswith('maze4_c5_pairs')]
        runs = [first, *other]
        header = 'JobID|JobIDRaw|CPUTimeRAW|AllocCPUS|State|\n'
        lines = '40_0|41|390|1|FAILED|\n40_1|40|0|0|CANCELLED|\n40_2|40|0|0|CANCELLED|\n40_3|42|125|1|COMPLETED|\n'
        audit = pilot_accounting([dict(array_id=40, runs=runs)], header + lines)
        self.assertEqual(audit['states'], dict(FAILED=1, CANCELLED=2, COMPLETED=1))
        self.assertEqual(audit['spent_seconds'], 515)
        self.assertEqual(audit['reserved_seconds'], 0)
        retry = retry_runs(plan, audit)
        self.assertEqual({run['previous_attempt_id'] for run in retry}, {run['attempt_id'] for run in runs[:3]})
        self.assertTrue(all(run['seed'] == 42 and run['attempt_id'].endswith('_a02') for run in retry))
        self.assertEqual(sorted(run['limit_seconds'] for run in retry), [720, 720, 840])
        self.assertEqual(sum(n for _, n, _ in grouped_runs(retry, 40)), 3)
        with self.assertRaisesRegex(ValueError, 'duplicated or invalid charged'):
            pilot_accounting([dict(array_id=40, runs=runs)], header + lines.replace('40_3|42', '40_3|41'))

    def test_grid_limits_use_each_cells_last_checkpoint_and_route_heavy_seeds_through_gate(self):
        settings, _, _, _ = pilot_registration()
        study, _, configs = approved_registration()
        pilot = []
        for task, cfg in configs.items():
            targets = [step for step in cfg['targets'] if step <= settings['tasks'][task]['last_target']]
            early = [dict(step=step, process=120, utility=20, reference=50) for step in targets[:-1]]
            last = dict(step=targets[-1], process=50, utility=10, reference=80)
            for arm in ARMS:
                pilot.append(dict(task=task, agent=arm, points=[*early, last]))
        inputs = dict(final_plan_sha256=PLAN_SHA256, pilot_manifest_sha256=PILOT_SHA256,
                      archive={task: {str(seed): 800 if seed == 42 else 1000 for seed in study['seeds']}
                               for task in configs}, pilot=pilot)
        plan = derive(inputs)
        runs = {(run['task'], run['agent'], run['seed']): run for run in plan['runs']}
        self.assertEqual(runs['woods14', TRIAL, 43]['limit_seconds'], 1080)
        self.assertEqual(runs['mazee1_alcs', TRIAL, 42]['limit_seconds'], 1860)
        self.assertEqual(runs['mazee2_alcs', TRIAL, 42]['limit_seconds'], 1440)
        self.assertEqual(runs['mazee3_alcs', TRIAL, 43]['limit_seconds'], 1860)
        self.assertEqual(conditional_limit(1250, 1.25), 1680)
        self.assertEqual(len(plan['reused']), 48)
        batches = grid_batches(plan)
        scheduled = {run['run_id'] for batch in batches for job in batch['jobs'] for run in job['runs']}
        self.assertEqual(scheduled, {run['run_id'] for run in runs.values()})
        scouts = {batch['name'] for batch in batches if batch['phase'] == 'scout'}
        self.assertEqual(len(scouts), 6)
        gate = next(batch for batch in batches if batch['phase'] == 'gate')
        self.assertEqual(set(gate['dependencies']), scouts)
        leading = [batch for batch in batches if 'scout-gate' in batch['dependencies']]
        self.assertEqual(len(leading), 2)
        self.assertTrue(all(batch['phase'] == 'conditional' and batch['throttle'] == 19 for batch in leading))

    def test_cost_projection_preserves_anchor_and_models_fixed_and_candidate_work(self):
        point = dict(
            archive_cpu=60.0, process_cpu=96.0, steps=10000, retained_steps=9995, tu_draws=10030,
            cost=dict(evaluations=260, candidates_sampled=1500, candidate_slots_at_four=890,
                      cpu_seconds=12.0, scoring_cpu_seconds=5.0, insertion_evaluations=100,
                      insertion_candidates=700, insertion_candidate_slots_at_four=390,
                      insertion_cpu_seconds=8.0, insertion_scoring_cpu_seconds=3.0),
        )
        self.assertEqual(len(MENU), 6)
        self.assertEqual(known_ratio(point, 8, 2000), (1.6, 12.0, 0.026))
        self.assertEqual(projected_utility(point, 8, None), (8.0, 0.01))
        cheap_cpu, cheap_rate = projected_utility(point, 4, None)
        self.assertAlmostEqual(cheap_cpu, 5 + 3 * 390 / 700)
        self.assertEqual(cheap_rate, 0.01)
        mid_cpu, mid_rate = projected_utility(point, 8, 5000)
        self.assertTrue(8 < mid_cpu < 12)
        self.assertTrue(0.01 < mid_rate < 0.026)
        self.assertAlmostEqual(expected_refreshes(9995, 5000, 0.0005), 1 - 2.718281828459045 ** -2.4975)
        self.assertGreater(known_ratio(point, 4, 2000)[0], known_ratio(point, 4, 5000)[0])

    def test_real_rows_and_per_pair_records_use_exactly_the_registered_arms(self):
        binary = REPOSITORY / 'target/release/acs2-measure'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rows.jsonl'
            subprocess.run([
                str(binary), '--task', 'maze4', '--cap', '5', '--protocol', 'maze-pairs-v1',
                '--agents', ','.join((*ARMS, S1)), '--seeds', '42', '--targets', '100', '--per-start',
                '--tu-candidates', '8', '--tu-refresh', '2000', '--out', str(path),
            ], check=True)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        _, _, configs, _ = pilot_registration()
        config = configs['maze4_c5_pairs']
        self.assertEqual({row['agent'] for row in rows}, {*ARMS, S1})
        for row in rows:
            row['source_state'] = 'clean'
            validate_row(row, config, row['agent'], 42, row['commit'], 100)
        trial = next(row for row in rows if row['agent'] == TRIAL)
        changed = copy.deepcopy(trial)
        changed['match_formations_train'] += changed['trajectory_utility']['cost']['match_sets']
        with self.assertRaisesRegex(ValueError, 'measurement separation'):
            validate_row(changed, config, TRIAL, 42, trial['commit'], 100)
        changed = copy.deepcopy(trial)
        changed['agent_parameters']['trajectory_utility']['refresh_steps'] = 5000
        with self.assertRaisesRegex(ValueError, 'unregistered utility parameters'):
            validate_row(changed, config, TRIAL, 42, trial['commit'], 100)
        with self.assertRaisesRegex(ValueError, 'exactly the three TU arms and S\\+1'):
            validate_row(dict(trial, agent='acs2her_shortened'), config, 'acs2her_shortened', 42, trial['commit'], 100)


if __name__ == '__main__':
    unittest.main()
