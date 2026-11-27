from pathlib import Path
import unittest
from unittest.mock import patch

import recovery_plan
from grid_plan import MAX_SECONDS
from proportions import MEAN, SCOUTS


class RecoveryPlanTests(unittest.TestCase):
    def test_reservation_charges_previous_spend_and_every_missing_run(self):
        stop = {'total_cpu_seconds': 4_202_368}
        with patch.object(recovery_plan, 'sealed_stop', return_value=({}, stop, [])):
            plan = recovery_plan.registration(Path('sealed'), 14 * 3600)
            self.assertEqual(plan['maximum_reserved_and_spent_cpu_seconds'], 8_083_168)
            self.assertEqual(plan['cumulative_cpu_limit_seconds'], MAX_SECONDS)
            with self.assertRaisesRegex(ValueError, 'exceed'):
                recovery_plan.registration(Path('sealed'), 50 * 3600)

    def test_selects_only_unfinished_cells_and_binds_their_prefix(self):
        task = SCOUTS[0]
        config = dict(id='maze', targets=[1000, 3000, 10000])
        runs = [dict(run_id=f'run-{seed}', task=task, agent=MEAN, arm='mean', seed=seed,
                     tu_share=.25, phase='conditional', pilot=None) for seed in (43, 44, 45)]
        old = [dict(run, path=f"attempts/{run['run_id']}/measurement/maze_{MEAN}_s{run['seed']}",
                    outcome=outcome, prior_limit_seconds=7200, checkpoints=count, last_checkpoint=last,
                    rows_sha256=digest)
               for run, outcome, count, last, digest in
               zip(runs, ('full', 'partial', 'not_started'), (3, 2, 0), (10000, 3000, None),
                   ('a', 'b', None))]
        summary = dict(total_cpu_seconds=490000, grid_cpu_seconds=490000-53037, trigger=old[1])
        before = {f"{old[1]['path']}/rows.jsonl": 'b'}
        with (patch.object(recovery_plan, 'sealed_stop', return_value=(before, summary, old)),
              patch.object(recovery_plan, 'RETRY_COUNT', 2),
              patch.object(recovery_plan, 'PREFIX_COUNT', 2)):
            plan = recovery_plan.registration(Path('sealed'), 14*3600)
            grid = dict(runs=runs, configs={task: config}, reused=[])
            retry = recovery_plan.derive_recovery(grid, Path('sealed'), plan)
            self.assertEqual([run['seed'] for run in retry['runs']], [44, 45])
            self.assertEqual(retry['runs'][0]['previous'],
                             dict(path=f"{old[1]['path']}/rows.jsonl", sha256='b', checkpoints=2))
            self.assertIsNone(retry['runs'][1]['previous'])
            old[1]['last_checkpoint'] = 1000
            with self.assertRaisesRegex(ValueError, 'preserved retry prefix'):
                recovery_plan.derive_recovery(grid, Path('sealed'), plan)

    def test_recovery_arrays_are_submitted_as_one_bounded_chain(self):
        plan = dict(runs=[dict(limit_seconds=50400) for _ in range(77)])
        first, second = recovery_plan.recovery_schedule(plan)
        self.assertEqual([first['throttle'], second['throttle']], [40, 37])
        self.assertEqual(first['dependencies'], [])
        self.assertEqual(second['dependencies'], [first['name']])
        self.assertEqual(len(first['jobs'])+len(second['jobs']), 77)

    def admit_stopped_cell(self, limit, *, outcome, **evidence):
        task = SCOUTS[0]
        run = dict(run_id='maze-43', task=task, agent=MEAN, arm='mean', seed=43,
                   tu_share=.25, phase='conditional', pilot=None)
        path = f'attempts/maze-43/measurement/maze_{MEAN}_s43'
        row = dict(run, path=path, outcome=outcome, prior_limit_seconds=7200,
                   checkpoints=0, last_checkpoint=None, rows_sha256=None)
        row.update(evidence)
        manifest = {f'{path}/rows.jsonl': 'previous-rows'}
        summary = dict(total_cpu_seconds=490000, grid_cpu_seconds=436963, trigger=row)
        plan = dict(runs=[run], configs={task: dict(id='maze', targets=[1000, 3000])}, reused=[])
        root = Path('sealed')
        with (patch.object(recovery_plan, 'sealed_stop', return_value=(manifest, summary, [row])),
              patch.object(recovery_plan, 'RETRY_COUNT', 1),
              patch.object(recovery_plan, 'PREFIX_COUNT', int(outcome == 'partial'))):
            approved = recovery_plan.registration(root, limit)
            return recovery_plan.derive_recovery(plan, root, approved)['runs'][0]

    def test_recovery_limit_must_exceed_the_stopped_allocation(self):
        evidence = dict(outcome='partial', checkpoints=1, last_checkpoint=1000, rows_sha256='previous-rows')
        self.assertEqual(self.admit_stopped_cell(7260, **evidence)['limit_seconds'], 7260)
        for limit in (7140, 7200):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, 'wall-clock limit differs'):
                    self.admit_stopped_cell(limit, **evidence)

    def test_cancelled_cell_must_have_no_measurement_evidence(self):
        self.assertIsNone(self.admit_stopped_cell(7260, outcome='not_started')['previous'])
        for evidence in (dict(checkpoints=1), dict(rows_sha256='previous-rows')):
            with self.subTest(evidence=evidence):
                with self.assertRaisesRegex(ValueError, 'cancelled slot has measurement evidence'):
                    self.admit_stopped_cell(7260, outcome='not_started', **evidence)
