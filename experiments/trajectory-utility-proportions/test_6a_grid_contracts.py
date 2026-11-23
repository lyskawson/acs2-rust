import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from proportions import MEAN, MEASURES, RANDOM, S1, read
import sys

sys.path.insert(0, str(MEASURES))
import grid_analysis
import grid_measure
import grid_run
from registration import TRIAL
sys.path.remove(str(MEASURES))


class PriorAnalysisContracts(unittest.TestCase):
    def test_odd_group_requires_ceiling_of_half(self):
        self.assertEqual(grid_analysis.verdict(1, 0, 3), 'no clear effect')
        self.assertEqual(grid_analysis.verdict(0, 1, 3), 'no clear effect')
        self.assertEqual(grid_analysis.verdict(2, 0, 3), 'improves')
        self.assertEqual(grid_analysis.verdict(0, 2, 3), 'worsens')

    def calculate(self, aliasing_trial):
        groups = dict(plain='no_aliasing', aliased='aliasing')
        targets = [1000, 3000, 10000, 30000]
        seeds = range(42, 62)
        study = dict(seeds=seeds, analysis=dict(
            critical_value=2.093024054408263, block='test', contrasts=[(TRIAL, MEAN), (MEAN, RANDOM)]))
        configs = {task: dict(group=group, targets=targets, registration={'blocks': {'test': {
            'random_floor': 0.1, 'random_floor_exact': '1/10', 'reachable_within_cap': 1.0
        }}}) for task, group in groups.items()}
        inputs = {}
        for seed in seeds:
            noise = (seed - 42) * 0.0002
            for task in groups:
                inputs[task, S1, seed] = [0.2, 0.4, 0.45, 0.8]
                inputs[task, RANDOM, seed] = [0.3, 0.55, 0.67, 0.98]
                inputs[task, MEAN, seed] = [0.4, 0.96, 0.81, 0.996]
                inputs[task, TRIAL, seed] = ([0.8 + noise, 0.97, 0.98, 0.998] if task == 'plain'
                                            else [aliasing_trial + noise, 0.79, 0.85, 0.996])
        return grid_analysis.calculate(study, configs, groups, inputs, {task: False for task in groups})

    def test_carry_is_blocked_when_another_group_worsens(self):
        result, carry = self.calculate(0.1)
        grouped = [row['verdict'] for row in result['grouped']
                   if row['contrast'] == f'{TRIAL} - {MEAN}' and row['measure'] == 'primary'
                   and row['group'] != 'all']
        self.assertEqual(grouped, ['improves', 'worsens'])
        self.assertEqual(carry, MEAN)
        self.assertEqual(self.calculate(0.8)[1], TRIAL)

    def test_sustained_and_contrast_specific_ceiling(self):
        result, _ = self.calculate(0.1)
        rows = {(row['task'], row['arm']): row for row in result['sustained-095']}
        self.assertEqual(rows['plain', MEAN]['sustained_095'], 30000)
        self.assertEqual(rows['plain', TRIAL]['sustained_095'], 3000)
        contrasts = {row['contrast']: row for row in result['comparisons']
                     if row['task'] == 'plain' and row['measure'] == 'final'}
        self.assertTrue(contrasts[f'{TRIAL} - {MEAN}']['ceiling'])
        self.assertFalse(contrasts[f'{MEAN} - {RANDOM}']['ceiling'])
        self.assertTrue(contrasts[f'{MEAN} - {RANDOM}']['counted'])


class PriorGridContracts(unittest.TestCase):
    def test_worker_uses_retry_deadline_when_allocation_can_run_longer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = dict(task='maze', seed=42, agent=MEAN, per_pair=True, attempt_id='retry-a02',
                       relative_directory='attempts/retry/measurement/output', limit_seconds=360, pilot=None)
            batch = dict(phase='retry', jobs=[dict(runs=[run])])
            config = dict(id='maze')
            plan = dict(configs={'maze': config}, preset={'epsilon': 0.8})
            path = grid_run.assignment_name(0, root)
            path.write_text(json.dumps(dict(index=0, grid_plan_sha256='pinned', batch=batch)))
            entry = dict(assignment_sha256=grid_run.checksum(path), phase='retry', limit_seconds=1200)
            allocation = dict(TimeLimit='00:20:00', RunTime='00:02:00', Partition='lem-cpu-normal',
                              NumCPUs='1')
            env = dict(SLURM_JOB_ID='900', SLURM_ARRAY_JOB_ID='890', SLURM_ARRAY_TASK_ID='0',
                       SLURM_MEM_PER_NODE=str(grid_run.MEMORY_MIB))
            with (
                patch.dict(grid_run.os.environ, env),
                patch.object(grid_run, 'current_plan', return_value=('pinned', plan)),
                patch.object(grid_run, 'scheduled_batches', return_value=[batch]),
                patch.object(grid_run, 'ledger_for_assignment', return_value=entry),
                patch.object(grid_run, 'allocation', return_value=('allocation', allocation)),
                patch.object(grid_run, 'gate_result', return_value=None),
                patch.object(grid_run.subprocess, 'run') as command,
            ):
                grid_run.worker(path, root)
            manifest = read(root / 'attempts/retry/manifest.json')
            self.assertEqual(manifest['process_timeout_seconds'], 330)
            self.assertEqual(manifest['allocation_limit_seconds'], 360)
            self.assertIn('--time=6', command.call_args.args[0])

    def test_retry_comparison_and_seal_require_the_preserved_checkpoint_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'attempts/retry/measurement/output'
            output.mkdir(parents=True)
            previous = dict(checkpoints=3, rows_sha256='source')
            run = dict(previous=previous, relative_directory=str(output.relative_to(root)), seed=42, task='maze')
            with patch.object(grid_measure, 'compare_prefix', return_value=dict(rows=2, differences=0)):
                with self.assertRaisesRegex(ValueError, 'reproduce all earlier checkpoints'):
                    grid_measure.compare_retry_prefix(root, run, {}, output)
                self.assertFalse((output / 'retry-prefix-identity.json').exists())
            with patch.object(grid_measure, 'compare_prefix', return_value=dict(rows=3, differences=0)):
                grid_measure.compare_retry_prefix(root, run, {}, output)
            proof = output / 'retry-prefix-identity.json'
            self.assertEqual(read(proof)['rows'], 3)
            proof.write_text(json.dumps(dict(rows=2, differences=0, previous_sha256='source')))
            plan = dict(reused=[], runs=[run], configs={'maze': {}}, earlier_grid_cpu_seconds=300,
                        earlier_completed_runs=2)
            ledger = [dict(run_count=1, name='retry', phase='retry')]
            (root / 'ledger.json').write_text(json.dumps(ledger))
            grid_run.assignment_name(0, root).write_text(json.dumps(dict(batch=dict(jobs=[dict(runs=[run])]))))
            account = dict(attempts=[dict(job_slot='retry:0', state='COMPLETED')], reserved_seconds=0,
                           spent_seconds=60, states={'COMPLETED': 1})
            with (
                patch.object(grid_run, 'RETRY', root),
                patch.object(grid_run, 'current_plan', return_value=('pinned', plan)),
                patch.object(grid_run, 'gate_result', return_value=None),
                patch.object(grid_run, 'accounting', return_value=('accounting', account)),
                patch.object(grid_run, 'checked_result'),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(ValueError, 'retry prefix differs'):
                    grid_run.status(seal=True, root=root)
                proof.write_text(json.dumps(dict(rows=3, differences=0, previous_sha256='source')))
                grid_run.status(seal=True, root=root)
            self.assertTrue((root / 'files.sha256.json').is_file())

    def test_grid_selects_a_retry_only_for_its_own_unfinished_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            grid, retry = root / 'first', root / 'retry'
            grid.mkdir()
            retry.mkdir()
            inputs = grid / 'grid-inputs.json'
            inputs.write_text('{}')
            original = [dict(run_id='done', phase='ordinary', relative_directory='first/done'),
                        dict(run_id='open', phase='scout', relative_directory='first/open')]
            retried = dict(run_id='open', phase='retry', relative_directory='retry/open')
            manifest = {'grid-inputs.json': grid_run.checksum(inputs)}
            second = {'retry/open/rows.jsonl': 'retried-sha'}
            (retry / 'files.sha256.json').write_text(json.dumps(second))
            summary = dict(earlier_completed_runs=1, grid_cpu_seconds=110, retry_cpu_seconds=40,
                           total_charged_cpu_seconds=120)
            with (
                patch.object(grid_analysis, 'stopped_evidence', return_value=(
                    manifest, dict(grid_cpu_seconds=70, total_cpu_seconds=80))),
                patch.object(grid_analysis, 'derive', return_value=dict(runs=original)),
                patch.object(grid_analysis, 'retry_plan', return_value=dict(runs=[retried])),
                patch.object(grid_analysis, 'completed_accounting', return_value=summary),
                patch.object(grid_analysis, 'step_records', return_value={}),
            ):
                selected = grid_analysis.grid_sources(grid, grid_analysis.STOPPED_MANIFEST_SHA256,
                                                      retry, grid_run.checksum(retry / 'files.sha256.json'))[2]
            self.assertEqual(selected, [original[0], retried])
