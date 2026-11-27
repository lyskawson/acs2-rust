import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import grid_results
from proportions import MEAN, checksum


class GridSourceTests(unittest.TestCase):
    def seal_file(self, root, manifest, name, contents):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contents) + '\n')
        manifest[name] = checksum(path)

    def seal_measurement(self, root, manifest, run):
        attempt = f"attempts/{run['run_id']}"
        folder = f"{attempt}/measurement/small_{run['agent']}_s{run['seed']}"
        rows = f'{folder}/rows.jsonl'
        self.seal_file(root, manifest, rows, {'nominal_step': 1000})
        self.seal_file(root, manifest, f'{folder}/completion.json',
                       dict(rows_seen=1, sha256={'rows.jsonl': manifest[rows]}))
        self.seal_file(root, manifest, f'{attempt}/step.json',
                       dict(job_id='50_0', job_id_raw='51', step_id='2', limit_seconds=600))
        return folder

    def measurement_step(self):
        records = {'51.2': dict(JobID='50_0.2', CPUTimeRAW='330', AllocCPUS='1', ExitCode='0:0', State='COMPLETED')}
        return records, {'50_0': '51'}

    def test_one_slurm_step_cannot_back_two_runs(self):
        first = dict(run_id='first', agent=MEAN, seed=43, pilot=None)
        second = dict(run_id='second', agent=MEAN, seed=44, pilot=None)
        manifest, used_steps = {}, set()
        config, steps = dict(id='small'), self.measurement_step()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.seal_measurement(root, manifest, first)
            self.seal_measurement(root, manifest, second)
            _, _, _, charge, _ = grid_results.complete_source(first, config, root, manifest, steps, used_steps)
            self.assertEqual(charge, 330)
            with self.assertRaisesRegex(ValueError, 'not a distinct registered Slurm charge'):
                grid_results.complete_source(second, config, root, manifest, steps, used_steps)

    def test_recovery_proof_requires_previous_sha256(self):
        reference = dict(checkpoints=1, sha256='preserved-rows')
        run = dict(run_id='retried', agent=MEAN, seed=43, pilot=None, previous=reference)
        with TemporaryDirectory() as directory:
            root, manifest = Path(directory), {}
            folder = self.seal_measurement(root, manifest, run)
            name = f'{folder}/recovery-prefix-identity.json'
            proof = dict(rows=1, differences=0)
            self.seal_file(root, manifest, name, dict(proof, pilot_sha256=reference['sha256']))
            config, steps = dict(id='small'), self.measurement_step()
            with self.assertRaisesRegex(ValueError, 'preserved prefix proof does not match its source'):
                grid_results.complete_source(run, config, root, manifest, steps, set())
            self.seal_file(root, manifest, name, dict(proof, previous_sha256=reference['sha256']))
            _, _, _, _, reproduced = grid_results.complete_source(run, config, root, manifest, steps, set())
            self.assertEqual(reproduced, 1)

    def test_retried_cells_are_read_from_the_recovery_archive(self):
        pilot, grid, recovery = map(Path, ('pilot', 'grid', 'recovery'))
        grid_manifest, retry_manifest = {'archive': 'first-grid'}, {'archive': 'recovery-grid'}
        grid_steps, retry_steps = ('grid-slurm',), ('recovery-slurm',)
        completion = dict(rows_seen=8, sha256={'rows.jsonl': 'row-sha'}, user_cpu_seconds=24,
                          system_cpu_seconds=1, peak_rss_kib=100)
        pilot_manifest = {'pilot-summary.json': 'pilot-summary-sha'}
        sealed = {pilot / 'files.sha256.json': pilot_manifest, pilot / 'pilot-summary.json': {},
                  recovery / 'files.sha256.json': retry_manifest}
        reused = []
        for index in range(105):
            folder = f'pilot-{index}'
            name = f'{folder}/completion.json'
            sealed[pilot / name] = completion
            pilot_manifest[name] = 'pilot-completion-sha'
            reused.append(dict(task='small', arm='pilot-arm', seed=42 + index,
                               pilot=dict(path=f'{folder}/rows.jsonl', sha256='row-sha', slurm_cpu_seconds=80)))
        runs = [dict(run_id=f'run-{index}', phase='conditional', task='small', arm='new-arm', seed=42 + index)
                for index in range(3255)]
        retries = {run['run_id']: dict(run, attempt_id=f"{run['run_id']}_a02", phase='recovery', prefix_count=count)
                   for run, count in zip(runs[-77:], (7,)*7 + (6,)*13 + (0,)*57)}
        data = dict(reused=reused, runs=runs, configs={'small': dict(targets=range(8))})
        original = dict(grid_cpu_seconds=100000, total_cpu_seconds=153037)
        summary = dict(stopped=False, completed_runs=77, submitted_runs=77, reserved_seconds=0, reused=105,
                       earlier_completed_runs=3178, earlier_grid_cpu_seconds=100000, reproduced_checkpoints=127,
                       grid_cpu_seconds=50000, grid_total_pilot_and_spent_cpu_seconds=203037)
        recovery_files = {'accounting-summary.json': summary, 'operations.json': dict(plan_sha256=grid_results.FIRST_PLAN),
                          'operations/experiments/trajectory-utility-proportions/recovery-registration.json': {}}

        def read_cell(run, config, root, manifest, steps, used_steps):
            identifier = run['run_id']
            if identifier in retries:
                self.assertEqual(root, recovery, f'{identifier} must use the recovery archive')
                self.assertEqual((run, manifest, steps), (retries[identifier], retry_manifest, retry_steps))
                charge, reproduced = 720, run['prefix_count']
            else:
                self.assertEqual((root, manifest, steps), (grid, grid_manifest, grid_steps))
                charge, reproduced = 360, 0
            used_steps.add(identifier)
            return completion, root / identifier / 'rows.jsonl', 'row-sha', charge, reproduced

        with (patch.object(grid_results, 'load_plan', side_effect=lambda path, digest: sealed[path]),
              patch.object(grid_results, 'derive', return_value=data),
              patch.object(grid_results, 'sealed_stop', return_value=(grid_manifest, original, [])),
              patch.object(grid_results, 'sealed_reader', return_value=recovery_files.__getitem__),
              patch.object(grid_results, 'derive_recovery', return_value=dict(runs=list(retries.values()))),
              patch.object(grid_results, 'measurement_steps', side_effect=[retry_steps, grid_steps]),
              patch.object(grid_results, 'complete_source', side_effect=read_cell)):
            sources, _, _ = grid_results.measured_sources(
                pilot, grid, grid_results.FIRST_MANIFEST, {'pilot_manifest_sha256': 'pilot-sha'}, recovery, 'retry-sha')

        first = sources['small', 'new-arm', 42]
        retried = sources['small', 'new-arm', 42 + 3178]
        self.assertEqual((first['path'], first['source'], first['allocated']), (grid/'run-0/rows.jsonl', 'grid', 360))
        self.assertEqual((retried['path'], retried['source'], retried['allocated']),
                         (recovery/'run-3178/rows.jsonl', 'recovery', 720))
