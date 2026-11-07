import copy
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from protocol import ROOT, NEW_ARMS, configurations, pilot, analysis_registration
from execution import authorization_manifest, require_authorization, registered_runs, scope
from operations import submit_pilot
from schema import validate_row
from grid_protocol import atomic_json
from archive import deduplicate


class ExecutionTests(unittest.TestCase):
    def fixture(self, temporary):
        root = Path(temporary) / 'study'
        root.mkdir()
        repository = Path(temporary) / 'repo'
        binary = repository / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'static test binary')
        oracle = repository / 'fixtures/goal_measurement_regression.json'
        oracle.parent.mkdir()
        oracle.write_bytes((ROOT / 'fixtures/goal_measurement_regression.json').read_bytes())
        identity = dict(measurement_commit='a' * 40, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            preset=json.loads(oracle.read_text())['configurations'][0]['rows'][0]['preset'])
        for name in ('reference-index', 'model-index'):
            atomic_json(root / (name + '.json'), [])
        registration = dict(manifest=pilot(identity), destination=str(root),
            **{name.replace('-', '_') + '_sha256': hashlib.sha256((root / (name + '.json')).read_bytes()).hexdigest()
               for name in ('reference-index', 'model-index')})
        atomic_json(root / 'pilot-registration.json', registration)
        digest = hashlib.sha256((root / 'pilot-registration.json').read_bytes()).hexdigest()
        atomic_json(root / 'pilot-scope.json', scope(digest, root))
        operation_file = root / 'operations/experiments/per-goal-reward/job.sh'
        operation_file.parent.mkdir(parents=True)
        operation_file.write_text('test operations\n')
        operations = root / 'operations/manifest.json'
        atomic_json(operations, dict(commit='b' * 40, files={'experiments/per-goal-reward/job.sh':
            hashlib.sha256(operation_file.read_bytes()).hexdigest()}))
        manifest = dict(pilot(identity), authorization_plan=str(root / 'pilot-registration.json'),
            authorization_sha256=digest, budget_scope=str(root / 'pilot-scope.json'),
            budget_scope_sha256=hashlib.sha256((root / 'pilot-scope.json').read_bytes()).hexdigest(),
            stop_path=str(root / 'STOP.json'), operations_commit='b' * 40, operations_manifest=str(operations),
            operations_manifest_sha256=hashlib.sha256(operations.read_bytes()).hexdigest(), job_name='tu-per-goal-reward')
        atomic_json(root / 'pilot-manifest.json', manifest)
        return root, repository, identity, manifest

    def executor(self, identity, calls, grant='1000', queue=''):
        def execute(command):
            calls.append(command)
            if command[0] == '/usr/bin/git':
                return identity['measurement_commit'] if 'rev-parse' in command else ''
            if command[0] == 'sshare': return grant
            if command[0] == 'squeue': return queue
            if command[0] == 'sbatch': return 'valid scheduler preview' if '--test-only' in command else '12345'
            raise AssertionError(command)
        return execute

    def test_manifest_derivation_fixes_both_arms_all_protocol_fields_seed42_and_four_hour_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, identity, manifest = self.fixture(temporary)
            with patch('execution.identity', return_value=identity):
                require_authorization(root / 'pilot', manifest)
                for field, value in [('agents', [NEW_ARMS[0]]), ('seeds', [43]), ('phase_cpu_hour_limit', 5),
                        ('configurations', list(reversed(manifest['configurations']))), ('extra', True)]:
                    changed = dict(copy.deepcopy(manifest), **{field: value})
                    with self.assertRaises(ValueError): require_authorization(root / 'pilot', changed)
                for field, value in [('cap', 99), ('targets', [1]), ('budget_steps', 1), ('per_start_diagnostics', False),
                                     ('evaluation_blocks', ['test']), ('agent_settings', {'x': {}})]:
                    changed = copy.deepcopy(manifest)
                    changed['configurations'][0][field] = value
                    with self.assertRaises(ValueError): require_authorization(root / 'pilot', changed)
                for stage in ['approved_grid', 'grid_pending_approval', 'unknown']:
                    with self.assertRaises(ValueError): authorization_manifest(dict(manifest, stage=stage))
                with self.assertRaises(ValueError): require_authorization(root / 'different', manifest)

    def test_submission_derives_twelve_jobs_checks_grant_and_preview_and_records_full_reservations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, repository, identity, manifest = self.fixture(temporary)
            calls = []
            with patch('execution.identity', return_value=identity):
                result = submit_pilot(root / 'pilot', root / 'pilot-manifest.json', repository,
                                      self.executor(identity, calls))
                self.assertEqual(result, dict(array_id=12345, runs=12, bound_cpu_hours=4))
                ledger = json.loads((root / 'pilot/ledger.json').read_text())
                self.assertEqual(ledger[0]['runs'], registered_runs(manifest))
                self.assertEqual(sum(run['limit_seconds'] for run in ledger[0]['runs']), 14400)
                self.assertEqual({run['seed'] for run in ledger[0]['runs']}, {42})
                submits = [command for command in calls if command[0] == 'sbatch']
                self.assertEqual(len(submits), 2)
                self.assertIn('--test-only', submits[0])
                self.assertNotIn('--test-only', submits[1])
                self.assertIn('--array=0-11%20', submits[1])
                self.assertEqual(Path(submits[1][-4]).name, 'job.sh')
                self.assertTrue(any(command[0] == 'sshare' for command in calls))
                self.assertTrue(any(command[0] == 'squeue' and '--array' in command for command in calls))
                with self.assertRaises(ValueError): submit_pilot(root / 'pilot', root / 'pilot-manifest.json', repository,
                                                                self.executor(identity, calls))

    def test_grant_includes_expanded_pending_reservations_and_failed_admission_submits_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, repository, identity, _ = self.fixture(temporary)
            calls = []
            queue = '10_0|other|PD|20:00:00|1\n10_1|other|PD|20:00:00|1\n'
            grant = str(15000 * 3600 - 14400 - 100000)
            with patch('execution.identity', return_value=identity):
                with self.assertRaises(ValueError):
                    submit_pilot(root / 'pilot', root / 'pilot-manifest.json', repository,
                                 self.executor(identity, calls, grant, queue))
            self.assertFalse(any(command[0] == 'sbatch' for command in calls))

    def test_scheduler_preview_cannot_change_frozen_inputs_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, repository, identity, _ = self.fixture(temporary)
            calls = []
            execute = self.executor(identity, calls)
            def mutate(command):
                value = execute(command)
                if '--test-only' in command:
                    (root / 'reference-index.json').write_text('[1]\n')
                return value
            with patch('execution.identity', return_value=identity):
                with self.assertRaises(ValueError): submit_pilot(root / 'pilot', root / 'pilot-manifest.json', repository, mutate)
            self.assertEqual(sum(command[0] == 'sbatch' and '--test-only' not in command for command in calls), 0)

    def test_collector_adapter_installs_the_study_validator_without_changing_shared_files(self):
        import collect
        original = collect.validate_row
        with patch.object(collect, 'validate_row', original), patch.object(collect, 'main') as main:
            runpy.run_path(str(Path(__file__).with_name('collector.py')), run_name='__main__')
            main.assert_called_once_with()
            self.assertIs(collect.validate_row, validate_row)
        self.assertIs(collect.validate_row, original)

    def test_accounting_deduplicates_identical_job_id_raw_and_refuses_conflicts(self):
        row = dict(JobIDRaw='123', CPUTimeRAW='10')
        self.assertEqual(deduplicate([row, dict(row)]), [row])
        with self.assertRaises(ValueError): deduplicate([row, dict(row, CPUTimeRAW='11')])

    def test_registration_includes_exact_contrasts_metrics_and_separate_verdicts(self):
        registered = analysis_registration()
        self.assertEqual(registered['success']['comparisons'], ['S+1 - S', 'E+1 - E', 'interaction', 'S+1 - G'])
        self.assertEqual(registered['success']['block'], 'test')
        self.assertEqual(registered['mechanism']['metrics'][-1], 'within_start_sd_estimate')
        self.assertEqual(registered['verdicts']['measures'], ['primary', 'final'])
        self.assertEqual(len(registered['verdicts']['mazes']), 6)
        self.assertEqual(configurations()[0]['targets'][0], 1000)
