import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from protocol import ROOT, NEW_ARMS
from execution_grid import (APPROVED_SHA256, approved, batch_name, checksum, manifest_for,
                            registered_runs, require_authorization, scope, validate_inputs)
from prepare_grid import prepare
from dispatch import manifest, submit
from grid_protocol import atomic_json


class GridTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / 'study'
        root.mkdir()
        for name in ('plan.json', 'approval.json'):
            (root / ('approved-plan.json' if name == 'plan.json' else name)).write_bytes(
                (ROOT / 'experiments/per-goal-reward' / name).read_bytes())
        plan, _ = approved(root / 'approved-plan.json')
        (root / 'continuation').mkdir()
        atomic_json(root / 'continuation/scope.json', scope(root, plan))
        operations = root / 'execution-operations'
        operations.mkdir()
        (operations / 'checked.txt').write_text('frozen operations')
        atomic_json(operations / 'manifest.json', dict(commit='c' * 40,
            files={'checked.txt': checksum(operations / 'checked.txt')}))
        repository = Path(directory) / 'repo'
        binary = repository / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
        binary.parent.mkdir(parents=True)
        binary.write_text('test binary')
        return root, repository, binary, plan

    def test_forecast_derivation_has_exactly_228_unique_replications_and_no_pilot(self):
        plan, _ = approved()
        keys, names = [], []
        for index, array in enumerate(plan['forecast']['arrays']):
            definition = manifest_for(plan, index)
            runs = registered_runs(definition)
            self.assertEqual(len(runs), array['runs'])
            self.assertEqual(sum(run['limit_seconds'] for run in runs), array['reservation_cpu_seconds'])
            self.assertEqual(definition['agents'], [array['agent']])
            self.assertEqual(definition['seeds'], array['seeds'])
            self.assertEqual(definition['phase_cpu_hour_limit'], 29)
            self.assertEqual({config.get('per_start_diagnostics', False) for config in definition['configurations']},
                             {array['kind'] == 'per_pair'})
            self.assertTrue(all(run['seed'] > 42 for run in runs))
            keys.extend((run['configuration_id'], run['agent'], run['seed']) for run in runs)
            names.append(batch_name(array))
        self.assertEqual(len(keys), 228)
        self.assertEqual(len(set(keys)), 228)
        self.assertEqual(len(set(names)), 8)
        for index in (-1, 8, True, '0'):
            with self.assertRaises(ValueError): manifest_for(plan, index)

    def test_authorization_refuses_seed_arm_block_limit_budget_and_extra_field_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _, _ = self.fixture(directory)
            with patch('execution_grid.validate_inputs'):
                definition = manifest(root, 0)
                batch = root / 'episodic-pairs'
                require_authorization(batch, definition)
                for field, value in [('seeds', [42]), ('agents', [NEW_ARMS[1]]), ('phase_cpu_hour_limit', 30),
                                     ('binary_sha256', '0' * 64), ('extra', 1)]:
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        require_authorization(batch, dict(copy.deepcopy(definition), **{field: value}))
                for field, value in [('targets', [1]), ('per_start_diagnostics', False), ('evaluation_blocks', ['test']),
                                     ('limits_minutes', {NEW_ARMS[0]: 1}), ('agent_settings', {'x': {}})]:
                    changed = copy.deepcopy(definition)
                    changed['configurations'][0][field] = value
                    with self.subTest(field=field), self.assertRaises(ValueError): require_authorization(batch, changed)
                with self.assertRaises(ValueError): require_authorization(root / 'pilot', definition)

    def test_cumulative_submission_counts_pilot_and_checks_grant_preview_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, repo, binary, plan = self.fixture(directory)
            calls = []
            def execute(command):
                calls.append(command)
                if command[0] == '/usr/bin/git': return plan['measurement_commit'] if 'rev-parse' in command else ''
                if command[0] == 'sshare': return '1000'
                if command[0] == 'squeue': return ''
                if command[0] == 'sbatch': return 'test passed' if '--test-only' in command else '12345'
                raise AssertionError(command)
            real_checksum = checksum
            def digest(path): return plan['binary_sha256'] if Path(path) == binary else real_checksum(path)
            state = dict(spent_seconds=1648, reserved_seconds=0, batch_states={})
            with patch('execution_grid.validate_inputs'), patch('dispatch.snapshot', return_value=state), \
                    patch('dispatch.checksum', side_effect=digest):
                result = submit(root, repo, 0, execute)
                self.assertEqual(result['runs'], 24)
                self.assertEqual(result['bound_cpu_hours'], (1648 + 24 * 8 * 60) / 3600)
                ledger = json.loads((root / 'episodic-pairs/ledger.json').read_text())
                self.assertEqual(len(ledger[0]['runs']), 24)
                commands = [command for command in calls if command[0] == 'sbatch']
                self.assertEqual(len(commands), 2)
                self.assertIn('--test-only', commands[0])
                self.assertNotIn('--test-only', commands[1])
                self.assertTrue(commands[1][-4].endswith('/job_grid.sh'))
                self.assertIn('--array=0-23%20', commands[1])
                with self.assertRaises(ValueError): submit(root, repo, 0, execute)

    def rejected_submission(self, state, reason, index=0):
        with tempfile.TemporaryDirectory() as directory:
            root, repo, binary, plan = self.fixture(directory)
            calls = []
            def execute(command):
                calls.append(command)
                if command[0] == '/usr/bin/git': return plan['measurement_commit'] if 'rev-parse' in command else ''
                if command[0] == 'sshare': return '1000\n'
                if command[0] == 'squeue': return '999|other-study|R|00:01:00|1\n'
                if command[0] == 'sbatch': return 'test passed' if '--test-only' in command else '12345'
                raise AssertionError(command)
            real_checksum = checksum
            digest = lambda path: plan['binary_sha256'] if Path(path) == binary else real_checksum(path)
            with patch('execution_grid.validate_inputs'), patch('dispatch.checksum', side_effect=digest), \
                    patch('dispatch.snapshot', return_value=state):
                try:
                    with self.assertRaises(ValueError) as error: submit(root, repo, index, execute)
                    self.assertEqual(str(error.exception), reason)
                finally:
                    self.assertEqual([command[0] for command in calls], ['/usr/bin/git', '/usr/bin/git'])
                    self.assertFalse((root / batch_name(plan['forecast']['arrays'][index])).exists())
                    self.assertFalse((root / 'continuation/admission-decisions.jsonl').exists())

    def test_budget_bound_rejects_before_grant_queue_or_scheduler_calls(self):
        self.rejected_submission(dict(spent_seconds=29 * 3600, reserved_seconds=0, batch_states={}),
            'phase budget would be exceeded: 32.200000 hours')

    def test_two_active_streams_reject_before_grant_queue_or_scheduler_calls(self):
        self.rejected_submission(dict(spent_seconds=1648, reserved_seconds=20,
            batch_states={'one': dict(reserved_seconds=10), 'two': dict(reserved_seconds=10)}),
            'two active streams already reserve forty running slots')

    def test_unfinished_previous_array_rejects_before_grant_queue_or_scheduler_calls(self):
        self.rejected_submission(dict(spent_seconds=1648, reserved_seconds=0, batch_states={}),
            'previous array of this arm must finish and validate first', index=1)

    def test_actual_preparation_output_is_authorizable_and_preserves_pilot_bytes(self):
        plan, _ = approved()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'study'
            root.mkdir()
            (root / 'registered-plan.json').write_bytes((ROOT / 'experiments/per-goal-reward/plan.json').read_bytes())
            (root / 'pilot').mkdir()
            (root / 'pilot/evidence.txt').write_text('immutable pilot')
            before = checksum(root / 'pilot/evidence.txt')
            def git(command, **kwargs):
                if 'status' in command: return ''
                if 'rev-parse' in command: return 'd' * 40
                if 'ls-files' in command: return 'experiments/per-goal-reward/execution_grid.py\n'
                raise AssertionError(command)
            with patch('prepare_grid.subprocess.check_output', side_effect=git), patch('prepare_grid.validate_inputs'), \
                    patch('execution_grid.validate_inputs'):
                result = prepare(root, root)
                self.assertFalse(result['measurement_rebuilt'])
                require_authorization(root / 'episodic-pairs', manifest(root, 0))
                self.assertEqual(checksum(root / 'pilot/evidence.txt'), before)
                preserved = json.loads((root / 'pilot-inventory.json').read_text())
                self.assertEqual(preserved['pilot/evidence.txt'], before)
                self.assertEqual(json.loads((root / 'continuation/scope.json').read_text())['immutable_batch_roots'], [str(root / 'pilot')])

    def test_registered_input_checksums_and_approval_bytes_are_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _, plan = self.fixture(directory)
            for name in ('pilot-summary', 'pilot-registration', 'reference-index', 'model-index'):
                (root / (name + '.json')).write_text(name)
            synthetic = dict(cost_pilot={name + '_sha256': checksum(root / (name + '.json'))
                for name in ('pilot-summary', 'pilot-registration', 'reference-index', 'model-index')})
            validate_inputs(root, synthetic)
            (root / 'model-index.json').write_text('changed')
            with self.assertRaises(ValueError): validate_inputs(root, synthetic)
            approval = json.loads((root / 'approval.json').read_text())
            approval['phase_cpu_hour_limit'] = 30
            atomic_json(root / 'approval.json', approval)
            with self.assertRaises(ValueError): approved(root / 'approved-plan.json')
