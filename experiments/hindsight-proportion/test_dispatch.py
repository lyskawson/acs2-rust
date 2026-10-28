import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from approved_execution import APPROVED_SHA256, approved
from grid_dispatch import dispatch, pending_assignment, recovery_arrays, require_prefix, require_stream_slot, verify_completed
from grid_scope import snapshot
from protocol import ROOT


class DispatchTests(unittest.TestCase):
    def test_approval_requires_exact_plan_and_forty_five_hour_confirmation(self):
        registration = Path(__file__).with_name('plan.json')
        self.assertEqual(hashlib.sha256(registration.read_bytes()).hexdigest(), APPROVED_SHA256)
        expected = json.loads(Path(__file__).with_name('approval.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'approved-plan.json').write_bytes(registration.read_bytes())
            for field, value in [('phase_cpu_hour_limit', 46), ('pilot_allocation_cpu_seconds', 0), ('approved', False), ('plan_sha256', 'a' * 64)]:
                confirmation = dict(expected, **{field: value})
                (root / 'approval.json').write_text(json.dumps(confirmation))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    approved(root / 'approved-plan.json')
            (root / 'approval.json').write_text(json.dumps(expected))
            self.assertEqual(approved(root / 'approved-plan.json')['candidates'], [1 / 3, 0.5, 2 / 3, 0.8])

    def test_two_stream_guard_counts_reservations_and_blocks_own_pending_array(self):
        own = str(Path('/study/tuning').resolve())
        other = str(Path('/study/control').resolve())
        third = str(Path('/study/test-maze4').resolve())
        state = dict(batch_states={other: dict(reserved_seconds=1), own: dict(reserved_seconds=0)})
        require_stream_slot(own, state)
        state['batch_states'][own]['reserved_seconds'] = 1
        with self.assertRaisesRegex(ValueError, 'previous array'):
            require_stream_slot(own, state)
        with self.assertRaisesRegex(ValueError, 'two streams'):
            require_stream_slot(third, state)

    def test_prefix_guard_refuses_wrong_seed_proportion_config_or_hard_limit(self):
        config = dict(id='maze_p1of2', limits_minutes={'acs2her': 4})
        plan = dict(stage='approved_grid', configurations=[config], agents=['acs2her'], seeds=list(range(42, 62)), batch_size=20)
        ledger = [dict(runs=[dict(configuration_id=config['id'], agent='acs2her', seed=seed, limit_seconds=240) for seed in plan['seeds']])]
        require_prefix(Path('/study/control'), plan, ledger)
        for field, value in [('seed', 62), ('configuration_id', 'maze_p2of3'), ('limit_seconds', 300)]:
            altered = copy.deepcopy(ledger)
            altered[0]['runs'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_prefix(Path('/study/control'), plan, altered)

    def dispatch_fixture(self, root):
        repository = root / 'source'
        binary = repository / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'frozen binary')
        stream = root / 'tuning'
        stream.mkdir()
        approval = root / 'approval.json'
        approval.write_text('{}')
        (stream / 'approval.json').write_bytes(approval.read_bytes())
        plan = dict(stage='approved_grid', experiment='hindsight-proportion',
            configurations=[dict(id=name, limits_minutes={'acs2her': 3}) for name in ('maze4_p1of2', 'maze5_p1of2')],
            agents=['acs2her'], seeds=list(range(42, 62)), batch_size=20, concurrency=20,
            measurement_commit='a' * 40, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            phase_cpu_hour_limit=45, pilot_allocation_cpu_seconds=0, memory_mib=512, operations_commit='b' * 40,
            operations_manifest=str(root / 'frozen-operations/manifest.json'), budget_scope=str(root / 'scope.json'), stop_path=str(root / 'STOP.json'), job_name='test-dispatch')
        data = (json.dumps(plan, indent=2) + '\n').encode()
        (stream / 'plan.json').write_bytes(data)
        (stream / 'plan.sha256').write_text(hashlib.sha256(data).hexdigest() + '\n')
        (stream / 'ledger.json').write_text('[]')
        manifest = root / 'manifest.json'
        manifest.write_bytes(data)
        state = dict(spent_seconds=4784, reserved_seconds=0, batch_states={str(stream.resolve()): dict(reserved_seconds=0)})
        return repository, stream, approval, manifest, state

    def test_dispatch_submits_one_array_and_checks_test_only_before_allocating(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository, stream, approval, manifest, state = self.dispatch_fixture(Path(temporary))
            commands = []
            def executor(command):
                commands.append(command)
                if command[0] == 'git':
                    return 'a' * 40 if 'rev-parse' in command else ''
                if command[0] == 'sshare':
                    return '0'
                if command[0] == 'sbatch':
                    return 'test-only passed' if '--test-only' in command else '1234'
                return ''
            with patch('grid_dispatch.require_authorization'), patch('grid_dispatch.require_batch_root'), patch('grid_dispatch.verify_operations'), patch('grid_dispatch.scoped_state', return_value=state):
                result = dispatch(stream, manifest, repository, approval, executor)
            self.assertEqual(result['array_id'], 1234)
            self.assertEqual(result['runs'], 20)
            actual = [command for command in commands if command[0] == 'sbatch' and '--test-only' not in command]
            tested = [command for command in commands if command[0] == 'sbatch' and '--test-only' in command]
            self.assertEqual(len(actual), 1)
            self.assertEqual(len(tested), 1)
            self.assertLess(commands.index(tested[0]), commands.index(actual[0]))
            self.assertEqual(len(json.loads((stream / 'ledger.json').read_text())), 1)
            self.assertFalse((stream / 'assignment-01.json').exists())
            state['batch_states'][str(stream.resolve())]['reserved_seconds'] = 3600
            commands.clear()
            with patch('grid_dispatch.require_authorization'), patch('grid_dispatch.require_batch_root'), patch('grid_dispatch.verify_operations'), patch('grid_dispatch.scoped_state', return_value=state):
                with self.assertRaisesRegex(ValueError, 'previous array'):
                    dispatch(stream, manifest, repository, approval, executor)
            self.assertFalse(any(command[0] == 'sbatch' for command in commands))

    def test_budget_cap_is_enforced_even_if_manifest_authorization_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository, stream, approval, manifest, state = self.dispatch_fixture(Path(temporary))
            state['spent_seconds'] = 45 * 3600 - 10
            commands = []
            def executor(command):
                commands.append(command)
                return 'a' * 40 if 'rev-parse' in command else ''
            with patch('grid_dispatch.require_authorization'), patch('grid_dispatch.require_batch_root'), patch('grid_dispatch.verify_operations'), patch('grid_dispatch.scoped_state', return_value=state):
                with self.assertRaisesRegex(ValueError, 'budget would be exceeded'):
                    dispatch(stream, manifest, repository, approval, executor)
            self.assertFalse(any(command[0] in ('sbatch', 'sshare') for command in commands))
            self.assertEqual(json.loads((stream / 'ledger.json').read_text()), [])

    def test_retry_keeps_logical_array_order_without_skipping_the_next_assignment(self):
        configs = [dict(id=name, limits_minutes={'acs2her': 3}) for name in ('first_p1of2', 'second_p1of2')]
        plan = dict(stage='approved_grid', configurations=configs, agents=['acs2her'], seeds=list(range(42, 62)), batch_size=20)
        def runs(config):
            return [dict(configuration_id=config['id'], agent='acs2her', seed=seed, limit_seconds=180) for seed in plan['seeds']]
        ledger = [dict(array_id=1, runs=runs(configs[0])), dict(array_id=2, runs=runs(configs[1])),
                  dict(array_id=3, registered_assignment=1, retry_of=2, runs=runs(configs[1]))]
        batches = require_prefix(Path('/study/control'), plan, ledger)
        completed = {('acs2her', seed): {} for seed in plan['seeds']}
        def selected(root, ledger, config):
            return (completed if config['id'] == configs[0]['id'] else {}), []
        with patch('grid_dispatch.select_attempts', side_effect=selected):
            self.assertEqual(pending_assignment(Path('/study/control'), plan, ledger, batches), 1)
        with patch('grid_dispatch.select_attempts', return_value=(completed, [])):
            self.assertEqual(pending_assignment(Path('/study/control'), plan, ledger, batches), 2)
        ledger[-1]['retry_of'] = 99
        with self.assertRaises(ValueError):
            require_prefix(Path('/study/control'), plan, ledger)

    def test_recovery_keeps_scientific_identity_and_requires_preserved_stop_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'stop.json').write_text('controller socket timeout')
            recovery = dict(plan_sha256='a' * 64, binary_sha256='b' * 64, phase_cpu_hour_limit=45,
                resolved=True, archived_stop='stop.json', stop_sha256=hashlib.sha256((root / 'stop.json').read_bytes()).hexdigest(), failed_arrays=[7, 8])
            path = root / 'recovery.json'
            path.write_text(json.dumps(recovery))
            plan = dict(authorization_sha256='a' * 64, binary_sha256='b' * 64, phase_cpu_hour_limit=45)
            self.assertEqual(recovery_arrays(path, plan), {7, 8})
            for field, value in [('phase_cpu_hour_limit', 46), ('resolved', False), ('binary_sha256', 'c' * 64), ('stop_sha256', 'd' * 64)]:
                path.write_text(json.dumps(dict(recovery, **{field: value})))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    recovery_arrays(path, plan)

    def test_retried_complete_run_reproduces_every_partial_row_in_binary64_bits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / 'failed', root / 'complete'
            old.mkdir()
            new.mkdir()
            (old / 'completion.json').write_text(json.dumps(dict(sha256={})))
            (new / 'identity-checks.json').write_text('[]')
            config = dict(id='maze', targets=[1000, 3000], evaluation_blocks=['validation'])
            plan = dict(configurations=[config], measurement_commit='a' * 40)
            attempts = [dict(path=old, array_id=1, configuration_id='maze', agent='acs2her', seed=42),
                        dict(path=new, array_id=2, configuration_id='maze', agent='acs2her', seed=42)]
            rows = [dict(seed=42, nominal_step=point, knowledge=0.9, evaluation_blocks=dict(validation=dict(success=0.5))) for point in config['targets']]
            partial = copy.deepcopy(rows[:1])
            def inspect(attempt, config):
                return (dict(status='incomplete'), partial) if attempt['array_id'] == 1 else (dict(status='complete'), rows)
            with patch('grid_dispatch.attempts', return_value=attempts), patch('grid_dispatch.inspect_attempt', side_effect=inspect), patch('grid_dispatch.validate_row'), patch('grid_dispatch.verify_references', return_value=[]):
                verify_completed(root, plan, [], {1})
                partial[0]['knowledge'] = 0.9000000000000001
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    verify_completed(root, plan, [], {1})

    def test_accounting_keeps_pilot_bytes_and_charges_every_scope_root_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot, tuning = root / 'pilot', root / 'tuning'
            for directory, array in [(pilot, 1), (tuning, 2)]:
                directory.mkdir()
                (directory / 'ledger.json').write_text(json.dumps([dict(array_id=array, runs=[dict(limit_seconds=600)])]))
            frozen = b'original terminal pilot accounting\n'
            (pilot / 'accounting.psv').write_bytes(frozen)
            scope = dict(authorization_sha256='a' * 64, pilot_allocation_cpu_seconds=0,
                         accounting_since='2026-10-28', batch_roots=[str(pilot), str(tuning)])
            path = root / 'scope.json'
            path.write_text(json.dumps(scope))
            plan = dict(budget_scope=str(path), budget_scope_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        authorization_sha256='a' * 64, pilot_allocation_cpu_seconds=0)
            records = 'JobID|JobIDRaw|CPUTimeRAW|State|\n1_0|101|100|COMPLETED|\n2_0|102|30|RUNNING|\n'
            state = snapshot(plan, lambda command: records, preserve_roots=[pilot])
            self.assertEqual((pilot / 'accounting.psv').read_bytes(), frozen)
            self.assertEqual((tuning / 'accounting.psv').read_text(), records)
            self.assertEqual((root / 'shared-accounting.psv').read_text(), records)
            self.assertEqual(state['spent_seconds'], 130)
            self.assertEqual(state['reserved_seconds'], 570)


if __name__ == '__main__':
    unittest.main()
