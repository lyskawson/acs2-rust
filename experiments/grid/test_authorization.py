import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grid_authorization import derive_manifest
from grid_protocol import atomic_json
from submit import prepare, require_authorization, submit

ROOT = Path(__file__).resolve().parents[2]


def tuning_fixture(base):
    registration = base / 'approved-plan.json'
    registration.write_bytes((ROOT / 'experiments/her-variants/plan.json').read_bytes())
    digest = hashlib.sha256(registration.read_bytes()).hexdigest()
    atomic_json(base / 'approval.json', dict(approved=True, plan_sha256=digest))
    manifest = derive_manifest(dict(experiment='her-variants', stage='tuning_pending_approval'), registration)
    manifest.update(authorization_plan=str(registration), authorization_sha256=digest)
    scope = base / 'scope.json'
    atomic_json(scope, dict(batch_roots=[str(base / 'tuning')],
        authorization_sha256=digest, pilot_allocation_cpu_seconds=manifest['pilot_allocation_cpu_seconds'],
        accounting_since='2026-10-12'))
    manifest.update(budget_scope=str(scope), budget_scope_sha256=hashlib.sha256(scope.read_bytes()).hexdigest(),
                    stop_path=str(base / 'STOP.json'))
    return manifest


def changed_manifests(plan):
    changes = [(['configurations'], plan['configurations'][:-1]), (['agents'], ['her_values']),
               (['seeds'], [42]), (['phase_cpu_hour_limit'], 59), (['concurrency'], 40),
               (['batch_by'], 'configuration'), (['batch_size'], 20),
               (['configurations', 0, 'task'], 'maze5'), (['configurations', 0, 'cap'], 99),
               (['configurations', 0, 'budget_steps'], 1), (['configurations', 0, 'targets'], [1000]),
               (['configurations', 0, 'limits_minutes', 'her_values'], 99),
               (['configurations', 0, 'agent_settings', 'her_values', 'relabeled_proportion'], 0.9),
               (['configurations', 0, 'evaluation_blocks'], ['test'])]
    for path, value in changes:
        changed = copy.deepcopy(plan)
        container = changed
        for key in path[:-1]:
            container = container[key]
        container[path[-1]] = value
        yield path, changed


class AuthorizationTests(unittest.TestCase):
    def test_submission_rederives_every_run_before_any_scheduler_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan = tuning_fixture(base)
            require_authorization(base / 'tuning', plan, 'child', base / 'approval.json')
            manifest = base / 'manifest.json'
            for path, changed in changed_manifests(plan):
                with self.subTest(field=path):
                    atomic_json(manifest, changed)
                    with patch('submit.execute') as execute:
                        with self.assertRaises(ValueError):
                            submit(base / 'tuning', manifest, base / 'clone', base / 'approval.json')
                        execute.assert_not_called()
                    self.assertFalse((base / 'tuning').exists())

    def test_registered_cost_pilots_need_no_separate_approval_and_refuse_seed42_impostors(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for experiment, name in [('her-variants', 'pilot.json'), ('maze-baselines', 'pilot.json'),
                                     ('maze-baselines', 'pair-pilot.json')]:
                plan = json.loads((ROOT / 'experiments' / experiment / name).read_text())
                require_authorization(base, plan, 'resolved-pilot')
                for field, value in [('agents', ['her_values']), ('seeds', [43]),
                                     ('configurations', plan['configurations'][:-1]), ('phase_cpu_hour_limit', 1)]:
                    with self.subTest(experiment=experiment, pilot=name, field=field):
                        with self.assertRaises(ValueError):
                            require_authorization(base, dict(plan, **{field: value}), 'resolved-pilot')
                for field, value in [('targets', [1000]), ('budget_steps', 1), ('limits_minutes', {})]:
                    changed = copy.deepcopy(plan)
                    changed['configurations'][0][field] = value
                    with self.subTest(experiment=experiment, pilot=name, field=field):
                        with self.assertRaises(ValueError):
                            require_authorization(base, changed, 'resolved-pilot')

    def test_prepare_also_rederives_after_resolving_measurement_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            binary = base / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'binary')
            (base / 'fixtures').mkdir()
            atomic_json(base / 'fixtures/goal_measurement_regression.json',
                        dict(configurations=[dict(rows=[dict(preset={})])]))
            plan = json.loads((ROOT / 'experiments/her-variants/pilot.json').read_text())
            plan['configurations'][0]['targets'] = [1000]
            manifest = base / 'manifest.json'
            atomic_json(manifest, plan)
            with patch('submit.execute', side_effect=['frozen', '']):
                with self.assertRaises(ValueError):
                    prepare(base / 'pilot', manifest, base)
            self.assertFalse((base / 'pilot').exists())

    def test_grant_reserves_every_pending_array_element_before_test_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            binary = base / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'binary')
            (base / 'fixtures').mkdir()
            atomic_json(base / 'fixtures/goal_measurement_regression.json',
                        dict(configurations=[dict(rows=[dict(preset={})])]))
            manifest = base / 'manifest.json'
            manifest.write_bytes((ROOT / 'experiments/her-variants/pilot.json').read_bytes())
            commands = []
            def execute(command):
                commands.append(command)
                if command[0] == 'git':
                    return 'frozen' if command[-1] == 'HEAD' else ''
                if command[0] == 'sshare':
                    return str(15000 * 3600 - 100000)
                if command[0] == 'squeue':
                    if '--array' in command:
                        return ''.join(f'7_{i}|other|PD|01:00:00|1\n' for i in range(40))
                    return '7_[0-39]|other|PD|01:00:00|1\n'
                if command[0] == 'sbatch':
                    return 'accepted' if '--test-only' in command else '99'
                raise AssertionError(command)
            with patch('submit.execute', side_effect=execute):
                with self.assertRaisesRegex(ValueError, 'shared grant'):
                    submit(base / 'pilot', manifest, base)
            self.assertFalse(any(command[0] == 'sbatch' for command in commands))


if __name__ == '__main__':
    unittest.main()
