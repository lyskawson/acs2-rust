import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from execution import APPROVED_SHA256, BINARY_SHA256, MEASUREMENT_COMMIT, approved, authorization_manifest, batches, study_scope
from protocol import ROOT
from grid_authorization import require_authorization, require_derivation
from grid_protocol import atomic_json, load_plan
from submit import run_batches, submit


class ExecutionTests(unittest.TestCase):
    def test_genuine_grid_approval_cannot_submit_a_fresh_sibling_outside_its_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.registration(directory)
            with patch('execution.pilot_references', return_value={}):
                with patch('grid_authorization.derive_manifest', side_effect=authorization_manifest):
                    manifest = authorization_manifest(dict(stage='approved_grid', grid_batch=0,
                        authorization_plan=str(root / 'approved-plan.json')), root / 'approved-plan.json')
                    registered = root / 'maze4_c5_pairs_acs2vcp_k2'
                    require_authorization(registered, manifest, 'child', root / 'approval.json')
                    path = root / 'manifest.json'
                    atomic_json(path, manifest)
                    with patch('submit.execute') as execute:
                        with self.assertRaisesRegex(ValueError, 'batch root.*accounting scope'):
                            submit(root / 'sibling', path, root / 'clone', root / 'approval.json')
                        execute.assert_not_called()
            self.assertFalse((root / 'sibling').exists())
            self.assertFalse(registered.exists())
            self.assertFalse((root / 'admission-decisions.jsonl').exists())

    def test_registered_grid_root_still_reaches_the_fake_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.registration(directory)
            with patch('execution.pilot_references', return_value={}):
                with patch('grid_authorization.derive_manifest', side_effect=authorization_manifest):
                    manifest = authorization_manifest(dict(stage='approved_grid', grid_batch=0,
                        authorization_plan=str(root / 'approved-plan.json')), root / 'approved-plan.json')
                    path = root / 'manifest.json'
                    atomic_json(path, manifest)
                    registered = root / 'maze4_c5_pairs_acs2vcp_k2'
                    def prepare(batch, *_):
                        batch.mkdir()
                        (batch / 'logs').mkdir()
                        atomic_json(batch / 'ledger.json', [])
                        return manifest, 'child', root / 'binary'
                    def execute(command):
                        if command[0] == 'sshare':
                            return '0'
                        if command[0] == 'squeue':
                            return ''
                        if command[0] == 'sbatch':
                            return 'accepted' if '--test-only' in command else '123'
                        raise AssertionError(command)
                    with patch('submit.prepare', side_effect=prepare), patch('submit.execute', side_effect=execute) as calls:
                        submit(registered, path, root / 'clone', root / 'approval.json')
                    self.assertEqual(sum(call.args[0][0] == 'sbatch' and '--test-only' not in call.args[0]
                                         for call in calls.call_args_list), 1)
                    self.assertEqual(json.loads((registered / 'ledger.json').read_text())[0]['array_id'], 123)
                    self.assertEqual(json.loads((root / 'admission-decisions.jsonl').read_text())['root'], str(registered.resolve()))

    def registration(self, directory):
        root = Path(directory)
        source = Path(__file__).parent
        shutil.copyfile(source / 'plan-draft.json', root / 'approved-plan.json')
        shutil.copyfile(source / 'approval.json', root / 'approval.json')
        plan = load_plan(root / 'approved-plan.json', APPROVED_SHA256)
        (root / 'scope.json').write_text(json.dumps(study_scope(plan, root)))
        return root, plan

    def test_only_exact_approved_bytes_confirmation_and_fixed_grid_batches_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.registration(directory)
            seen = set()
            with patch('execution.pilot_references', return_value={}):
                for index in range(24):
                    manifest = authorization_manifest(dict(stage='approved_grid', grid_batch=index,
                        authorization_plan=str(root / 'approved-plan.json')), root / 'approved-plan.json')
                    batch = run_batches(manifest)
                    self.assertEqual(len(batch), 1)
                    self.assertEqual(len(batch[0]), 20)
                    seen.update((config['id'], agent, seed) for config, agent, seed in batch[0])
                    self.assertEqual(manifest['phase_cpu_hour_limit'], 110)
                    self.assertEqual(manifest['pilot_allocation_cpu_seconds'], 16041)
                    self.assertEqual(manifest['measurement_commit'], MEASUREMENT_COMMIT)
                    self.assertEqual(manifest['binary_sha256'], BINARY_SHA256)
                    for field, value in [('binary_sha256', '0' * 64), ('measurement_commit', '1' * 40),
                                         ('phase_cpu_hour_limit', 111), ('seeds', [42])]:
                        with self.assertRaises(ValueError):
                            require_derivation(dict(manifest, **{field: value}), manifest)
            self.assertEqual(len(seen), 480)
            self.assertEqual(len(batches(plan)), 24)
            content = (root / 'approved-plan.json').read_bytes()
            (root / 'approved-plan.json').write_bytes(content + b'\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                approved(root / 'approved-plan.json')
            (root / 'approved-plan.json').write_bytes(content)
            confirmation = json.loads((root / 'approval.json').read_text())
            for field, value in [('approved', False), ('phase_cpu_hour_limit', 111), ('plan_sha256', '0' * 64)]:
                (root / 'approval.json').write_text(json.dumps(dict(confirmation, **{field: value})))
                with self.assertRaises(ValueError):
                    approved(root / 'approved-plan.json')

    def test_accounting_cannot_omit_another_batch_or_pilot_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.registration(directory)
            original = study_scope(plan, root)
            for change in [dict(original, batch_roots=original['batch_roots'][:-1]),
                           dict(original, pilot_allocation_cpu_seconds=0)]:
                (root / 'scope.json').write_text(json.dumps(change))
                with self.assertRaisesRegex(ValueError, 'accounting'):
                    authorization_manifest(dict(stage='approved_grid', grid_batch=0,
                        authorization_plan=str(root / 'approved-plan.json')), root / 'approved-plan.json')


if __name__ == '__main__':
    unittest.main()
