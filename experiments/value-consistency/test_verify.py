import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from execution import approved, study_scope
from retry_batch import RECOVERY_SHA256
from verify import verify_admissions, verify_attempt_scope, verify_snapshot, verify_study


class SnapshotTests(unittest.TestCase):
    def admissions_fixture(self, root):
        source = Path(__file__).parent
        for name, target in [('plan-draft.json', 'approved-plan.json'), ('approval.json', 'approval.json')]:
            shutil.copyfile(source / name, root / target)
        scope = study_scope(approved(root / 'approved-plan.json'), Path('/registered/cluster/study'))
        (root / 'scope.json').write_text(json.dumps(scope))
        admissions = [dict(root=name, assignment=0) for name in scope['batch_roots']]
        (root / 'admission-decisions.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in admissions))
        recovery = root / 'recovery'
        recovery.mkdir()
        shutil.copyfile(source / 'recovery.json', recovery / 'approved-recovery.json')
        shutil.copyfile(source / 'recovery-approval.json', recovery / 'approval.json')
        proposal = json.loads((recovery / 'approved-recovery.json').read_text())
        command = ['sbatch', '--array=0-6%20', '/registered/cluster/study/maze4_c5_pairs_acs2vcp_k2']
        resolution = dict(event='approved_timeout_resolution', recovery_sha256=RECOVERY_SHA256,
            state=dict(array_ids=[proposal['cause']['array_id']]), command=command,
            new_limit_seconds=proposal['accounting']['new_retry_hard_seconds'],
            whole_study_hard_bound_seconds=proposal['accounting']['whole_study_upper_bound_seconds'])
        (recovery / 'resolution.json').write_text(json.dumps(resolution))
        batch = root / 'maze4_c5_pairs_acs2vcp_k2'
        batch.mkdir()
        ledger = [dict(array_id=proposal['cause']['array_id']),
                  dict(array_id=6109917, submission_response='6109917', command=command, recovery_sha256=RECOVERY_SHA256)]
        (batch / 'ledger.json').write_text(json.dumps(ledger))
        return scope, admissions

    def test_admissions_use_declared_cluster_roots_and_the_approved_second_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, _ = self.admissions_fixture(root)
            verify_admissions(root, scope)
            path = root / 'maze4_c5_pairs_acs2vcp_k2/ledger.json'
            ledger = json.loads(path.read_text())
            ledger[1]['array_id'] = 6109918
            path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, 'recovery resolution'):
                verify_admissions(root, scope)

    def test_study_refuses_extra_missing_and_duplicate_admissions_before_raw_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, admissions = self.admissions_fixture(root)
            changes = [admissions + [dict(root='/registered/cluster/study/sibling')],
                       admissions[:-1], admissions + [admissions[0]]]
            for changed in changes:
                with self.subTest(roots=[item['root'] for item in changed]):
                    (root / 'admission-decisions.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in changed))
                    with self.assertRaisesRegex(ValueError, 'admissions.*exactly once'):
                        verify_study(root)

    def test_attempt_accounting_refuses_an_extra_seed_or_a_substituted_limit(self):
        plan = dict(configurations=[dict(id='maze', limits_minutes=dict(acs2vcp_k2=6))],
                    agents=['acs2vcp_k2'], seeds=list(range(42, 62)))
        runs = [dict(configuration_id='maze', agent='acs2vcp_k2', seed=seed, limit_seconds=360,
                     attempt_id=f'maze_acs2vcp_k2_s{seed}_a01') for seed in range(42, 62)]
        verify_attempt_scope(plan, [dict(runs=runs)], Path('unused'))
        for key, changed in [('seed', 62), ('limit_seconds', 600)]:
            altered = [dict(run) for run in runs]
            altered[0][key] = changed
            with self.assertRaisesRegex(ValueError, 'scope or hard limits'):
                verify_attempt_scope(plan, [dict(runs=altered)], Path('unused'))
        with self.assertRaisesRegex(ValueError, 'no approval'):
            verify_attempt_scope(plan, [dict(runs=runs), dict(runs=runs)], Path('unused'))

    def test_archived_operations_are_verified_by_content_and_cannot_escape_the_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operations = root / 'operations'
            operations.mkdir()
            source = operations / 'operation.py'
            source.write_text('value = 1\n')
            manifest = operations / 'manifest.json'
            manifest.write_text(json.dumps(dict(commit='frozen', files={'operation.py': hashlib.sha256(source.read_bytes()).hexdigest()})))
            child = dict(operations_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(verify_snapshot(root, child)['commit'], 'frozen')
            source.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                verify_snapshot(root, child)
            manifest.write_text(json.dumps(dict(files={'../outside.py': 'unused'})))
            child['operations_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'outside'):
                verify_snapshot(root, child)


if __name__ == '__main__':
    unittest.main()
