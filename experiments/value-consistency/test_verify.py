import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify import verify_attempt_scope, verify_snapshot


class SnapshotTests(unittest.TestCase):
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
