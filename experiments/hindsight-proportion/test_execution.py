import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from execution import authorization_manifest, scope
from protocol import pilot, scope_roots
from submit import prepare


class ExecutionTests(unittest.TestCase):
    def test_shared_submission_has_independent_five_hour_pilot_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'source'
            binary = repo / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'frozen-test-binary')
            (repo / 'fixtures').mkdir()
            (repo / 'fixtures/goal_measurement_regression.json').write_text(json.dumps(
                dict(configurations=[dict(rows=[dict(preset={})])])))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(pilot(0), phase_cpu_hour_limit=6)))
            with patch('submit.execute', side_effect=['a' * 40, '']), patch('submit.require_authorization'):
                with self.assertRaisesRegex(ValueError, 'wrong partition, concurrency or CPU-hour limit'):
                    prepare(root / 'batch', manifest, repo)

    def test_one_scope_contains_all_pilot_tuning_control_and_conditional_test_roots_once(self):
        value = scope('a' * 64, '/external/study')
        self.assertEqual(len(value['batch_roots']), len(set(value['batch_roots'])))
        self.assertEqual([Path(name).name for name in value['batch_roots']], scope_roots())
        self.assertEqual(value['pilot_allocation_cpu_seconds'], 0)
        self.assertEqual(len(scope_roots()), 13)

    def test_stage_one_cannot_authorize_tuning_control_or_test(self):
        for purpose in ('tuning', 'control', 'test'):
            with self.assertRaises((ValueError, FileNotFoundError)):
                authorization_manifest(dict(stage='approved_grid', purpose=purpose,
                    authorization_plan='/unapproved/plan.json'), '/unapproved/plan.json')


if __name__ == '__main__':
    unittest.main()
