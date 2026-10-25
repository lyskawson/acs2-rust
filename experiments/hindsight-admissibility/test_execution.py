import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from execution import authorization_manifest
from protocol import pilot
from grid_authorization import require_derivation
from submit import prepare


class ExecutionTests(unittest.TestCase):
    def test_submission_refuses_pilot_limits_above_two_cpu_hours(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'source'
            binary = repo / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'frozen-test-binary')
            (repo / 'fixtures').mkdir()
            (repo / 'fixtures/goal_measurement_regression.json').write_text(json.dumps(
                dict(configurations=[dict(rows=[dict(preset={})])])))
            for hours in (3, 5):
                manifest = root / 'manifest.json'
                manifest.write_text(json.dumps(dict(pilot(), phase_cpu_hour_limit=hours)))
                with self.subTest(hours=hours), patch('submit.execute', side_effect=['a' * 40, '']), patch('submit.require_authorization'):
                    with self.assertRaisesRegex(ValueError, 'wrong partition, concurrency or CPU-hour limit'):
                        prepare(root / ('batch' + str(hours)), manifest, repo)

    def test_pilot_manifest_is_derived_from_the_fixed_protocol(self):
        with patch('execution.measurement_identity', return_value=dict(measurement_commit='a' * 40, binary_sha256='b' * 64)):
            expected = authorization_manifest(pilot())
        require_derivation(expected, expected)
        for key, value in [('phase_cpu_hour_limit', 3), ('seeds', [42, 43]), ('agents', ['acs2her']), ('unknown', True)]:
            manifest = dict(expected, **{key: value})
            with self.assertRaises(ValueError):
                require_derivation(manifest, expected)

    def test_stage_one_adapter_cannot_authorize_a_grid(self):
        with self.assertRaises(ValueError):
            authorization_manifest(dict(stage='approved_grid'), 'unapproved-plan.json')


if __name__ == '__main__':
    unittest.main()
