import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from previous_pilot import pilot_batch, previous_path, reference_path, verify_previous
from protocol import pilot_roots


class PreviousPilotTests(unittest.TestCase):
    def test_previous_round_verifies_raw_bytes_and_recomputes_accounting_with_its_frozen_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = dict(measurement_commit='a' * 40, binary_sha256='b' * 64,
                allocations=60, rows=490, identity_rows=49, allocation_cpu_seconds=5035)
            (root / 'pilot-summary.json').write_text(json.dumps(summary))
            (root / 'pilot-rows.jsonl').write_text('{}\n')
            (root / 'pilot-registration.json').write_text('{}\n')
            files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
            (root / 'checksums.json').write_text(json.dumps(dict(files=files)))
            checked = {key: summary[key] for key in ('allocations', 'rows', 'identity_rows', 'allocation_cpu_seconds')}
            completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(checked))
            with patch('previous_pilot.subprocess.run', return_value=completed) as command:
                self.assertEqual(verify_previous(root)['allocation_cpu_seconds'], 5035)
                self.assertEqual(Path(command.call_args.args[0][2]), root / 'operations/experiments/replay-schedule-and-order/archive.py')
                (root / 'pilot-rows.jsonl').write_text('{"changed":true}\n')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    verify_previous(root)
                (root / 'pilot-rows.jsonl').write_text('{}\n')
                completed.stdout = json.dumps(dict(checked, allocation_cpu_seconds=5034))
                with self.assertRaisesRegex(ValueError, 'independent verification'):
                    verify_previous(root)

    def test_b_identity_uses_the_first_round_and_c_and_default_use_the_new_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root, old = base / 'uniform-episode-pilot', base / 'study'
            root.mkdir()
            registration = dict(destination='/cluster/study/uniform-episode-pilot',
                previous_pilot=dict(root='/cluster/study', mirror_relative='../study'))
            (root / 'pilot-registration.json').write_text(json.dumps(registration))
            self.assertEqual(previous_path(root), old)
            self.assertEqual(pilot_batch(root, 0), old / pilot_roots()[0])
            self.assertEqual(pilot_batch(root, 3), root / pilot_roots()[3])
            self.assertEqual(pilot_batch(root, 9), root / pilot_roots()[9])
            declared = registration['destination']
            self.assertEqual(reference_path(root, declared, old / 'b/rows.jsonl'), Path('/cluster/study/b/rows.jsonl'))
            self.assertEqual(reference_path(root, declared, root / 'c/rows.jsonl'), Path(declared) / 'c/rows.jsonl')
            with self.assertRaisesRegex(ValueError, 'outside the two registered rounds'):
                reference_path(root, declared, base / 'unregistered/rows.jsonl')


if __name__ == '__main__':
    unittest.main()
