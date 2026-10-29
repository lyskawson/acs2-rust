import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_pilot
from protocol import CORRECTED_PILOT_INDICES


class PreparationTests(unittest.TestCase):
    def test_complete_preparation_returns_reference_count_and_freezes_clean_operations(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            repo = directory / 'source'
            name = 'experiments/replay-schedule-and-order/execution.py'
            source = repo / name
            source.parent.mkdir(parents=True)
            source.write_text('immutable operation')
            archive = directory / 'prior'
            archive.mkdir()
            (archive / 'selections.json').write_text('{}\n')
            (archive / 'pilot-scope.json').write_text(json.dumps(dict(batch_roots=['/cluster/previous/pilot-base'])))
            data = b'{}\n'
            refs = [dict(purpose='test', configuration_id='maze4_c5_pairs', seed=42,
                         data=data, source='/prior/rows.jsonl', sha256=hashlib.sha256(data).hexdigest())]
            output = directory / 'prepared'
            with patch.object(prepare_pilot, 'ROOT', repo), \
                 patch.object(prepare_pilot, 'reference_rows', return_value=refs), \
                 patch.object(prepare_pilot, 'verify_previous', return_value=dict(allocation_cpu_seconds=5035)), \
                 patch.object(prepare_pilot, 'pilot_manifest', side_effect=lambda index, *_: dict(pilot_index=index)), \
                 patch.object(prepare_pilot.subprocess, 'check_output', side_effect=['', 'a' * 40, name]):
                result = prepare_pilot.prepare(output, '/cluster/study', archive, archive)
            self.assertEqual(result['reference_cells'], 1)
            self.assertEqual(result['operations_commit'], 'a' * 40)
            snapshot = json.loads((output / 'operations/manifest.json').read_text())
            self.assertEqual(snapshot['files'], {name: hashlib.sha256(source.read_bytes()).hexdigest()})
            registration = output / 'pilot-registration.json'
            self.assertEqual(result['registration_sha256'], hashlib.sha256(registration.read_bytes()).hexdigest())
            registered = json.loads(registration.read_text())
            self.assertEqual(registered['pilot_indices'], CORRECTED_PILOT_INDICES)
            self.assertEqual(registered['previous_pilot']['allocation_cpu_seconds'], 5035)
            accounting = json.loads((output / 'pilot-scope.json').read_text())
            self.assertEqual(len(accounting['batch_roots']), 45)
            self.assertEqual(accounting['pilot_allocation_cpu_seconds'], 0)
            for index in CORRECTED_PILOT_INDICES:
                manifest = json.loads((output / f'pilot-manifest-{index}.json').read_text())
                self.assertEqual(manifest['pilot_index'], index)
                self.assertEqual(manifest['authorization_sha256'], result['registration_sha256'])
                self.assertEqual(manifest['operations_commit'], 'a' * 40)
            for index in set(range(10)) - set(CORRECTED_PILOT_INDICES):
                self.assertIsNone(registered['manifests'][index])
                self.assertFalse((output / f'pilot-manifest-{index}.json').exists())


if __name__ == '__main__':
    unittest.main()
