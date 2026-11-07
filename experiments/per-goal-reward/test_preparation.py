import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from protocol import ROOT, configurations
from prepare_pilot import prepare
from execution import require_authorization
from grid_protocol import atomic_json


class PreparationTests(unittest.TestCase):
    def test_actual_pilot_preparation_output_passes_the_actual_authorizer_and_frozen_input_guards(self):
        preset = json.loads((ROOT / 'fixtures/goal_measurement_regression.json').read_text())['configurations'][0]['rows'][0]['preset']
        identity = dict(measurement_commit='a' * 40, binary_sha256='b' * 64, preset=preset)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, models = root / 'references', root / 'models'
            references.mkdir()
            models.mkdir()
            atomic_json(references / 'reference-index.json', [])
            index = []
            for config in configurations():
                path = models / (config['task'] + '.json')
                atomic_json(path, dict(task=config['task'], cap=config['cap'], gamma=.95, reward=1000, actions=8, pairs=[]))
                index.append(dict(task=config['task'], file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            atomic_json(models / 'model-index.json', index)
            def git(command, **kwargs):
                if 'status' in command: return ''
                if 'rev-parse' in command: return 'c' * 40
                if 'ls-files' in command: return 'experiments/per-goal-reward/protocol.py\n'
                raise AssertionError(command)
            output = root / 'study'
            with patch('prepare_pilot.subprocess.check_output', side_effect=git), \
                    patch('prepare_pilot.identity', return_value=identity), patch('execution.identity', return_value=identity):
                result = prepare(output, output, references, models)
                self.assertEqual(result['runs'], 12)
                manifest = json.loads((output / 'pilot-manifest.json').read_text())
                require_authorization(output / 'pilot', manifest)
                registration = json.loads((output / 'pilot-registration.json').read_text())
                self.assertEqual(registration['reference_index_sha256'], hashlib.sha256((output / 'reference-index.json').read_bytes()).hexdigest())
                self.assertEqual(registration['model_index_sha256'], hashlib.sha256((output / 'model-index.json').read_bytes()).hexdigest())
                (output / 'reference-index.json').write_text('[1]\n')
                with self.assertRaises(ValueError): require_authorization(output / 'pilot', manifest)
