import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from freeze_dose import derive
from protocol import ROOT, ARM, G
from prepare_pilot import prepare
from execution import authorization_manifest


class InputTests(unittest.TestCase):
    def test_archive_integer_sums_require_twenty_final_points_and_match_binary64(self):
        expected=json.loads(Path(__file__).with_name('probabilities.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            entries=[]
            for maze,counts in expected['mazes'].items():
                for i,seed in enumerate(range(42,62)):
                    counters={}
                    for field in ('with_goal','skipped_without_goal'):
                        base,remainder=divmod(counts[field],20)
                        counters[field]=base+int(i<remainder)
                    row=dict(task=maze,agent=G,seed=seed,nominal_step=4_000_000 if maze=='woods14' else 2_000_000,
                             replay_diagnostics=dict(reward_updates=dict(relabeled=counters)))
                    path=root/f'{maze}-{seed}.jsonl'
                    path.write_text(json.dumps(row)+'\n')
                    entries.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),agent=G,seed=seed))
            verification=dict(verified=True,failed_attempts=0,plan_sha256=expected['source_plan_sha256'],runs_index=entries)
            path=root/'verification.json'
            path.write_text(json.dumps(verification))
            self.assertEqual(derive(root),expected)
            broken=copy.deepcopy(verification)
            broken['runs_index'].append(broken['runs_index'][0])
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError): derive(root)
            path.write_text(json.dumps(verification))
            (root/entries[0]['file']).write_text('{}\n')
            with self.assertRaises(ValueError): derive(root)

    def test_pilot_preparation_and_authorization_bind_identity_inputs_scope_and_seed(self):
        identity=dict(measurement_commit='a'*40,binary_sha256='b'*64,preset={})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            references=root/'references'
            references.mkdir()
            (references/'rows.jsonl').write_text('{}\n')
            entry=dict(file='rows.jsonl',sha256=hashlib.sha256((references/'rows.jsonl').read_bytes()).hexdigest())
            (references/'reference-index.json').write_text(json.dumps([entry]))
            study=root/'study'
            def command(args, **kwargs):
                if 'status' in args:return ''
                if 'rev-parse' in args:return 'c'*40+'\n'
                if 'ls-files' in args:return 'experiments/grid/grid_authorization.py\n'
                return b'{}\n'
            with patch('prepare_pilot.identity',return_value=identity),patch('prepare_pilot.subprocess.check_output',side_effect=command):
                measured=prepare(study,study,references)
            self.assertEqual(measured['hard_reservations_cpu_hours'],2)
            manifest=json.loads((study/'pilot-manifest.json').read_text())
            with patch('execution.identity',return_value=identity):
                expected=authorization_manifest(manifest)
                self.assertEqual(expected['agents'],[ARM])
                self.assertEqual(expected['seeds'],[42])
                self.assertTrue(all(c['per_start_diagnostics'] for c in expected['configurations']))
                with self.assertRaises(ValueError): authorization_manifest(manifest,study/'approved-plan.json')
                (study/'model-index.json').write_text('[]')
                with self.assertRaises(ValueError): authorization_manifest(manifest)
