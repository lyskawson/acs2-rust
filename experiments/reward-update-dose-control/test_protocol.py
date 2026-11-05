import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from protocol import ARM, ROOT, configurations, pilot, recorded
from execution import authorization_manifest
from grid_authorization import require_derivation
from grid_identity import fields
from row_schema import validate_blocks, validate_replay
from random_rule_schema import probability


class ProtocolTests(unittest.TestCase):
    def test_exact_geometry_budgets_three_blocks_and_two_hour_pilot(self):
        source = json.loads((ROOT / 'experiments/hindsight-admissibility/plan.json').read_text())['configurations']
        self.assertEqual(configurations(), source)
        manifest = pilot(dict(measurement_commit='a'*40, binary_sha256='b'*64, preset={}))
        self.assertEqual(manifest['seeds'], [42])
        self.assertEqual(manifest['agents'], [ARM])
        self.assertEqual(manifest['phase_cpu_hour_limit'], 2)
        self.assertEqual(sum(c['limits_minutes'][ARM] for c in manifest['configurations']), 120)
        self.assertTrue(all(c['per_start_diagnostics'] for c in manifest['configurations']))
        for changed in [dict(manifest, seeds=[43]), dict(manifest, phase_cpu_hour_limit=3)]:
            with self.assertRaises(ValueError): require_derivation(changed, manifest)
        with self.assertRaises(ValueError): authorization_manifest(dict(stage='approved_grid'))

    def test_cli_and_validators_fix_the_probability_of_each_maze(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'rows.jsonl'
            for config in configurations():
                command = [str(ROOT/'target/release/acs2-measure'), '--task',config['task'],'--cap',str(config['cap']),
                    '--protocol','maze-pairs-v1','--agents',ARM,'--seeds','42','--targets','30,60','--per-start','--out',str(output)]
                subprocess.run(command, check=True, capture_output=True, timeout=30)
                for row in map(json.loads, output.read_text().splitlines()):
                    validate_replay(row, recorded(config), ARM)
                    validate_blocks(row, recorded(config))
                    self.assertEqual(row['agent_parameters']['reward_skip_probability'].hex(), probability(config['task']).hex())
                    for field, value in [('reward_skip_probability', probability(config['task'])+1e-12),
                            ('reward_skip_stream',7), ('copy_update','GoalRules'), ('relabeled_proportion',.5), ('replay_updates_per_step',1)]:
                        changed = copy.deepcopy(row)
                        changed['agent_parameters'][field] = value
                        with self.subTest(maze=config['task'], field=field), self.assertRaises(ValueError):
                            validate_replay(changed, recorded(config), ARM)
                    changed = copy.deepcopy(row)
                    changed['replay_diagnostics']['reward_updates']['original']['skipped_with_goal'] = 1
                    with self.assertRaises(ValueError): validate_replay(changed, recorded(config), ARM)
                for extra in [['--relabel-proportion','0.5'], ['--reward-skip-probability','0.1'], ['--replay-order','backward']]:
                    self.assertNotEqual(subprocess.run(command+extra, capture_output=True).returncode, 0)

    def test_all_existing_arms_and_goal_rule_counters_keep_every_deterministic_bit(self):
        fixture = json.loads((ROOT/'fixtures/reward_dose_existing_arms.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            for config in fixture['configurations']:
                output = Path(directory)/'rows.jsonl'
                subprocess.run([str(ROOT/'target/release/acs2-measure'),'--task',config['task'],'--cap',str(config['cap']),
                    '--protocol','maze-pairs-v1','--agents',config['arm'],'--seeds','42','--targets','1000,3000','--per-start','--out',str(output)],
                    check=True,capture_output=True,timeout=30)
                observed = [hashlib.sha256(json.dumps(fields(json.loads(line),['training','validation','test']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
                            for line in output.read_text().splitlines()]
                self.assertEqual(observed,config['rows'],(config['task'],config['arm']))
