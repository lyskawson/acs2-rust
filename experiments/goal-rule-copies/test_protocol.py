import copy
import unittest
from protocol import ARM, ROOT, configurations, pilot, recorded
from grid_authorization import require_derivation
from row_schema import validate_replay, validate_blocks
from grid_identity import field_bits
import json
import subprocess
import tempfile
from pathlib import Path


class GoalRuleProtocolTests(unittest.TestCase):
    def test_registration_preserves_geometry_protocol_and_budgets_and_pilot_limit(self):
        source = json.loads((ROOT / 'experiments/hindsight-admissibility/plan.json').read_text())['configurations']
        self.assertEqual(configurations(), source)
        manifest = pilot(dict(measurement_commit='a'*40, binary_sha256='b'*64, preset={}))
        self.assertEqual(manifest['seeds'], [42])
        self.assertEqual(manifest['agents'], [ARM])
        self.assertEqual(sum(c['limits_minutes'][ARM] for c in manifest['configurations']), 120)
        self.assertEqual(manifest['phase_cpu_hour_limit'], 2)
        self.assertTrue(all(c['per_start_diagnostics'] for c in manifest['configurations']))
        with self.assertRaises(ValueError):
            require_derivation(dict(manifest, seeds=[43]), manifest)

    def test_cli_and_shared_validators_accept_exact_fixed_arm_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'rows.jsonl'
            command = [str(ROOT/'target/release/acs2-measure'), '--task','maze4','--cap','5', '--protocol','maze-pairs-v1',
                '--agents', ARM, '--seeds','42','--targets','30,60','--per-start','--out',str(path)]
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            config = recorded(configurations()[0])
            for row in rows:
                validate_replay(row, config, ARM)
                validate_blocks(row, config)
                for field, value in [('relabeled_proportion', .5), ('copy_update','Full'), ('replay_updates_per_step',1)]:
                    changed = copy.deepcopy(row)
                    changed['agent_parameters'][field] = value
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        validate_replay(changed, config, ARM)
            for changed in [command[:command.index('--agents')+1]+[ARM+'_extra']+command[command.index('--agents')+2:],
                            command+['--relabel-proportion','0.5']]:
                self.assertNotEqual(subprocess.run(changed, capture_output=True).returncode, 0)
