import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from protocol import ROOT, E1, S1, NEW_ARMS, recorded, configurations
from grid_identity import field_bits
from schema import validate_row, validate_sequence


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT / 'fixtures/per_goal_reward_legacy.json').read_text())
        cls.legacy_arms = list(dict.fromkeys(row['agent'] for row in cls.fixture['rows']))
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        output = Path(cls.directory.name) / 'rows.jsonl'
        subprocess.run([str(ROOT / 'target/release/acs2-measure'), '--task', 'maze4', '--cap', '5',
            '--protocol', 'maze-pairs-v1', '--agents', ','.join([*cls.legacy_arms, *NEW_ARMS]),
            '--seeds', '42', '--targets', '10,50', '--out', str(output)], check=True, capture_output=True)
        cls.rows = [json.loads(line) for line in output.read_text().splitlines()]
        cls.config = dict(recorded(configurations()[0]), targets=[10, 50])
        del cls.config['per_start_diagnostics']

    def test_every_existing_arm_including_goal_and_random_rule_counters_is_bit_identical(self):
        observed = []
        for row in self.rows:
            if row['agent'] in NEW_ARMS:
                continue
            value = {key: value for key, value in row.items() if key not in {
                'commit', 'source_state', 'host', 'cpu_model', 'wall_seconds_train', 'wall_seconds_eval', 'wall_seconds_total'}}
            observed.append(dict(agent=row['agent'], seed=row['seed'], nominal_step=row['nominal_step'],
                learning_sha256=hashlib.sha256(json.dumps(field_bits(value), sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()))
        self.assertEqual(observed, self.fixture['rows'])
        self.assertEqual(len(self.legacy_arms), 15)

    def test_new_measurement_rows_have_exact_presets_routes_and_live_entry_counts(self):
        for arm in NEW_ARMS:
            rows = [dict(row, source_state='clean') for row in self.rows if row['agent'] == arm]
            validate_sequence(rows, self.config, arm, 42, rows[0]['commit'])
            for row in rows:
                counts = row['replay_diagnostics']['per_goal_rewards']
                self.assertGreater(counts['updated_entries'], 0)
                self.assertEqual(counts['existing_pairs'], row['population_classifiers'] * 27)
                self.assertGreater(counts['reward_updates']['original'], 0)
                if arm == S1:
                    self.assertGreater(counts['reward_updates']['relabeled'], 0)
                else:
                    self.assertEqual(counts['reward_updates']['relabeled'], 0)

    def test_validator_rejects_other_arms_ga_incomplete_or_misdirected_values_and_changed_presets(self):
        row = next(copy.deepcopy(row) for row in self.rows if row['agent'] == S1)
        row['source_state'] = 'clean'
        def rejected(mutator):
            changed = copy.deepcopy(row)
            mutator(changed)
            with self.assertRaises(ValueError):
                validate_row(changed, self.config, changed['agent'], 42, row['commit'], 10)
        for arm in [*self.legacy_arms, S1 + '_extra', E1 + '_extra']:
            rejected(lambda value, arm=arm: value.update(agent=arm))
        rejected(lambda value: value['preset'].update(do_ga=True))
        rejected(lambda value: value['preset'].update(initial_r=0.0))
        rejected(lambda value: value['agent_parameters'].update(restore_supported=True))
        rejected(lambda value: value['agent_parameters'].update(reward_goal_count=26))
        rejected(lambda value: value['agent_parameters'].update(relabeled_proportion=.5))
        rejected(lambda value: value['replay_diagnostics'].pop('per_goal_rewards'))
        rejected(lambda value: value['replay_diagnostics']['per_goal_rewards'].update(existing_pairs=1))
        rejected(lambda value: value['replay_diagnostics']['per_goal_rewards'].update(updated_entries=10**9))
        rejected(lambda value: value['replay_diagnostics']['per_goal_rewards']['reward_updates'].update(original=-1))

    def test_cli_refuses_restore_ga_and_proportion_overrides_for_both_new_arms(self):
        for arm in NEW_ARMS:
            for option, argument in [('--restore', 'missing.checkpoint'), ('--do-ga', 'true'),
                                     ('--relabel-proportion', '0.5')]:
                result = subprocess.run([str(ROOT / 'target/release/acs2-measure'), '--task', 'maze4',
                    '--cap', '5', '--agents', arm, '--targets', '1', '--out', str(Path(self.directory.name) / 'rejected.jsonl'),
                    option, argument], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)
