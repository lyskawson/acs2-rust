import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from protocol import ROOT, HER, SHORTENED, configurations, setting
from collect import named_row, validate_row
from row_schema import validate_replay


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        with tempfile.TemporaryDirectory() as temporary:
            for agent, p in [(SHORTENED, 1 / 3), (SHORTENED, 0.5), (HER, 0.5)]:
                path = Path(temporary) / f'{agent}-{p}.jsonl'
                subprocess.run([str(ROOT / 'target/release/acs2-measure'), '--task', 'maze4', '--cap', '5',
                    '--protocol', 'maze-pairs-v1', '--agents', agent, '--relabel-proportion', str(p), '--seeds', '42',
                    '--targets', '1000', '--evaluation-blocks', 'validation', '--out', str(path)], check=True, capture_output=True)
                config = setting(configurations()[0], agent, p, ['validation'])
                row = named_row(json.loads(path.read_text()), config)
                row['source_state'] = 'clean'
                cls.rows[agent, p] = config, row

    def test_measured_distinct_settings_pass_shared_collection(self):
        for (agent, p), (config, row) in self.rows.items():
            validate_row(row, config, agent, 42, row['commit'], 1000)
            self.assertEqual(row['agent_parameters']['relabeled_proportion'], p)
        self.assertNotEqual(self.rows[SHORTENED, 1 / 3][1]['replay_diagnostics'], self.rows[SHORTENED, 0.5][1]['replay_diagnostics'])

    def test_schema_rejects_row_p_different_from_manifest_without_changing_arm(self):
        for (agent, p), (config, reference) in self.rows.items():
            row = copy.deepcopy(reference)
            row['agent_parameters']['relabeled_proportion'] = 0.8
            with self.assertRaises(ValueError):
                validate_replay(row, config, agent)

    def test_other_preset_fields_and_goal_rule_are_still_fixed(self):
        for (agent, p), (config, reference) in self.rows.items():
            for key, value in [('copy_update', 'ValuesOnly'), ('strategy', 'episode'),
                               ('admissibility', 'CounterfactualEpisode' if agent == HER else 'EveryTransition'),
                               ('replay_warmup', 2), ('candidate_filter', True)]:
                row = copy.deepcopy(reference)
                row['agent_parameters'][key] = value
                with self.subTest(agent=agent, key=key), self.assertRaises(ValueError):
                    validate_replay(row, config, agent)
            for invalid in [float('nan'), 1.1, -0.1, True]:
                altered = copy.deepcopy(config)
                altered['agent_settings'][agent]['relabeled_proportion'] = invalid
                with self.assertRaises(ValueError):
                    validate_replay(reference, altered, agent)


if __name__ == '__main__':
    unittest.main()
