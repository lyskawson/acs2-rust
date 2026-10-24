import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from protocol import AGENTS, ROOT, configurations
from collect import validate_row
from row_schema import validate_replay


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        cls.config = configurations()[0]
        with tempfile.TemporaryDirectory() as temporary:
            for agent in AGENTS:
                output = Path(temporary) / (agent + '.jsonl')
                subprocess.run([str(ROOT / 'target/release/acs2-measure'), '--task', 'maze4', '--cap', '5',
                    '--protocol', 'maze-pairs-v1', '--agents', agent, '--seeds', '42', '--targets', '1000',
                    '--out', str(output)], check=True, capture_output=True)
                row = json.loads(output.read_text())
                row['source_state'] = 'clean'
                cls.rows[agent] = row

    def test_both_new_names_pass_shared_collection(self):
        for agent, row in self.rows.items():
            validate_row(row, self.config, agent, 42, row['commit'], 1000)

    def test_fixed_presets_reject_every_loosened_parameter(self):
        mutations = [('admissibility', 'EveryTransition'), ('relabeled_proportion', 1 / 3),
                     ('copy_update', 'ValuesOnly'), ('strategy', 'episode'), ('candidate_filter', True),
                     ('replay_capacity', 9999), ('replay_capacity', 10000.0), ('candidate_filter', 0),
                     ('replay_updates_per_step', 1), ('unknown_parameter', 1)]
        for agent, reference in self.rows.items():
            for key, value in mutations:
                with self.subTest(agent=agent, key=key):
                    row = copy.deepcopy(reference)
                    row['agent_parameters'][key] = value
                    with self.assertRaises(ValueError):
                        validate_replay(row, self.config, agent)
            row = copy.deepcopy(reference)
            row['agent_parameters'].pop('truncation')
            with self.assertRaises(ValueError):
                validate_replay(row, self.config, agent)

    def test_manifests_cannot_override_the_fixed_proportion(self):
        for agent, row in self.rows.items():
            config = dict(self.config, agent_settings={agent: dict(relabeled_proportion=1 / 3)})
            with self.assertRaises(ValueError):
                validate_replay(row, config, agent)

    def test_wall_counters_and_zero_inadmissible_shares_are_required(self):
        for agent, reference in self.rows.items():
            for key, value in [('relabeled_draws', reference['replay_updates']), ('unchanged_state_terminal', 1),
                               ('unchanged_state', -1), ('proposals_equal_draws', False), ('unchanged_state', True)]:
                row = copy.deepcopy(reference)
                row['replay_diagnostics']['wall_bumps'][key] = value
                with self.assertRaises(ValueError):
                    validate_replay(row, self.config, agent)
            row = copy.deepcopy(reference)
            row['replay_diagnostics'].pop('wall_bumps')
            with self.assertRaises(ValueError):
                validate_replay(row, self.config, agent)


if __name__ == '__main__':
    unittest.main()
