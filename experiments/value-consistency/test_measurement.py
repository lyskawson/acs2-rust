import copy
import json
from pathlib import Path
import struct
import sys
import unittest

from protocol import ROOT, configurations

sys.path.append(str(ROOT / 'experiments/grid'))
from collect import validate_row


def decode(value):
    if isinstance(value, str) and value.startswith('f64:'):
        return struct.unpack('>d', bytes.fromhex(value[4:]))[0]
    if isinstance(value, list):
        return list(map(decode, value))
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    return value


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT / 'fixtures/vcp_pair_regression.json').read_text())
        self.configs = {config['task']: dict(config, evaluation_blocks=['validation']) for config in configurations()}
        self.rows = [dict(decode(row), commit='pinned', source_state='clean', host='fixture-host',
                         cpu_model='AMD EPYC 9554 64-Core Processor')
                     for config in self.fixture['configurations'] for row in config['rows']]

    def validate(self, row):
        validate_row(row, self.configs[row['task']], row['agent'], row['seed'], 'pinned', row['nominal_step'])

    def test_actual_pilot_rows_pass_shared_collection_guards(self):
        self.assertGreaterEqual(len(self.rows), 8)
        for row in self.rows:
            self.validate(row)

    def test_ensemble_cost_and_behavior_defects_are_refused(self):
        row = next(row for row in self.rows if row['agent'] == 'acs2vcp_k4')
        mutations = [
            (['population_classifiers'], row['population_classifiers'] * 2),
            (['replay_diagnostics', 'priority', 'value_queries'], row['replay_diagnostics']['priority']['value_queries'] + 1),
            (['replay_diagnostics', 'priority', 'accepted_draws'], row['replay_updates'] + 1),
            (['agent_parameters', 'priority_clock'], 'after_learning'),
            (['agent_parameters', 'sampler_stream'], 1),
            (['agent_parameters', 'temperature'], 2),
            (['agent_parameters', 'importance_weights'], 'inverse_probability'),
            (['match_formations_train'], row['match_formations_train'] - 1),
            (['replay_diagnostics', 'objective_evaluations_by_purpose', 'scoring', 'objective_evaluations'], 0),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                changed = copy.deepcopy(row)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    self.validate(changed)

    def test_a_head_zero_only_metric_or_an_acceptance_coin_in_the_control_is_refused(self):
        row = copy.deepcopy(next(row for row in self.rows if any(
            abs(head['success'] - row['evaluation_blocks']['validation']['success']) > 1e-12
            for head in row['evaluation_blocks']['validation']['heads'][:1])))
        row['evaluation_blocks']['validation']['success'] = row['evaluation_blocks']['validation']['heads'][0]['success']
        with self.assertRaises(ValueError):
            self.validate(row)
        control = copy.deepcopy(next(row for row in self.rows if row['agent'].endswith('_uniform')))
        control['replay_diagnostics']['priority']['acceptance_coins'] = 1
        with self.assertRaises(ValueError):
            self.validate(control)


if __name__ == '__main__':
    unittest.main()
