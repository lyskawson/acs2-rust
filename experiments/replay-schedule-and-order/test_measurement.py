import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from protocol import ROOT, PILOT_SETTINGS, SHORTENED, ER, configurations, setting
from collect import named_row, validate_row
from replay_schema import registered_axes


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        base = configurations()[0]
        with tempfile.TemporaryDirectory() as temporary:
            for index, (agent, mode) in enumerate(PILOT_SETTINGS):
                config = setting(base, agent, mode, ['validation'])
                path = Path(temporary) / f'{index}.jsonl'
                command = [str(ROOT / 'target/release/acs2-measure'), '--task', 'maze4', '--cap', '5',
                    '--protocol', 'maze-pairs-v1', '--agents', agent, '--seeds', '42', '--targets', '1000,30100',
                    '--evaluation-blocks', 'validation', '--out', str(path)]
                settings = config['agent_settings'][agent]
                if agent == SHORTENED:
                    command += ['--relabel-proportion', str(settings['relabeled_proportion'])]
                if mode != 'base':
                    command += ['--replay-schedule', settings['replay_schedule'], '--replay-order', settings['replay_order']]
                subprocess.run(command, check=True, capture_output=True)
                cls.rows[agent, mode] = config, [dict(named_row(json.loads(line), config), source_state='clean')
                                              for line in path.read_bytes().splitlines()]

    def test_every_registered_setting_passes_shared_collection(self):
        for (agent, mode), (config, rows) in self.rows.items():
            for row in rows:
                validate_row(row, config, agent, 42, row['commit'], row['nominal_step'])
                if agent == SHORTENED:
                    self.assertEqual(row['replay_diagnostics']['routes']['relabeled']['already_reached'], 0)
                    self.assertEqual(row['replay_diagnostics']['routes']['relabeled']['after_counterfactual_end'], 0)

    def test_validator_refuses_different_or_extra_axes_parameters(self):
        for (agent, mode), (config, rows) in self.rows.items():
            if mode == 'base':
                continue
            for key, value in [('replay_order', 'forward'), ('replay_schedule', dict(kind='constant', extra=True)),
                               ('copy_update', 'ValuesOnly'), ('strategy', 'final'), ('extra', True)]:
                row = copy.deepcopy(rows[-1])
                row['agent_parameters'][key] = value
                with self.subTest(agent=agent, mode=mode, key=key), self.assertRaises(ValueError):
                    validate_row(row, config, agent, 42, row['commit'], row['nominal_step'])

    def test_switch_volume_and_er_control_schedule_cannot_be_ignored(self):
        for agent in (SHORTENED, ER):
            config, rows = self.rows[agent, '3to1at30000']
            row = rows[-1]
            self.assertLess(row['replay_updates'], 3 * row['actual_steps'])
            for change in ('rate', 'clock', 'parameters'):
                altered = copy.deepcopy(row)
                if change == 'rate':
                    altered['replay_updates'] = 3 * altered['actual_steps']
                elif change == 'clock':
                    altered['replay_diagnostics']['draw_schedule']['early_steps'] += 1
                else:
                    altered['agent_parameters']['replay_schedule']['late_rate'] = True
                with self.subTest(agent=agent, change=change), self.assertRaises(ValueError):
                    validate_row(altered, config, agent, 42, row['commit'], row['nominal_step'])

    def test_manifest_refuses_unregistered_combined_and_partial_settings(self):
        base = configurations()[0]
        config = setting(base, ER, 'backward', ['validation'])
        for settings in [dict(replay_order='backward'), dict(replay_schedule='3to1at30000', replay_order='backward'),
                         dict(replay_schedule='3to1at40000', replay_order='independent'),
                         dict(replay_schedule='constant', replay_order='independent'),
                         dict(replay_schedule='constant', replay_order='backward', relabeled_proportion=0.0)]:
            altered = copy.deepcopy(config)
            altered['agent_settings'][ER] = settings
            with self.assertRaises(ValueError):
                registered_axes(altered, ER)

    def test_pass_diagnostics_reconcile_exact_volume_and_cannot_be_dropped(self):
        for agent, mode in [(SHORTENED, 'backward'), (SHORTENED, 'random-pass'), (ER, 'backward')]:
            config, rows = self.rows[agent, mode]
            row = rows[-1]
            passes = row['replay_diagnostics']['passes']
            self.assertEqual(passes['emitted_steps'], row['replay_updates'])
            for field, value in [('passes', None), ('emitted_steps', 0), ('requested_relabel', -1)]:
                altered = copy.deepcopy(row)
                if field == 'passes':
                    del altered['replay_diagnostics']['passes']
                else:
                    altered['replay_diagnostics']['passes'][field] = value
                with self.assertRaises(ValueError):
                    validate_row(altered, config, agent, 42, row['commit'], row['nominal_step'])


if __name__ == '__main__':
    unittest.main()
