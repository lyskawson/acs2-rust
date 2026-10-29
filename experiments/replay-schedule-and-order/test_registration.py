import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import execution
from execution import measurement_identity
from protocol import ROOT, study
from grid_identity import OMIT


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(Path(__file__).with_name('plan.json').read_text())

    def test_registered_plan_reproduces_the_exact_pilot_forecast_and_scientific_protocol(self):
        evidence = self.plan['cost_pilot']
        cells = {(cell['configuration_id'], cell['arm']): {
            key: value for key, value in cell.items() if key not in ('configuration_id', 'arm')}
            for cell in evidence['cells']}
        expected = study(cells, evidence['allocation_cpu_seconds'], measurement_identity(), evidence)
        self.assertEqual(self.plan, expected)
        self.assertEqual(self.plan['stage'], 'grid_pending_approval')
        self.assertEqual(self.plan['screening']['runs'], 120)
        self.assertEqual(self.plan['test']['maximum_runs'], 600)
        self.assertEqual(self.plan['analysis']['paired_interval']['critical_value'], 2.093024054408263)
        self.assertEqual(evidence['runs'], 60)
        self.assertEqual(evidence['rows'], 490)
        self.assertEqual(evidence['identity_rows'], 49)
        self.assertEqual(evidence['allocations'], 84)
        self.assertEqual([item['allocations'] for item in evidence['rounds']], [60, 24])
        self.assertEqual(sum(item['allocation_cpu_seconds'] for item in evidence['rounds']), evidence['allocation_cpu_seconds'])
        self.assertLessEqual(evidence['allocation_cpu_seconds'], 12 * 3600)
        for name, digest in evidence['generators'].items():
            self.assertEqual(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(), digest)

    def test_new_fixture_pins_every_new_arm_without_changing_default_row_fields(self):
        path = ROOT / 'fixtures/replay_schedule_and_order_regression.json'
        evidence = self.plan['cost_pilot']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), evidence['fixture_sha256'])
        fixture = json.loads(path.read_text())
        self.assertEqual(set(fixture['excluded']), OMIT)
        self.assertEqual(len(fixture['configurations']), 54)
        self.assertEqual(sum(len(config['rows']) for config in fixture['configurations']), 108)
        schedules = [config for config in fixture['configurations'] if config['mode'].startswith('3to1')]
        digest = hashlib.sha256(json.dumps(schedules, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, '49e0e0def3609801a5246adf9e502ffddfb2c85617e98944298351dcc4d9942e')
        for config in fixture['configurations']:
            self.assertNotEqual(config['mode'], 'base')
            self.assertEqual([row['nominal_step'] for row in config['rows']], [1000, 3000])
            for row in config['rows']:
                self.assertEqual(set(row['evaluation_blocks']), {'training', 'validation', 'test'})
                self.assertEqual(row['agent'], config['agent'])
                self.assertIn('replay_order', row['agent_parameters'])
                self.assertIn('replay_schedule', row['agent_parameters'])

    def test_approval_cannot_replace_the_registered_pilot_source_or_binary(self):
        registered = Path(__file__).with_name('plan.json').read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'approved-plan.json'
            path.write_bytes(registered)
            (root / 'approval.json').write_text(json.dumps(dict(approved=True,
                plan_sha256=hashlib.sha256(registered).hexdigest(), phase_cpu_hour_limit=100)))
            for field, value in [('measurement_commit', 'a' * 40), ('binary_sha256', 'b' * 64), ('preset', {})]:
                identity = dict(measurement_identity(), **{field: value})
                with self.subTest(field=field), patch.object(execution, 'measurement_identity', return_value=identity):
                    with self.assertRaisesRegex(ValueError, 'registered measurement identity differs'):
                        execution.authorization_manifest(dict(stage='approved_grid', purpose='screening',
                            authorization_plan=str(path)), path)


if __name__ == '__main__':
    unittest.main()
