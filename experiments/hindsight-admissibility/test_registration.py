import hashlib
import json
from pathlib import Path
import unittest

from execution import measurement_identity
from protocol import AGENTS, configurations, study
from submit import run_batches


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parent
        cls.plan = json.loads((cls.directory / 'plan.json').read_text())

    def test_registered_plan_reproduces_every_scientific_and_cost_choice(self):
        evidence = self.plan['cost_pilot']
        cells = {(cell['configuration_id'], cell['agent']): {key: value for key, value in cell.items()
                  if key not in ('configuration_id', 'agent')} for cell in evidence['cells']}
        expected = study(cells, evidence['allocation_cpu_seconds'], measurement_identity(), evidence)
        self.assertEqual(self.plan, expected)

    def test_generator_and_fixture_hashes_match_the_registered_bytes(self):
        for name, digest in self.plan['cost_pilot']['generator_sha256'].items():
            self.assertEqual(hashlib.sha256((self.directory / name).read_bytes()).hexdigest(), digest)
        fixture_path = self.directory.parents[1] / 'fixtures/hindsight_admissibility_regression.json'
        self.assertEqual(hashlib.sha256(fixture_path.read_bytes()).hexdigest(), self.plan['cost_pilot']['fixture_sha256'])
        fixture = json.loads(fixture_path.read_text())
        self.assertEqual(fixture['measurement_commit'], self.plan['measurement_commit'])
        self.assertEqual(fixture['binary_sha256'], self.plan['binary_sha256'])
        self.assertEqual(len(fixture['configurations']), 12)
        self.assertEqual(sum(len(config['rows']) for config in fixture['configurations']), 24)

    def test_hard_forecast_uses_all_twelve_actual_twenty_seed_batches(self):
        reservations = 0
        for config in self.plan['configurations']:
            for agent in AGENTS:
                manifest = dict(stage='approved_grid', configurations=[config], agents=[agent], seeds=list(range(42, 62)), batch_size=20)
                batches = run_batches(manifest)
                self.assertEqual(len(batches), 1)
                self.assertEqual(len(batches[0]), 20)
                reservations += 20 * max(item['limits_minutes'][arm] for item, arm, _ in batches[0]) * 60
        forecast = self.plan['forecast']
        self.assertEqual(reservations, forecast['grid_reservation_cpu_seconds'])
        self.assertEqual(forecast['hard_bound_with_pilot_cpu_hours'], (reservations + self.plan['cost_pilot']['allocation_cpu_seconds']) / 3600)
        self.assertGreaterEqual(self.plan['requested_cpu_hour_limit'], forecast['hard_bound_with_pilot_cpu_hours'])
        self.assertGreaterEqual(self.plan['requested_cpu_hour_limit'], forecast['doubled_forecast_with_pilot_cpu_hours'])

    def test_grid_is_pending_and_reuses_all_reference_arms_without_tuning(self):
        self.assertEqual(self.plan['stage'], 'grid_pending_approval')
        self.assertFalse((self.directory / 'approval.json').exists())
        self.assertEqual(self.plan['agents'], AGENTS)
        self.assertEqual(self.plan['runs'], 240)
        self.assertEqual(self.plan['references']['agents'], ['acs2her', 'acs2er_episodic', 'acs2', 'acs2er'])
        self.assertEqual(self.plan['evaluation_blocks'], ['training', 'validation', 'test'])
        self.assertEqual([config['registration'] for config in self.plan['configurations']],
                         [config['registration'] for config in configurations()])


if __name__ == '__main__':
    unittest.main()
