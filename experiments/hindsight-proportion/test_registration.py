import hashlib
import json
from pathlib import Path
import unittest

from protocol import HER, SHORTENED, candidates, study
from execution import derive, measurement_identity
from submit import run_batches


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parent
        cls.plan = json.loads((cls.directory / 'plan.json').read_text())

    def test_plan_reproduces_scientific_choices_and_both_cost_forecasts(self):
        evidence = self.plan['cost_pilot']
        cells = {(cell['configuration_id'], cell['arm']): {key: value for key, value in cell.items()
                  if key not in ('configuration_id', 'arm')} for cell in evidence['cells']}
        self.assertEqual(self.plan, study(cells, evidence['allocation_cpu_seconds'], measurement_identity(), evidence, self.plan['candidates']))
        self.assertEqual(self.plan['candidates'], candidates(self.plan['candidates']))
        self.assertEqual(self.plan['stage'], 'grid_pending_approval')
        self.assertEqual(evidence['rows'], 245)
        self.assertEqual(evidence['identity_rows'], 49)
        self.assertLessEqual(evidence['allocation_cpu_seconds'], 18000)

    def test_generator_and_new_fixture_bytes_match_registration(self):
        for name, checksum in self.plan['cost_pilot']['generators'].items():
            self.assertEqual(hashlib.sha256((self.directory / name).read_bytes()).hexdigest(), checksum)
        fixture_path = self.directory.parents[1] / 'fixtures/hindsight_proportion_regression.json'
        self.assertEqual(hashlib.sha256(fixture_path.read_bytes()).hexdigest(), self.plan['cost_pilot']['fixture_sha256'])
        fixture = json.loads(fixture_path.read_text())
        self.assertEqual(len(fixture['configurations']), 24)
        self.assertEqual(sum(len(config['rows']) for config in fixture['configurations']), 48)
        self.assertEqual(fixture['measurement_commit'], self.plan['measurement_commit'])
        self.assertEqual(fixture['binary_sha256'], self.plan['binary_sha256'])

    def test_forecast_reservations_cover_derived_tuning_control_and_worst_test_manifests(self):
        forecast = self.plan['forecasts']['with_four_fifths']
        reservation = 0
        for purpose in ('tuning', 'control'):
            manifest = derive(self.plan, purpose)
            reservation += sum(60 * len(batch) * max(config['limits_minutes'][agent] for config, agent, _ in batch)
                               for batch in run_batches(manifest))
        for config in self.plan['configurations']:
            possible = []
            for p in self.plan['candidates']:
                if p == 2 / 3:
                    continue
                manifest = derive(self.plan, 'test', dict(selections={config['id']: dict(proportion=p)}), config['id'])
                possible.append(sum(60 * len(batch) * max(item['limits_minutes'][agent] for item, agent, _ in batch)
                                    for batch in run_batches(manifest)))
            reservation += max(possible, default=0)
        self.assertEqual(forecast['hard_reservations_cpu_hours'], (reservation + self.plan['cost_pilot']['allocation_cpu_seconds']) / 3600)
        for value in self.plan['forecasts'].values():
            self.assertGreaterEqual(value['requested_cpu_hour_limit'], value['hard_reservations_cpu_hours'])
            self.assertGreaterEqual(value['requested_cpu_hour_limit'], value['doubled_forecast_cpu_hours'])

    def test_registered_control_and_tuning_use_only_their_own_arm_and_blocks(self):
        tuning = derive(self.plan, 'tuning')
        control = derive(self.plan, 'control')
        self.assertEqual(tuning['agents'], [SHORTENED])
        self.assertEqual(control['agents'], [HER])
        self.assertEqual(tuning['seeds'], list(range(42, 47)))
        self.assertEqual(control['seeds'], list(range(42, 62)))
        self.assertTrue(all(config['evaluation_blocks'] == ['validation'] for config in tuning['configurations']))
        self.assertTrue(all(config['agent_settings'][HER]['relabeled_proportion'] == 0.5 for config in control['configurations']))
        self.assertFalse((self.directory / 'approval.json').exists())


if __name__ == '__main__':
    unittest.main()
