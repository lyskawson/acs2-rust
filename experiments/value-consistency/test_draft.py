import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

from execution import measurement_identity, pilot_measurement_identity
from protocol import AGENTS, PRIORITIZED, ROOT

sys.path.append(str(ROOT / 'experiments/grid'))
from submit import run_batches


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(Path(__file__).with_name('plan-draft.json').read_text())

    def test_reviewed_artifact_has_all_four_fixed_arms_nine_contrasts_and_no_tuning(self):
        plan = self.plan
        self.assertEqual(plan['agents'], AGENTS)
        self.assertEqual(plan['seeds'], list(range(42, 62)))
        self.assertEqual(plan['final_grid']['runs'], 480)
        self.assertEqual(plan['final_grid']['evaluation_blocks'], ['training', 'validation', 'test'])
        self.assertEqual(plan['temperature']['prioritized'], 24)
        self.assertEqual(plan['temperature']['controls'], 0)
        self.assertIs(plan['temperature']['tuning'], False)
        self.assertNotIn('tuning', plan)
        self.assertNotIn('cost_reduction', plan)
        self.assertIs(plan['stage2_authorized'], False)
        self.assertTrue(plan['hindsight']['confirmation_required_before_registration'])
        expected = [(AGENTS[0], AGENTS[2]), (AGENTS[1], AGENTS[3]),
                    (AGENTS[0], 'acs2her'), (AGENTS[1], 'acs2her'),
                    (AGENTS[2], 'acs2her'), (AGENTS[3], 'acs2her'),
                    (AGENTS[0], 'acs2er_episodic'), (AGENTS[1], 'acs2er_episodic'),
                    (AGENTS[1], AGENTS[0])]
        self.assertEqual(list(map(tuple, plan['analysis']['comparisons'])), expected)
        self.assertEqual(plan['analysis']['views'], ['checkpoint_mean_within_seed', 'final_point'])
        self.assertEqual(len(plan['configurations']), 6)
        for config in plan['configurations']:
            self.assertEqual(config['agent_settings'], {agent: dict(temperature=24) for agent in PRIORITIZED})
            self.assertEqual(config['evaluation_blocks'], ['training', 'validation', 'test'])
        her = json.loads((ROOT / 'experiments/her-variants/plan.json').read_text())
        for field in ['thresholds', 'baseline_threshold_crossing_checks']:
            self.assertEqual(plan['analysis'][field], her['analysis'][field])

    def test_forecast_uses_frozen_calibration_and_real_batches_with_conservative_maze6_limits(self):
        plan, prediction = self.plan, self.plan['forecast']
        identity = measurement_identity()
        for field in ['measurement_commit', 'binary_sha256', 'preset']:
            self.assertEqual(plan[field], identity[field])
        self.assertNotEqual(identity['measurement_commit'], pilot_measurement_identity()['measurement_commit'])
        calibration = Path(__file__).with_name('control-work-calibration.json')
        self.assertEqual(hashlib.sha256(calibration.read_bytes()).hexdigest(), plan['cost_pilot']['control_work_calibration_sha256'])
        self.assertEqual(json.loads(calibration.read_text())['measurement_commit'], identity['measurement_commit'])
        for name, digest in plan['cost_pilot']['generator_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(), digest)
        batches = run_batches(plan)
        self.assertEqual(len(batches), 24)
        hard = sum(len(batch) * 60 * max(config['limits_minutes'][agent] for config, agent, _ in batch) for batch in batches)
        self.assertEqual(hard, prediction['grid_reservation_cpu_seconds'])
        self.assertEqual(hard, sum(batch['reservation_cpu_seconds'] for batch in prediction['grid_batches']))
        jobs = {(config['id'], agent, seed) for batch in batches for config, agent, seed in batch}
        forecast_jobs = {(job['configuration_id'], job['agent'], job['seed']) for batch in prediction['grid_batches'] for job in batch['jobs']}
        self.assertEqual(jobs, forecast_jobs)
        self.assertEqual(len(jobs), 480)
        cells = prediction['cells']
        self.assertEqual(len(cells), 24)
        expected = 20 * sum(cell['expected_allocation_seconds'] for cell in cells)
        self.assertAlmostEqual(expected / 3600, prediction['grid_expected_cpu_hours'])
        self.assertAlmostEqual((16041 + 2 * expected) / 3600, prediction['doubled_forecast_with_pilots_cpu_hours'])
        self.assertAlmostEqual((16041 + hard) / 3600, prediction['hard_bound_with_pilots_cpu_hours'])
        self.assertEqual(plan['requested_cpu_hour_limit'], 5 * math.ceil(max(prediction['doubled_forecast_with_pilots_cpu_hours'], prediction['hard_bound_with_pilots_cpu_hours']) / 5))
        for cell in cells:
            if cell['temperature'] == 0:
                calibration = cell['calibration']
                self.assertEqual(cell['scale'], calibration['revised_classifier_tests'] / calibration['pilot_classifier_tests'])
                self.assertEqual(calibration['pilot_classifier_tests'] - calibration['revised_classifier_tests'], calibration['avoided_classifier_tests'])
                self.assertGreaterEqual(cell['limit_minutes'], cell['pilot_derived_limit_minutes'])
            elif cell['task'] == 'maze6':
                ratio = 68 / 58 if cell['agent'] == AGENTS[0] else 296 / 195
                self.assertEqual(cell['scale'], ratio)
                self.assertGreaterEqual(cell['limit_minutes'] * 60, 1.5 * cell['expected_body_seconds'] + 30)


if __name__ == '__main__':
    unittest.main()
