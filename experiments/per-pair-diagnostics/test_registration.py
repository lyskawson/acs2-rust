import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from protocol import ARMS, BLOCKS, analysis_registration, configurations
from execution import authorization_manifest, measurement_identity
from grid_protocol import load_plan

REGISTERED_SHA256 = 'ec83bffbba7ca0d14315138abf03d6b485900d6f3c63dcfa0172ebe74a53d8d5'


class RegistrationTests(unittest.TestCase):
    def test_single_plan_pins_identity_runs_analysis_and_hard_reservations(self):
        path = Path(__file__).with_name('plan.json')
        plan = load_plan(path, REGISTERED_SHA256)
        self.assertEqual(plan['configurations'], configurations())
        self.assertEqual(plan['agents'], list(ARMS))
        self.assertEqual(plan['seeds'], [43, 44, 45, 46])
        self.assertEqual(plan['runs'], 120)
        self.assertEqual(plan['evaluation_blocks'], BLOCKS)
        self.assertEqual(plan['analysis'], analysis_registration())
        self.assertTrue(plan['per_start_diagnostics'])
        for name, value in measurement_identity().items():
            self.assertEqual(plan[name], value)
        forecast = plan['forecast']
        self.assertEqual(forecast['requested_cpu_hour_limit'], 15)
        self.assertGreaterEqual(15, forecast['hard_reservations_cpu_hours'])
        self.assertEqual(forecast['doubled_forecast_cpu_hours'], 9.18)
        self.assertEqual(plan['cost_pilot']['identity_rows'], 245)

    def test_unapproved_or_different_plan_cannot_authorize_replications(self):
        content = Path(__file__).with_name('plan.json').read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registered = root / 'approved-plan.json'
            registered.write_bytes(content)
            proposed = dict(stage='approved_grid', purpose='replication', authorization_plan=str(registered),
                            authorization_sha256=REGISTERED_SHA256)
            for approval in [dict(approved=False, plan_sha256=REGISTERED_SHA256, phase_cpu_hour_limit=15),
                             dict(approved=True, plan_sha256='a' * 64, phase_cpu_hour_limit=15),
                             dict(approved=True, plan_sha256=REGISTERED_SHA256, phase_cpu_hour_limit=-1)]:
                (root / 'approval.json').write_text(json.dumps(approval))
                with self.assertRaisesRegex(ValueError, 'explicit plan'):
                    authorization_manifest(proposed, registered)
            changed = json.loads(content)
            changed['seeds'] = [43]
            registered.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                authorization_manifest(proposed, registered)


if __name__ == '__main__':
    unittest.main()
