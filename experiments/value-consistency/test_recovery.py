import json
from pathlib import Path
import tempfile
import unittest

from retry_batch import approved_recovery


class RecoveryTests(unittest.TestCase):
    def test_only_the_exact_approved_seven_seed_recovery_is_authorized(self):
        directory = Path(__file__).parent
        proposal = directory / 'recovery.json'
        approval = directory / 'recovery-approval.json'
        value = approved_recovery(proposal, approval)
        self.assertEqual(value['recovery']['seeds'], [44, 46, 49, 52, 53, 59, 61])
        self.assertEqual(value['accounting']['whole_study_upper_bound_seconds'], 391875)
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / 'altered.json'
            altered.write_text(proposal.read_text().replace('600', '900'))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                approved_recovery(altered, approval)
            changed = json.loads(approval.read_text())
            for key, substitute in [('approved', False), ('phase_cpu_hour_limit', 115), ('plan_sha256', 'foreign')]:
                altered.write_text(json.dumps(dict(changed, **{key: substitute})))
                with self.assertRaisesRegex(ValueError, 'exact approved'):
                    approved_recovery(proposal, altered)


if __name__ == '__main__':
    unittest.main()
