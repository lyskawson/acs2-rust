import copy
import json
from pathlib import Path
import tempfile
import unittest

from execution import authorization_manifest


class RegistrationIdentityTests(unittest.TestCase):
    def test_external_registration_is_compared_with_the_committed_pair_registration(self):
        registered = json.loads(Path(__file__).with_name('grid-plan.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'registration.json'
            path.write_text(json.dumps(registered))
            self.assertEqual(authorization_manifest(registered, path), registered)
            substituted = copy.deepcopy(registered)
            substituted['configurations'][0]['budget_steps'] += 1
            path.write_text(json.dumps(substituted))
            with self.assertRaisesRegex(ValueError, 'registered pair grid'):
                authorization_manifest(substituted, path)


if __name__ == '__main__':
    unittest.main()
