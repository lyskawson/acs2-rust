import unittest
from unittest.mock import patch

from execution import authorization_manifest
from protocol import pilot
from grid_authorization import require_derivation


class ExecutionTests(unittest.TestCase):
    def test_pilot_manifest_is_derived_from_the_fixed_protocol(self):
        with patch('execution.measurement_identity', return_value=dict(measurement_commit='a' * 40, binary_sha256='b' * 64)):
            expected = authorization_manifest(pilot())
        require_derivation(expected, expected)
        for key, value in [('phase_cpu_hour_limit', 3), ('seeds', [42, 43]), ('agents', ['acs2her']), ('unknown', True)]:
            manifest = dict(expected, **{key: value})
            with self.assertRaises(ValueError):
                require_derivation(manifest, expected)

    def test_stage_one_adapter_cannot_authorize_a_grid(self):
        with self.assertRaises(ValueError):
            authorization_manifest(dict(stage='approved_grid'), 'unapproved-plan.json')


if __name__ == '__main__':
    unittest.main()
