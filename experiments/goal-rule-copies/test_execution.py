import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from execution import authorization_manifest, scope
from protocol import pilot
from grid_authorization import require_derivation
from grid_protocol import atomic_json


class PilotAuthorizationTests(unittest.TestCase):
    def test_pilot_exact_manifest_scope_inputs_and_no_stage2(self):
        identity = dict(measurement_commit='a'*40, binary_sha256='b'*64, preset={})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registered = pilot(identity)
            for name in ('reference-index', 'model-index'):
                atomic_json(root/(name+'.json'), [])
            atomic_json(root/'pilot-registration.json', dict(manifest=registered, destination=str(root), **{
                name+'_sha256': hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest()
                for name in ('reference-index','model-index')}))
            digest = hashlib.sha256((root/'pilot-registration.json').read_bytes()).hexdigest()
            atomic_json(root/'pilot-scope.json', scope(digest,root))
            proposed = dict(stage='authorized_cost_pilot', authorization_plan=str(root/'pilot-registration.json'), authorization_sha256=digest)
            with patch('execution.identity', return_value=identity):
                expected = authorization_manifest(proposed)
                for field, value in [('seeds',[43]), ('agents',['acs2her_shortened']), ('phase_cpu_hour_limit',3),
                        ('binary_sha256','x'*64), ('configurations',registered['configurations'][:-1])]:
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        require_derivation(dict(expected,**{field:value}),expected)
                with self.assertRaisesRegex(ValueError, 'stage2 approval'):
                    authorization_manifest(dict(proposed,stage='approved_grid'),root/'plan.json')
                (root/'reference-index.json').write_text('[1]')
                with self.assertRaisesRegex(ValueError, 'input index changed'):
                    authorization_manifest(proposed)
