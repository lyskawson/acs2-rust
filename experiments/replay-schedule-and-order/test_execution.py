import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import execution


class ExecutionTests(unittest.TestCase):
    def test_saved_selection_requires_its_mechanically_registered_screening_manifest(self):
        expected = dict(stage='tuning_pending_approval', purpose='screening', agents=['acs2her_shortened'],
            seeds=list(range(42, 47)), configurations=[dict(id='registered',
                agent_settings=dict(acs2her_shortened=dict(relabeled_proportion=0.5,
                    replay_schedule='3to1at30000', replay_order='independent')))])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'screening').mkdir()
            def save(manifest):
                data = (json.dumps(manifest) + '\n').encode()
                (root / 'screening/plan.json').write_bytes(data)
                (root / 'screening/plan.sha256').write_text(hashlib.sha256(data).hexdigest())
            save(expected)
            with patch.object(execution, 'authorization_manifest', return_value=expected):
                self.assertEqual(execution.verified_screening(root), expected)
                for kind in ('seed', 'schedule', 'proportion'):
                    changed = copy.deepcopy(expected)
                    if kind == 'seed':
                        changed['seeds'][-1] = 61
                    else:
                        settings = changed['configurations'][0]['agent_settings']['acs2her_shortened']
                        settings['replay_schedule' if kind == 'schedule' else 'relabeled_proportion'] = '3to1at100000' if kind == 'schedule' else 0.8
                    save(changed)
                    with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'mechanical registration'):
                        execution.verified_screening(root)


if __name__ == '__main__':
    unittest.main()
