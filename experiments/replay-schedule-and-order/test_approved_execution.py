import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import approved_execution as execution
from protocol import ROOT


class ApprovedExecutionTests(unittest.TestCase):
    def approval(self, root):
        directory = ROOT / 'experiments/replay-schedule-and-order'
        (root / 'approved-plan.json').write_bytes((directory / 'plan.json').read_bytes())
        (root / 'approval.json').write_bytes((directory / 'approval.json').read_bytes())
        return root / 'approved-plan.json'

    def test_exact_approval_and_frozen_measurement_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.approval(root)
            self.assertEqual(execution.approved(path)['cost_pilot']['allocation_cpu_seconds'], 8147)
            valid = json.loads((root / 'approval.json').read_text())
            for key, value in [('phase_cpu_hour_limit', 61), ('pilot_allocation_cpu_seconds', 0),
                               ('approved', False), ('plan_sha256', 'a' * 64)]:
                changed = dict(valid, **{key: value})
                (root / 'approval.json').write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    execution.approved(path)
            (root / 'approval.json').write_text(json.dumps(valid))
            with patch.object(execution, 'measurement_identity', return_value=dict(binary_sha256='b' * 64)):
                with self.assertRaisesRegex(ValueError, 'frozen measurement'):
                    execution.approved(path)

    def test_catalog_cache_is_scoped_to_each_fresh_derivation_and_restores_on_failure(self):
        with patch.object(execution.archive, 'catalog', side_effect=[('first', {}), ('second', {})]) as reader:
            with execution.catalog_cache():
                self.assertEqual(execution.archive.catalog('/same'), execution.archive.catalog('/same'))
            with self.assertRaisesRegex(ValueError, 'stop'):
                with execution.catalog_cache():
                    self.assertEqual(execution.archive.catalog('/same')[0], 'second')
                    raise ValueError('stop')
            self.assertIs(execution.archive.catalog, reader)
            self.assertEqual(reader.call_count, 2)

    def test_every_command_rederives_one_manifest_and_dispatches_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [dict(stage='approved_grid', selection='first'), dict(stage='approved_grid', selection='second')]
            with patch.object(execution, 'manifest', side_effect=copy.deepcopy(values)) as derive, \
                 patch.object(execution, 'dispatch', return_value=dict(submitted=True)) as dispatch:
                for expected in values:
                    execution.execute(root, Path('/measurement'), 'test', 'maze4_c5_pairs', 'C', 'er')
                    saved = json.loads((root / 'manifests/test-C-maze4_c5_pairs-er.json').read_text())
                    self.assertEqual(saved, expected)
                self.assertEqual(derive.call_count, 2)
                self.assertEqual(dispatch.call_count, 2)
                self.assertEqual(dispatch.call_args.args[:2],
                    (root / 'test-C-maze4_c5_pairs-er', root / 'manifests/test-C-maze4_c5_pairs-er.json'))


if __name__ == '__main__':
    unittest.main()
