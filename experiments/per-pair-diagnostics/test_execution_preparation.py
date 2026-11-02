import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare_execution import approved, freeze_operations, OPERATIONS_COMMIT


class PreparationTests(unittest.TestCase):
    def test_approval_pins_the_plan_limit_and_original_operations(self):
        plan, approval = approved()
        self.assertEqual(approval['phase_cpu_hour_limit'], 15)
        self.assertEqual(plan['cost_pilot']['allocation_cpu_seconds'], 3672)
        self.assertEqual(approval['operations_commit'], OPERATIONS_COMMIT)

    def test_freeze_preserves_every_existing_operation_and_only_adds_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'base'
            base.mkdir()
            (base / 'operation.py').write_bytes(b'unchanged operation\n')
            files = {'operation.py': hashlib.sha256((base / 'operation.py').read_bytes()).hexdigest()}
            (base / 'manifest.json').write_text(json.dumps(dict(commit=OPERATIONS_COMMIT, files=files)))
            plan, approval = root / 'plan.json', root / 'approval.json'
            plan.write_text('{}\n')
            approval.write_text('{"approved": true}\n')
            count, additions = freeze_operations(base, root / 'copy', plan, approval)
            self.assertEqual(count, 1)
            self.assertEqual(len(additions), 2)
            self.assertEqual((root / 'copy/operation.py').read_bytes(), (base / 'operation.py').read_bytes())
            manifest = json.loads((root / 'copy/manifest.json').read_text())
            self.assertEqual(manifest['files']['operation.py'], files['operation.py'])
            (base / 'operation.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                freeze_operations(base, root / 'bad', plan, approval)


if __name__ == '__main__':
    unittest.main()
