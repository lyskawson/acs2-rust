import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify import verify_snapshot


class SnapshotTests(unittest.TestCase):
    def test_archived_operations_are_verified_by_content_and_cannot_escape_the_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operations = root / 'operations'
            operations.mkdir()
            source = operations / 'operation.py'
            source.write_text('value = 1\n')
            manifest = operations / 'manifest.json'
            manifest.write_text(json.dumps(dict(commit='frozen', files={'operation.py': hashlib.sha256(source.read_bytes()).hexdigest()})))
            child = dict(operations_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(verify_snapshot(root, child)['commit'], 'frozen')
            source.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                verify_snapshot(root, child)
            manifest.write_text(json.dumps(dict(files={'../outside.py': 'unused'})))
            child['operations_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'outside'):
                verify_snapshot(root, child)


if __name__ == '__main__':
    unittest.main()
