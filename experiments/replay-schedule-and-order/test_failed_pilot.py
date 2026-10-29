import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from failed_pilot import compare_retry
from protocol import BLOCKS, SHORTENED, pilot_roots


class FailedPilotTests(unittest.TestCase):
    def test_retry_matches_every_partial_row_and_refuses_a_binary64_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, failed = base / 'retry', base / 'failed'
            root.mkdir()
            (root / 'pilot-registration.json').write_text(json.dumps(dict(destination='/cluster/retry',
                failed_pilot=dict(root='/cluster/failed', mirror_relative='../failed'))))
            batch = failed / pilot_roots()[-1]
            batch.mkdir(parents=True)
            (batch / 'plan.json').write_text(json.dumps(dict(configurations=[dict(id='registered')])) )
            (batch / 'ledger.json').write_text('[]')
            partial = [dict(seed=42, nominal_step=1000, knowledge=0.5,
                evaluation_blocks={block: dict(success=0.25) for block in BLOCKS})]
            entries = {('registered', SHORTENED, 42): dict(rows=copy.deepcopy(partial))}
            with patch('failed_pilot.attempts', return_value=[dict(configuration_id='registered')]), \
                 patch('failed_pilot.inspect_attempt', return_value=(dict(status='failed'), partial)):
                self.assertEqual(compare_retry(root, entries), 1)
                entries['registered', SHORTENED, 42]['rows'][0]['knowledge'] = 0.5000000000000001
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    compare_retry(root, entries)


if __name__ == '__main__':
    unittest.main()
