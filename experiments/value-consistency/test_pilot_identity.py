import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from grid_identity import verify_references
from pilot_identity import compare_rows
import test_measurement as measurement_data


class PilotIdentityTests(unittest.TestCase):
    def test_node_comparison_checks_full_overlapping_rows_and_refuses_missing_points_or_changed_knowledge(self):
        helper = measurement_data.MeasurementTests()
        helper.setUp()
        for agent in ['acs2vcp_k2', 'acs2vcp_k4_uniform']:
            old = [row for row in helper.rows if row['task'] == 'maze4' and row['agent'] == agent][:2]
            revised = [helper.revised_row(row) for row in old]
            targets = [row['nominal_step'] for row in old]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'pilot.jsonl'
                path.write_text(''.join(json.dumps(row) + '\n' for row in old))
                key = 'maze4_c5_pairs|' + agent + '|42'
                reference = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 targets=targets, blocks=['validation'], comparison='vcp_pilot')
                plan = dict(identity_references={key: [reference]})
                entry = dict(configuration_id='maze4_c5_pairs', agent=agent, seed=42)
                result = verify_references(plan, entry, revised)
                self.assertEqual(result[0]['rows'], 2)
                self.assertEqual(result[0]['differences'], 0)
                changed = copy.deepcopy(revised)
                changed[-1]['knowledge'] += 0.01
                with self.assertRaisesRegex(ValueError, 'knowledge'):
                    verify_references(plan, entry, changed)
                with self.assertRaisesRegex(ValueError, 'checkpoint'):
                    compare_rows(revised[:-1], old, 42, targets)
                with self.assertRaisesRegex(ValueError, 'seed 42'):
                    compare_rows(revised, old, 43, targets)


if __name__ == '__main__':
    unittest.main()
