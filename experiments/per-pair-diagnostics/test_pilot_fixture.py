import json
import unittest

from protocol import ARMS, BLOCKS, ROOT, configurations
from pair_schema import PAIR_FIELDS


class PilotFixtureTests(unittest.TestCase):
    def test_fixture_pins_thirty_runs_all_blocks_and_six_complete_distance_rows(self):
        fixture = json.loads((ROOT / 'fixtures/per_pair_diagnostics_regression.json').read_text())
        configs = configurations()
        expected = {(config['task'], arm) for config in configs for arm in ARMS}
        self.assertEqual({(item['task'], item['agent']) for item in fixture['configurations']}, expected)
        self.assertEqual(len(fixture['configurations']), 30)
        for item in fixture['configurations']:
            self.assertEqual(item['nominal_step'], 1000)
            self.assertEqual(set(item['records']), set(BLOCKS))
            for block, records in item['records'].items():
                self.assertEqual(len(records), 2)
                self.assertEqual(records[0]['index'], 0)
                self.assertEqual(records[-1]['index'] + 1, item['shared_fields']['evaluation_blocks'][block]['evaluation_pairs'])
                for sample in records:
                    self.assertEqual(set(sample['record']), PAIR_FIELDS)
        self.assertEqual({item['task'] for item in fixture['distance_rows']}, {config['task'] for config in configs})
        for item in fixture['distance_rows']:
            config = next(config for config in configs if config['task'] == item['task'])
            for block, distances in item['distances'].items():
                self.assertEqual(len(distances), config['registration']['blocks'][block]['evaluation_pairs'])


if __name__ == '__main__':
    unittest.main()
