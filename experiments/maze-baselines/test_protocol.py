from fractions import Fraction
import json
from pathlib import Path
import unittest

from register_protocol import BLOCKS, ROOT, build_protocol, partition


class GoalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.entries = {entry['id']: entry for entry in json.loads((ROOT / 'fixtures/maze_knowledge.json').read_text())['mazes']}
        self.protocol = json.loads(Path(__file__).with_name('protocol.json').read_text())

    def test_registration_is_reproducible_without_learning_data(self):
        regenerated = build_protocol()
        self.assertEqual(regenerated, self.protocol)

    def test_every_block_covers_easy_and_hard_goals_with_disjoint_complete_pools(self):
        for config in self.protocol['configurations']:
            entry = self.entries[config['geometry']]
            all_cells = {tuple(cell) for cell in entry['walkable_cells']}
            pools, strata, sums = partition(entry)
            seen = set()
            for block in BLOCKS:
                cells = {tuple(cell) for cell in config['blocks'][block]['goals']}
                self.assertFalse(seen & cells)
                seen |= cells
                self.assertGreaterEqual(len(cells), 3)
                self.assertGreaterEqual(config['blocks'][block]['evaluation_pairs'], 40)
                for stratum in strata:
                    self.assertTrue(cells & set(stratum))
                self.assertLess(min(sums[cell] for cell in cells), max(sums[cell] for cell in cells))
                self.assertEqual(cells, set(pools[block]))
            self.assertEqual(seen, all_cells)

    def test_caps_preserve_sparse_floors_and_pin_the_only_unreachable_pairs(self):
        self.assertEqual([config['cap'] for config in self.protocol['configurations']], [5, 8, 10, 10, 18, 9])
        exceptions = []
        for config in self.protocol['configurations']:
            for name, block in config['blocks'].items():
                self.assertLess(Fraction(block['random_floor_exact']), Fraction(1, 5))
                self.assertEqual(float(Fraction(block['random_floor_exact'])), block['random_floor'])
                exceptions.extend((config['task'], name, pair) for pair in block['beyond_cap_pairs'])
        self.assertEqual(len(exceptions), 2)
        self.assertTrue(all(task == 'maze4' and pair['distance'] == 6 for task, _, pair in exceptions))


if __name__ == '__main__':
    unittest.main()
