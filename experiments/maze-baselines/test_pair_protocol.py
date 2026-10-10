from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest

from register_pairs import BLOCKS, ROOT, build_protocol, partition


class PairProtocolTests(unittest.TestCase):
    def setUp(self):
        self.entries = {entry['id']: entry for entry in json.loads((ROOT / 'fixtures/maze_knowledge.json').read_text())['mazes']}
        self.protocol = json.loads(Path(__file__).with_name('pair-protocol.json').read_text())

    def test_geometry_registration_is_reproducible_and_preserves_goal_protocol(self):
        self.assertEqual(build_protocol(), self.protocol)
        self.assertEqual(hashlib.sha256(Path(__file__).with_name('protocol.json').read_bytes()).hexdigest(),
                         '7baac478cbf6f7d4c2ab86d2784ce63c60399817fc22f8d20cee2bddcb9b8eff')

    def test_every_goal_has_near_and_far_starts_in_all_disjoint_blocks(self):
        for config in self.protocol['configurations']:
            entry = self.entries[config['geometry']]
            cells = [tuple(cell) for cell in entry['walkable_cells']]
            pools, strata = partition(entry)
            seen = set()
            for block in BLOCKS:
                registered = config['blocks'][block]
                goals = {tuple(item['goal']): {tuple(start) for start in item['starts']}
                         for item in registered['goal_starts']}
                self.assertEqual(set(goals), set(cells))
                self.assertGreaterEqual(registered['evaluation_pairs'], 40)
                self.assertEqual(sum(map(len, goals.values())), registered['evaluation_pairs'])
                self.assertEqual(Fraction(registered['proportion']),
                                 Fraction(registered['evaluation_pairs'], len(cells) * (len(cells) - 1)))
                for goal, starts in goals.items():
                    self.assertEqual(starts, set(pools[block][goal]))
                    self.assertGreaterEqual(len(starts), 3)
                    self.assertEqual(len(starts), registered['starts_per_goal'])
                    self.assertNotIn(goal, starts)
                    for stratum in strata[goal]:
                        self.assertTrue(starts & set(stratum))
                    distances = [entry['distances'][cells.index(start)][cells.index(goal)] for start in starts]
                    self.assertLess(min(distances), max(distances))
                    for start in starts:
                        self.assertNotIn((start, goal), seen)
                        seen.add((start, goal))
            self.assertEqual(seen, {(start, goal) for start in cells for goal in cells if start != goal})

    def test_counts_caps_and_beyond_cap_pairs_are_pinned(self):
        self.assertEqual([config['cap'] for config in self.protocol['configurations']], [5, 8, 10, 10, 18, 9])
        self.assertEqual([[config['blocks'][block]['evaluation_pairs'] for block in BLOCKS]
                          for config in self.protocol['configurations']],
                         [[378, 162, 162], [888, 222, 222], [888, 222, 222],
                          [828, 216, 216], [228, 57, 57], [120, 45, 45]])
        exceptions = []
        for config in self.protocol['configurations']:
            for block, registered in config['blocks'].items():
                count = registered['evaluation_pairs']
                beyond = registered['beyond_cap_pairs']
                self.assertLess(Fraction(registered['random_floor_exact']), Fraction(1, 5))
                self.assertEqual(float(Fraction(registered['random_floor_exact'])), registered['random_floor'])
                self.assertEqual(Fraction(registered['reachable_within_cap_exact']), Fraction(count - len(beyond), count))
                exceptions.extend((config['task'], block, item) for item in beyond)
        self.assertEqual(len(exceptions), 2)
        self.assertTrue(all(task == 'maze4' and item['distance'] == 6 for task, _, item in exceptions))


if __name__ == '__main__':
    unittest.main()
