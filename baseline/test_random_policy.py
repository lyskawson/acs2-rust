import json
import unittest
from fractions import Fraction
from itertools import product

from baseline.compute_random_policy import (
    ROOT, bit_flipping_probability, hand_eye_probabilities,
    maze_probabilities, taxi_probabilities,
)


class RandomPolicyOracles(unittest.TestCase):
    def test_bit_flipping_matches_every_action_sequence_at_four_bits(self):
        successes = 0
        for start in range(1, 16):
            for actions in product(range(4), repeat=4):
                state = start
                reached = False
                for action in actions:
                    state ^= 1 << action
                    reached |= state == 0
                successes += reached
        self.assertEqual(bit_flipping_probability(4, 4), Fraction(successes, 15 * 4 ** 4))
        self.assertEqual(bit_flipping_probability(1, 1), 1)

    def test_maze_one_step_matches_the_reference_edges_plus_reward_departures(self):
        entries = json.loads((ROOT / "fixtures/maze_knowledge.json").read_text())["mazes"]
        entry = next(e for e in entries if e["id"] == "Maze4-v0")
        reward = next(cell for cell in entry["walkable_cells"] if entry["grid"][cell[0]][cell[1]] == 9)
        departures = sum(end == reward for _, _, end, _, _ in entry["transitions"])
        expected = Fraction(len(entry["transitions"]) + departures, 27 * 26 * 8)
        self.assertEqual(maze_probabilities(entry, [1])[1], expected)

    def test_research_caps_match_pinned_rational_floors(self):
        entries = {entry['id']: entry for entry in json.loads((ROOT / 'fixtures/maze_knowledge.json').read_text())['mazes']}
        for name, cap, exact in [('Maze5-v0', 8, '925501357/11173625856'),
                                 ('Woods14-v0', 18, '398825337379297111/3080462145121419264'),
                                 ('MazeMA-v0', 9, '29933983/234881024')]:
            self.assertEqual(maze_probabilities(entries[name], [cap])[cap], Fraction(exact))

    def test_restricted_new_maze_floor_matches_every_two_action_sequence(self):
        entries = {entry['id']: entry for entry in json.loads((ROOT / 'fixtures/maze_knowledge.json').read_text())['mazes']}
        offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        for name in ['Woods14-v0', 'MazeMA-v0']:
            entry = entries[name]
            cells = list(map(tuple, entry['walkable_cells']))
            goals = [cells[0], cells[-1]]
            successes = 0
            for goal in goals:
                for start in cells:
                    if start == goal:
                        continue
                    for actions in product(range(8), repeat=2):
                        state = start
                        reached = False
                        for action in actions:
                            dr, dc = offsets[action]
                            target = (state[0] + dr, state[1] + dc)
                            if target in cells:
                                state = target
                            reached |= state == goal
                        successes += reached
            self.assertEqual(maze_probabilities(entry, [2], goals=goals)[2],
                             Fraction(successes, 2 * (len(cells) - 1) * 8 ** 2))

    def test_hand_eye_one_step_uses_the_half_held_distribution(self):
        entry = json.loads((ROOT / "fixtures/hand_eye.json").read_text())["grids"][0]
        moves = sum(before[4] and before[2:4] != after[2:4]
                    for before, _, after, _, _ in entry["probes"])
        self.assertEqual(hand_eye_probabilities(entry, [1])[1], Fraction(moves, 2 * 9 * 8 * 6))

    def test_taxi_cannot_deliver_in_one_step_from_its_initial_distribution(self):
        data = json.loads((ROOT / "fixtures/taxi.json").read_text())
        self.assertEqual(taxi_probabilities(data, [1, 2])[1], 0)
        self.assertEqual(taxi_probabilities(data, [1, 2])[2], 0)

    def test_taxi_first_positive_horizon_counts_two_minimal_sequences(self):
        data = json.loads((ROOT / "fixtures/taxi.json").read_text())
        self.assertEqual(taxi_probabilities(data, [5, 6])[5], 0)
        self.assertEqual(taxi_probabilities(data, [5, 6])[6], Fraction(2, 300 * 6 ** 6))


if __name__ == "__main__":
    unittest.main()
