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
