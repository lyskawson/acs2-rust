import argparse
import json
from fractions import Fraction
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
MAZE_CAPS = [1, 2, 5, 10, 20, 25, 50]
HAND_EYE_CAPS = [1, 2, 5, 10, 20, 30, 50, 100]
TAXI_CAPS = [50, 75, 100, 150, 200]


def absorbing_counts(successors, reached, caps):
    actions = len(successors[0])
    assert all(len(row) == actions for row in successors)
    counts = [int(success) for success in reached]
    sequences = 1
    result = {}
    for cap in range(1, max(caps) + 1):
        sequences *= actions
        counts = [sequences if reached[state] else sum(counts[next_state] for next_state in row)
                  for state, row in enumerate(successors)]
        if cap in caps:
            result[cap] = counts.copy()
    return result


def weighted_success(successors, goal_cases, caps):
    actions = len(successors[0])
    numerators = {cap: 0 for cap in caps}
    denominator = 0
    for reached, weights in goal_cases:
        assert not any(success and weight for success, weight in zip(reached, weights))
        counts = absorbing_counts(successors, reached, caps)
        denominator += sum(weights)
        for cap in caps:
            numerators[cap] += sum(count * weight for count, weight in zip(counts[cap], weights))
    return {cap: Fraction(numerators[cap], denominator * actions ** cap) for cap in caps}


def maze_probabilities(entry, caps=MAZE_CAPS):
    cells = [tuple(cell) for cell in entry["walkable_cells"]]
    indices = {cell: i for i, cell in enumerate(cells)}
    successors = [[indices.get((row + dr, col + dc), i) for dr, dc in OFFSETS]
                  for i, (row, col) in enumerate(cells)]
    cases = [([i == goal for i in range(len(cells))],
              [int(i != goal) for i in range(len(cells))]) for goal in range(len(cells))]
    return weighted_success(successors, cases, caps)


def bit_flipping_probability(n, cap):
    counts = [int(distance == 0) for distance in range(n + 1)]
    sequences = 1
    for _ in range(cap):
        sequences *= n
        counts = [sequences if distance == 0 else
                  distance * counts[distance - 1] +
                  ((n - distance) * counts[distance + 1] if distance < n else 0)
                  for distance in range(n + 1)]
    numerator = sum(comb(n, distance) * counts[distance] for distance in range(1, n + 1))
    return Fraction(numerator, (2 ** n - 1) * n ** cap)


def hand_eye_probabilities(entry, caps=HAND_EYE_CAPS):
    side = entry["side"]
    cells = side * side
    states = [tuple(state) for state in entry["states"]]
    indices = {state: i for i, state in enumerate(states)}
    successors = [[None] * 6 for _ in states]
    for before, action, after, _, _ in entry["probes"]:
        successors[indices[tuple(before)]][action] = indices[tuple(after)]
    cases = []
    for goal in range(cells):
        reached = [state[2:4] == (goal % side, goal // side) for state in states]
        weights = [0 if reached[i] else cells if state[4] else 1 for i, state in enumerate(states)]
        assert sum(weights) == 2 * cells * (cells - 1)
        cases.append((reached, weights))
    return weighted_success(successors, cases, caps)


def taxi_probabilities(data, caps=TAXI_CAPS):
    states = [tuple(state) for state in data["states"]]
    indices = {state: i for i, state in enumerate(states)}
    successors = [[None] * 6 for _ in states]
    for before, action, after, _, _ in data["probes"]:
        index = indices[tuple(before[:3])]
        target = indices[tuple(after[:3])]
        assert successors[index][action] in [None, target]
        successors[index][action] = target
    cases = [([state[2] == goal for state in states],
              [int(state[2] < 4 and state[2] != goal) for state in states]) for goal in range(4)]
    return weighted_success(successors, cases, caps)


def row(environment, cap, probability):
    return {"environment": environment, "cap": cap, "exact": str(probability),
            "success_probability": float(probability)}


def calculate():
    mazes = json.loads((ROOT / "fixtures" / "maze_knowledge.json").read_text())["mazes"]
    hands = json.loads((ROOT / "fixtures" / "hand_eye.json").read_text())["grids"]
    taxi = json.loads((ROOT / "fixtures" / "taxi.json").read_text())
    rows = []
    for name in ["Maze4-v0", "Maze5-v0", "Maze6-v0", "Maze7-v0"]:
        entry = next(entry for entry in mazes if entry["id"] == name)
        rows.extend(row(name, cap, probability) for cap, probability in maze_probabilities(entry).items())
    rows.extend(row(f"BitFlipping-{n}", n, bit_flipping_probability(n, n)) for n in range(4, 17))
    for entry in hands:
        rows.extend(row(f"HandEye-{entry['side']}", cap, probability)
                    for cap, probability in hand_eye_probabilities(entry).items())
    rows.extend(row("Taxi", cap, probability) for cap, probability in taxi_probabilities(taxi).items())
    return {"method": "Exact integer absorbing-state dynamic programming; uniform actions, success by the cap; environment-specific start distributions conditioned on start != goal.",
            "hand_eye": "Half held, otherwise uniform independent gripper; a held block at the desired position succeeds without release.",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.out.resolve()
    if output.is_relative_to(ROOT):
        parser.error("one-off analytical tables belong outside the checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = calculate()
    output.write_text(json.dumps(result, indent=2) + "\n")
    for item in result["rows"]:
        print(f"{item['environment']:16} cap={item['cap']:3} success={100 * item['success_probability']:.6f}%")


if __name__ == "__main__":
    main()
