import json
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline.compute_random_policy import maze_probabilities

TASKS = [('maze4', 'Maze4-v0'), ('maze5', 'Maze5-v0'), ('maze6', 'Maze6-v0'),
         ('maze7', 'Maze7-v0'), ('woods14', 'Woods14-v0'), ('mazema', 'MazeMA-v0')]
BLOCKS = ('training', 'validation', 'test')
LEGACY_CAPS = {'maze4': 5, 'maze6': 10, 'maze7': 10}


def chunks(values, count):
    quotient, remainder = divmod(len(values), count)
    offset = 0
    result = []
    for index in range(count):
        size = quotient + (index < remainder)
        result.append(values[offset:offset + size])
        offset += size
    return result


def partition(entry):
    cells = [tuple(cell) for cell in entry['walkable_cells']]
    distances = entry['distances']
    sums = {goal: sum(row[index] for row in distances) for index, goal in enumerate(cells)}
    ranked = sorted(cells, key=lambda goal: (sums[goal], *goal))
    strata = chunks(ranked, 3)
    pools = {block: [] for block in BLOCKS}
    for stratum_index, stratum in enumerate(strata):
        holdout = max(1, (len(stratum) + 2) // 5)
        for segment_index, segment in enumerate(chunks(stratum, holdout)):
            first, last = ('validation', 'test') if (stratum_index + segment_index) % 2 == 0 else ('test', 'validation')
            pools[first].append(segment[0])
            pools[last].append(segment[-1])
            pools['training'].extend(segment[1:-1])
    if any(len(pools[block]) < 3 or len(pools[block]) * (len(cells) - 1) < 40 for block in BLOCKS):
        raise ValueError('goal split is too small; register a separate pair-split protocol')
    for pool in pools.values():
        if not all(set(pool).intersection(stratum) for stratum in strata):
            raise ValueError('a block lacks a difficulty stratum')
    return pools, strata, sums


def configuration(task, entry):
    pools, strata, sums = partition(entry)
    cells = [tuple(cell) for cell in entry['walkable_cells']]
    cap = LEGACY_CAPS.get(task, max(max(row) for row in entry['distances']))
    blocks = {}
    for block, pool in pools.items():
        goals = sorted(pool)
        pairs = len(goals) * (len(cells) - 1)
        beyond = [{'start': list(start), 'goal': list(goal), 'distance': entry['distances'][cells.index(start)][cells.index(goal)]}
                  for goal in goals for start in cells
                  if entry['distances'][cells.index(start)][cells.index(goal)] > cap]
        floor = maze_probabilities(entry, [cap], goals=goals)[cap]
        if float(floor) >= 0.2:
            raise ValueError('random floor violates the preregistered sparsity check')
        blocks[block] = {'goals': [list(goal) for goal in goals], 'evaluation_pairs': pairs,
                         'pair_weight': f'1/{pairs}', 'random_floor_exact': str(floor),
                         'random_floor': float(floor), 'reachable_within_cap_exact': str(Fraction(pairs - len(beyond), pairs)),
                         'reachable_within_cap': (pairs - len(beyond)) / pairs,
                         'beyond_cap_pairs': beyond}
    return {'id': f'{task}_c{cap}_split', 'task': task, 'geometry': entry['id'], 'cap': cap,
            'encoding': 'coordinates', 'split_unit': 'goal_cells', 'walkable_cells': len(cells),
            'diameter': max(max(row) for row in entry['distances']),
            'goal_difficulty': [{'goal': list(goal), 'distance_sum': sums[goal],
                                 'mean_distance': sums[goal] / (len(cells) - 1),
                                 'stratum': index}
                                for index, stratum in enumerate(strata) for goal in stratum],
            'blocks': blocks}


def build_protocol():
    entries = json.loads((ROOT / 'fixtures/maze_knowledge.json').read_text())['mazes']
    return {'schema': 1, 'id': 'maze-goals-v1',
            'reference_commit': '79d9f862e35385ba6fbad3a75415c96b9b116500',
            'cap_rule': 'Keep Maze4=5, Maze6=10, Maze7=10. For Maze5, Woods14 and MazeMA use the eight-action graph diameter: the smallest cap admitting every distinct start-goal pair. Every block must have random floor < 0.20. List all beyond-cap pairs, including legacy exceptions.',
            'split_rule': 'Rank goal cells by the integer sum of shortest distances from all other walkable starts, then row and column. Split ranks into three contiguous balanced strata (earlier strata receive remainders). In each stratum use h=max(1,floor((size+2)/5)) balanced contiguous segments. Assign segment endpoints to validation and test, alternating their order with stratum+segment parity; assign interiors to training. Sort final pools by row,column. No seed or learning result enters this rule.',
            'goal_split_criterion': 'Each block has at least three goals and forty evaluation pairs, with at least one goal from each difficulty stratum. Otherwise a separate start-goal-pair protocol requires registration; none of these six mazes needs it.',
            'start_law': 'Uniform over every walkable cell except the desired goal, including the original 9 landmark; desired goals uniform within the block.',
            'evaluation': 'Read-only, separate training/validation/test blocks at every point; exhaustive exact equal pair weights, no adaptation from test diagnostics.',
            'configurations': [configuration(task, next(entry for entry in entries if entry['id'] == geometry))
                               for task, geometry in TASKS]}


if __name__ == '__main__':
    result = build_protocol()
    Path(__file__).with_name('protocol.json').write_text(json.dumps(result, indent=2) + '\n')
