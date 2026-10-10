import json
from fractions import Fraction
from pathlib import Path

from register_protocol import BLOCKS, ROOT, TASKS, chunks
from baseline.compute_random_policy import OFFSETS, weighted_success

CAPS = {'maze4': 5, 'maze5': 8, 'maze6': 10, 'maze7': 10, 'woods14': 18, 'mazema': 9}
PROTOCOL_ID = 'maze-pairs-v1'
BUDGET_RULE = 'Two million interactions times ceil(eight-action graph diameter / 10), equal for both agents. This geometry rule is fixed before learning under this protocol: 2M for Maze4, Maze5, Maze6, Maze7 and MazeMA; 4M for Woods14. It is a measurement horizon, not a convergence claim.'


def partition(entry):
    cells = [tuple(cell) for cell in entry['walkable_cells']]
    indices = {cell: index for index, cell in enumerate(cells)}
    pools = {block: {} for block in BLOCKS}
    strata_by_goal = {}
    for goal_index, goal in enumerate(cells):
        ranked = sorted((cell for cell in cells if cell != goal),
                        key=lambda start: (entry['distances'][indices[start]][goal_index], *start))
        strata = chunks(ranked, 3)
        strata_by_goal[goal] = strata
        selected = {block: [] for block in BLOCKS}
        for stratum_index, stratum in enumerate(strata):
            holdout = max(1, (len(stratum) + 2) // 5)
            for segment_index, segment in enumerate(chunks(stratum, holdout)):
                first, last = ('validation', 'test') if (goal_index + stratum_index + segment_index) % 2 == 0 else ('test', 'validation')
                selected[first].append(segment[0])
                selected[last].append(segment[-1])
                selected['training'].extend(segment[1:-1])
        for block, starts in selected.items():
            if len(starts) < 3 or any(not set(starts).intersection(stratum) for stratum in strata):
                raise ValueError('each goal needs three starts and all three distance strata in every block')
            distances = [entry['distances'][indices[start]][goal_index] for start in starts]
            if min(distances) == max(distances):
                raise ValueError('each goal needs distinct near and far distances in every block')
            pools[block][goal] = sorted(starts)
    if any(sum(map(len, pool.values())) < 40 for pool in pools.values()):
        raise ValueError('each block needs at least forty distinct pairs')
    return pools, strata_by_goal


def exact_floor(entry, starts_by_goal, cap):
    cells = [tuple(cell) for cell in entry['walkable_cells']]
    indices = {cell: index for index, cell in enumerate(cells)}
    successors = [[indices.get((row + dr, col + dc), index) for dr, dc in OFFSETS]
                  for index, (row, col) in enumerate(cells)]
    cases = [([cell == goal for cell in cells], [int(cell in starts) for cell in cells])
             for goal, starts in starts_by_goal.items()]
    return weighted_success(successors, cases, [cap])[cap]


def configuration(task, entry):
    pools, strata = partition(entry)
    cells = [tuple(cell) for cell in entry['walkable_cells']]
    indices = {cell: index for index, cell in enumerate(cells)}
    cap = CAPS[task]
    blocks = {}
    for block, pool in pools.items():
        count = sum(map(len, pool.values()))
        beyond = [{'start': list(start), 'goal': list(goal),
                   'distance': entry['distances'][indices[start]][indices[goal]]}
                  for goal, starts in pool.items() for start in starts
                  if entry['distances'][indices[start]][indices[goal]] > cap]
        floor = exact_floor(entry, pool, cap)
        if floor >= Fraction(1, 5):
            raise ValueError('random floor violates the registered sparsity criterion')
        blocks[block] = {
            'goal_starts': [{'goal': list(goal), 'starts': [list(start) for start in starts]}
                            for goal, starts in pool.items()],
            'goals': len(pool), 'starts_per_goal': len(next(iter(pool.values()))),
            'evaluation_pairs': count, 'pair_weight': f'1/{count}',
            'proportion': str(Fraction(count, len(cells) * (len(cells) - 1))),
            'random_floor_exact': str(floor), 'random_floor': float(floor),
            'reachable_within_cap_exact': str(Fraction(count - len(beyond), count)),
            'reachable_within_cap': (count - len(beyond)) / count,
            'beyond_cap_pairs': beyond,
        }
    return {
        'id': f'{task}_c{cap}_pairs', 'task': task, 'geometry': entry['id'], 'cap': cap,
        'encoding': 'coordinates', 'split_unit': 'start_goal_pairs',
        'walkable_cells': len(cells), 'diameter': max(max(row) for row in entry['distances']),
        'distance_stratum_sizes': [len(stratum) for stratum in next(iter(strata.values()))],
        'blocks': blocks,
    }


def build_protocol():
    entries = json.loads((ROOT / 'fixtures/maze_knowledge.json').read_text())['mazes']
    return {
        'schema': 1, 'id': PROTOCOL_ID,
        'reference_commit': '79d9f862e35385ba6fbad3a75415c96b9b116500',
        'split_rule': 'For each goal in row,column order rank every other walkable start by shortest eight-action distance, row and column. Divide ranks into three balanced contiguous strata, with remainders in earlier strata. In each stratum use h=max(1,floor((size+2)/5)) balanced contiguous segments. Put segment endpoints into validation/test, alternating their order by goal-index+stratum-index+segment-index parity; put interiors into training. Sort selected starts by row,column. No seed or learning result enters this rule.',
        'proportions': 'Approximately 60/20/20, rounded within each of three per-goal distance strata. Actual rational proportions and starts per goal are recorded for every block; each goal has the same number of starts in a given block.',
        'adequacy_criterion': 'Every goal has at least three starts in each block, including one from each of the three distance-rank strata and at least two distinct distances. Each maze block has at least forty pairs. The disjoint union is every ordered distinct walkable pair, including the original 9 landmark.',
        'cap_rule': 'Unchanged from maze-goals-v1: Maze4=5, Maze5=8, Maze6=10, Maze7=10, Woods14=18, MazeMA=9. Keep every beyond-cap pair explicit. Every block has exact random floor below 0.20.',
        'training_law': 'Goal uniform over all walkable cells, then start uniform over that goal\'s training starts. Identical for both agents. Only training pairs and all goal identifiers may be handed to the training boundary; held-out pair sets remain in the measurement layer.',
        'evaluation': 'Read-only exhaustive evaluation of each registered pair set with equal exact pair weights and its own exact random floor and ceiling. Test diagnostics do not adapt learning or selection.',
        'knowledge': 'Read-only secondary metric over all goals and all multi-goal walkable transitions, excluding wall bumps and transitions starting at their own goal. Join the goal on both transition sides; use core anticipation_fraction with quality strictly above theta_r and complete goal-condition matching. Outside training cost.',
        'budget_rule': BUDGET_RULE,
        'configurations': [configuration(task, next(entry for entry in entries if entry['id'] == geometry))
                           for task, geometry in TASKS],
    }


if __name__ == '__main__':
    Path(__file__).with_name('pair-protocol.json').write_text(json.dumps(build_protocol(), indent=2) + '\n')
