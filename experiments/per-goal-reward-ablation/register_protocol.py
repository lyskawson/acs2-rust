import argparse
import json
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLOCKS = ('training', 'validation', 'test')
PROTOCOL_ID = 'alcs-maze-pairs-v1'
CHECKPOINTS = [1000, 3000, 10000, 30000, 100000, 300000, 1000000, 2000000]


def chunks(values, count):
    q, r = divmod(len(values), count)
    offset = 0
    result = []
    for i in range(count):
        size = q + (i < r)
        result.append(values[offset:offset + size])
        offset += size
    return result


def partition(entry):
    cells = list(map(tuple, entry['walkable_cells']))
    index = {cell: i for i, cell in enumerate(cells)}
    pools = {block: {} for block in BLOCKS}
    strata_by_goal, rankings = {}, {}
    for gi, goal in enumerate(cells):
        ranked = sorted((cell for cell in cells if cell != goal and entry['distances'][index[cell]][gi] is not None),
                        key=lambda cell: (entry['distances'][index[cell]][gi], *cell))
        rankings[goal] = ranked
        strata = chunks(ranked, 3)
        strata_by_goal[goal] = strata
        selected = {block: [] for block in BLOCKS}
        for si, stratum in enumerate(strata):
            for pi, segment in enumerate(chunks(stratum, max(1, (len(stratum) + 2) // 5))):
                first, last = ('validation', 'test') if (gi + si + pi) % 2 == 0 else ('test', 'validation')
                selected[first].append(segment[0])
                selected[last].append(segment[-1])
                selected['training'].extend(segment[1:-1])
        for block in BLOCKS:
            pools[block][goal] = sorted(selected[block])
    def valid():
        expected = {(start, goal) for goal, starts in rankings.items() for start in starts}
        selected = [(start, goal) for pool in pools.values() for goal, starts in pool.items() for start in starts]
        return (len(selected) == len(set(selected)) and set(selected) == expected
                and all(starts for pool in pools.values() for starts in pool.values()))
    rule = 'maze-pairs-v1'
    if not valid():
        rule = 'second-nearest-second-farthest'
        for gi, goal in enumerate(cells):
            ranked = rankings[goal]
            if len(ranked) < 4:
                raise ValueError('fallback cannot give three disjoint nonempty blocks')
            near, far = ranked[1], ranked[-2]
            first, last = ('validation', 'test') if gi % 2 == 0 else ('test', 'validation')
            pools[first][goal] = [near]
            pools[last][goal] = [far]
            pools['training'][goal] = sorted(set(ranked) - {near, far})
        if not valid():
            raise ValueError('fallback violates a binding partition condition')
    return pools, strata_by_goal, rule


def exact_floor(entry, pool, cap):
    cells = list(map(tuple, entry['walkable_cells']))
    index = {cell: i for i, cell in enumerate(cells)}
    total = 0
    for goal, starts in pool.items():
        gi = index[goal]
        counts = [int(i == gi) for i in range(len(cells))]
        for step in range(1, cap + 1):
            counts = [8 ** step if i == gi else sum(counts[j] for j in successors)
                      for i, successors in enumerate(entry['successors'])]
        total += sum(counts[index[start]] for start in starts)
    return Fraction(total, 8 ** cap * sum(map(len, pool.values())))


def configuration(entry):
    cells = list(map(tuple, entry['walkable_cells']))
    indices = {cell: i for i, cell in enumerate(cells)}
    pools, strata, rule = partition(entry)
    diameter = max(d for row in entry['distances'] for d in row if d is not None)
    components = []
    remaining = set(cells)
    while remaining:
        first = min(remaining)
        component = sorted(cell for cell in remaining if entry['distances'][indices[first]][indices[cell]] is not None)
        components.append(component)
        remaining.difference_update(component)
    total = sum(len(c) * (len(c) - 1) for c in components)
    blocks = {}
    for block, pool in pools.items():
        count = sum(map(len, pool.values()))
        floor = exact_floor(entry, pool, diameter)
        adequacy = dict(three_starts_per_goal=all(len(starts) >= 3 for starts in pool.values()),
            all_strata=all(all(set(starts).intersection(s) for s in strata[goal]) for goal, starts in pool.items()),
            two_distances_per_goal=all(len({entry['distances'][indices[start]][indices[goal]] for start in starts}) >= 2 for goal, starts in pool.items()),
            forty_pairs=count >= 40)
        blocks[block] = dict(goal_starts=[dict(goal=goal, starts=starts) for goal, starts in pool.items()],
            goals=len(cells), starts_per_goal=len(next(iter(pool.values()))), evaluation_pairs=count,
            pair_weight=f'1/{count}', proportion=str(Fraction(count, total)),
            random_floor_exact=str(floor), random_floor=float(floor), floor_at_least_020=floor >= Fraction(1,5),
            reachable_within_cap_exact='1', reachable_within_cap=1.0, beyond_cap_pairs=[], adequacy=adequacy)
    return dict(id=entry['task'] + f'_c{diameter}_pairs', task=entry['task'], geometry=entry['id'],
        cap=diameter, diameter=diameter, encoding='coordinates', split_unit='start_goal_pairs',
        walkable_cells=len(cells), components=components, split_rule_used=rule, blocks=blocks,
        aliased_cells=entry['aliased_cells'], aliased_share=entry['aliased_share'],
        group='aliasing' if entry['aliased_cells'] else 'no_aliasing', budget=2000000 * ((diameter + 9)//10),
        checkpoints=CHECKPOINTS)


def build():
    geometry = json.loads((HERE / 'geometry.json').read_text())
    expected = {'maze4','maze5','maze6','maze7','woods14','mazema','mazea_alcs','mazef1_alcs','mazef2_alcs'}
    if {e['task'] for e in geometry if not e['aliased_cells']} != expected:
        raise ValueError('aliasing groups differ from the binding list')
    old = json.loads((HERE.parent / 'maze-baselines/pair-protocol.json').read_text())
    value = dict(schema=1, id=PROTOCOL_ID, alcs_commit='ed0e76282a121fa2bad7dac82b744b2cc763eb72',
        split_rule=old['split_rule'], fallback='Only on a binding-condition violation: second-nearest and second-farthest starts go to validation/test by goal-index parity; all others to training.',
        pair_universe='All ordered distinct walkable pairs in the same eight-action component, including every 9 cell.',
        cap_rule='Largest finite shortest eight-action distance.', budget_rule=old['budget_rule'],
        training_law=old['training_law'], evaluation=old['evaluation'], knowledge=old['knowledge'],
        adequacy='Reported separately for each block; never an exclusion or rejection criterion.',
        configurations=[configuration(entry) for entry in geometry if entry['task'].endswith('_alcs')])
    return json.loads(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    value = build()
    path = HERE / 'pair-protocol.json'
    if args.check:
        if json.loads(path.read_text()) != value:
            raise SystemExit('registered protocol differs from geometry-only rebuilding')
    else:
        path.write_text(json.dumps(value, indent=2) + '\n')
