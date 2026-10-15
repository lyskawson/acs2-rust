from fractions import Fraction
import math
import statistics

T19 = 2.093024054408263


def exact_success(block):
    pairs = block['evaluation_pairs']
    heads = block.get('heads', [block])
    counts = []
    for head in heads:
        count = head['success'] * pairs
        integer = round(count)
        if abs(count - integer) > 1e-8 or not 0 <= integer <= pairs:
            raise ValueError('success is not an exhaustive per-head integer count')
        counts.append(integer)
    value = Fraction(sum(counts), pairs * len(heads))
    if abs(float(value) - block['success']) > 1e-12:
        raise ValueError('mean success disagrees with the head counts')
    return value


def seed_scores(rows, targets, seeds=range(42, 62), block='test'):
    cells = {}
    for row in rows:
        key = row['seed'], row['nominal_step']
        if key in cells:
            raise ValueError('duplicate seed/checkpoint')
        cells[key] = exact_success(row['evaluation_blocks'][block])
    if set(cells) != {(seed, point) for seed in seeds for point in targets}:
        raise ValueError('missing or extra registered seed/checkpoint')
    return {seed: dict(checkpoint_mean_within_seed=sum((cells[seed, point] for point in targets), Fraction()) / len(targets),
                       final_point=cells[seed, targets[-1]]) for seed in seeds}


def summarize(values):
    if len(values) != 20 or any(not math.isfinite(float(value)) for value in values):
        raise ValueError('analysis requires twenty finite seed values')
    values = list(map(float, values))
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(20)
    return dict(n=20, mean=mean, se=se, ci_low=mean - T19 * se, ci_high=mean + T19 * se)


def paired(left, right):
    if set(left) != set(range(42, 62)) or set(left) != set(right):
        raise ValueError('paired comparison requires the registered twenty seeds')
    return {view: summarize([left[seed][view] - right[seed][view] for seed in range(42, 62)])
            for view in ['checkpoint_mean_within_seed', 'final_point']}


def threshold_crossing(rows, targets, threshold):
    scores = seed_scores(rows, targets)
    if len(scores) != 20:
        raise ValueError('threshold requires twenty seeds')
    by_point = {point: [] for point in targets}
    for row in rows:
        by_point[row['nominal_step']].append(exact_success(row['evaluation_blocks']['test']))
    for point in targets:
        if sum(by_point[point], Fraction()) / 20 >= Fraction(threshold):
            return point
    return None


def head_spread(block):
    values = [head['success'] for head in block['heads']]
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) * (value - mean) for value in values) / len(values))
