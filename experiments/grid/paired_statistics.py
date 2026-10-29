from fractions import Fraction
import math
import statistics

SEEDS = tuple(range(42, 62))
VIEWS = ('checkpoint_mean', 'final_point')
T19 = 2.093024054408263


def success(block):
    pairs, value = block['evaluation_pairs'], block['success']
    if (type(pairs) is not int or pairs <= 0 or type(value) not in (int, float)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('success requires exhaustive integer pair counts')
    count = round(pairs * value)
    if abs(pairs * value - count) > 1e-8:
        raise ValueError('success is not an integer exhaustive count')
    return Fraction(count, pairs)


def indexed(rows, targets, seeds=SEEDS, blocks=('training', 'validation', 'test')):
    expected = {(seed, point) for seed in seeds for point in targets}
    result = {}
    pairs = {name: set() for name in blocks}
    for row in rows:
        key = row['seed'], row['nominal_step']
        if key not in expected or key in result or set(row['evaluation_blocks']) != set(blocks):
            raise ValueError('unregistered, duplicate or incomplete analysis row')
        for name in blocks:
            block = row['evaluation_blocks'][name]
            if block.get('evaluation_distribution') != 'exhaustive_exact_weights':
                raise ValueError('analysis requires exhaustive exact pair weights')
            success(block)
            pairs[name].add(block['evaluation_pairs'])
        result[key] = row
    if set(result) != expected or any(len(values) != 1 for values in pairs.values()):
        raise ValueError('analysis requires complete seeds, points and fixed pair counts')
    return result


def scores(rows, targets, view, block='test', seeds=SEEDS, blocks=('training', 'validation', 'test')):
    if view not in VIEWS:
        raise ValueError('unregistered analysis view')
    data = indexed(rows, targets, seeds, blocks)
    points = targets if view == 'checkpoint_mean' else targets[-1:]
    return {seed: sum((success(data[seed, point]['evaluation_blocks'][block]) for point in points), Fraction()) / len(points)
            for seed in seeds}


def summary(values):
    present = [value for value in values if value is not None]
    if any(not math.isfinite(value) for value in present):
        raise ValueError('summary contains nonfinite values')
    n = len(present)
    return dict(n=n, mean=float(statistics.mean(present)) if n else None,
                se=statistics.stdev(present) / math.sqrt(n) if n > 1 else None)


def success_summary(values):
    if set(values) != set(SEEDS) or any(not isinstance(value, Fraction) for value in values.values()):
        raise ValueError('success summary requires twenty exact paired seeds')
    return dict(summary(list(values.values())), exact_mean=str(sum(values.values(), Fraction()) / len(SEEDS)),
                seed_scores={str(seed): str(values[seed]) for seed in SEEDS})


def contrast(arms, terms):
    if any(set(arms[name]) != set(SEEDS) for name, _ in terms):
        raise ValueError('paired contrasts require exactly seeds 42--61')
    differences = {seed: sum((weight * arms[name][seed] for name, weight in terms), Fraction()) for seed in SEEDS}
    if any(not isinstance(value, Fraction) for value in differences.values()):
        raise ValueError('paired contrasts require exact seed scores')
    mean = sum(differences.values(), Fraction()) / len(SEEDS)
    se = statistics.stdev(differences.values()) / math.sqrt(len(SEEDS))
    return dict(n=20, mean=float(mean), exact_mean=str(mean), se=se,
        ci_low=float(mean) - T19 * se, ci_high=float(mean) + T19 * se,
        seed_differences={str(seed): str(differences[seed]) for seed in SEEDS})


def crossing(rows, targets, threshold):
    data = indexed(rows, targets)
    for point in targets:
        rate = sum((success(data[seed, point]['evaluation_blocks']['test']) for seed in SEEDS), Fraction()) / len(SEEDS)
        if rate >= Fraction(threshold):
            return point
    return None
