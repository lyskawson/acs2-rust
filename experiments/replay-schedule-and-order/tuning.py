from fractions import Fraction
import math


def select(rows_by_setting, targets, candidates):
    if list(rows_by_setting) != ['base'] + list(candidates):
        raise ValueError('screening must contain exactly the registered base and settings')
    table = {}
    for setting, rows in rows_by_setting.items():
        indexed = {}
        for row in rows:
            key = row['seed'], row['nominal_step']
            if set(row['evaluation_blocks']) != {'validation'} or key in indexed:
                raise ValueError('selection accepts unique validation-only rows')
            block = row['evaluation_blocks']['validation']
            pairs = block['evaluation_pairs']
            count = round(block['success'] * pairs)
            if not 0 <= count <= pairs or abs(count / pairs - block['success']) > 1e-12:
                raise ValueError('screening success is not an exhaustive count')
            indexed[key] = Fraction(count, pairs)
        if set(indexed) != {(seed, point) for seed in range(42, 47) for point in targets}:
            raise ValueError('screening requires every registered point and five seeds')
        scores = [sum((indexed[seed, point] for point in targets), Fraction()) / len(targets) for seed in range(42, 47)]
        mean = sum(scores, Fraction()) / 5
        se = math.sqrt(float(sum((score - mean) ** 2 for score in scores) / 20))
        table[setting] = dict(mean=float(mean), exact_mean=str(mean), se=se,
            seed_scores={str(seed): str(score) for seed, score in zip(range(42, 47), scores)})
    best = max(Fraction(cell['exact_mean']) for cell in table.values())
    selected = next(setting for setting, cell in table.items() if float(best - Fraction(cell['exact_mean'])) <= 1e-12)
    return dict(setting=selected, table=table)
