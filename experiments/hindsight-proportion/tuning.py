import math
import statistics

from protocol import SHORTENED, candidates, label


def select(rows, proportions, targets):
    proportions = candidates(proportions)
    if not targets or sorted(set(targets)) != list(targets):
        raise ValueError('distinct ordered registered points are required')
    expected = {(p, seed, point) for p in proportions for seed in range(42, 47) for point in targets}
    indexed = {}
    for row in rows:
        if set(row['evaluation_blocks']) != {'validation'}:
            raise ValueError('tuning requires validation-only rows; test access is forbidden')
        p = row['agent_parameters']['relabeled_proportion']
        key = p, row['seed'], row['nominal_step']
        if (row['agent'] != SHORTENED or row.get('arm') != label(SHORTENED, p)
                or key not in expected or key in indexed):
            raise ValueError('unexpected arm, proportion, seed, duplicate or point in tuning')
        success = row['evaluation_blocks']['validation']['success']
        if type(success) not in (float, int) or not math.isfinite(success) or not 0 <= success <= 1:
            raise ValueError('invalid validation success')
        indexed[key] = success
    if set(indexed) != expected:
        raise ValueError('incomplete registered tuning candidates, seeds or points')
    table = []
    for p in proportions:
        scores = [statistics.mean(indexed[p, seed, point] for point in targets) for seed in range(42, 47)]
        table.append(dict(proportion=p, mean=statistics.mean(scores), se=statistics.stdev(scores) / math.sqrt(5),
                          n=5, seed_scores=scores))
    best = max(item['mean'] for item in table)
    selected = min(item['proportion'] for item in table if best - item['mean'] <= 1e-12)
    return dict(proportion=selected, validation_scores=table)
