import statistics

from protocol import select_proportion


def select(rows, proportions, targets):
    expected={(proportion,seed,point) for proportion in proportions for seed in range(42,47) for point in targets}
    indexed={}
    for row in rows:
        if set(row['evaluation_blocks']) != {'validation'}:
            raise ValueError('tuning must not access the test block')
        key=row['agent_parameters']['relabeled_proportion'],row['seed'],row['nominal_step']
        if key not in expected or key in indexed:
            raise ValueError('unexpected or duplicate tuning row')
        indexed[key]=row['evaluation_blocks']['validation']['success']
    if set(indexed)!=expected:
        raise ValueError('incomplete tuning points or seeds')
    scores={proportion:statistics.mean(statistics.mean(indexed[proportion,seed,point] for point in targets) for seed in range(42,47)) for proportion in proportions}
    return dict(proportion=select_proportion(scores,proportions),validation_scores=[dict(proportion=p,mean=scores[p]) for p in sorted(scores)])
