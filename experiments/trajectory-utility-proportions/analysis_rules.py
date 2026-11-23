import math
from statistics import mean

from proportions import MEAN, S1, SHARES, label


def carry_share(study, configs, rows):
    shares = (0.0, 0.25, 0.5, *SHARES[1:])
    seeds = study['seeds']
    groups = sorted({config['group'] for config in configs.values()})
    result = {}
    for group in groups:
        scores = {}
        for share in shares:
            arm = S1 if share == 0 else label(MEAN, share)
            maze_scores = []
            for task, config in configs.items():
                if config['group'] != group:
                    continue
                seed_scores = []
                for seed in seeds:
                    records = rows[task, arm, seed]
                    if [record['nominal_step'] for record in records] != config['targets']:
                        raise ValueError('carry requires every registered validation checkpoint')
                    values = [record['evaluation_blocks']['validation']['success'] for record in records]
                    if not all(math.isfinite(value) and 0 <= value <= 1 + 1e-12 for value in values):
                        raise ValueError('invalid validation success')
                    seed_scores.append(mean(values))
                maze_scores.append(mean(seed_scores))
            scores[share] = mean(maze_scores)
        best = max(scores.values())
        result[group] = dict(tu_share=min(share for share, score in scores.items() if best - score <= 1e-12),
                             validation_primary_by_share=scores)
    return result
