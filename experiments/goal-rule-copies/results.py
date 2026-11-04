import math
import statistics

from protocol import ARM, REFERENCES, T4, T19


def paired(left, right, seeds):
    if set(left) != set(seeds) or set(right) != set(seeds):
        raise ValueError('paired comparison requires every registered seed')
    values = [None if left[seed] is None or right[seed] is None else left[seed] - right[seed] for seed in seeds]
    if any(value is None for value in values):
        return dict(n=sum(value is not None for value in values), mean=None, se=None, ci_low=None, ci_high=None, seed_differences=values)
    if any(not math.isfinite(value) for value in values) or len(values) not in (5, 20):
        raise ValueError('expected five or twenty finite paired differences')
    n = len(values)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    t = T4 if n == 5 else T19
    return dict(n=n, mean=mean, se=se, ci_low=mean-t*se, ci_high=mean+t*se, seed_differences=values)


def contrasts(success, mechanisms, configs):
    rows = []
    for config in configs:
        maze = config['task']
        for reference in REFERENCES:
            for group, metrics, source, seeds in [
                ('success', ['primary', 'final'], success, list(range(42, 62))),
                ('mechanism', ['F', 'F_mod', 'estimate_minus_q', 'estimate_above_v', 'within_start_slope'], mechanisms, list(range(42, 47)))]:
                for metric in metrics:
                    values = lambda arm: {seed: source[maze, arm, seed][metric] for seed in seeds}
                    rows.append(dict(maze=maze, reference=reference, comparison=ARM + ' - ' + reference,
                        group=group, metric=metric, registered=True, **paired(values(ARM), values(reference), seeds)))
    return rows


def decisions(rows, mazes):
    def interval(maze, group, metric):
        matches = [row for row in rows if row['maze'] == maze and row['reference'] == 'acs2her_shortened'
                   and row['group'] == group and row['metric'] == metric]
        if len(matches) != 1:
            raise ValueError('outcome requires one exact registered contrast per maze and metric')
        return matches[0]
    def sign(row, positive):
        bound = row['ci_low'] if positive else row['ci_high']
        return bound is not None and (bound > 0 if positive else bound < 0)
    positive = {metric: sum(sign(interval(maze, 'success', metric), True) for maze in mazes) for metric in ('primary', 'final')}
    negative = {metric: sum(sign(interval(maze, 'success', metric), False) for maze in mazes) for metric in ('primary', 'final')}
    if (positive['final'] >= 3 and negative['primary'] >= 1) or (positive['primary'] >= 3 and negative['final'] >= 1):
        outcome = 'trade-off'
    elif positive['primary'] >= 3 and negative['primary'] == 0:
        outcome = 'helps'
    else:
        outcome = 'does not help'
    confirmed = [maze for maze in ('maze4', 'maze5', 'maze6', 'maze7')
                 if sign(interval(maze, 'mechanism', 'F_mod'), True)
                 and sign(interval(maze, 'mechanism', 'estimate_above_v'), False)]
    return dict(outcome=outcome, positive=positive, negative=negative,
                mechanism_confirmed=len(confirmed) >= 3, mechanism_mazes=confirmed)
