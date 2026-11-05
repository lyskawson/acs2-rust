import math
import statistics

from protocol import ARM, G, S, T4, T19

MECHANISM_METRICS = ['F', 'F_mod', 'estimate_minus_q', 'estimate_above_v', 'within_start_slope', 'winner_goal_mass']
DECISION_MAZES = ('maze4', 'maze5', 'maze6', 'maze7', 'mazema')


def paired(left, right, seeds):
    if set(left) != set(seeds) or set(right) != set(seeds):
        raise ValueError('paired comparison requires every registered seed')
    values = [None if left[seed] is None or right[seed] is None else left[seed] - right[seed] for seed in seeds]
    if len(values) not in (5, 20):
        raise ValueError('registered contrasts require five or twenty seeds')
    if any(value is None for value in values):
        return dict(n=sum(value is not None for value in values), mean=None, se=None, ci_low=None, ci_high=None, seed_differences=values)
    if any(not math.isfinite(value) for value in values):
        raise ValueError('finite paired differences required')
    n = len(values)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    t = T4 if n == 5 else T19
    return dict(n=n, mean=mean, se=se, ci_low=mean-t*se, ci_high=mean+t*se, seed_differences=values)


def contrasts(success, mechanisms, configs):
    rows = []
    for config in configs:
        maze = config['task']
        for left, right in [(G, ARM), (ARM, S)]:
            for group, metrics, source, seeds in [
                ('success', ['primary', 'final'], success, list(range(42, 62))),
                ('mechanism', MECHANISM_METRICS, mechanisms, list(range(42, 47)))]:
                for metric in metrics:
                    values = lambda arm: {seed: source[maze, arm, seed][metric] for seed in seeds}
                    rows.append(dict(maze=maze, left=left, right=right, comparison=left+' - '+right,
                        group=group, metric=metric, registered=True, **paired(values(left), values(right), seeds)))
    return rows


def decisions(rows):
    def interval(maze, left, right, group, metric):
        found = [row for row in rows if (row['maze'], row['left'], row['right'], row['group'], row['metric'])
                 == (maze, left, right, group, metric)]
        if len(found) != 1:
            raise ValueError('decision requires one exact registered contrast per maze and metric')
        return found[0]
    def sign(row, positive):
        bound = row['ci_low'] if positive else row['ci_high']
        return bound is not None and (bound > 0 if positive else bound < 0)
    a_positive = sum(sign(interval(maze, G, ARM, 'success', 'primary'), True) for maze in DECISION_MAZES)
    a_negative = sum(sign(interval(maze, G, ARM, 'success', 'primary'), False) for maze in DECISION_MAZES)
    b_positive = sum(sign(interval(maze, ARM, S, 'success', 'primary'), True) for maze in DECISION_MAZES)
    if a_positive >= 3 and a_negative == 0:
        outcome = 'rule choice beyond fewer updates' if b_positive >= 3 else 'rule choice'
    elif a_positive < 3 and b_positive >= 3:
        outcome = 'fewer updates'
    else:
        outcome = 'unresolved'
    confirmed = [maze for maze in DECISION_MAZES[:4]
        if all(sign(interval(maze, G, ARM, 'mechanism', metric), True) for metric in ('F_mod', 'winner_goal_mass'))]
    return dict(outcome=outcome, A_positive=a_positive, A_negative=a_negative, B_positive=b_positive,
        mechanism_separated=len(confirmed) >= 3, mechanism_mazes=confirmed)
