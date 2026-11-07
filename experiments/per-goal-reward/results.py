import math
import statistics

from protocol import (E, E1, S, S1, G, MAZES, MECHANISM_MAZES, SUCCESS_CONTRASTS,
                      MECHANISM_CONTRASTS, MECHANISM_METRICS, T4, T19)


def interval(values, seeds):
    if set(values) != set(seeds) or len(seeds) not in (5, 20):
        raise ValueError('interval requires exactly the registered five or twenty paired seeds')
    ordered = [values[seed] for seed in seeds]
    if any(value is None for value in ordered):
        return dict(n=sum(value is not None for value in ordered), mean=None, se=None,
                    ci_low=None, ci_high=None, seed_differences=ordered)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError('finite paired differences required')
    n = len(ordered)
    mean = statistics.mean(ordered)
    se = statistics.stdev(ordered) / math.sqrt(n)
    t = T4 if n == 5 else T19
    return dict(n=n, mean=mean, se=se, ci_low=mean - t * se, ci_high=mean + t * se,
                seed_differences=ordered)


def seed_difference(values, comparison):
    required = {'S+1 - S': (S1, S), 'E+1 - E': (E1, E),
                'interaction': (S1, S, E1, E), 'S+1 - G': (S1, G)}
    if comparison not in required:
        raise ValueError('unregistered contrast')
    if any(values[arm] is None for arm in required[comparison]):
        return None
    if comparison == 'interaction':
        return (values[S1] - values[S]) - (values[E1] - values[E])
    left, right = required[comparison]
    return values[left] - values[right]


def contrasts(success, mechanisms, configs):
    rows = []
    for config in configs:
        maze = config['task']
        for group, metrics, comparisons, source, seeds in [
            ('success', ('primary', 'final'), SUCCESS_CONTRASTS, success, list(range(42, 62))),
            ('mechanism', MECHANISM_METRICS, MECHANISM_CONTRASTS, mechanisms, list(range(42, 47)))]:
            for comparison in comparisons:
                arms = (S1, S, E1, E) if comparison == 'interaction' else {
                    'S+1 - S': (S1, S), 'E+1 - E': (E1, E), 'S+1 - G': (S1, G)}[comparison]
                for metric in metrics:
                    values = {seed: seed_difference({arm: source[maze, arm, seed][metric] for arm in arms}, comparison)
                              for seed in seeds}
                    rows.append(dict(maze=maze, comparison=comparison, group=group, metric=metric,
                                     registered=True, **interval(values, seeds)))
    return rows


def decisions(rows):
    indexed = {}
    for row in rows:
        key = row['maze'], row['comparison'], row['group'], row['metric']
        if key in indexed:
            raise ValueError('duplicated registered contrast')
        indexed[key] = row
    def positive(maze, comparison, group, metric):
        value = indexed[maze, comparison, group, metric]['ci_low']
        return value is not None and value > 0
    verdicts = []
    for metric in ('primary', 'final'):
        for comparison in SUCCESS_CONTRASTS:
            a_positive = sum(positive(maze, comparison, 'success', metric) for maze in MAZES)
            a_negative = sum(indexed[maze, comparison, 'success', metric]['ci_high'] is not None
                             and indexed[maze, comparison, 'success', metric]['ci_high'] < 0 for maze in MAZES)
            if a_positive >= 3 and a_negative == 0:
                verdict = 'copies gain more from per-goal values' if comparison == 'interaction' else 'improves'
            elif a_negative >= 3 and a_positive == 0:
                verdict = 'copies gain less from per-goal values' if comparison == 'interaction' else 'worsens'
            elif a_positive >= 1 and a_negative >= 1:
                verdict = 'mixed'
            else:
                verdict = 'no clear difference' if comparison == 'interaction' else 'no clear effect'
            verdicts.append(dict(comparison=comparison, metric=metric, A_positive=a_positive,
                                 A_negative=a_negative, verdict=verdict))
    mechanism = []
    for comparison in MECHANISM_CONTRASTS:
        mazes = [maze for maze in MECHANISM_MAZES if all(positive(maze, comparison, 'mechanism', metric)
                                                      for metric in ('F_mod', 'within_start_slope'))]
        mechanism.append(dict(comparison=comparison, confirmed=len(mazes) >= 3, mazes=mazes))
    return dict(verdicts=verdicts, mechanism=mechanism)
