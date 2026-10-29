import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path

from approved_execution import approved
from protocol import ER, SHORTENED, label


def read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, tolerance=1e-10):
    if expected is None:
        if actual not in ('', None):
            raise ValueError('empty statistical group must remain null')
    elif actual in ('', None) or not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError('exported statistic differs from independent reconstruction')


def mean_se(values):
    values = [float(value) for value in values]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    se = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1) / len(values)) if len(values) > 1 else None
    return mean, se


def validate(root, output):
    root, output = Path(root), Path(output)
    plan = approved(root / 'approved-plan.json')
    result = json.loads((output / 'results.json').read_text())
    selections = result['selections']['selections']
    raw = read(output / 'counts.csv')
    counts = {}
    for row in raw:
        key = row['configuration_id'], row['arm'], int(row['seed']), int(row['nominal_step']), row['block']
        if key in counts:
            raise ValueError('duplicate exported count')
        counts[key] = Fraction(int(row['successes']), int(row['evaluation_pairs']))
    seed_rows = read(output / 'seed-scores.csv')
    scores = {}
    for row in seed_rows:
        config = next(config for config in plan['configurations'] if config['id'] == row['configuration_id'])
        points = config['targets'] if row['view'] == 'checkpoint_mean' else config['targets'][-1:]
        value = sum((counts[config['id'], row['arm'], int(row['seed']), point, 'test'] for point in points), Fraction()) / len(points)
        if value != Fraction(row['score_exact']):
            raise ValueError('seed score is not the registered point mean or final point')
        key = config['id'], row['arm'], int(row['seed']), row['view']
        if key in scores:
            raise ValueError('duplicate exported seed score')
        scores[key] = value
    definitions = {
        'S_X - S': lambda values: values['S_X'] - values['S'],
        'E_X - E': lambda values: values['E_X'] - values['E'],
        '(S_X - S) - (E_X - E)': lambda values: (values['S_X'] - values['S']) - (values['E_X'] - values['E']),
        'S_X - E_X': lambda values: values['S_X'] - values['E_X'],
        'S_backward - S_order_control': lambda values: values['S_X'] - values['order_control'],
    }
    expected_comparisons = set()
    for config in plan['configurations']:
        for axis in ('B', 'C'):
            if selections[config['id']][axis]['setting'] != 'base':
                for view in ('checkpoint_mean', 'final_point'):
                    for name in plan['analysis']['comparisons'] + ([plan['analysis']['C_additional']] if axis == 'C' else []):
                        expected_comparisons.add((config['id'], axis, view, name))
    comparisons = read(output / 'comparisons.csv')
    observed = set()
    expected_differences = {}
    for row in comparisons:
        key = row['configuration_id'], row['axis'], row['view'], row['comparison']
        if key not in expected_comparisons or key in observed:
            raise ValueError('missing, duplicate or unregistered comparison')
        observed.add(key)
        config = next(config for config in plan['configurations'] if config['id'] == row['configuration_id'])
        mode = selections[config['id']][row['axis']]['setting']
        aliases = dict(S=label(config, SHORTENED, 'base'), E=ER,
            S_X=label(config, SHORTENED, mode), E_X=label(config, ER, mode))
        if row['axis'] == 'C':
            aliases['order_control'] = label(config, SHORTENED, 'random-pass')
        values = []
        for seed in range(42, 62):
            difference = definitions[row['comparison']]({alias: scores[config['id'], arm, seed, row['view']] for alias, arm in aliases.items()})
            values.append(difference)
            expected_differences[key + (seed,)] = difference
        exact = sum(values, Fraction()) / 20
        if Fraction(row['exact_mean']) != exact or int(row['n']) != 20:
            raise ValueError('paired comparison exact mean or sample count differs')
        mean, se = mean_se(values)
        close(row['mean'], mean)
        close(row['se'], se)
        close(row['ci_low'], mean - 2.093024054408263 * se)
        close(row['ci_high'], mean + 2.093024054408263 * se)
    if observed != expected_comparisons:
        raise ValueError('a registered comparison is absent')
    actual_differences = {(row['configuration_id'], row['axis'], row['view'], row['comparison'], int(row['seed'])): Fraction(row['difference_exact'])
                          for row in read(output / 'paired-differences.csv')}
    if actual_differences != expected_differences:
        raise ValueError('exported paired seed differences differ')
    crossings = read(output / 'thresholds.csv')
    expected_thresholds = {(config['id'], axis, threshold) for config in plan['configurations'] for axis in ('B', 'C')
        if selections[config['id']][axis]['setting'] != 'base' for threshold in plan['analysis']['thresholds'][config['id']]}
    seen_thresholds = set()
    for row in crossings:
        key = row['configuration_id'], row['axis'], row['threshold']
        if key not in expected_thresholds or key in seen_thresholds:
            raise ValueError('unregistered or duplicate threshold')
        seen_thresholds.add(key)
        config = next(config for config in plan['configurations'] if config['id'] == row['configuration_id'])
        arm = label(config, SHORTENED, selections[config['id']][row['axis']]['setting'])
        threshold = Fraction(plan['analysis']['thresholds'][config['id']][row['threshold']])
        expected = next((point for point in config['targets'] if sum((counts[config['id'], arm, seed, point, 'test'] for seed in range(42, 62)), Fraction()) / 20 >= threshold), None)
        if row['arm'] != arm or (int(row['first_nominal_step']) if row['first_nominal_step'] else None) != expected:
            raise ValueError('threshold does not use the selected arm and inclusive exact mean')
    if seen_thresholds != expected_thresholds:
        raise ValueError('registered threshold is missing')
    screening_counts = read(output / 'screening-counts.csv')
    screening_scores = {}
    for config in plan['configurations']:
        for mode in ('base', '3to1at30000', '3to1at100000', '3to1at300000', 'backward'):
            for seed in range(42, 47):
                values = {int(row['nominal_step']): Fraction(int(row['successes']), int(row['evaluation_pairs'])) for row in screening_counts
                          if (row['configuration_id'], row['setting'], int(row['seed'])) == (config['id'], mode, seed)}
                if set(values) != set(config['targets']):
                    raise ValueError('screening count grid differs')
                screening_scores[config['id'], mode, seed] = sum(values.values(), Fraction()) / len(values)
    for row in read(output / 'screening-table.csv'):
        values = [screening_scores[row['configuration_id'], row['setting'], seed] for seed in range(42, 47)]
        exact = sum(values, Fraction()) / 5
        mean, se = mean_se(values)
        if Fraction(row['exact_mean']) != exact or int(row['n']) != 5:
            raise ValueError('screening mean or sample count differs')
        close(row['mean'], mean)
        close(row['se'], se)
    metrics = read(output / 'checkpoint-metrics.csv')
    trajectories = {}
    for row in metrics:
        trajectories.setdefault(row['trajectory'], []).append(row)
        for block in ('training', 'validation', 'test'):
            if block + '_evaluation_pairs' not in row or not row[block + '_evaluation_pairs']:
                continue
            n, hits = int(row[block + '_evaluation_pairs']), int(row[block + '_successes'])
            all_gap = float(row[block + '_value_gap_all'])
            good = float(row[block + '_value_gap_successful']) if hits else None
            failed = (n * all_gap - hits * good) / (n - hits) if 0 < hits < n else all_gap if hits == 0 else None
            close(row[block + '_value_gap_failed'], failed, 1e-8)
    actual_intervals = {(row['trajectory'], int(row['nominal_step'])): row for row in read(output / 'update-intervals.csv')}
    for trajectory, rows in trajectories.items():
        previous_steps = previous_updates = 0
        for row in sorted(rows, key=lambda value: int(value['nominal_step'])):
            steps = int(row['actual_steps'])
            updates = int(row['online_updates']) + int(row['replay_updates'])
            exact = Fraction(updates - previous_updates, steps - previous_steps)
            exported = actual_intervals[trajectory, int(row['nominal_step'])]
            if Fraction(exported['updates_per_interaction_exact']) != exact:
                raise ValueError('checkpoint interval rate is not the counter difference')
            close(exported['updates_per_interaction'], float(exact))
            previous_steps, previous_updates = steps, updates
    final_metrics = {}
    for row in metrics:
        if row['purpose'] in ('test', 'reused'):
            final_metrics.setdefault((row['configuration_id'], row['arm'], int(row['seed'])), []).append(row)
    for row in read(output / 'diagnostics.csv'):
        if row['purpose'] != 'final_arms':
            continue
        config = next(config for config in plan['configurations'] if config['id'] == row['configuration_id'])
        points = config['targets'] if row['view'] == 'checkpoint_mean' else config['targets'][-1:]
        values = []
        for seed in range(42, 62):
            candidates = [value[row['metric']] for value in final_metrics[config['id'], row['arm'], seed]
                          if int(value['nominal_step']) in points and value[row['metric']] != '']
            if candidates:
                values.append(sum(map(float, candidates)) / len(candidates))
        mean, se = mean_se(values)
        if int(row['n']) != len(values):
            raise ValueError('diagnostic contributing seed count differs')
        close(row['mean'], mean, 1e-8)
        close(row['se'], se, 1e-8)
    return dict(verified=True, counts=len(raw), seed_scores=len(seed_rows), comparisons=len(comparisons),
        paired_differences=len(actual_differences), thresholds=len(crossings), intervals=len(actual_intervals),
        diagnostics='independent within-seed reconstruction from checkpoint metrics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.out)))
