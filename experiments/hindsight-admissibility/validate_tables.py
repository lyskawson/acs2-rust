import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path

from execution import PILOT_SECONDS, approved

SEEDS = tuple(range(42, 62))
REFERENCES = ('acs2her', 'acs2er_episodic', 'acs2', 'acs2er')
CRITICAL = 2.093024054408263


def verify_summary(scores, observed):
    if set(scores) != set(SEEDS) or observed['n'] != 20:
        raise ValueError('published success summary has the wrong seed count')
    mean = sum(scores.values(), Fraction()) / 20
    variance = sum(((value - mean) ** 2 for value in scores.values()), Fraction()) / (20 * 19)
    se = math.sqrt(float(variance))
    expected = dict(mean=float(mean), se=se, ci_low=float(mean) - CRITICAL * se, ci_high=float(mean) + CRITICAL * se)
    if observed['mean_exact'] != str(mean):
        raise ValueError('published exact mean differs from independent pair counts')
    for name, value in expected.items():
        if not math.isclose(observed[name], value, rel_tol=0, abs_tol=1e-13):
            raise ValueError('published mean, standard error or paired interval differs: ' + name)
    if observed['seed_scores'] != {str(seed): str(scores[seed]) for seed in SEEDS}:
        raise ValueError('published seed scores differ from independent pair counts')


def first_crossing(curve, targets, threshold):
    return next((point for point in targets if curve[point] >= Fraction(threshold)), None)


def verify_allocation_cost(costs, execution):
    new = sum(int(row['allocation_cpu_seconds']) for row in costs if row['reused_reference'] == 'False')
    historical = sum(int(row['allocation_cpu_seconds']) for row in costs if row['reused_reference'] == 'True')
    if (execution['grid_allocated_cpu_seconds'] != new or execution['pilot_allocated_cpu_seconds'] != PILOT_SECONDS
            or execution['allocated_cpu_seconds'] != new + PILOT_SECONDS or new + PILOT_SECONDS > 30 * 3600):
        raise ValueError('new allocation total charges the wrong attempts or historical references')
    return historical


def validate(directory, plan):
    directory = Path(directory)
    results = json.loads((directory / 'results.json').read_text())
    counts = list(csv.DictReader((directory / 'success-counts.csv').open()))
    comparisons = list(csv.DictReader((directory / 'comparisons.csv').open()))
    costs = list(csv.DictReader((directory / 'costs.csv').open()))
    if len(counts) != 5880 or len(comparisons) != 60 or len(costs) != 720 or results['analysis_rules'] != plan['analysis']:
        raise ValueError('published exports omit registered rows, comparisons or analysis choices')
    indexed = {}
    for row in counts:
        key = row['configuration_id'], row['agent'], int(row['seed']), int(row['nominal_step'])
        successes, pairs = int(row['successes']), int(row['evaluation_pairs'])
        if key in indexed or not 0 <= successes <= pairs or pairs <= 0:
            raise ValueError('duplicate or invalid exported pair count')
        indexed[key] = Fraction(successes, pairs)
    published = {item['configuration_id']: item for item in results['mazes']}
    checked = 0
    for config in plan['configurations']:
        identifier, points = config['id'], config['targets']
        maze = published[identifier]
        names = plan['agents'] + list(REFERENCES)
        scores = {}
        for name in names:
            curve = {point: sum((indexed[identifier, name, seed, point] for seed in SEEDS), Fraction()) / 20 for point in points}
            for view in ('checkpoint_mean', 'final_point'):
                selected = points if view == 'checkpoint_mean' else points[-1:]
                values = {seed: sum((indexed[identifier, name, seed, point] for point in selected), Fraction()) / len(selected) for seed in SEEDS}
                scores[name, view] = values
                verify_summary(values, maze['arms'][name]['success'][view])
            for threshold, value in plan['analysis']['thresholds'][identifier].items():
                if maze['arms'][name]['threshold_crossings'][threshold] != first_crossing(curve, points, value):
                    raise ValueError('published threshold uses a rounded or strict crossing')
            if name in ('acs2', 'acs2er'):
                if curve[points[-1]] != Fraction(plan['analysis']['thresholds'][identifier][name + '_final']):
                    raise ValueError('archived final baseline threshold differs from its own pair counts')
        expected = {(view, left, right) for view in ('checkpoint_mean', 'final_point') for left, right in plan['analysis']['comparisons']}
        actual = {(item['view'], item['left'], item['right']) for item in maze['comparisons']}
        if actual != expected or len(maze['comparisons']) != 10:
            raise ValueError('published comparisons differ from the five registered contrasts in both views')
        for item in maze['comparisons']:
            view, left, right = item['view'], item['left'], item['right']
            values = {seed: scores[left, view][seed] - scores[right, view][seed] for seed in SEEDS}
            verify_summary(values, item['difference'])
            exported = [row for row in comparisons if (row['configuration_id'], row['view'], row['left'], row['right']) == (identifier, view, left, right)]
            if len(exported) != 1:
                raise ValueError('CSV is missing or duplicating a registered contrast')
            record = dict(item['difference'])
            record.update({name: float(exported[0][name]) for name in ('mean', 'se', 'ci_low', 'ci_high')})
            record['mean_exact'] = exported[0]['mean_exact']
            verify_summary(values, record)
            checked += 1
    historical = verify_allocation_cost(costs, results['execution'])
    if historical != results['reference_reuse']['selected_reference_allocation_cpu_seconds'] or results['reference_reuse']['charged_to_this_study'] is not False:
        raise ValueError('reused reference accounting changed')
    return dict(success_count_rows=len(counts), independently_checked_contrasts=checked,
                exact_threshold_checks=72, historical_allocations_charged=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.analysis, approved(args.plan))
    (args.analysis / 'independent-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
