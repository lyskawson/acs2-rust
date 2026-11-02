import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from protocol import ARMS
from prepare_execution import APPROVED_SHA256


def number(value):
    return float(value) if value not in ('', None) else None


def interval(values):
    if len(values) != 5:
        raise ValueError('five seeds required')
    if any(value is None for value in values):
        return dict(n=sum(value is not None for value in values), mean=None, se=None, ci_low=None, ci_high=None)
    mean = math.fsum(values) / 5
    se = math.sqrt(math.fsum((value - mean)**2 for value in values) / 20)
    margin = 2.7764451051977987 * se
    return dict(n=5, mean=mean, se=se, ci_low=mean - margin, ci_high=mean + margin)


def compare_interval(row, values):
    expected = interval(values)
    for key, value in expected.items():
        actual = int(row[key]) if key == 'n' else number(row[key])
        if value is None:
            if actual is not None:
                raise ValueError('undefined interval was filled')
        elif actual is None or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-11):
            raise ValueError('exported interval differs: ' + key)


def seed_index(rows):
    result = {}
    for row in rows:
        key = row['maze'], row['arm'], int(row['seed']), int(row['nominal_step']), row['view']
        if key in result or row['post_hoc'] != 'True':
            raise ValueError('duplicate or unlabelled seed metric')
        result[key] = row
        if number(row['F']) is not None:
            gain = number(row['F']) - number(row['F_mod'])
            if not math.isclose(gain, number(row['F_minus_F_mod']), abs_tol=1e-12):
                raise ValueError('modal gain is inconsistent')
            share = number(row['nonmodal_share_eligible'])
            if share:
                decomposition = share * (number(row['nonmodal_optimal']) - number(row['nonmodal_modal_optimal']))
                if not math.isclose(gain, decomposition, abs_tol=1e-12):
                    raise ValueError('nonmodal decomposition is inconsistent')
        attributes = [number(row['winner_goal_attributes_' + str(count)]) for count in range(3)]
        if not math.isclose(sum(attributes), 1 - number(row['random_fallback_share']), abs_tol=1e-12):
            raise ValueError('goal attribute masses do not partition winner decisions')
        if not math.isclose(sum(attributes[1:]), number(row['winner_goal_mass']), abs_tol=1e-12):
            raise ValueError('goal attribute split differs from the registered profile')
    return result


def validate_registered(rows, seeds, plan):
    expected = {}
    def values(maze, arm, point, metric):
        return [number(seeds[maze, arm, seed, point, 'pooled'][metric]) for seed in range(42, 47)]
    def difference(left, right):
        return [a - b if a is not None and b is not None else None for a, b in zip(left, right)]
    for config in plan['configurations']:
        maze, final = config['task'], config['targets'][-1]
        for arm in ARMS:
            for point in (30000, 100000, final):
                expected[maze, arm + ': F - C', 'F-C', point] = difference(values(maze, arm, point, 'F'), values(maze, arm, point, 'C'))
            expected[maze, arm, 'winner_goal_mass', final] = values(maze, arm, final, 'winner_goal_mass')
        for left, right in plan['analysis']['final_policy_contrasts']:
            for metric in ('F', 'M', 'D'):
                expected[maze, left + ' - ' + right, metric, final] = difference(values(maze, left, final, metric), values(maze, right, final, metric))
        for left, right in plan['analysis']['final_value_contrasts']:
            expected[maze, left + ' - ' + right, 'estimate_minus_Q*', final] = difference(values(maze, left, final, 'estimate_minus_q'), values(maze, right, final, 'estimate_minus_q'))
    seen = set()
    for row in rows:
        key = row['maze'], row['comparison'], row['metric'], int(row['nominal_step'])
        if key in seen or key not in expected or row['registered'] != 'True' or row['view'] != 'pooled':
            raise ValueError('registered table differs from the exact registration')
        compare_interval(row, expected[key])
        seen.add(key)
    if seen != set(expected):
        raise ValueError('registered table is incomplete')
    return len(seen)


def validate_post_hoc(rows, seeds, final_points, expected_view):
    for row in rows:
        if row['post_hoc'] != 'True' or row['view'] != expected_view or int(row['nominal_step']) != final_points[row['maze']]:
            raise ValueError('post-hoc table lacks its correct label, view or point')
        values = []
        for seed in range(42, 47):
            prefix = row['maze'], seed, int(row['nominal_step']), expected_view
            if row['metric'] == 'F-C':
                arm = row['comparison'].split(':')[0]
                source = seeds[prefix[0], arm, prefix[1], prefix[2], prefix[3]]
                left, right = number(source['F']), number(source['C'])
            else:
                left_arm, right_arm = row['comparison'].split(' - ')
                left = number(seeds[prefix[0], left_arm, prefix[1], prefix[2], prefix[3]][row['metric']])
                right = number(seeds[prefix[0], right_arm, prefix[1], prefix[2], prefix[3]][row['metric']])
            values.append(left - right if left is not None and right is not None else None)
        compare_interval(row, values)
    return len(rows)


def validate(root, plan_path):
    root = Path(root)
    content = Path(plan_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != APPROVED_SHA256:
        raise ValueError('validation requires exact approved registration')
    plan = json.loads(content)
    summary = json.loads((root / 'analysis-summary.json').read_text())
    for name, digest in summary['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('analysis artifact checksum changed: ' + name)
    def read(name):
        with (root / name).open() as stream:
            return list(csv.DictReader(stream))
    seeds = seed_index(read('post-hoc-seed-metrics.csv'))
    registered = validate_registered(read('registered-comparisons.csv'), seeds, plan)
    final = {config['task']: config['targets'][-1] for config in plan['configurations']}
    uniform = validate_post_hoc(read('post-hoc-uniform-comparisons.csv'), seeds, final, 'uniform_pairs')
    modal = validate_post_hoc(read('post-hoc-modal-comparisons.csv'), seeds, final, 'pooled')
    curves = read('post-hoc-curves.csv')
    for row in curves:
        if row['post_hoc'] != 'True':
            raise ValueError('curve lacks post-hoc label')
        values = [number(seeds[row['maze'], row['arm'], seed, int(row['nominal_step']), row['view']][row['metric']]) for seed in range(42, 47)]
        compare_interval(row, values)
    return dict(verified=True, registered_comparisons=registered, uniform_post_hoc=uniform, modal_post_hoc=modal,
                curve_intervals=len(curves), seed_cells=len(seeds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--plan', default=str(Path(__file__).with_name('plan.json')))
    args = parser.parse_args()
    result = validate(args.root, args.plan)
    (Path(args.root) / 'export-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
