import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

CRITICAL = 2.093024054408263
SEEDS = tuple(range(42, 62))
PLAN_SHA256 = '894596d2f792a2327ff5fd061c4344a17bcb96b01b9a96dd4f58a61a1e3a2888'


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def moments(values):
    n = len(values)
    mean = sum(values, Fraction()) / n
    variance = sum((value - mean) ** 2 for value in values) / (n * (n - 1))
    return mean, math.sqrt(float(variance))


def close(actual, expected):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'numeric table mismatch: {actual} versus {expected}')


def seed_scores(counts, configuration, arm, targets, view):
    if view not in ('checkpoint_mean', 'final_point'):
        raise ValueError('unregistered score view')
    points = targets if view == 'checkpoint_mean' else targets[-1:]
    return {seed: sum((counts[configuration, arm, seed, point] for point in points), Fraction()) / len(points) for seed in SEEDS}


def validate_difference(record, left, right):
    if set(left) != set(SEEDS) or set(right) != set(SEEDS):
        raise ValueError('paired twenty-seed scores required')
    values = [left[seed] - right[seed] for seed in SEEDS]
    mean, se = moments(values)
    if int(record['n']) != 20 or Fraction(record['exact_mean']) != mean:
        raise ValueError('paired exact mean differs')
    close(record['mean'], mean)
    close(record['se'], se)
    close(record['ci_low'], float(mean) - CRITICAL * se)
    close(record['ci_high'], float(mean) + CRITICAL * se)
    return mean


def select_exact(seed_values):
    means = {p: sum(values, Fraction()) / 5 for p, values in seed_values.items()}
    best = max(means.values())
    return min(p for p, value in means.items() if float(best - value) <= 1e-12)


def validate(directory, plan_path):
    directory, plan_path = Path(directory), Path(plan_path)
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != PLAN_SHA256:
        raise ValueError('registered plan bytes differ')
    plan = json.loads(plan_path.read_text())
    results = json.loads((directory / 'results.json').read_text())
    if results['plan_sha256'] != PLAN_SHA256:
        raise ValueError('analysis uses another registration')
    configs = {config['id']: config for config in plan['configurations']}
    counts, expected_keys = {}, set()
    selected = {key: value['proportion'] for key, value in results['selections']['selections'].items()}
    selected_arms = {maze['configuration_id']: maze['aliases']['S@selected'] for maze in results['mazes']}
    arms = {}
    for config_id, config in configs.items():
        base = {'acs2her_shortened_p2of3', 'acs2her_p2of3', 'acs2her_p1of2', 'acs2er_episodic', 'acs2', 'acs2er'}
        base.add(selected_arms[config_id])
        arms[config_id] = base
        expected_keys.update((config_id, arm, seed, point) for arm in base for seed in SEEDS for point in config['targets'])
    for row in read_csv(directory / 'success-counts.csv'):
        key = row['configuration_id'], row['arm'], int(row['seed']), int(row['nominal_step'])
        n, total = int(row['successes']), int(row['evaluation_pairs'])
        if key in counts or key not in expected_keys or total <= 0 or not 0 <= n <= total:
            raise ValueError('invalid or duplicate integer test-success row')
        counts[key] = Fraction(n, total)
    if set(counts) != expected_keys:
        raise ValueError('success count export is incomplete')
    score_records = {}
    for row in read_csv(directory / 'seed-scores.csv'):
        key = row['configuration_id'], row['arm'], row['view'], int(row['seed'])
        config = configs[key[0]]
        expected = seed_scores(counts, key[0], key[1], config['targets'], key[2])[key[3]]
        if key in score_records or Fraction(row['score_exact']) != expected:
            raise ValueError('seed score differs from exact test counts')
        score_records[key] = expected
    expected_scores = {(cfg, arm, view, seed) for cfg in configs for arm in arms[cfg]
                       for view in ('checkpoint_mean', 'final_point') for seed in SEEDS}
    if set(score_records) != expected_scores:
        raise ValueError('seed score export is incomplete')
    comparison_records, expected_comparisons = {}, set()
    for cfg in configs:
        for view in ('checkpoint_mean', 'final_point'):
            for left, right in plan['analysis']['comparisons']:
                if selected[cfg] == 2 / 3 and (left, right) == ('S@selected', 'S@2/3'):
                    continue
                expected_comparisons.add((cfg, view, left, right))
    for row in read_csv(directory / 'comparisons.csv'):
        key = row['configuration_id'], row['view'], row['left_alias'], row['right_alias']
        cfg, view, left_alias, right_alias = key
        aliases = {'S@selected': selected_arms[cfg], 'S@2/3': 'acs2her_shortened_p2of3',
                   'H@1/2': 'acs2her_p1of2', 'H@2/3': 'acs2her_p2of3', 'E': 'acs2er_episodic'}
        if key not in expected_comparisons or key in comparison_records or row['left'] != aliases[left_alias] or row['right'] != aliases[right_alias]:
            raise ValueError('comparison arm or proportion differs from registration')
        left = seed_scores(counts, cfg, row['left'], configs[cfg]['targets'], view)
        right = seed_scores(counts, cfg, row['right'], configs[cfg]['targets'], view)
        comparison_records[key] = validate_difference(row, left, right)
    if set(comparison_records) != expected_comparisons:
        raise ValueError('registered comparisons are missing or added post hoc without a label')
    for cfg in configs:
        for view in ('checkpoint_mean', 'final_point'):
            rule = comparison_records[cfg, view, 'S@2/3', 'H@1/2']
            dose = comparison_records[cfg, view, 'H@1/2', 'H@2/3']
            old_left = seed_scores(counts, cfg, 'acs2her_shortened_p2of3', configs[cfg]['targets'], view)
            old_right = seed_scores(counts, cfg, 'acs2her_p2of3', configs[cfg]['targets'], view)
            previous = sum((old_left[seed] - old_right[seed] for seed in SEEDS), Fraction()) / 20
            if rule + dose != previous:
                raise ValueError('reported rule-plus-dose decomposition differs exactly')
    observed_thresholds = set()
    for row in read_csv(directory / 'thresholds.csv'):
        cfg, alias, threshold = row['configuration_id'], row['alias'], row['threshold']
        key = cfg, alias, threshold
        arm = selected_arms[cfg] if alias == 'S@selected' else 'acs2her_p1of2' if alias == 'H@1/2' else None
        target = Fraction(plan['analysis']['thresholds'][cfg][threshold])
        if row['arm'] != arm or key in observed_thresholds or Fraction(row['threshold_exact']) != target:
            raise ValueError('threshold label or exact value differs')
        curve = {point: sum((counts[cfg, arm, seed, point] for seed in SEEDS), Fraction()) / 20 for point in configs[cfg]['targets']}
        first = next((point for point, value in curve.items() if value >= target), None)
        reported = int(row['first_nominal_step']) if row['first_nominal_step'] else None
        if first != reported:
            raise ValueError('first exact inclusive threshold crossing differs')
        observed_thresholds.add(key)
    if observed_thresholds != {(cfg, alias, threshold) for cfg in configs for alias in ('S@selected', 'H@1/2') for threshold in plan['analysis']['thresholds'][cfg]}:
        raise ValueError('threshold export is incomplete')
    baseline_checks = 0
    for cfg in configs:
        for threshold, agents in plan['analysis']['baseline_threshold_crossing_checks'][cfg].items():
            target = Fraction(plan['analysis']['thresholds'][cfg][threshold])
            for arm, expected in agents.items():
                curve = {point: sum((counts[cfg, arm, seed, point] for seed in SEEDS), Fraction()) / 20 for point in configs[cfg]['targets']}
                actual = next((point for point, value in curve.items() if value >= target), None)
                if actual != expected:
                    raise ValueError('registered baseline crossing differs')
                baseline_checks += 1
    tuning_scores = {}
    for row in read_csv(directory / 'tuning-seed-scores.csv'):
        key = row['configuration_id'], float(row['proportion']), int(row['seed'])
        if key in tuning_scores:
            raise ValueError('duplicate tuning seed score')
        tuning_scores[key] = Fraction(row['score_exact'])
    expected_tuning = {(cfg, p, seed) for cfg in configs for p in plan['candidates'] for seed in range(42, 47)}
    if set(tuning_scores) != expected_tuning:
        raise ValueError('incomplete validation-only tuning score export')
    tuning_means, tuning_keys = {}, set()
    for row in read_csv(directory / 'tuning.csv'):
        cfg, p = row['configuration_id'], float(row['proportion'])
        key = cfg, p
        values = [tuning_scores[cfg, p, seed] for seed in range(42, 47)]
        mean, se = moments(values)
        if key in tuning_keys or int(row['n']) != 5:
            raise ValueError('invalid tuning summary')
        close(row['mean'], mean)
        close(row['se'], se)
        if (row['selected'] == 'True') != (p == selected[cfg]):
            raise ValueError('tuning selection flag differs')
        tuning_keys.add(key)
        tuning_means[key] = mean
    if tuning_keys != {(cfg, p) for cfg in configs for p in plan['candidates']}:
        raise ValueError('incomplete tuning table')
    for cfg in configs:
        values = {p: [tuning_scores[cfg, p, seed] for seed in range(42, 47)] for p in plan['candidates']}
        if select_exact(values) != selected[cfg]:
            raise ValueError('saved selection differs from exact exported validation scores')
    across = read_csv(directory / 'tuning-across-mazes.csv')
    if {float(row['proportion']) for row in across} != set(plan['candidates']):
        raise ValueError('across-maze tuning table is incomplete')
    for row in across:
        p = float(row['proportion'])
        close(row['mean'], sum((tuning_means[cfg, p] for cfg in configs), Fraction()) / len(configs))
    allocations = read_csv(directory / 'allocation-attempts.csv')
    if len({row['job_id_raw'] for row in allocations}) != len(allocations):
        raise ValueError('allocation CPU was charged twice')
    total = sum(int(row['allocation_cpu_seconds']) for row in allocations)
    if total != results['execution']['allocated_cpu_seconds'] or total > 45 * 3600:
        raise ValueError('allocated CPU differs from scope or exceeds approved limit')
    purposes = read_csv(directory / 'allocation-purposes.csv')
    if {row['purpose'] for row in purposes} != {'pilot', 'tuning', 'control', 'test'}:
        raise ValueError('allocation purposes differ')
    for row in purposes:
        expected = sum(int(item['allocation_cpu_seconds']) for item in allocations if item['purpose'] == row['purpose'])
        if int(row['allocation_cpu_seconds']) != expected:
            raise ValueError('historical or missing allocation cost entered study purposes')
    if sum(int(row['allocation_cpu_seconds']) for row in allocations if row['purpose'] == 'pilot') != 4784:
        raise ValueError('pilot cost differs')
    proof = dict(plan_sha256=PLAN_SHA256, exact_success_counts=len(counts), exact_seed_scores=len(score_records),
                 paired_comparisons=len(comparison_records), registered_thresholds=len(observed_thresholds),
                 baseline_crossing_checks=baseline_checks, tuning_rows=len(tuning_keys), allocated_cpu_seconds=total,
                 exact_decomposition_checks=12, arithmetic='Independent Fraction reconstruction from integer CSV counts.',
                 csv_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob('*.csv'))})
    (directory / 'table-validation.json').write_text(json.dumps(proof, indent=2) + '\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    args = parser.parse_args()
    print(json.dumps(validate(args.analysis, args.plan)))
