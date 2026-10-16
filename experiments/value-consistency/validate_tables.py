import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from execution import APPROVED_SHA256, approved

REFERENCES = ['acs2her', 'acs2er_episodic', 'acs2', 'acs2er']
VIEWS = ['checkpoint_mean_within_seed', 'final_point']
T19 = 2.093024054408263


def table(root, name):
    with (root / (name + '.csv')).open() as stream:
        return list(csv.DictReader(stream))


def indexed(rows, fields, expected=None):
    result = {}
    for row in rows:
        key = tuple(int(row[field]) if field in ['seed', 'nominal_step'] else row[field] for field in fields)
        if key in result:
            raise ValueError('duplicate published table cell')
        result[key] = row
    if expected is not None and set(result) != set(expected):
        raise ValueError('published table scope differs from the registration')
    return result


def close(actual, expected, name):
    actual = float(actual)
    if not math.isfinite(actual) or not math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError('published statistic differs: ' + name)


def summary(row, values):
    n = len(values)
    if n != 20 or int(row['n']) != n:
        raise ValueError('published summary must use twenty seeds')
    mean = sum(values, Fraction()) / n
    se = math.sqrt(float(sum(((value-mean)**2 for value in values), Fraction()) / (n*(n-1))))
    for name, expected in dict(mean=float(mean), se=se, ci_low=float(mean)-T19*se, ci_high=float(mean)+T19*se).items():
        close(row[name], expected, name)


def validate(root, plan):
    root = Path(root)
    hashes = json.loads((root / 'files.sha256.json').read_text())
    for name, digest in hashes.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('derived table checksum differs')
    result = json.loads((root / 'results.json').read_text())
    if result['approved_plan_sha256'] != APPROVED_SHA256:
        raise ValueError('published analysis has a different registration')
    configurations = {config['id']: config for config in plan['configurations']}
    arms, seeds = plan['agents'] + REFERENCES, plan['seeds']
    checkpoints = {(identifier, arm, seed, point) for identifier, config in configurations.items()
                   for arm in arms for seed in seeds for point in config['targets']}
    rows = indexed(table(root, 'checkpoint-scores'), ['configuration', 'arm', 'seed', 'nominal_step'], checkpoints)
    values = {}
    for (identifier, arm, seed, point), row in rows.items():
        config = configurations[identifier]
        heads = 2 if arm.startswith('acs2vcp_k2') else 4 if arm.startswith('acs2vcp_k4') else 1
        pairs = config['registration']['blocks']['test']['evaluation_pairs']
        count = int(row['head_successes_sum'])
        if int(row['heads']) != heads or int(row['evaluation_pairs']) != pairs or not 0 <= count <= heads*pairs:
            raise ValueError('published checkpoint head/pair counts differ')
        if not point <= int(row['actual_steps']) < point + config['cap']:
            raise ValueError('published checkpoint overshoot differs')
        value = Fraction(count, heads*pairs)
        if Fraction(row['exact_test_success']) != value:
            raise ValueError('published exact checkpoint fraction differs from the integer counts')
        values[identifier, arm, seed, point] = value
    scores, means = {}, {}
    for identifier, config in configurations.items():
        for arm in arms:
            for seed in seeds:
                points = [values[identifier, arm, seed, point] for point in config['targets']]
                scores[identifier, arm, seed, VIEWS[0]] = sum(points, Fraction()) / len(points)
                scores[identifier, arm, seed, VIEWS[1]] = points[-1]
            for point in config['targets']:
                means[identifier, arm, point] = sum((values[identifier, arm, seed, point] for seed in seeds), Fraction()) / 20
    stored_scores = indexed(table(root, 'seed-scores'), ['configuration', 'arm', 'seed', 'view'], scores)
    for key, exact in scores.items():
        if Fraction(stored_scores[key]['exact_score']) != exact:
            raise ValueError('published within-seed score differs')
        close(stored_scores[key]['score'], exact, 'seed score')
    expected = {(identifier, arm, view) for identifier in configurations for arm in arms for view in VIEWS}
    primary = indexed(table(root, 'primary'), ['configuration', 'arm', 'view'], expected)
    for (identifier, arm, view), row in primary.items():
        summary(row, [scores[identifier, arm, seed, view] for seed in seeds])
    expected = {(identifier, left, right, view) for identifier in configurations
                for left, right in plan['analysis']['comparisons'] for view in VIEWS}
    comparisons = indexed(table(root, 'comparisons'), ['configuration', 'left', 'right', 'view'], expected)
    for (identifier, left, right, view), row in comparisons.items():
        summary(row, [scores[identifier, left, seed, view]-scores[identifier, right, seed, view] for seed in seeds])
    expected = {(identifier, arm, threshold) for identifier in configurations for arm in arms
                for threshold in plan['analysis']['thresholds'][identifier]}
    thresholds = indexed(table(root, 'thresholds'), ['configuration', 'arm', 'threshold'], expected)
    for (identifier, arm, threshold), row in thresholds.items():
        exact = Fraction(plan['analysis']['thresholds'][identifier][threshold])
        if Fraction(row['exact_threshold']) != exact:
            raise ValueError('published threshold differs from the registration')
        crossing = next((point for point in configurations[identifier]['targets'] if means[identifier, arm, point] >= exact), None)
        if row['first_registered_nominal_step'] != (str(crossing) if crossing is not None else ''):
            raise ValueError('published exact threshold crossing differs')
    expected = {(identifier, arm, point) for identifier, config in configurations.items()
                for arm in arms for point in config['targets']}
    curves = indexed([row for row in table(root, 'learning-curves') if row['block'] == 'test'],
                     ['configuration', 'arm', 'nominal_step'], expected)
    for (identifier, arm, point), row in curves.items():
        summary(row, [values[identifier, arm, seed, point] for seed in seeds])
    expected = {(identifier, arm, seed) for identifier in configurations for arm in arms for seed in seeds}
    costs = indexed(table(root, 'run-costs'), ['configuration', 'arm', 'seed'], expected)
    selected = 0
    for (_, arm, _), row in costs.items():
        scope = 'new_selected_run' if arm in plan['agents'] else 'reused_reference'
        if row['allocation_scope'] != scope:
            raise ValueError('published reference cost was charged to the new study')
        if scope == 'new_selected_run':
            selected += int(row['allocation_cpu_seconds'])
    verification = result['verification']
    failures = sum(attempt['allocation_cpu_seconds'] for batch in verification['batches'].values()
                   for attempt in batch['incomplete_attempts'])
    if (verification['runs'] != len(configurations)*len(plan['agents'])*20
            or selected + failures != verification['grid_allocated_cpu_seconds']
            or verification['pilot_allocated_cpu_seconds'] != 16041
            or selected + failures + 16041 != verification['allocated_cpu_seconds']
            or verification['allocated_cpu_seconds'] > 110*3600):
        raise ValueError('published allocation accounting differs')
    diagnostics = indexed(table(root, 'diagnostics'), ['configuration', 'arm', 'view', 'metric'])
    for identifier in configurations:
        for arm in [name for name in plan['agents'] if name.endswith('_uniform')]:
            for view in VIEWS:
                row = diagnostics[identifier, arm, view, 'replay_diagnostics.priority.value_queries']
                summary(row, [Fraction()] * 20)
                for key in diagnostics:
                    if key[:3] == (identifier, arm, view) and key[3].startswith(('replay_diagnostics.priority.proposal_variance.',
                                                                             'replay_diagnostics.priority.accepted_variance.')):
                        raise ValueError('published control has an unmeasured training variance')
    return dict(checkpoints=len(rows), seed_scores=len(scores), primary=len(primary), comparisons=len(comparisons),
                thresholds=len(thresholds), costs=len(costs), allocated_cpu_seconds=verification['allocated_cpu_seconds'],
                method='independent rational arithmetic from integer head success counts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.analysis, approved(Path(__file__).with_name('plan-draft.json')))))
