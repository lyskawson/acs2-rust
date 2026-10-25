from fractions import Fraction
import math
import statistics
import sys

from diagnostics import value_groups
from protocol import ROOT
from grid_protocol import summarize

sys.path.append(str(ROOT / 'experiments/her-variants'))
from thresholds import crossings

SEEDS = list(range(42, 62))
VIEWS = ['checkpoint_mean', 'final_point']
REFERENCES = ['acs2her', 'acs2er_episodic', 'acs2', 'acs2er']


def exact_success(block):
    pairs, probability = block['evaluation_pairs'], block['success']
    if (type(pairs) is not int or pairs <= 0 or not math.isfinite(probability) or not 0 <= probability <= 1
            or block['evaluation_distribution'] != 'exhaustive_exact_weights'):
        raise ValueError('success requires a positive exhaustive pair count')
    count = round(probability * pairs)
    if abs(probability * pairs - count) > 1e-8:
        raise ValueError('success is not an exhaustive integer pair count')
    return Fraction(count, pairs)


def indexed(rows, targets):
    if not targets or sorted(set(targets)) != list(targets):
        raise ValueError('registered points must be increasing and unique')
    expected = {(seed, point) for seed in SEEDS for point in targets}
    result = {}
    for row in rows:
        key = row['seed'], row['nominal_step']
        if key not in expected or key in result or set(row['evaluation_blocks']) != {'training', 'validation', 'test'}:
            raise ValueError('unexpected, duplicate or incomplete analysis row')
        result[key] = row
    if set(result) != expected:
        raise ValueError('analysis requires all twenty seeds and every registered point')
    return result


def seed_scores(rows, targets, view, block='test'):
    if view not in VIEWS:
        raise ValueError('unregistered analysis view')
    data = indexed(rows, targets)
    points = targets if view == 'checkpoint_mean' else targets[-1:]
    return {seed: sum((exact_success(data[seed, point]['evaluation_blocks'][block]) for point in points), Fraction()) / len(points)
            for seed in SEEDS}


def success_summary(scores):
    if set(scores) != set(SEEDS):
        raise ValueError('success needs the twenty registered seed scores')
    return dict(summarize([float(scores[seed]) for seed in SEEDS]),
                mean_exact=str(sum(scores.values(), Fraction()) / 20),
                seed_scores={str(seed): str(scores[seed]) for seed in SEEDS})


def paired(left, right):
    if set(left) != set(SEEDS) or set(right) != set(SEEDS):
        raise ValueError('paired contrasts require matching registered seeds')
    differences = {seed: left[seed] - right[seed] for seed in SEEDS}
    return success_summary(differences)


def conditional_summary(values):
    observed = [value for value in values if value is not None]
    if not all(math.isfinite(value) for value in observed):
        raise ValueError('nonfinite diagnostic')
    count = len(observed)
    return dict(n=count, mean=statistics.mean(observed) if count else None,
                se=statistics.stdev(observed) / math.sqrt(count) if count > 1 else None)


def metrics(row):
    counts = row.get('learning_update_counts') or {}
    full = counts.get('full', row['online_updates'] + row['replay_updates'])
    result = {key: row[key] for key in ['actual_steps', 'knowledge', 'population_classifiers', 'population_numerosity',
        'population_known_bytes_lower_bound', 'replay_logical_bytes', 'trajectory_logical_bytes',
        'wall_seconds_train', 'wall_seconds_eval', 'wall_seconds_total']}
    result['memory_bytes_lower_bound'] = row['population_known_bytes_lower_bound'] + row['replay_logical_bytes'] + row['trajectory_logical_bytes']
    result['full_steps_per_environment_step'] = full / row['actual_steps']
    for key in ['realized_relabeled_share', 'trivial_copy_share', 'after_end_copy_share', 'fallback_share', 'rewarded_copy_share',
                'wall_copy_share', 'terminal_wall_copy_share', 'terminal_given_wall_copy_share']:
        result[key] = None
    for purpose in ['scoring', 'selection', 'provenance']:
        for name in ['reward_evaluations', 'reach_evaluations', 'objective_evaluations']:
            result['objective_' + purpose + '_' + name] = None
            result['objective_' + purpose + '_' + name + '_per_draw'] = None
    diagnostics = row.get('replay_diagnostics')
    if diagnostics is not None:
        draws = diagnostics['draws']
        copies = diagnostics['routes']['relabeled']
        number = copies['draws']
        result['realized_relabeled_share'] = number / draws
        result['fallback_share'] = diagnostics['fallbacks'] / draws
        if number:
            if copies['reward_sum'] != 1000 * copies['done']:
                raise ValueError('maze copy rewards differ from terminal-goal counts')
            result.update(trivial_copy_share=copies['already_reached'] / number,
                after_end_copy_share=copies['after_counterfactual_end'] / number,
                rewarded_copy_share=copies['done'] / number)
        walls = diagnostics.get('wall_bumps')
        if walls is not None:
            result['wall_copy_share'] = walls['unchanged_state'] / number if number else None
            result['terminal_wall_copy_share'] = walls['unchanged_state_terminal'] / number if number else None
            result['terminal_given_wall_copy_share'] = walls['unchanged_state_terminal'] / walls['unchanged_state'] if walls['unchanged_state'] else None
        for purpose, cost in diagnostics['objective_evaluations_by_purpose'].items():
            for name, value in cost.items():
                result['objective_' + purpose + '_' + name] = value
                result['objective_' + purpose + '_' + name + '_per_draw'] = value / draws
    groups = value_groups(row['evaluation_blocks']['test'])
    for group in ['all', 'successful', 'failed']:
        result['test_value_gap_' + group] = groups[group]['mean_estimate_minus_return'] if groups[group] is not None else None
    return result


def diagnostic_summaries(rows, targets, view):
    data = indexed(rows, targets)
    if view not in VIEWS:
        raise ValueError('unregistered diagnostic view')
    points = targets if view == 'checkpoint_mean' else targets[-1:]
    measured = {key: metrics(row) for key, row in data.items()}
    names = set(next(iter(measured.values())))
    if any(set(value) != names for value in measured.values()):
        raise ValueError('diagnostic schema changes inside an arm')
    result = {}
    for name in sorted(names):
        seed_values = []
        point_counts = {}
        for seed in SEEDS:
            values = [measured[seed, point][name] for point in points if measured[seed, point][name] is not None]
            point_counts[str(seed)] = len(values)
            seed_values.append(statistics.mean(values) if values else None)
        result[name] = dict(conditional_summary(seed_values), contributing_points_by_seed=point_counts)
    return result


def analyze_arm(rows, config, thresholds):
    targets = config['targets']
    data = indexed(rows, targets)
    scores = {view: seed_scores(rows, targets, view) for view in VIEWS}
    diagnostics = {view: diagnostic_summaries(rows, targets, view) for view in VIEWS}
    curve = []
    for point in targets:
        curve.append(dict(nominal_step=point,
            blocks={name: success_summary({seed: exact_success(data[seed, point]['evaluation_blocks'][name]) for seed in SEEDS})
                    for name in ['training', 'validation', 'test']},
            diagnostics={name: conditional_summary([metrics(data[seed, point])[name] for seed in SEEDS])
                         for name in ['knowledge', 'population_classifiers', 'memory_bytes_lower_bound',
                                      'trivial_copy_share', 'after_end_copy_share', 'fallback_share', 'rewarded_copy_share',
                                      'wall_copy_share', 'terminal_wall_copy_share', 'test_value_gap_all',
                                      'test_value_gap_successful', 'test_value_gap_failed']}))
    return dict(success={view: success_summary(scores[view]) for view in VIEWS}, diagnostics=diagnostics,
        threshold_crossings=crossings(rows, targets, thresholds), points=curve,
        maximum_observed_population=max(row['population_classifiers'] for row in rows))


def analyze_maze(dataset, config, plan):
    if set(dataset) != set(plan['agents']) | set(REFERENCES):
        raise ValueError('every registered arm and all four archived references are required')
    thresholds = plan['analysis']['thresholds'][config['id']]
    arms = {name: analyze_arm(rows, config, thresholds) for name, rows in dataset.items()}
    for threshold, names in plan['analysis']['baseline_threshold_crossing_checks'][config['id']].items():
        for name, expected in names.items():
            if arms[name]['threshold_crossings'][threshold] != expected:
                raise ValueError('archived rational threshold check differs')
    comparisons = []
    for view in VIEWS:
        for left, right in plan['analysis']['comparisons']:
            differences = paired(seed_scores(dataset[left], config['targets'], view), seed_scores(dataset[right], config['targets'], view))
            comparisons.append(dict(view=view, left=left, right=right, difference=differences,
                diagnostics={name: arms[name]['diagnostics'][view] for name in (left, right)}))
    return dict(configuration_id=config['id'], task=config['task'], cap=config['cap'], budget_steps=config['budget_steps'],
                seeds=SEEDS, thresholds=thresholds, arms=arms, comparisons=comparisons)
