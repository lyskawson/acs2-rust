import math
import statistics
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'grid'))

from grid_protocol import summarize
from thresholds import BASELINE_CROSSINGS, crossings, mean_curve

SEEDS = list(range(42, 62))
VALUES_ARMS = {'her_values', 'her_values_shortened', 'her_values_p2of3'}


def update_counts(row):
    if 'learning_update_counts' in row:
        counts = row['learning_update_counts']
        return counts['full'], counts['values_only']
    return row['online_updates'] + row['replay_updates'], 0


def metrics(row):
    full, values = update_counts(row)
    steps = row['actual_steps']
    result = dict(success=row['evaluation_blocks']['test']['success'], knowledge=row['knowledge'],
        population=row['population_classifiers'], numerosity=row['population_numerosity'],
        memory_bytes_lower_bound=row['population_known_bytes_lower_bound'] + row['replay_logical_bytes'] + row['trajectory_logical_bytes'],
        full_steps_per_environment_step=full / steps, values_steps_per_environment_step=values / steps,
        full_updates=full, values_only_updates=values, train_wall_seconds=row['wall_seconds_train'],
        eval_wall_seconds=row['wall_seconds_eval'], total_wall_seconds=row['wall_seconds_total'])
    diagnostics = row.get('replay_diagnostics')
    if diagnostics is not None:
        draws = diagnostics['draws']
        result.update(realized_relabeled_share=diagnostics['realized_relabeled_share'],
                      fallback_share=diagnostics['fallbacks'] / draws)
        for route, counts in diagnostics['routes'].items():
            for field in ['draws', 'already_reached', 'after_counterfactual_end', 'done', 'outside_candidates', 'reward_sum']:
                result['route_' + route + '_' + field] = counts[field]
            for field in ['already_reached', 'after_counterfactual_end', 'done', 'outside_candidates']:
                result['route_' + route + '_' + field + '_share'] = counts[field] / counts['draws'] if counts['draws'] else None
            result['route_' + route + '_mean_reward'] = counts['mean_reward']
        for purpose, counts in diagnostics['objective_evaluations_by_purpose'].items():
            for field, value in counts.items():
                result['objective_' + purpose + '_' + field] = value
                result['objective_' + purpose + '_' + field + '_per_draw'] = value / draws
    return result


def indexed(rows, targets):
    expected = {(seed, point) for seed in SEEDS for point in targets}
    result = {}
    for row in rows:
        key = row['seed'], row['nominal_step']
        if key not in expected or key in result:
            raise ValueError('unexpected or duplicate final analysis row')
        if set(row['evaluation_blocks']) != {'training', 'validation', 'test'}:
            raise ValueError('final analysis requires twenty seeds and all three evaluation blocks')
        result[key] = row
    if set(result) != expected:
        raise ValueError('final analysis is missing seeds or registered points')
    return result


def aggregate(values):
    if all(value is None for value in values):
        return None
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError('incomplete or nonfinite secondary metric')
    return summarize(values)


def analyze_arm(rows, config, thresholds):
    targets = config['targets']
    data = indexed(rows, targets)
    measures = {key: metrics(row) for key, row in data.items()}
    keys = set(measures[SEEDS[0], targets[0]])
    if any(set(value) != keys for value in measures.values()):
        raise ValueError('metric schema changes inside one arm')
    points = []
    for point in targets:
        points.append(dict(nominal_step=point,
            metrics={name: aggregate([measures[seed, point][name] for seed in SEEDS]) for name in sorted(keys)}))
    seed_means = {seed: {name: None if measures[seed, targets[0]][name] is None else
                        statistics.mean(measures[seed, point][name] for point in targets)
                        for name in keys} for seed in SEEDS}
    checkpoint_means = {name: aggregate([seed_means[seed][name] for seed in SEEDS]) for name in sorted(keys)}
    exact_curve = mean_curve(rows, targets)
    return dict(primary=checkpoint_means['success'], checkpoint_means=checkpoint_means,
        final=points[-1]['metrics'], points=points,
        seed_scores=[dict(seed=seed, **seed_means[seed]) for seed in SEEDS],
        thresholds=thresholds, threshold_crossings=crossings(rows, targets, thresholds),
        exact_mean_test_curve={str(point): str(value) for point, value in exact_curve.items()},
        maximum_population=max(row['population_classifiers'] for row in rows))


def paired(left, right, field):
    left_seeds = {row['seed']: row[field] for row in left['seed_scores']}
    right_seeds = {row['seed']: row[field] for row in right['seed_scores']}
    if set(left_seeds) != set(SEEDS) or set(right_seeds) != set(SEEDS):
        raise ValueError('paired analysis needs the same twenty registered seeds')
    return summarize([left_seeds[seed] - right_seeds[seed] for seed in SEEDS])


def analyze_maze(dataset, config, plan):
    expected = set(plan['agents']) | {'her_values_p2of3', 'acs2', 'acs2er'}
    if set(dataset) != expected:
        raise ValueError('analysis must include every registered arm, matched control and both baselines')
    thresholds = plan['analysis']['thresholds'][config['id']]
    arms = {name: analyze_arm(rows, config, thresholds) for name, rows in dataset.items()}
    for threshold, expected_crossings in BASELINE_CROSSINGS[config['id']].items():
        for name, expected_point in expected_crossings.items():
            if arms[name]['threshold_crossings'][threshold] != expected_point:
                raise ValueError('archived baseline threshold crossing differs from the registered check')
    comparisons = []
    for left, right in plan['analysis']['comparisons']:
        comparison = dict(left=left, right=right, primary_difference=paired(arms[left], arms[right], 'success'),
                          knowledge_difference=paired(arms[left], arms[right], 'knowledge'))
        if left in VALUES_ARMS or right in VALUES_ARMS:
            comparison['model_updates_and_knowledge'] = {
                name: dict(full_steps_per_environment_step=arms[name]['checkpoint_means']['full_steps_per_environment_step'],
                           knowledge=arms[name]['checkpoint_means']['knowledge'],
                           points=[dict(nominal_step=point['nominal_step'],
                                        full_steps_per_environment_step=point['metrics']['full_steps_per_environment_step'],
                                        knowledge=point['metrics']['knowledge']) for point in arms[name]['points']])
                for name in [left, right]}
        comparisons.append(comparison)
    return dict(configuration=config['id'], task=config['task'], cap=config['cap'], budget_steps=config['budget_steps'],
                seeds=SEEDS, primary_block='test', primary_rule=plan['analysis']['primary'],
                interval_rule=plan['analysis']['paired_interval'], arms=arms, comparisons=comparisons)
