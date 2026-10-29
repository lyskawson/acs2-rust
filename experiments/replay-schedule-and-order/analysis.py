from fractions import Fraction
import statistics

from protocol import ER, SHORTENED, SCHEDULES, label
from paired_statistics import SEEDS, VIEWS, contrast, crossing, indexed, scores, success, success_summary, summary

CONTRASTS = {
    'S_X - S': [('S_X', 1), ('S', -1)],
    'E_X - E': [('E_X', 1), ('E', -1)],
    '(S_X - S) - (E_X - E)': [('S_X', 1), ('S', -1), ('E_X', -1), ('E', 1)],
    'S_X - E_X': [('S_X', 1), ('E_X', -1)],
}
ORDER_CONTRAST = [('S_X', 1), ('order_control', -1)]


def metrics(row):
    result = {name: row[name] for name in ('actual_steps', 'knowledge', 'population_classifiers', 'population_numerosity',
        'population_known_bytes_lower_bound', 'replay_logical_bytes', 'trajectory_logical_bytes',
        'wall_seconds_train', 'wall_seconds_eval', 'wall_seconds_total')}
    result['memory_bytes_lower_bound'] = sum(row[key] for key in
        ('population_known_bytes_lower_bound', 'replay_logical_bytes', 'trajectory_logical_bytes'))
    updates = row['online_updates'] + row['replay_updates']
    result['updates_per_interaction'] = updates / row['actual_steps']
    diag = row.get('replay_diagnostics')
    names = ['draws', 'copies', 'fallbacks', 'realized_copy_share', 'trivial_copy_share', 'post_end_copy_share',
             'rewarded_copy_share', 'fallbacks_per_draw', 'wall_copy_share', 'terminal_wall_copy_share']
    names += ['pass_' + name for name in ('passes', 'requested_relabel', 'fallbacks', 'cut_passes', 'available_steps', 'emitted_steps')]
    names += ['requested_pass_share', 'fallbacks_per_pass', 'cuts_per_pass', 'available_steps_per_pass']
    for name in names:
        result[name] = None
    for purpose in ('scoring', 'selection', 'provenance'):
        for name in ('reward_evaluations', 'reach_evaluations', 'objective_evaluations'):
            result['objective_' + purpose + '_' + name] = None
            result['objective_' + purpose + '_' + name + '_per_draw'] = None
    ratio = lambda value, denominator: value / denominator if denominator else None
    if diag:
        copy = diag['routes']['relabeled']
        draws, copies = diag['draws'], copy['draws']
        if copy['reward_sum'] != 1000 * copy['done']:
            raise ValueError('copied maze rewards differ from terminal goal counts')
        result.update(draws=draws, copies=copies, fallbacks=diag['fallbacks'],
            realized_copy_share=ratio(copies, draws), trivial_copy_share=ratio(copy['already_reached'], copies),
            post_end_copy_share=ratio(copy['after_counterfactual_end'], copies),
            rewarded_copy_share=ratio(copy['done'], copies), fallbacks_per_draw=ratio(diag['fallbacks'], draws))
        walls = diag.get('wall_bumps')
        if walls:
            result.update(wall_copy_share=ratio(walls['unchanged_state'], copies),
                          terminal_wall_copy_share=ratio(walls['unchanged_state_terminal'], copies))
        for purpose, counts in diag['objective_evaluations_by_purpose'].items():
            for name, value in counts.items():
                result['objective_' + purpose + '_' + name] = value
                result['objective_' + purpose + '_' + name + '_per_draw'] = ratio(value, draws)
        passes = diag.get('passes')
        if passes:
            result.update({'pass_' + key: value for key, value in passes.items()})
            result.update(requested_pass_share=ratio(passes['requested_relabel'], passes['passes']),
                fallbacks_per_pass=ratio(passes['fallbacks'], passes['passes']),
                cuts_per_pass=ratio(passes['cut_passes'], passes['passes']),
                available_steps_per_pass=ratio(passes['available_steps'], passes['passes']))
    for name, block in row['evaluation_blocks'].items():
        rate = success(block)
        diagnostics = block['value_diagnostics']
        all_gap = diagnostics['all']['mean_estimate_minus_return']
        successful = diagnostics['successful']['mean_estimate_minus_return'] if rate else None
        failed = (all_gap - float(rate) * successful) / (1 - float(rate)) if 0 < rate < 1 else all_gap if rate == 0 else None
        result.update({name + '_value_gap_' + group: value for group, value in
                       [('all', all_gap), ('successful', successful), ('failed', failed)]})
    return result


def diagnostic_summaries(rows, targets, view, seeds=SEEDS, blocks=('training', 'validation', 'test')):
    data = indexed(rows, targets, seeds, blocks)
    if view not in VIEWS:
        raise ValueError('unregistered diagnostic view')
    points = targets if view == 'checkpoint_mean' else targets[-1:]
    measured = {key: metrics(row) for key, row in data.items()}
    names = sorted(next(iter(measured.values())))
    result = {}
    for name in names:
        values, contributors = [], {}
        for seed in seeds:
            present = [measured[seed, point][name] for point in points if measured[seed, point][name] is not None]
            values.append(statistics.mean(present) if present else None)
            contributors[str(seed)] = len(present)
        result[name] = dict(summary(values), contributing_points_by_seed=contributors)
    return result


def intervals(rows):
    previous_steps = previous_updates = 0
    result = []
    for row in rows:
        steps, updates = row['actual_steps'], row['online_updates'] + row['replay_updates']
        if steps <= previous_steps or updates < previous_updates:
            raise ValueError('nonmonotone checkpoint counters')
        ratio = Fraction(updates - previous_updates, steps - previous_steps)
        result.append(dict(nominal_step=row['nominal_step'], actual_steps=steps,
            interval_interactions=steps - previous_steps, interval_updates=updates - previous_updates,
            updates_per_interaction=float(ratio), updates_per_interaction_exact=str(ratio)))
        previous_steps, previous_updates = steps, updates
    return result


def analyze_arm(rows, config):
    targets = config['targets']
    data = indexed(rows, targets)
    return dict(success={view: success_summary(scores(rows, targets, view)) for view in VIEWS},
        diagnostics={view: diagnostic_summaries(rows, targets, view) for view in VIEWS},
        points=[dict(nominal_step=point,
            blocks={block: success_summary({seed: success(data[seed, point]['evaluation_blocks'][block]) for seed in SEEDS})
                    for block in ('training', 'validation', 'test')},
            diagnostics={key: summary([metrics(data[seed, point])[key] for seed in SEEDS]) for key in
                ('knowledge', 'population_classifiers', 'memory_bytes_lower_bound', 'realized_copy_share',
                 'requested_pass_share', 'trivial_copy_share', 'post_end_copy_share',
                 'test_value_gap_all', 'test_value_gap_successful', 'test_value_gap_failed')}) for point in targets])


def analyze_maze(dataset, config, plan, selected):
    base = label(config, SHORTENED, 'base')
    expected = {base, ER, 'acs2', 'acs2er'}
    aliases = {}
    for axis in ('B', 'C'):
        mode = selected[axis]['setting']
        if mode == 'base':
            continue
        if mode not in (SCHEDULES if axis == 'B' else ['backward']):
            raise ValueError('unregistered selected setting')
        names = dict(S=base, E=ER, S_X=label(config, SHORTENED, mode), E_X=label(config, ER, mode))
        if axis == 'C':
            names['order_control'] = label(config, SHORTENED, 'random-pass')
        aliases[axis] = names
        expected.update(names.values())
    if set(dataset) != expected or list(CONTRASTS) != plan['analysis']['comparisons']:
        raise ValueError('registered final arms or comparisons differ')
    arms = {name: analyze_arm(rows, config) for name, rows in dataset.items()}
    comparisons, thresholds = [], {}
    for axis, names in aliases.items():
        for view in VIEWS:
            values = {alias: scores(dataset[name], config['targets'], view) for alias, name in names.items()}
            definitions = dict(CONTRASTS)
            if axis == 'C':
                definitions['S_backward - S_order_control'] = ORDER_CONTRAST
            for name, terms in definitions.items():
                comparisons.append(dict(axis=axis, setting=selected[axis]['setting'], view=view,
                    comparison=name, terms=[dict(arm=names[alias], weight=weight) for alias, weight in terms],
                    difference=contrast(values, terms)))
        thresholds[axis] = {name: crossing(dataset[names['S_X']], config['targets'], value)
                            for name, value in plan['analysis']['thresholds'][config['id']].items()}
    baseline_checks = 0
    for name, threshold in (('acs2', 'acs2_final'), ('acs2er', 'acs2er_final')):
        target = plan['analysis']['thresholds'][config['id']][threshold]
        if Fraction(arms[name]['success']['final_point']['exact_mean']) != Fraction(target):
            raise ValueError('baseline final mean differs from the registered threshold')
    for threshold, references in plan['analysis']['baseline_threshold_crossing_checks'][config['id']].items():
        value = plan['analysis']['thresholds'][config['id']][threshold]
        for name, expected_point in references.items():
            if crossing(dataset[name], config['targets'], value) != expected_point:
                raise ValueError('registered exact baseline crossing differs')
            baseline_checks += 1
    return dict(configuration_id=config['id'], task=config['task'], cap=config['cap'],
        selections=selected, aliases=aliases, arms=arms, comparisons=comparisons,
        thresholds=plan['analysis']['thresholds'][config['id']], threshold_crossings=thresholds,
        baseline_crossing_checks=baseline_checks)
