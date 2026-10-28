from fractions import Fraction
import math
import statistics

T19 = 2.093024054408263
COMPARISONS = [('S@selected', 'S@2/3'), ('S@selected', 'E'), ('S@selected', 'H@2/3'),
               ('S@2/3', 'H@1/2'), ('H@1/2', 'H@2/3')]


def success(block):
    pairs, value = block['evaluation_pairs'], block['success']
    if type(pairs) is not int or pairs <= 0 or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('integer exhaustive success counts are required')
    count = round(pairs * value)
    if not 0 <= count <= pairs or abs(pairs * value - count) > 1e-8:
        raise ValueError('integer exhaustive success counts are required')
    return Fraction(count, pairs)


def seed_score(rows, targets, view):
    indexed = {row['nominal_step']: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(targets) or view not in ('checkpoint_mean', 'final_point'):
        raise ValueError('complete registered points and view required')
    scores = [success(indexed[point]['evaluation_blocks']['test']) for point in targets]
    return sum(scores, Fraction()) / len(scores) if view == 'checkpoint_mean' else scores[-1]


def paired(left, right):
    if set(left) != set(range(42, 62)) or set(right) != set(left):
        raise ValueError('paired comparisons require seeds 42--61')
    differences = [left[seed] - right[seed] for seed in sorted(left)]
    mean = sum(differences, Fraction()) / 20
    se = statistics.stdev(differences) / math.sqrt(20)
    return dict(n=20, mean=float(mean), exact_mean=str(mean), se=se,
                ci_low=float(mean) - T19 * se, ci_high=float(mean) + T19 * se)


def contrasts(scores, selected):
    for seed in range(42, 62):
        rule = scores['S@2/3'][seed] - scores['H@1/2'][seed]
        dose = scores['H@1/2'][seed] - scores['H@2/3'][seed]
        total = scores['S@2/3'][seed] - scores['H@2/3'][seed]
        if not all(isinstance(value, Fraction) for value in (rule, dose, total)) or rule + dose != total:
            raise ValueError('exact rule plus dose decomposition failed')
    return {f'{left} - {right}': paired(scores[left], scores[right]) for left, right in COMPARISONS
            if selected != 2 / 3 or (left, right) != ('S@selected', 'S@2/3')}


def crossing(rows_by_seed, targets, threshold):
    if set(rows_by_seed) != set(range(42, 62)):
        raise ValueError('threshold crossing requires twenty paired seeds')
    for point in targets:
        values = [seed_score(rows, [point], 'final_point') for rows in
                  ([row for row in rows_by_seed[seed] if row['nominal_step'] == point] for seed in range(42, 62))]
        if sum(values, Fraction()) / 20 >= Fraction(threshold):
            return point
    return None

SEEDS = tuple(range(42, 62))
VIEWS = ('checkpoint_mean', 'final_point')


def indexed(rows, targets, seeds=SEEDS, blocks=('training', 'validation', 'test')):
    expected = {(seed, point) for seed in seeds for point in targets}
    result = {}
    pairs = {name: set() for name in blocks}
    for row in rows:
        key = row['seed'], row['nominal_step']
        if key not in expected or key in result or set(row['evaluation_blocks']) != set(blocks):
            raise ValueError('unexpected, duplicate or incomplete analysis row')
        for name in blocks:
            block = row['evaluation_blocks'][name]
            if block.get('evaluation_distribution') != 'exhaustive_exact_weights':
                raise ValueError('analysis requires exhaustive exact pair weights')
            success(block)
            pairs[name].add(block['evaluation_pairs'])
        result[key] = row
    if set(result) != expected or any(len(values) != 1 for values in pairs.values()):
        raise ValueError('analysis requires complete seeds, points and fixed pair counts')
    return result


def summary(values):
    observed = [value for value in values if value is not None]
    if not all(math.isfinite(value) for value in observed):
        raise ValueError('nonfinite summary value')
    n = len(observed)
    return dict(n=n, mean=float(statistics.mean(observed)) if n else None,
                se=statistics.stdev(observed) / math.sqrt(n) if n > 1 else None)


def success_summary(scores):
    if set(scores) != set(SEEDS):
        raise ValueError('success summaries require twenty registered seeds')
    return dict(summary([scores[seed] for seed in SEEDS]),
                exact_mean=str(sum(scores.values(), Fraction()) / 20),
                seed_scores={str(seed): str(scores[seed]) for seed in SEEDS})


def value_groups(block):
    rate = success(block)
    pairs = block['evaluation_pairs']
    count = rate * pairs
    if count.denominator != 1:
        raise ValueError('value groups require integer exhaustive successes')
    count = count.numerator
    values = block['value_diagnostics']
    if count == pairs:
        failed = None
    elif count == 0:
        failed = dict(values['all'])
    else:
        if values['successful'] is None:
            raise ValueError('successful value group is missing')
        mass = count / pairs
        failed = {key: (value - mass * values['successful'][key]) / (1 - mass)
                  for key, value in values['all'].items()}
    return dict(all=values['all'], successful=values['successful'] if count else None, failed=failed)


def metrics(row):
    result = {key: row[key] for key in ('actual_steps', 'knowledge', 'population_classifiers', 'population_numerosity',
        'population_known_bytes_lower_bound', 'replay_logical_bytes', 'trajectory_logical_bytes',
        'wall_seconds_train', 'wall_seconds_eval', 'wall_seconds_total')}
    result['memory_bytes_lower_bound'] = row['population_known_bytes_lower_bound'] + row['replay_logical_bytes'] + row['trajectory_logical_bytes']
    kinds = row.get('learning_update_counts') or {}
    result['full_steps_per_environment_step'] = kinds.get('full', row['online_updates'] + row['replay_updates']) / row['actual_steps']
    for name in ('realized_relabeled_share', 'trivial_copy_share', 'after_end_copy_share', 'fallback_share',
                 'rewarded_copy_share', 'wall_copy_share', 'terminal_wall_copy_share', 'terminal_given_wall_copy_share'):
        result[name] = None
    for purpose in ('scoring', 'selection', 'provenance'):
        for field in ('reward_evaluations', 'reach_evaluations', 'objective_evaluations'):
            result['objective_' + purpose + '_' + field] = None
            result['objective_' + purpose + '_' + field + '_per_draw'] = None
    diagnostics = row.get('replay_diagnostics')
    if diagnostics:
        draws = diagnostics['draws']
        copies = diagnostics['routes']['relabeled']
        n = copies['draws']
        if draws:
            result['realized_relabeled_share'] = n / draws
            result['fallback_share'] = diagnostics['fallbacks'] / draws
        if n:
            if copies['reward_sum'] != 1000 * copies['done']:
                raise ValueError('maze copied rewards do not equal terminal-goal counts')
            result.update(trivial_copy_share=copies['already_reached'] / n,
                          after_end_copy_share=copies['after_counterfactual_end'] / n,
                          rewarded_copy_share=copies['done'] / n)
        walls = diagnostics.get('wall_bumps')
        if walls:
            result.update(wall_copy_share=walls['unchanged_state'] / n if n else None,
                          terminal_wall_copy_share=walls['unchanged_state_terminal'] / n if n else None,
                          terminal_given_wall_copy_share=walls['unchanged_state_terminal'] / walls['unchanged_state'] if walls['unchanged_state'] else None)
        for purpose, counts in diagnostics['objective_evaluations_by_purpose'].items():
            for field, value in counts.items():
                result['objective_' + purpose + '_' + field] = value
                result['objective_' + purpose + '_' + field + '_per_draw'] = value / draws if draws else None
    for block in ('validation', 'test'):
        groups = value_groups(row['evaluation_blocks'][block]) if block in row['evaluation_blocks'] else {}
        for group in ('all', 'successful', 'failed'):
            value = groups.get(group)
            result[block + '_value_gap_' + group] = value['mean_estimate_minus_return'] if value is not None else None
    return result


def diagnostic_summaries(rows, targets, view, seeds=SEEDS, blocks=('training', 'validation', 'test')):
    data = indexed(rows, targets, seeds, blocks)
    if view not in VIEWS:
        raise ValueError('unregistered diagnostic view')
    points = targets if view == 'checkpoint_mean' else targets[-1:]
    measured = {key: metrics(row) for key, row in data.items()}
    names = set(next(iter(measured.values())))
    result = {}
    for name in sorted(names):
        values, contributors = [], {}
        for seed in seeds:
            observed = [measured[seed, point][name] for point in points if measured[seed, point][name] is not None]
            contributors[str(seed)] = len(observed)
            values.append(statistics.mean(observed) if observed else None)
        result[name] = dict(summary(values), contributing_points_by_seed=contributors)
    return result


def scores(rows, targets, view):
    indexed(rows, targets)
    return {seed: seed_score([row for row in rows if row['seed'] == seed], targets, view) for seed in SEEDS}


def analyze_arm(rows, config):
    targets = config['targets']
    data = indexed(rows, targets)
    return dict(success={view: success_summary(scores(rows, targets, view)) for view in VIEWS},
        diagnostics={view: diagnostic_summaries(rows, targets, view) for view in VIEWS},
        points=[dict(nominal_step=point,
            blocks={block: success_summary({seed: success(data[seed, point]['evaluation_blocks'][block]) for seed in SEEDS})
                    for block in ('training', 'validation', 'test')},
            diagnostics={name: summary([metrics(data[seed, point])[name] for seed in SEEDS]) for name in
                ('knowledge', 'population_classifiers', 'memory_bytes_lower_bound', 'realized_relabeled_share',
                 'trivial_copy_share', 'after_end_copy_share', 'fallback_share', 'rewarded_copy_share',
                 'wall_copy_share', 'test_value_gap_all', 'test_value_gap_successful', 'test_value_gap_failed')})
            for point in targets])


def analyze_maze(dataset, config, plan, selected):
    from protocol import HER, SHORTENED, label
    aliases = {'S@selected': label(SHORTENED, selected), 'S@2/3': label(SHORTENED, 2 / 3),
               'H@1/2': label(HER, 0.5), 'H@2/3': label(HER, 2 / 3), 'E': 'acs2er_episodic'}
    expected = set(aliases.values()) | {'acs2', 'acs2er'}
    if set(dataset) != expected or [list(pair) for pair in COMPARISONS] != plan['analysis']['comparisons']:
        raise ValueError('registered arms or comparison definitions differ')
    targets = config['targets']
    indexed_data = {name: indexed(rows, targets) for name, rows in dataset.items()}
    arms = {name: analyze_arm(rows, config) for name, rows in dataset.items()}
    thresholds = plan['analysis']['thresholds'][config['id']]
    crossings_by_arm = {}
    for alias in ('S@selected', 'H@1/2'):
        name = aliases[alias]
        by_seed = {seed: [row for row in dataset[name] if row['seed'] == seed] for seed in SEEDS}
        crossings_by_arm[alias] = {threshold: crossing(by_seed, targets, value) for threshold, value in thresholds.items()}
    for name, threshold in (('acs2', 'acs2_final'), ('acs2er', 'acs2er_final')):
        if Fraction(arms[name]['success']['final_point']['exact_mean']) != Fraction(thresholds[threshold]):
            raise ValueError('archived baseline final mean differs from registered threshold')
    baseline_checks = 0
    for threshold, references in plan['analysis']['baseline_threshold_crossing_checks'][config['id']].items():
        for name, expected_point in references.items():
            by_seed = {seed: [row for row in dataset[name] if row['seed'] == seed] for seed in SEEDS}
            if crossing(by_seed, targets, thresholds[threshold]) != expected_point:
                raise ValueError('registered exact baseline crossing differs')
            baseline_checks += 1
    point_checks = 0
    for seed in SEEDS:
        for point in targets:
            values = {alias: success(indexed_data[name][seed, point]['evaluation_blocks']['test']) for alias, name in aliases.items()}
            rule, dose = values['S@2/3'] - values['H@1/2'], values['H@1/2'] - values['H@2/3']
            if rule + dose != values['S@2/3'] - values['H@2/3']:
                raise ValueError('exact point-level decomposition differs')
            point_checks += 1
    comparisons, decompositions = [], {}
    for view in VIEWS:
        seed_values = {alias: scores(dataset[name], targets, view) for alias, name in aliases.items()}
        differences = contrasts(seed_values, selected)
        for left, right in COMPARISONS:
            key = left + ' - ' + right
            if key not in differences:
                continue
            comparisons.append(dict(view=view, left=aliases[left], right=aliases[right], left_alias=left, right_alias=right,
                difference=differences[key], seed_differences={str(seed): str(seed_values[left][seed] - seed_values[right][seed]) for seed in SEEDS},
                diagnostics={name: arms[name]['diagnostics'][view] for name in (aliases[left], aliases[right])}))
        rule = Fraction(differences['S@2/3 - H@1/2']['exact_mean'])
        dose = Fraction(differences['H@1/2 - H@2/3']['exact_mean'])
        total = sum((seed_values['S@2/3'][seed] - seed_values['H@2/3'][seed] for seed in SEEDS), Fraction()) / 20
        if rule + dose != total:
            raise ValueError('reported rule and dose contrasts do not sum to the previous contrast')
        decompositions[view] = dict(rule_exact=str(rule), dose_exact=str(dose), previous_contrast_exact=str(total), exact=True)
    return dict(configuration_id=config['id'], task=config['task'], cap=config['cap'], budget_steps=config['budget_steps'],
                selected_proportion=selected, aliases=aliases, seeds=list(SEEDS), arms=arms, comparisons=comparisons,
                thresholds=thresholds, threshold_crossings=crossings_by_arm,
                decomposition=dict(point_checks=point_checks, views=decompositions), baseline_crossing_checks=baseline_checks)
