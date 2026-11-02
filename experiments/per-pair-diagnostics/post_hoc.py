from collections import defaultdict
import math

from analysis import paired, summarize, weighted_mean
from protocol import ARMS, analysis_registration


def uniform_pairs(records):
    records = list(records)
    keys = {(record['start'], record['goal']) for record in records}
    if not records or len(keys) != len(records):
        raise ValueError('uniform sensitivity requires unique ordered pairs')
    return [dict(record, weight=1.0 / len(records)) for record in records]


def modal_statistics(records, actions=8):
    eligible = [record for record in records if not record['random_fallback']]
    by_start = defaultdict(list)
    for record in eligible:
        by_start[record['start']].append(record)
    modes = {start: max(range(actions), key=lambda action: (
        math.fsum(record['weight'] for record in group if record['first_action'] == action), -action))
        for start, group in by_start.items()}
    nonmodal = [record for record in eligible if record['first_action'] != modes[record['start']]]
    actual = weighted_mean(eligible, lambda record: record['first_action'] in record['optimal'])
    modal = weighted_mean(eligible, lambda record: modes[record['start']] in record['optimal'])
    eligible_weight = math.fsum(record['weight'] for record in eligible)
    nonmodal_weight = math.fsum(record['weight'] for record in nonmodal)
    total = math.fsum(record['weight'] for record in records)
    return dict(post_hoc=True, F_mod=modal, F_minus_F_mod=actual - modal if actual is not None else None,
        eligible_weight=eligible_weight, nonmodal_weight=nonmodal_weight,
        nonmodal_mass=nonmodal_weight / total if total else None,
        nonmodal_share_eligible=nonmodal_weight / eligible_weight if eligible_weight else None,
        nonmodal_optimal=weighted_mean(nonmodal, lambda record: record['first_action'] in record['optimal']),
        nonmodal_modal_optimal=weighted_mean(nonmodal, lambda record: modes[record['start']] in record['optimal']))


def winner_attributes(records):
    result = dict(post_hoc=True)
    for name, group in [('all', records), ('solved', [record for record in records if record['success']]),
                        ('failed', [record for record in records if not record['success']])]:
        winners = [record for record in group if not record['random_fallback']]
        def credit(record, count):
            return sum(winner['specified_goal_attributes'] == count for winner in record['winners']) / len(record['winners'])
        result[name] = dict(weight=math.fsum(record['weight'] for record in group), winner_weight=math.fsum(record['weight'] for record in winners),
            mass=[weighted_mean(group, lambda record: 0 if record['random_fallback'] else credit(record, count)) for count in range(3)],
            conditional_share=[weighted_mean(winners, lambda record: credit(record, count)) for count in range(3)])
    return result


def final_contrasts(metrics, configs, view, metric_names):
    result = []
    pairs = analysis_registration()['final_policy_contrasts']
    for config in configs:
        for left, right in pairs:
            for metric in metric_names:
                def values(arm):
                    return {seed: metrics[config['task'], arm, seed, config['targets'][-1], view][metric] for seed in range(42, 47)}
                result.append(dict(post_hoc=True, view=view, maze=config['task'], comparison=left + ' - ' + right,
                    metric=metric, nominal_step=config['targets'][-1], **paired(values(left), values(right))))
    return result


def curves(metrics, metric_names):
    cells = defaultdict(dict)
    for (maze, arm, seed, point, view), values in metrics.items():
        for metric in metric_names:
            cells[maze, arm, point, view, metric][seed] = values[metric]
    result = []
    for (maze, arm, point, view, metric), seeds in sorted(cells.items()):
        if set(seeds) != set(range(42, 47)):
            raise ValueError('curve requires all five seeds')
        result.append(dict(post_hoc=True, maze=maze, arm=arm, nominal_step=point, view=view, metric=metric,
                           seed_values=list(seeds.values()), **summarize(list(seeds.values()))))
    return result


def uniform_baselines(metrics, configs):
    result = []
    for config in configs:
        point = config['targets'][-1]
        for arm in ARMS:
            values = {seed: metrics[config['task'], arm, seed, point, 'uniform_pairs'] for seed in range(42, 47)}
            result.append(dict(post_hoc=True, view='uniform_pairs', maze=config['task'], comparison=arm + ': F - C',
                metric='F-C', nominal_step=point, **paired({seed: item['F'] for seed, item in values.items()},
                                                         {seed: item['C'] for seed, item in values.items()})))
    return result
