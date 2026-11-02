from collections import defaultdict
from itertools import combinations
import math
import statistics

from protocol import T4
from pair_schema import cell, goal_cell


def truth_index(model):
    if model['gamma'] != 0.95 or model['reward'] != 1000 or model['actions'] != 8:
        raise ValueError('ground truth uses a different objective or action space')
    result = {}
    for item in model['pairs']:
        key = tuple(item['start']), tuple(item['goal'])
        if key in result:
            raise ValueError('duplicate ground-truth pair')
        result[key] = dict(item, cap=model['cap'])
    return result


def attach(records, model):
    result = []
    for record in records:
        key = cell(record['start']), goal_cell(record['goal'])
        truth = model[key]
        distance = record['shortest_distance']
        value = 1000.0 * 0.95 ** (distance - 1) if 0 < distance <= truth['cap'] else 0.0
        if (distance != truth['shortest_distance'] or not math.isclose(truth['value'], value, rel_tol=0, abs_tol=2e-12)
                or not math.isclose(max(truth['action_values']), truth['value'], rel_tol=0, abs_tol=2e-12)):
            raise ValueError('recorded shortest distance and recomputed V* disagree; STOP')
        decision = record['first_decision']
        result.append(dict(record, start=key[0], goal=key[1], optimal=set(truth['optimal_actions']),
                           q_star=truth['action_values'][record['first_action']], v_star=truth['value'],
                           random_fallback=decision['random_fallback'], winners=decision['winners']))
    return result


def pool_blocks(blocks):
    if set(blocks) != {'training', 'validation', 'test'}:
        raise ValueError('primary pooling requires all three evaluation blocks')
    return [record for records in blocks.values() for record in records]


def weighted_mean(records, value):
    denominator = math.fsum(record['weight'] for record in records)
    return math.fsum(record['weight'] * value(record) for record in records) / denominator if denominator else None


def ceiling(records, actions=8):
    by_start = defaultdict(list)
    for record in records:
        by_start[record['start']].append(record)
    total = math.fsum(record['weight'] for record in records)
    numerator = math.fsum(max(math.fsum(record['weight'] for record in group if action in record['optimal'])
                             for action in range(actions)) for group in by_start.values())
    return numerator / total if total else None


def statistics_for(records, actions=8):
    records = list(records)
    eligible = [record for record in records if not record['random_fallback']]
    by_start = defaultdict(list)
    for record in eligible:
        by_start[record['start']].append(record)
    mass = math.fsum(record['weight'] for record in eligible)
    modes = math.fsum(max(math.fsum(record['weight'] for record in group if record['first_action'] == action)
                          for action in range(actions)) for group in by_start.values())
    discrimination_mass = different = both = 0.0
    for group in by_start.values():
        for left, right in combinations(group, 2):
            if left['goal'] == right['goal'] or left['optimal'] & right['optimal']:
                continue
            weight = left['weight'] * right['weight']
            discrimination_mass += weight
            different += weight * (left['first_action'] != right['first_action'])
            both += weight * (left['first_action'] in left['optimal'] and right['first_action'] in right['optimal'])
    values = {}
    for name, group in [('all', records), ('solved', [record for record in records if record['success']]),
                        ('failed', [record for record in records if not record['success']])]:
        winners = [record for record in group if not record['random_fallback']]
        credit = lambda record: sum(winner['specified_goal_attributes'] > 0 for winner in record['winners']) / len(record['winners'])
        values[name] = dict(pair_count=len(group), weight=math.fsum(record['weight'] for record in group),
            estimate_minus_q=weighted_mean(group, lambda record: record['estimated_first_action_value'] - record['q_star']),
            estimate_minus_v=weighted_mean(group, lambda record: record['estimated_first_action_value'] - record['v_star']),
            winner_goal_share=weighted_mean(winners, credit), winner_weight=math.fsum(record['weight'] for record in winners),
            winner_goal_mass=weighted_mean(group, lambda record: 0 if record['random_fallback'] else credit(record)))
    return dict(F=weighted_mean(eligible, lambda record: record['first_action'] in record['optimal']),
        C=ceiling(records, actions), random_reference=weighted_mean(records, lambda record: len(record['optimal']) / actions),
        C_eligible=ceiling(eligible, actions), random_reference_eligible=weighted_mean(eligible, lambda record: len(record['optimal']) / actions),
        M=modes / mass if mass else None, D=different / discrimination_mass if discrimination_mass else None,
        D_both_optimal=both / discrimination_mass if discrimination_mass else None,
        pair_count=len(records), eligible_count=len(eligible), eligible_weight=mass,
        D_weight=discrimination_mass, random_fallback_share=weighted_mean(records, lambda record: record['random_fallback']),
        values=values)


def summarize(values):
    if len(values) != 5:
        raise ValueError('registered interval requires exactly five paired seeds')
    if any(value is None for value in values):
        return dict(n=sum(value is not None for value in values), mean=None, se=None, ci_low=None, ci_high=None)
    if any(not math.isfinite(value) for value in values):
        raise ValueError('nonfinite seed statistic')
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(5)
    return dict(n=5, mean=mean, se=se, ci_low=mean - T4 * se, ci_high=mean + T4 * se)


def paired(left, right):
    if set(left) != set(range(42, 47)) or set(right) != set(left):
        raise ValueError('contrast requires matching seeds 42--46')
    values = [None if left[seed] is None or right[seed] is None else left[seed] - right[seed] for seed in sorted(left)]
    return dict(summarize(values), seed_differences=values)


def consistency(seed_records):
    if set(seed_records) != set(range(42, 47)):
        raise ValueError('consistency requires seeds 42--46')
    indexed = {seed: {(record['start'], record['goal']): record for record in records} for seed, records in seed_records.items()}
    keys = set(indexed[42])
    if any(set(records) != keys for records in indexed.values()):
        raise ValueError('consistency pair universes differ')
    histogram = [0.0] * 6
    expected = [0.0] * 6
    pairs = []
    for key in sorted(keys):
        records = [indexed[seed][key] for seed in sorted(indexed)]
        if len({record['weight'] for record in records}) != 1:
            raise ValueError('consistency pair weights differ')
        solved = sum(record['success'] for record in records)
        rate = solved / 5
        weight = records[0]['weight']
        histogram[solved] += weight
        for count in range(6):
            expected[count] += weight * math.comb(5, count) * rate**count * (1 - rate)**(5 - count)
        pairs.append(dict(start=key[0], goal=key[1], solved_seeds=solved, mean_rate=rate, weight=weight))
    total = math.fsum(histogram)
    return dict(pairs=pairs, observed=[value / total for value in histogram],
                binomial_plugin=[value / total for value in expected], post_hoc=True)
