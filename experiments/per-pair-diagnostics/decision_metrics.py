from collections import defaultdict
from itertools import combinations
import math

from analysis import statistics_for, weighted_mean
from post_hoc import modal_statistics


def within_start(records):
    groups = defaultdict(list)
    for record in records:
        groups[record['start']].append(record)
    covariance = estimate_variance = truth_variance = 0.0
    total = math.fsum(record['weight'] for record in records)
    for group in groups.values():
        estimate_mean = weighted_mean(group, lambda record: record['estimated_first_action_value'])
        truth_mean = weighted_mean(group, lambda record: record['q_star'])
        covariance += math.fsum(record['weight'] * (record['estimated_first_action_value'] - estimate_mean)
                                * (record['q_star'] - truth_mean) for record in group)
        estimate_variance += math.fsum(record['weight'] * (record['estimated_first_action_value'] - estimate_mean)**2 for record in group)
        truth_variance += math.fsum(record['weight'] * (record['q_star'] - truth_mean)**2 for record in group)
    return dict(within_start_slope=covariance / truth_variance if truth_variance else None,
        within_start_sd_estimate=math.sqrt(estimate_variance / total) if total else None,
        within_start_sd_q=math.sqrt(truth_variance / total) if total else None,
        within_start_correlation=covariance / math.sqrt(estimate_variance * truth_variance) if estimate_variance * truth_variance else None,
        within_start_weight=total)


def tie_discrimination(records):
    groups = defaultdict(list)
    for record in records:
        if not record['random_fallback']:
            groups[record['start']].append(record)
    eligible_mass = different_mass = tie_mass = 0.0
    for group in groups.values():
        for left, right in combinations(group, 2):
            if left['goal'] == right['goal'] or left['optimal'] & right['optimal']:
                continue
            weight = left['weight'] * right['weight']
            eligible_mass += weight
            if left['first_action'] != right['first_action']:
                different_mass += weight
                if all(any(winner['specified_goal_attributes'] == 0 for winner in record['winners']) for record in (left, right)):
                    tie_mass += weight
    return dict(D_tie_implied=tie_mass / eligible_mass if eligible_mass else None,
        D_tie_share=tie_mass / different_mass if different_mass else None,
        D_tie_weight=tie_mass, D_different_weight=different_mass)


def decision_metrics(records):
    records = list(records)
    measured = statistics_for(records)
    modal = modal_statistics(records)
    eligible = [record for record in records if not record['random_fallback']]
    f, c, fm = measured['F'], measured['C'], modal['F_mod']
    return dict(F=f, C=c, F_mod=fm, estimate_minus_q=measured['values']['all']['estimate_minus_q'],
        estimate_above_v=weighted_mean(eligible, lambda record: record['estimated_first_action_value'] > record['v_star']),
        estimate_above_v_weight=math.fsum(record['weight'] for record in eligible),
        F_minus_C=f - c if f is not None else None,
        F_mod_minus_C=fm - c if fm is not None else None,
        F_minus_F_mod=f - fm if f is not None else None,
        **within_start(records), **tie_discrimination(records))
