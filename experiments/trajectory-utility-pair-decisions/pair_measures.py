from collections import defaultdict
import math

from pair_sources import per_pair_definitions

ANALYSIS, DECISIONS = per_pair_definitions()
BINS = (('1-2', 1, 2), ('3-4', 3, 4), ('5-6', 5, 6), ('7+', 7, None))
MEASURES = ('F', 'fallback_share', 'value_error', 'absolute_value_error', 'overestimation', 'success')
FIELDS = ('pairs', 'fallbacks', 'eligible', 'optimal', 'error', 'absolute_error', 'overestimated', 'successes')
VERDICTS = ('supported', 'contradicted', 'inconclusive')


def distance_bin(distance):
    if type(distance) is not int or distance < 1:
        raise ValueError('a test pair needs a positive shortest distance')
    return next(name for name, low, high in BINS if low <= distance and (high is None or distance <= high))


def attach(records, truth):
    return ANALYSIS.attach(records, truth)


def distance_sums(attached):
    groups = defaultdict(list)
    for record in attached:
        groups[record['shortest_distance']].append(record)
    result = {}
    for distance, records in groups.items():
        eligible = [record for record in records if not record['random_fallback']]
        result[distance] = dict(
            pairs=len(records), fallbacks=len(records) - len(eligible), eligible=len(eligible),
            optimal=sum(record['first_action'] in record['optimal'] for record in eligible),
            error=math.fsum(record['estimated_first_action_value'] - record['q_star'] for record in records),
            absolute_error=math.fsum(abs(record['estimated_first_action_value'] - record['q_star']) for record in records),
            overestimated=sum(record['estimated_first_action_value'] > record['v_star'] for record in eligible),
            successes=sum(bool(record['success']) for record in records))
    return result


def combine(parts):
    parts = list(parts)
    result = {}
    for field in FIELDS:
        values = [part[field] for part in parts]
        result[field] = math.fsum(values) if field in ('error', 'absolute_error') else sum(values)
    return result


def measures(sums):
    pairs, eligible = sums['pairs'], sums['eligible']
    if not pairs:
        raise ValueError('a measured cell needs at least one pair')
    return dict(F=sums['optimal'] / eligible if eligible else None, fallback_share=sums['fallbacks'] / pairs,
                value_error=sums['error'] / pairs, absolute_value_error=sums['absolute_error'] / pairs,
                overestimation=sums['overestimated'] / eligible if eligible else None,
                success=sums['successes'] / pairs)


def summarize(values):
    return ANALYSIS.summarize(list(values))


def paired(left, right):
    return ANALYSIS.paired(left, right)


def judge(comparison, sign):
    if sign not in (1, -1):
        raise ValueError('a prediction has sign +1 or -1')
    if comparison['mean'] is None:
        return 'inconclusive'
    low, high = comparison['ci_low'], comparison['ci_high']
    if sign > 0 and low > 0 or sign < 0 and high < 0:
        return 'supported'
    if sign > 0 and high < 0 or sign < 0 and low > 0:
        return 'contradicted'
    return 'inconclusive'


def pooled(cell_sums, tasks, distances):
    parts = [cell_sums[task][distance] for task in tasks for distance in distances if distance in cell_sums[task]]
    return combine(parts) if parts else None
