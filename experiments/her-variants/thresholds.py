from fractions import Fraction
import math

THRESHOLDS = {
    'maze4_c5_pairs': {'acs2_final': '527/1620', 'acs2er_final': '92/405'},
    'maze5_c8_pairs': {'acs2_final': '523/2220', 'acs2er_final': '89/555'},
    'maze6_c10_pairs': {'acs2_final': '253/888', 'acs2er_final': '739/4440'},
    'maze7_c10_pairs': {'acs2_final': '1151/4320', 'acs2er_final': '259/1440'},
    'woods14_c18_pairs': {'acs2_final': '15/76', 'acs2er_final': '11/60'},
    'mazema_c9_pairs': {'acs2_final': '49/180', 'acs2er_final': '193/900'},
}
BASELINE_CROSSINGS = {
    'maze4_c5_pairs': {'acs2_final': {'acs2': 2000000, 'acs2er': None},
                      'acs2er_final': {'acs2': 300000, 'acs2er': 2000000}},
    'maze5_c8_pairs': {'acs2_final': {'acs2': 1000000, 'acs2er': None},
                      'acs2er_final': {'acs2': 300000, 'acs2er': 1000000}},
    'maze6_c10_pairs': {'acs2_final': {'acs2': 1000000, 'acs2er': None},
                      'acs2er_final': {'acs2': 300000, 'acs2er': 100000}},
    'maze7_c10_pairs': {'acs2_final': {'acs2': 2000000, 'acs2er': None},
                      'acs2er_final': {'acs2': 300000, 'acs2er': 100000}},
    'woods14_c18_pairs': {'acs2_final': {'acs2': 100000, 'acs2er': 30000},
                        'acs2er_final': {'acs2': 100000, 'acs2er': 30000}},
    'mazema_c9_pairs': {'acs2_final': {'acs2': 2000000, 'acs2er': None},
                      'acs2er_final': {'acs2': 100000, 'acs2er': 30000}},
}


def success_count(row):
    block = row['evaluation_blocks']['test']
    pairs, success = block['evaluation_pairs'], block['success']
    if (type(pairs) is not int or pairs <= 0 or not math.isfinite(success)
            or not 0 <= success <= 1 or block['evaluation_distribution'] != 'exhaustive_exact_weights'):
        raise ValueError('thresholds require exhaustive success counts over a positive pair count')
    count = round(success * pairs)
    if abs(success * pairs - count) > 1e-8:
        raise ValueError('success is not an integer count over the registered pairs')
    return count, pairs


def mean_curve(rows, targets, seeds=range(42, 62)):
    targets, seeds = tuple(targets), tuple(seeds)
    if (not targets or sorted(set(targets)) != list(targets) or len(seeds) != 20
            or set(seeds) != set(range(42, 62))):
        raise ValueError('thresholds require registered increasing points and seeds 42--61')
    expected = {(seed, point) for seed in seeds for point in targets}
    indexed, pair_counts = {}, set()
    for row in rows:
        key = row['seed'], row['nominal_step']
        if key not in expected or key in indexed:
            raise ValueError('unexpected or duplicate threshold row')
        if 'evaluation_blocks' in row:
            count, pairs = success_count(row)
        else:
            count, pairs = row['successes'], row['evaluation_pairs']
        if type(count) is not int or type(pairs) is not int or pairs <= 0 or not 0 <= count <= pairs:
            raise ValueError('invalid exact success count')
        pair_counts.add(pairs)
        indexed[key] = Fraction(count, pairs)
    if set(indexed) != expected or len(pair_counts) != 1:
        raise ValueError('incomplete seeds or points, or inconsistent test pair counts')
    return {point: sum((indexed[seed, point] for seed in seeds), Fraction()) / 20 for point in targets}


def crossings(rows, targets, thresholds):
    if any(not isinstance(value, str) or not 0 <= Fraction(value) <= 1 for value in thresholds.values()):
        raise ValueError('thresholds must be exact fraction strings between zero and one')
    curve = mean_curve(rows, targets)
    return {name: next((point for point, value in curve.items() if value >= Fraction(threshold)), None)
            for name, threshold in thresholds.items()}
