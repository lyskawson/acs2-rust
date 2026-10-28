from fractions import Fraction
import math
import statistics

T19 = 2.093024054408263
COMPARISONS = [('S@selected', 'S@2/3'), ('S@selected', 'E'), ('S@selected', 'H@2/3'),
               ('S@2/3', 'H@1/2'), ('H@1/2', 'H@2/3')]


def success(block):
    pairs, value = block['evaluation_pairs'], block['success']
    count = round(pairs * value)
    if type(pairs) is not int or pairs <= 0 or not 0 <= count <= pairs or abs(pairs * value - count) > 1e-8:
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
