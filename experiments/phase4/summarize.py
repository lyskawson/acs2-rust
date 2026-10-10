import argparse
import csv
import json
import math
from pathlib import Path

SCHEMA = 2
SEEDS = tuple(range(42, 62))
STRATEGIES = ('final', 'future', 'episode', 'uniform_real')
RULES = ('every_transition', 'from_non_goal_state', 'counterfactual_episode')
CONFIGURATIONS = (
    'maze4_c5_full', 'maze4_c5_p4', 'maze6_c10_full', 'maze6_c10_p4',
    'handeye4_c50_full', 'handeye4_c50_p4', 'taxi_c200_full',
    'bitflip8_c8_full', 'handeye5_c50_full', 'maze7_c10_full',
)
METRICS = (
    'admissible_share', 'already_reached', 'after_counterfactual_end', 'done',
    'outside_candidates', 'no_admissible_goal', 'mean_reward',
    'mean_objective_evaluations', 'original', 'relabeled', 'fallback',
    'mean_reward_evaluations', 'mean_reach_evaluations',
    'mean_scoring_evaluations', 'mean_selection_evaluations', 'mean_provenance_evaluations',
)
UNBOUNDED = (
    'mean_reward', 'mean_objective_evaluations', 'mean_reward_evaluations', 'mean_reach_evaluations',
    'mean_scoring_evaluations', 'mean_selection_evaluations', 'mean_provenance_evaluations',
)
PURPOSES = ('mean_scoring_evaluations', 'mean_selection_evaluations', 'mean_provenance_evaluations')
ROUTES = ('original', 'relabeled')
ROUTE_FLAGS = ('already_reached', 'after_counterfactual_end', 'done', 'outside_candidates')
ROUTE_METRICS = ROUTE_FLAGS + ('mean_reward',)
TOLERANCE = 1e-10


def mean_and_se(values):
    if len(values) < 2:
        raise ValueError('at least two independent seeds are needed')
    mean = math.fsum(values) / len(values)
    squared = math.fsum((value - mean) * (value - mean) for value in values)
    return mean, math.sqrt(squared / ((len(values) - 1) * len(values)))


def defined_mean_and_se(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return mean_and_se(values)


def close(actual, expected, scale=1.0):
    return abs(actual - expected) <= TOLERANCE * max(1.0, abs(scale))


def given_route(expected, route):
    joint = expected['routes'][route]
    share = joint['share']
    if share <= 0.0:
        return None
    result = {flag: joint[flag] / share for flag in ROUTE_FLAGS}
    result['mean_reward'] = joint['reward'] / share
    return result


def validate_routes(row, expected):
    routes = expected['routes']
    if set(routes) != set(ROUTES):
        raise ValueError('composition routes are incomplete')
    for route in ROUTES:
        joint = routes[route]
        if not close(joint['share'], expected[route]):
            raise ValueError('route share disagrees with the pooled route share')
        for name in ROUTE_FLAGS + ('reward',):
            if not math.isfinite(joint[name]) or joint[name] < -TOLERANCE:
                raise ValueError('invalid route composition')
        for flag in ROUTE_FLAGS:
            if joint[flag] > joint['share'] + TOLERANCE:
                raise ValueError('a route flag exceeds the route share')
    for flag in ROUTE_FLAGS:
        if not close(routes['original'][flag] + routes['relabeled'][flag], expected[flag]):
            raise ValueError('routes do not reproduce the pooled ' + flag)
    reward = routes['original']['reward'] + routes['relabeled']['reward']
    if not close(reward, expected['mean_reward'], expected['mean_reward']):
        raise ValueError('routes do not reproduce the pooled mean reward')
    if not close(routes['original']['share'], expected['fallback']):
        raise ValueError('at proportion one the original route is the fallback')
    original = routes['original']
    if max(original['already_reached'], original['after_counterfactual_end'], original['outside_candidates']) > TOLERANCE:
        raise ValueError('an original goal was reached before its episode ended or is not a candidate')
    relabeled = routes['relabeled']
    if row['rule'] != 'every_transition' and relabeled['already_reached'] > TOLERANCE:
        raise ValueError('non-goal admissibility relabeled with a reached start')
    if row['rule'] == 'counterfactual_episode' and relabeled['after_counterfactual_end'] > TOLERANCE:
        raise ValueError('counterfactual admissibility relabeled after the end')
    if row['candidate_filter'] and relabeled['outside_candidates'] > TOLERANCE:
        raise ValueError('candidate filter relabeled with an outside goal')


def validate_purposes(row, expected):
    total = expected['mean_objective_evaluations']
    if not close(sum(expected[name] for name in PURPOSES), total, total):
        raise ValueError('objective evaluations by purpose do not reproduce the total')
    if not close(expected['mean_reward_evaluations'], 1.0):
        raise ValueError('every draw is scored by exactly one reward evaluation')
    scoring = expected['mean_scoring_evaluations']
    if scoring < expected['mean_reward_evaluations'] - TOLERANCE or scoring > 2 * expected['mean_reward_evaluations'] + TOLERANCE:
        raise ValueError('scoring is one reward and at most one reach evaluation per draw')
    if row['rule'] == 'every_transition' and expected['mean_selection_evaluations'] > TOLERANCE:
        raise ValueError('every-transition admissibility reads no objective call')


def validate(row):
    if row['strategy'] not in STRATEGIES or row['rule'] not in RULES or type(row['candidate_filter']) is not bool:
        raise ValueError('unexpected selection')
    if not 0 <= row['actual_steps'] - row['target_steps'] < row['cap']:
        raise ValueError('episode overshoot outside the measurement contract')
    if row['target_steps'] != 20000 or row['composition_uses_rng'] or row['relabeled_proportion'] != 1.0:
        raise ValueError('unexpected measurement design')
    expected = row['expected']
    for metric in METRICS:
        if not math.isfinite(expected[metric]) or expected[metric] < -TOLERANCE:
            raise ValueError('invalid composition metric')
        if metric not in UNBOUNDED and expected[metric] > 1.0 + TOLERANCE:
            raise ValueError('share outside [0, 1]')
    if abs(expected['original'] + expected['relabeled'] - 1.0) > TOLERANCE:
        raise ValueError('original and relabeled routes do not exhaust the draws')
    if abs(expected['fallback'] - expected['no_admissible_goal']) > TOLERANCE:
        raise ValueError('empty goal distributions must fall back at proportion one')
    if abs(expected['mean_objective_evaluations'] - expected['mean_reward_evaluations'] - expected['mean_reach_evaluations']) > 1e-8:
        raise ValueError('objective cost components disagree')
    if row['candidate_filter'] and expected['outside_candidates'] > TOLERANCE:
        raise ValueError('candidate filter leaked an outside goal')
    if row['rule'] != 'every_transition' and expected['already_reached'] > TOLERANCE:
        raise ValueError('non-goal admissibility leaked a reached start')
    if row['rule'] == 'counterfactual_episode' and expected['after_counterfactual_end'] > TOLERANCE:
        raise ValueError('counterfactual admissibility leaked a post-end transition')
    validate_purposes(row, expected)
    validate_routes(row, expected)


def summarize(rows, seeds=SEEDS, configurations=CONFIGURATIONS):
    groups = {}
    runs = {}
    for row in rows:
        if row['schema'] != SCHEMA or row['seed'] not in seeds or row['configuration'] not in configurations:
            raise ValueError('unexpected schema, seed or configuration')
        validate(row)
        key = row['configuration'], row['strategy'], row['rule'], row['candidate_filter']
        group = groups.setdefault(key, {})
        if row['seed'] in group:
            raise ValueError('duplicate seed in a composition cell')
        group[row['seed']] = row
        run = row['configuration'], row['seed']
        identity = tuple(row[name] for name in ('actual_steps', 'episodes', 'goal_pool', 'candidate_count_at_end', 'commit', 'source_state'))
        if runs.setdefault(run, identity) != identity:
            raise ValueError('selection variants did not use the same trajectory')
    if len(groups) != len(configurations) * len(STRATEGIES) * len(RULES) * 2:
        raise ValueError('incomplete selection grid')
    result = []
    for key in sorted(groups):
        group = groups[key]
        if set(group) != set(seeds):
            raise ValueError('incomplete seed set')
        summary = dict(zip(('configuration', 'strategy', 'rule', 'candidate_filter'), key))
        summary['seeds'] = len(seeds)
        for metric in METRICS:
            mean, se = mean_and_se([group[seed]['expected'][metric] for seed in seeds])
            summary[metric + '_mean'] = mean
            summary[metric + '_se'] = se
        for route in ROUTES:
            given = [given_route(group[seed]['expected'], route) for seed in seeds]
            given = [values for values in given if values is not None]
            summary[route + '_only_seeds'] = len(given)
            for metric in ROUTE_METRICS:
                mean, se = defined_mean_and_se([values[metric] for values in given])
                summary[f'{route}_only_{metric}_mean'] = mean
                summary[f'{route}_only_{metric}_se'] = se
        result.append(summary)
    return result


def cell(row, name, scale):
    mean, se = row[name + '_mean'], row[name + '_se']
    if mean is None:
        return '—'
    if se is None:
        return f'{mean * scale:.4f}'
    return f'{mean * scale:.4f} ± {se * scale:.4f}'


def report(summaries):
    selection = ['Konfiguracja', 'Strategia', 'Reguła', 'Filtr']
    lines = [
        '# Skład trajektorii przy polityce losowej — faza 4, schemat 2', '',
        '10 konfiguracji 3b, ziarna 42–61, po co najmniej 20 000 kroków; kończymy ostatni epizod.',
        'Średnia ± SE między 20 ziarnami. Udziały w procentach; nagroda w skali zadania.',
        'Skład obejmuje fallback do oryginalnego celu. Pula kandydatów: cele otrzymane do końca danego epizodu.',
        'Koszt oznacza oczekiwaną liczbę publicznych wywołań celu przez sampler na próbkę; analiza ma oddzielny licznik.', '',
        '## Skład łączny', '',
        '| ' + ' | '.join(selection + ['Dopuszczalne %', 'Już osiągnięte %', 'Po końcu %', 'Done %', 'Poza kandydatami %', 'Brak celu %', 'Nagroda', 'Ewaluacje celu']) + ' |',
        '|' + '---|' * 12,
    ]
    for row in summaries:
        cells = [row['configuration'], row['strategy'], row['rule'], str(row['candidate_filter'])]
        for metric in METRICS[:8]:
            cells.append(cell(row, metric, 1.0 if metric in UNBOUNDED else 100.0))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += [
        '', '## Koszt według celu wywołania', '',
        'Ocena próbki: nagroda i zakończenie losowanej próbki. Dobór: wywołania, które czyta reguła dopuszczalności.',
        'Proweniencja: pozostałe, tylko do flag diagnostycznych. Suma odtwarza „Ewaluacje celu”.', '',
        '| ' + ' | '.join(selection + ['Ocena próbki', 'Dobór celu', 'Ocena + dobór', 'Proweniencja', 'Razem']) + ' |',
        '|' + '---|' * 9,
    ]
    for row in summaries:
        cells = [row['configuration'], row['strategy'], row['rule'], str(row['candidate_filter'])]
        for metric in ('mean_scoring_evaluations', 'mean_selection_evaluations'):
            cells.append(cell(row, metric, 1.0))
        cells.append(f"{row['mean_scoring_evaluations_mean'] + row['mean_selection_evaluations_mean']:.4f}")
        cells.append(cell(row, 'mean_provenance_evaluations', 1.0))
        cells.append(cell(row, 'mean_objective_evaluations', 1.0))
        lines.append('| ' + ' | '.join(cells) + ' |')
    for route, title in (('relabeled', 'Tylko próbki z podmienionym celem'), ('original', 'Tylko próbki z drogi oryginalnej (fallback)')):
        lines += [
            '', f'## {title}', '',
            'Na ziarno: iloraz wielkości łącznych ważonych krokami przez udział drogi; potem średnia ± SE po ziarnach,',
            'w których droga ma losowania (kolumna „Ziarna”).', '',
            '| ' + ' | '.join(selection + ['Udział drogi %', 'Ziarna', 'Już osiągnięte %', 'Po końcu %', 'Done %', 'Poza kandydatami %', 'Nagroda']) + ' |',
            '|' + '---|' * 11,
        ]
        for row in summaries:
            cells = [row['configuration'], row['strategy'], row['rule'], str(row['candidate_filter'])]
            cells.append(cell(row, route, 100.0))
            cells.append(str(row[route + '_only_seeds']))
            for metric in ROUTE_METRICS:
                cells.append(cell(row, f'{route}_only_{metric}', 1.0 if metric == 'mean_reward' else 100.0))
            lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines()]
    summaries = summarize(rows)
    destination = args.out.resolve()
    if any((parent / '.git').exists() for parent in (destination, *destination.parents)):
        raise ValueError('measurement output belongs outside a checkout')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    with (destination / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (destination / 'composition-report.md').write_text(report(summaries))
    print(f'{len(rows)} raw rows, {len(summaries)} complete summary cells')


if __name__ == '__main__':
    main()
