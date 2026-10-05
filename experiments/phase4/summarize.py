import argparse
import csv
import json
import math
from pathlib import Path

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
)


def mean_and_se(values):
    if len(values) < 2:
        raise ValueError('at least two independent seeds are needed')
    mean = math.fsum(values) / len(values)
    squared = math.fsum((value - mean) * (value - mean) for value in values)
    return mean, math.sqrt(squared / ((len(values) - 1) * len(values)))


def summarize(rows, seeds=SEEDS, configurations=CONFIGURATIONS):
    groups = {}
    runs = {}
    for row in rows:
        if row['schema'] != 1 or row['seed'] not in seeds or row['configuration'] not in configurations:
            raise ValueError('unexpected schema, seed or configuration')
        if row['strategy'] not in STRATEGIES or row['rule'] not in RULES or type(row['candidate_filter']) is not bool:
            raise ValueError('unexpected selection')
        if not 0 <= row['actual_steps'] - row['target_steps'] < row['cap']:
            raise ValueError('episode overshoot outside the measurement contract')
        if row['target_steps'] != 20000 or row['composition_uses_rng'] or row['relabeled_proportion'] != 1.0:
            raise ValueError('unexpected measurement design')
        expected = row['expected']
        for metric in METRICS:
            if not math.isfinite(expected[metric]) or expected[metric] < -1e-10:
                raise ValueError('invalid composition metric')
            if metric not in ('mean_reward', 'mean_objective_evaluations', 'mean_reward_evaluations', 'mean_reach_evaluations') and expected[metric] > 1.0 + 1e-10:
                raise ValueError('share outside [0, 1]')
        if abs(expected['original'] + expected['relabeled'] - 1.0) > 1e-10:
            raise ValueError('original and relabeled routes do not exhaust the draws')
        if abs(expected['fallback'] - expected['no_admissible_goal']) > 1e-10:
            raise ValueError('empty goal distributions must fall back at proportion one')
        if abs(expected['mean_objective_evaluations'] - expected['mean_reward_evaluations'] - expected['mean_reach_evaluations']) > 1e-8:
            raise ValueError('objective cost components disagree')
        if row['candidate_filter'] and expected['outside_candidates'] > 1e-10:
            raise ValueError('candidate filter leaked an outside goal')
        if row['rule'] != 'every_transition' and expected['already_reached'] > 1e-10:
            raise ValueError('non-goal admissibility leaked a reached start')
        if row['rule'] == 'counterfactual_episode' and expected['after_counterfactual_end'] > 1e-10:
            raise ValueError('counterfactual admissibility leaked a post-end transition')
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
        result.append(summary)
    return result


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
    lines = [
        '# Skład trajektorii przy polityce losowej — faza 4', '',
        '10 konfiguracji 3b, ziarna 42–61, po co najmniej 20 000 kroków; kończymy ostatni epizod.',
        'Średnia ± SE między 20 ziarnami. Udziały w procentach; nagroda w skali zadania.',
        'Skład obejmuje fallback do oryginalnego celu. Pula kandydatów: cele otrzymane do końca danego epizodu.',
        'Koszt oznacza oczekiwaną liczbę publicznych wywołań celu przez sampler na próbkę; analiza ma oddzielny licznik.', '',
        '| Konfiguracja | Strategia | Reguła | Filtr | Dopuszczalne % | Już osiągnięte % | Po końcu % | Done % | Poza kandydatami % | Brak celu % | Nagroda | Ewaluacje celu |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for row in summaries:
        cells = [row['configuration'], row['strategy'], row['rule'], str(row['candidate_filter'])]
        for metric in METRICS[:8]:
            scale = 100.0 if metric not in ('mean_reward', 'mean_objective_evaluations') else 1.0
            cells.append(f"{row[metric + '_mean'] * scale:.4f} ± {row[metric + '_se'] * scale:.4f}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    (destination / 'composition-report.md').write_text('\n'.join(lines) + '\n')
    print(f'{len(rows)} raw rows, {len(summaries)} complete summary cells')


if __name__ == '__main__':
    main()
