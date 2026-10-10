import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'grid'))
from collect import complete_rows, outside_repository, validate_row
from grid_attempts import select_attempts
from grid_protocol import atomic_json, compare_pilot, load_plan, summarize, verdict
from row_schema import evaluation_blocks


def analyze_values(series, targets, floor, ceiling, threshold=0.5):
    seeds = list(range(42, 62))
    expected = {(agent, seed) for agent in ('acs2', 'acs2er') for seed in seeds}
    if set(series) != expected or any(len(values) != len(targets) for values in series.values()):
        raise ValueError('registered analysis needs both agents, all twenty seeds and every point')
    points = {agent: [{'nominal_step': point, **summarize([series[agent, seed][index] for seed in seeds])}
                      for index, point in enumerate(targets)] for agent in ('acs2', 'acs2er')}
    scores = {key: statistics.mean(values) for key, values in series.items()}
    differences = [scores['acs2er', seed] - scores['acs2', seed] for seed in seeds]
    crossings = {agent: next((point['nominal_step'] for point in points[agent] if point['mean'] >= threshold * ceiling), None)
                 for agent in ('acs2', 'acs2er')}
    return {'points': points, 'primary_score': {agent: summarize([scores[agent, seed] for seed in seeds])
                                               for agent in ('acs2', 'acs2er')},
            'paired_primary_difference': summarize(differences),
            'seed_scores': [{'seed': seed, 'acs2': scores['acs2', seed], 'acs2er': scores['acs2er', seed],
                             'paired_difference': differences[index]} for index, seed in enumerate(seeds)],
            'final_success': {agent: points[agent][-1] for agent in points},
            'verdict': verdict(points['acs2'][-1], points['acs2er'][-1], points['acs2'][0], floor, ceiling),
            'random_floor': floor, 'ceiling': ceiling, 'threshold_fraction_of_ceiling': threshold,
            'first_nominal_interactions_at_threshold': crossings}


def analyze(root, pilot):
    digest = (root / 'plan.sha256').read_text().strip()
    plan = load_plan(root / 'plan.json', digest)
    if plan['stage'] not in ('approved_grid', 'grid_pending_approval'):
        raise ValueError('cost pilots are not final baseline results')
    approval = json.loads((root / 'approval.json').read_text())
    if not approval.get('approved') or approval['plan_sha256'] != digest:
        raise ValueError('no matching grid approval')
    ledger = json.loads((root / 'ledger.json').read_text())
    results = []
    for config in plan['configurations']:
        selected, inventory = select_attempts(root, ledger, config)
        if set(selected) != {(agent, seed) for agent in plan['agents'] for seed in range(42, 62)}:
            raise ValueError(f"incomplete configuration: {config['id']}")
        series = {name: {} for name in ('training', 'validation', 'test')}
        costs, compared = [], 0
        for key, attempt in sorted(selected.items()):
            agent, seed = key
            rows, tail = complete_rows(attempt['path'] / 'rows.jsonl')
            if tail:
                raise ValueError('incomplete raw tail in selected attempt')
            for row, target in zip(rows, config['targets']):
                validate_row(row, config, agent, seed, plan['measurement_commit'], target)
                if row['cpu_model'] != plan['cpu_model'] or row['preset'] != plan['preset']:
                    raise ValueError('CPU model or preset differs')
            for name in series:
                series[name][key] = [evaluation_blocks(row)[name]['success'] for row in rows]
            if seed == 42:
                pilot_paths = list(pilot.glob(f"attempts/*/measurement/{config['id']}_{agent}_s42/rows.jsonl"))
                if len(pilot_paths) != 1:
                    raise ValueError('pilot seed must have one complete reference')
                reference, reference_tail = complete_rows(pilot_paths[0])
                if reference_tail:
                    raise ValueError('pilot reference has an incomplete tail')
                by_target = {row['nominal_step']: row for row in reference}
                compared += compare_pilot(rows, [by_target[target] for target in config['targets']])
            resources, tail = complete_rows(attempt['path'] / 'resources.jsonl')
            if tail:
                raise ValueError('incomplete resource tail')
            train_cpu = sum(item['train_cpu_seconds_estimate'] for item in resources)
            completion = json.loads((attempt['path'] / 'completion.json').read_text())
            costs.append({'agent': agent, 'seed': seed, 'training_cpu_seconds_estimate': train_cpu,
                          'process_cpu_seconds': completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                          'peak_rss_kib': completion['peak_rss_kib'],
                          'training_steps_per_cpu_second_estimate': rows[-1]['actual_steps'] / train_cpu})
        blocks = {name: analyze_values(series[name], config['targets'],
                    config['registration']['blocks'][name]['random_floor'],
                    config['registration']['blocks'][name]['reachable_within_cap'], plan['threshold_fraction_of_ceiling'])
                  for name in series}
        results.append({'configuration': config['id'], 'primary_block': 'test', 'blocks': blocks,
                        'costs': costs, 'attempts': inventory, 'seed42_learning_fields_compared': compared})
    destination = outside_repository(root / 'analysis')
    destination.mkdir(exist_ok=True)
    atomic_json(destination / 'results.json', {'plan_sha256': digest, 'configurations': results})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.pilot)
