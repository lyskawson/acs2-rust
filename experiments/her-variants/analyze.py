import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'grid'))
from analysis import analyze_maze
from execution import APPROVED_SHA256, approved, catalog, choose, control_manifest, grid_manifest, selected_proportions
from grid_protocol import atomic_json, load_plan
from identity import field_bits
from protocol import BASELINE_SHA256
from verify import verify


def reference_resolver(study, pilots):
    def resolve(value):
        path = Path(value)
        prefix = Path('/home/alelys2099/tu_runs/5')
        if not path.is_relative_to(prefix):
            raise ValueError('identity reference is outside this study line')
        relative = path.relative_to(prefix)
        if relative.parts[0] == 'approved-her-study':
            return Path(study).joinpath(*relative.parts[1:])
        if relative.parts[0] in {'cost-pilot', 'baseline-regression', 'u1-observer-pilot'}:
            return Path(pilots) / relative
        raise ValueError('unrecognized identity reference root')
    return resolve


def entries(root, resolver, save=True):
    checked = verify(root, resolver)
    values, _ = catalog(root, validate=False)
    if save:
        atomic_json(Path(root) / 'verification.json', checked)
    return values, checked


def require_derivation(child, expected):
    for field, value in expected.items():
        if child.get(field) != value:
            raise ValueError(f'execution manifest differs from mechanical registration: {field}')


def matched_entries(identifier, proportion, main, controls, seeds):
    source = main if proportion == 2 / 3 else controls
    if len(seeds) != 20 or set(seeds) != set(range(42, 62)):
        raise ValueError('matched comparison requires twenty final-grid seeds')
    try:
        group = [source[identifier, 'her_values', seed] for seed in seeds]
    except KeyError as error:
        raise ValueError('matched comparison is missing a final-grid seed') from error
    if any(set(row['evaluation_blocks']) != {'training', 'validation', 'test'} for entry in group for row in entry['rows']):
        raise ValueError('matched comparison cannot reuse validation-only tuning rows')
    return group


def cost(entry, root):
    path = Path(entry['path']).parent
    completion = json.loads((path / 'completion.json').read_text())
    resources = [json.loads(line) for line in (path / 'resources.jsonl').read_text().splitlines()]
    train_cpu = sum(row['train_cpu_seconds_estimate'] for row in resources)
    return dict(process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                peak_rss_kib=completion['peak_rss_kib'], training_cpu_seconds_estimate=train_cpu,
                training_steps_per_cpu_second_estimate=entry['rows'][-1]['actual_steps'] / train_cpu,
                source_directory=str(path.relative_to(Path(root))), root=str(root))


def collect(study, baseline, pilots):
    study, baseline, pilots = Path(study), Path(baseline), Path(pilots)
    if (study / 'STOP.json').exists():
        raise ValueError('STOP blocks analysis')
    plan = approved(study / 'approved-plan.json')
    approval = json.loads((study / 'approval.json').read_text())
    if approval.get('approved') is not True or approval['plan_sha256'] != APPROVED_SHA256:
        raise ValueError('analysis requires the approved registration')
    resolver = reference_resolver(study, pilots)
    tuning, tune_checked = entries(study / 'tuning', resolver)
    selected = json.loads((study / 'selections.json').read_text())
    if field_bits(choose(plan, tuning)) != field_bits(selected):
        raise ValueError('saved tuning selection differs from the registered mechanical rule')
    main, main_checked = entries(study / 'main-grid', resolver)
    child = load_plan(study / 'main-grid/plan.json', (study / 'main-grid/plan.sha256').read_text().strip())
    require_derivation(child, grid_manifest(plan, selected))
    choices = selected_proportions(selected)
    control_entries, checked = {}, {'tuning': tune_checked, 'main-grid': main_checked}
    for config in plan['configurations']:
        if choices[config['id']]['her_values'] == 2 / 3:
            continue
        root = study / ('control-' + config['id'])
        values, result = entries(root, resolver)
        child = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
        require_derivation(child, control_manifest(plan, selected, config['id']))
        control_entries.update(values)
        checked[root.name] = result
    if (baseline / 'plan.sha256').read_text().strip() != BASELINE_SHA256:
        raise ValueError('baseline archive is not the registered pair grid')
    baseline_entries, baseline_checked = entries(baseline, None, save=False)
    phase_seconds = 5445 + sum(result['allocation_cpu_seconds'] for result in checked.values())
    if phase_seconds > 60 * 3600:
        raise ValueError('allocated CPU-hours exceed the approved study limit')
    datasets, costs, reuse = {}, {}, {}
    for config in plan['configurations']:
        identifier = config['id']
        datasets[identifier], costs[identifier] = {}, {}
        for name in plan['agents']:
            group = [main[identifier, name, seed] for seed in plan['seeds']]
            datasets[identifier][name] = [row for entry in group for row in entry['rows']]
            costs[identifier][name] = [dict(seed=seed, **cost(entry, study / 'main-grid')) for seed, entry in zip(plan['seeds'], group)]
        for name in ['acs2', 'acs2er']:
            group = [baseline_entries[identifier, name, seed] for seed in plan['seeds']]
            datasets[identifier][name] = [row for entry in group for row in entry['rows']]
            costs[identifier][name] = [dict(seed=seed, **cost(entry, baseline)) for seed, entry in zip(plan['seeds'], group)]
        if choices[identifier]['her_values'] == 2 / 3:
            matched_entries(identifier, 2 / 3, main, control_entries, plan['seeds'])
            datasets[identifier]['her_values_p2of3'] = datasets[identifier]['her_values']
            costs[identifier]['her_values_p2of3'] = costs[identifier]['her_values']
            reuse[identifier] = dict(source='final-grid her_values', seeds=plan['seeds'], evaluation_blocks=['training', 'validation', 'test'], additional_runs=0)
        else:
            group = matched_entries(identifier, choices[identifier]['her_values'], main, control_entries, plan['seeds'])
            datasets[identifier]['her_values_p2of3'] = [row for entry in group for row in entry['rows']]
            costs[identifier]['her_values_p2of3'] = [dict(seed=seed, **cost(entry, study / ('control-' + identifier))) for seed, entry in zip(plan['seeds'], group)]
            reuse[identifier] = dict(source='additional fixed-proportion control', seeds=plan['seeds'], evaluation_blocks=['training', 'validation', 'test'], additional_runs=20)
    return plan, datasets, costs, selected, checked, reuse, phase_seconds


def write_tables(destination, results):
    with (destination / 'primary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['configuration','arm','seeds','primary_mean','primary_se','final_test_mean','final_test_se','knowledge_mean','full_steps_per_environment_step','realized_relabeled_share','threshold_acs2','threshold_acs2er'])
        for maze in results['configurations']:
            for name, arm in maze['arms'].items():
                share = arm['checkpoint_means'].get('realized_relabeled_share')
                writer.writerow([maze['configuration'],name,20,arm['primary']['mean'],arm['primary']['se'],arm['final']['success']['mean'],arm['final']['success']['se'],arm['checkpoint_means']['knowledge']['mean'],arm['checkpoint_means']['full_steps_per_environment_step']['mean'],share['mean'] if share else None,arm['threshold_crossings']['acs2_final'],arm['threshold_crossings']['acs2er_final']])
    with (destination / 'comparisons.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['configuration','left','right','seeds','mean_difference','se','ci_low','ci_high','knowledge_difference'])
        for maze in results['configurations']:
            for pair in maze['comparisons']:
                value = pair['primary_difference']
                writer.writerow([maze['configuration'],pair['left'],pair['right'],20,value['mean'],value['se'],value['ci_low'],value['ci_high'],pair['knowledge_difference']['mean']])


def analyze(study, baseline, pilots, destination):
    plan, datasets, costs, selections, verification, reuse, seconds = collect(study, baseline, pilots)
    configs = [analyze_maze(datasets[config['id']], config, plan) for config in plan['configurations']]
    results = dict(approved_plan_sha256=APPROVED_SHA256, allocated_cpu_seconds=seconds,
                   pilot_allocated_cpu_seconds=5445, selected_parameters=selections,
                   control_sources=reuse, configurations=configs, costs=costs,
                   verification={name: {key: checked[key] for key in ['runs','rows','allocation_cpu_seconds']} for name,checked in verification.items()},
                   horizon='registered finite 2M/4M interactions; no convergence claim',
                   identity_checks='every node record independently reproduced before analysis')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / 'results.json', results)
    write_tables(destination, results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--pilots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.study, args.baseline, args.pilots, args.out)
    print(json.dumps(dict(allocated_cpu_hours=result['allocated_cpu_seconds']/3600,configurations=len(result['configurations']))))
