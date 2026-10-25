import argparse
import csv
import hashlib
import json
from pathlib import Path

from analysis import REFERENCES, SEEDS, VIEWS, analyze_maze, conditional_summary, exact_success
from execution import APPROVED_SHA256, approved
from references import reused
from verify import verify_study
from collect import outside_repository
from grid_protocol import atomic_json


def write_csv(path, columns, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_run(entry):
    path = entry['path']
    rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
    completion = json.loads((path / 'completion.json').read_text())
    cost = dict(allocation_cpu_seconds=entry['allocation_cpu_seconds'],
        process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
        elapsed_seconds=completion['elapsed_seconds'], peak_rss_kib=completion['peak_rss_kib'],
        train_wall_seconds=rows[-1]['wall_seconds_train'], eval_wall_seconds=rows[-1]['wall_seconds_eval'])
    return rows, cost


def analyze(root, her, baseline, output=None):
    root = Path(root)
    entries, execution = verify_study(root)
    plan = approved(root / 'approved-plan.json')
    reference_entries = reused(plan, her, baseline)
    if set(entries) & set(reference_entries):
        raise ValueError('new runs and archived references overlap')
    all_entries = entries | reference_entries
    output = outside_repository(output or root / 'analysis')
    output.mkdir(parents=True, exist_ok=True)
    names = plan['agents'] + REFERENCES
    results = dict(plan_sha256=APPROVED_SHA256, seeds=SEEDS, views=VIEWS,
        analysis_rules=plan['analysis'], execution=execution, mazes=[], input_checksums={},
        reference_reuse=dict(runs=len(reference_entries), rerun=False,
            selected_reference_allocation_cpu_seconds=sum(item['allocation_cpu_seconds'] for item in reference_entries.values()),
            charged_to_this_study=False))
    counts, contrasts, thresholds, diagnostics, costs, scores = [], [], [], [], [], []
    for config in plan['configurations']:
        dataset, cost_records = {}, {}
        for name in names:
            dataset[name], cost_records[name] = [], []
            for seed in SEEDS:
                entry = all_entries[config['id'], name, seed]
                rows, cost = load_run(entry)
                dataset[name].extend({key: value for key, value in row.items() if key not in ('goal_protocol', 'preset')} for row in rows)
                cost_records[name].append(cost)
                label = '|'.join([config['id'], name, str(seed)])
                results['input_checksums'][label] = {file: hashlib.sha256((entry['path'] / file).read_bytes()).hexdigest()
                                                       for file in ('rows.jsonl', 'completion.json')}
                costs.append(dict(configuration_id=config['id'], agent=name, seed=seed,
                                  reused_reference=name in REFERENCES, cpu_model=rows[0]['cpu_model'], **cost))
                for row in rows:
                    block = row['evaluation_blocks']['test']
                    count = exact_success(block) * block['evaluation_pairs']
                    if count.denominator != 1:
                        raise ValueError('integer success export is inconsistent')
                    counts.append(dict(configuration_id=config['id'], agent=name, seed=seed,
                        nominal_step=row['nominal_step'], successes=count.numerator, evaluation_pairs=block['evaluation_pairs']))
            dataset[name].sort(key=lambda row: (row['seed'], row['nominal_step']))
        maze = analyze_maze(dataset, config, plan)
        for name in names:
            maze['arms'][name]['cost'] = {field: conditional_summary([record[field] for record in cost_records[name]])
                                         for field in cost_records[name][0]}
            for threshold, point in maze['arms'][name]['threshold_crossings'].items():
                thresholds.append(dict(configuration_id=config['id'], agent=name, threshold=threshold,
                    threshold_exact=maze['thresholds'][threshold], first_nominal_step=point))
            for view in VIEWS:
                success = maze['arms'][name]['success'][view]
                for seed, score in success['seed_scores'].items():
                    scores.append(dict(configuration_id=config['id'], agent=name, seed=int(seed), view=view, score_exact=score))
                for metric, summary in maze['arms'][name]['diagnostics'][view].items():
                    diagnostics.append(dict(configuration_id=config['id'], agent=name, view=view, metric=metric,
                                            n=summary['n'], mean=summary['mean'], se=summary['se']))
        for comparison in maze['comparisons']:
            difference = comparison['difference']
            contrasts.append(dict(configuration_id=config['id'], view=comparison['view'], left=comparison['left'],
                right=comparison['right'], n=difference['n'], mean=difference['mean'], se=difference['se'],
                ci_low=difference['ci_low'], ci_high=difference['ci_high'], mean_exact=difference['mean_exact']))
        results['mazes'].append(maze)
    if len(counts) != 5880 or len(contrasts) != 60 or len(costs) != 720:
        raise ValueError('analysis export does not cover the approved grid and all reused references')
    write_csv(output / 'success-counts.csv', ['configuration_id', 'agent', 'seed', 'nominal_step', 'successes', 'evaluation_pairs'], counts)
    write_csv(output / 'seed-scores.csv', ['configuration_id', 'agent', 'seed', 'view', 'score_exact'], scores)
    write_csv(output / 'comparisons.csv', ['configuration_id', 'view', 'left', 'right', 'n', 'mean', 'se', 'ci_low', 'ci_high', 'mean_exact'], contrasts)
    write_csv(output / 'thresholds.csv', ['configuration_id', 'agent', 'threshold', 'threshold_exact', 'first_nominal_step'], thresholds)
    write_csv(output / 'diagnostics.csv', ['configuration_id', 'agent', 'view', 'metric', 'n', 'mean', 'se'], diagnostics)
    write_csv(output / 'costs.csv', list(costs[0]), costs)
    results['analysis_sources'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(Path(__file__).parent.glob('*.py'))}
    atomic_json(output / 'results.json', results)
    atomic_json(output / 'verification.json', execution)
    return dict(new_runs=len(entries), reference_runs=len(reference_entries), success_rows=len(counts),
                comparisons=len(contrasts), allocated_cpu_hours=execution['allocated_cpu_seconds'] / 3600,
                output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--her-root', required=True, type=Path)
    parser.add_argument('--baseline-root', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.her_root, args.baseline_root, args.out)))
