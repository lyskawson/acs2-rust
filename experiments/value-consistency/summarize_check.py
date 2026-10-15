import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(values):
    return dict(n=len(values), mean=statistics.mean(values), se=statistics.stdev(values) / math.sqrt(len(values)))


def metrics(row):
    priority = row['replay_diagnostics']['priority']
    result = dict(success=row['evaluation_blocks']['validation']['success'], knowledge=row['knowledge'],
        population=row['population_classifiers'], memory_bytes=row['ensemble']['agent_known_bytes_lower_bound'],
        train_seconds=row['wall_seconds_train'], evaluation_seconds=row['wall_seconds_eval'],
        full_updates=row['learning_update_counts']['full'], values_only_updates=row['learning_update_counts']['values_only'],
        proposals_per_draw=priority['proposals_per_draw'], acceptance=priority['acceptance_share'],
        proposal_variance=priority['proposal_variance']['mean'], accepted_variance=priority['accepted_variance']['mean'],
        variance_maximum=priority['variance_maximum'], relabeled_share=row['replay_diagnostics']['realized_relabeled_share'],
        priority_queries_per_interaction=priority['value_queries'] / row['actual_steps'])
    for purpose, value in row['replay_diagnostics']['objective_evaluations_by_purpose'].items():
        result[purpose + '_queries_per_interaction'] = value['objective_evaluations'] / row['actual_steps']
    for head, detail in enumerate(row['ensemble']['heads']):
        result[f'head_{head}_population'] = detail['population_classifiers']
        result[f'head_{head}_knowledge'] = detail['knowledge']
        result[f'head_{head}_success'] = row['evaluation_blocks']['validation']['heads'][head]['success']
        result[f'head_{head}_memory_bytes'] = detail['head_known_bytes_lower_bound']
    return result


def summarize_check(root, output):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / 'manifest.json').read_text())
    if len(manifest['runs']) != 180:
        raise ValueError('local check requires six mazes, six arm/temperature settings and five seeds')
    groups = defaultdict(list)
    for entry in manifest['runs']:
        path = root / entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('local check hash mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [row['nominal_step'] for row in rows] != [1000, 10000, 20000]:
            raise ValueError('local check points differ')
        for row in rows:
            if set(row['evaluation_blocks']) != {'validation'}:
                raise ValueError('local check cannot read the test block')
            groups[row['task'], row['agent'], row['agent_parameters']['temperature'], row['nominal_step']].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        if {row['seed'] for row in rows} != set(range(42, 47)) or len(rows) != 5:
            raise ValueError('missing local check seed')
        values = [metrics(row) for row in rows]
        result.append(dict(task=key[0], agent=key[1], temperature=key[2], nominal_step=key[3],
                           metrics={name: summarize([value[name] for value in values]) for name in values[0]}))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(dict(runs=180, rows=540, cells=result,
        interpretation='sequential validation-only operational check; no agent ranking'), indent=2) + '\n')
    names = sorted({name for cell in result for name in cell['metrics']})
    with (output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task', 'agent', 'temperature', 'nominal_step'] + [name + suffix for name in names for suffix in ['_mean', '_se']])
        writer.writeheader()
        for cell in result:
            row = {key: cell[key] for key in ['task', 'agent', 'temperature', 'nominal_step']}
            row.update({name + suffix: value[field] for name, value in cell['metrics'].items() for suffix, field in [('_mean', 'mean'), ('_se', 'se')]})
            writer.writerow(row)
    print(json.dumps(dict(runs=180, rows=540, cells=len(result))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    summarize_check(args.root, args.out)
