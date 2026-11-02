import argparse
import csv
import hashlib
import json
from pathlib import Path

from analysis import attach, consistency, statistics_for
from archive import audit, catalog, models
from comparisons import registered_results
from execution import replication
from protocol import ARMS, ROOT, configurations
from collect import outside_repository
from grid_protocol import atomic_json, load_plan


def analyze(root, output):
    root, output = Path(root), outside_repository(output)
    registered = Path(__file__).with_name('plan.json')
    digest = hashlib.sha256(registered.read_bytes()).hexdigest()
    plan = load_plan(root / 'approved-plan.json', digest)
    approval = json.loads((root / 'approval.json').read_text())
    if approval.get('approved') is not True or approval.get('plan_sha256') != digest:
        raise ValueError('analysis needs the approved registration')
    pilot_entries, _ = audit(root)
    more, _ = catalog(root / 'replication', root)
    manifest = json.loads((root / 'replication/plan.json').read_text())
    expected = replication(plan, approval['phase_cpu_hour_limit'])
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('replication manifest differs from registration')
    entries = dict(pilot_entries)
    entries.update(more)
    truth = models(root)
    metrics, final_records = {}, {}
    diagnostics = []
    for config in configurations():
        for arm in ARMS:
            for seed in range(42, 47):
                entry = entries[config['id'], arm, seed]
                for row in entry['rows']:
                    point = row['nominal_step']
                    views = {name: attach(block['starts'], truth[config['task']]) for name, block in row['evaluation_blocks'].items()}
                    views['pooled'] = [record for records in views.values() for record in records]
                    for view, records in views.items():
                        measured = statistics_for(records)
                        metrics[config['task'], arm, seed, point, view] = measured
                        diagnostics.append(dict(maze=config['task'], arm=arm, seed=seed, nominal_step=point, view=view,
                            post_hoc=True, metrics=measured, distances={str(distance): statistics_for([record for record in records if record['shortest_distance'] == distance])
                                for distance in sorted({record['shortest_distance'] for record in records})}))
                    if point == config['targets'][-1]:
                        final_records.setdefault((config['task'], arm), {})[seed] = views['pooled']
    results = registered_results(metrics, configurations())
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'registered-comparisons.json', dict(plan_sha256=digest, comparisons=results))
    atomic_json(output / 'post-hoc-diagnostics.json', diagnostics)
    atomic_json(output / 'post-hoc-consistency.json', [dict(maze=maze, arm=arm, **consistency(records))
                for (maze, arm), records in sorted(final_records.items())])
    with (output / 'registered-comparisons.csv').open('w', newline='') as stream:
        columns = ['maze', 'comparison', 'metric', 'nominal_step', 'view', 'registered', 'n', 'mean', 'se', 'ci_low', 'ci_high']
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    return dict(registered_comparisons=len(results), diagnostic_cells=len(diagnostics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.out)))
