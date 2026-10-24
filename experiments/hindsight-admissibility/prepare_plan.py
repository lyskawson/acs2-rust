import argparse
import hashlib
import json
from pathlib import Path

from archive import audit
from execution import measurement_identity
from protocol import AGENTS, ROOT, configurations, study
from grid_identity import OMIT, field_bits
from grid_protocol import atomic_json


def pin(summary, rows, output):
    result = dict(scope='Both full-update admissibility presets, all six seed-42 pilot mazes, first two registered points and all evaluation blocks.',
        nominal_points=[1000, 3000], excluded=sorted(OMIT),
        source_pilot_allocation_cpu_seconds=summary['allocation_cpu_seconds'],
        measurement_commit=summary['measurement_commit'], binary_sha256=summary['binary_sha256'], configurations=[])
    for config in configurations():
        for agent in AGENTS:
            selected = [row for row in rows if row['task'] == config['task'] and row['agent'] == agent and row['nominal_step'] in result['nominal_points']]
            if [row['nominal_step'] for row in selected] != result['nominal_points']:
                raise ValueError('missing pilot regression points')
            source = next(run for run in summary['runs'] if (run['configuration_id'], run['agent']) == (config['id'], agent))
            result['configurations'].append(dict(task=config['task'], cap=config['cap'], agent=agent,
                source_rows_sha256=source['rows_sha256'],
                rows=[field_bits({key: value for key, value in row.items() if key not in OMIT}) for row in selected]))
    atomic_json(output, result)


def prepare(root, output, fixture):
    root = Path(root)
    summary, rows, cells = audit(root)
    identity = measurement_identity()
    if any(summary[key] != identity[key] for key in ('measurement_commit', 'binary_sha256')):
        raise ValueError('pilot differs from frozen measurement')
    rows_path = root / 'pilot-rows.jsonl'
    rows_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    atomic_json(root / 'pilot-summary.json', summary)
    pin(summary, rows, fixture)
    evidence = dict(runs=12, rows=len(rows), allocations=summary['allocations'],
        allocation_cpu_seconds=summary['allocation_cpu_seconds'],
        rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((root / 'pilot-summary.json').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
        generator_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ('protocol.py', 'archive.py', 'prepare_plan.py')})
    plan = study(cells, summary['allocation_cpu_seconds'], identity, evidence)
    atomic_json(output, plan)
    return dict(plan_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(), forecast=plan['forecast'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('plan.json'))
    parser.add_argument('--fixture', type=Path, default=ROOT / 'fixtures/hindsight_admissibility_regression.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.out, args.fixture)))
