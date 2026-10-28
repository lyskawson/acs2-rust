import argparse
import hashlib
import json
from pathlib import Path

from archive import audit
from protocol import ROOT, BLOCKS, CANDIDATES, study, configurations, label, PILOT_SETTINGS
from execution import measurement_identity
from grid_identity import OMIT, field_bits
from grid_protocol import atomic_json


def pin(summary, rows, output):
    result = dict(scope='Four new full-copy settings, six seed42 mazes, first two registered points and all three evaluation blocks.',
        nominal_points=[1000, 3000], excluded=sorted(OMIT),
        measurement_commit=summary['measurement_commit'], binary_sha256=summary['binary_sha256'],
        source_pilot_allocation_cpu_seconds=summary['allocation_cpu_seconds'], configurations=[])
    for config in configurations():
        for agent, p, blocks in PILOT_SETTINGS:
            if blocks != BLOCKS:
                continue
            arm = label(agent, p)
            selected = [row for row in rows if row['task'] == config['task'] and row['arm'] == arm and row['nominal_step'] in result['nominal_points']]
            if [row['nominal_step'] for row in selected] != result['nominal_points']:
                raise ValueError('missing pilot regression points')
            source = next(run for run in summary['runs'] if (run['configuration_id'], run['arm']) == (config['id'], arm))
            result['configurations'].append(dict(task=config['task'], cap=config['cap'], agent=agent,
                arm=arm, proportion=p, source_rows_sha256=source['source_rows_sha256'],
                rows=[field_bits({key: value for key, value in row.items() if key not in OMIT}) for row in selected]))
    atomic_json(output, result)


def prepare(root, output, fixture, proportions=CANDIDATES):
    root = Path(root)
    summary, rows, cells = audit(root)
    identity = measurement_identity()
    if any(summary[key] != identity[key] for key in identity):
        raise ValueError('pilot measurement differs from frozen identity')
    rows_path = root / 'pilot-rows.jsonl'
    rows_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    atomic_json(root / 'pilot-summary.json', summary)
    pin(summary, rows, fixture)
    evidence = dict(runs=30, allocations=summary['allocations'], rows=len(rows), identity_rows=summary['identity_rows'],
        allocation_cpu_seconds=summary['allocation_cpu_seconds'],
        cells=[dict(configuration_id=config_id, arm=arm, **cost) for (config_id, arm), cost in cells.items()],
        rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((root / 'pilot-summary.json').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
        references_sha256=hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest(),
        generators={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
                    ('protocol.py', 'archive.py', 'prepare_plan.py', 'execution.py', 'tuning.py')})
    plan = study(cells, summary['allocation_cpu_seconds'], identity, evidence, proportions)
    atomic_json(output, plan)
    return dict(plan_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(), forecasts=plan['forecasts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--proportions', type=lambda value: [float(item) for item in value.split(',')], default=CANDIDATES)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('plan.json'))
    parser.add_argument('--fixture', type=Path, default=ROOT / 'fixtures/hindsight_proportion_regression.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.out, args.fixture, args.proportions)))
