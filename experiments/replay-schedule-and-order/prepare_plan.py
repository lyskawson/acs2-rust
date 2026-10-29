import argparse
import hashlib
import json
from pathlib import Path

from archive import audit
from execution import measurement_identity
from protocol import ROOT, PILOT_SETTINGS, configurations, label, study
from grid_identity import OMIT, field_bits
from grid_protocol import atomic_json


def prepare(root, output, fixture):
    root = Path(root)
    summary, rows, cells = audit(root)
    identity = measurement_identity()
    if any(summary[key] != value for key, value in identity.items()):
        raise ValueError('pilot source or binary differs from its frozen identity')
    result = dict(scope='Nine new replay settings, six seed42 mazes, first two points and all three blocks.',
        nominal_points=[1000, 3000], excluded=sorted(OMIT), **identity,
        source_pilot_allocation_cpu_seconds=summary['allocation_cpu_seconds'], configurations=[])
    for config in configurations():
        for agent, mode in PILOT_SETTINGS:
            if mode == 'base':
                continue
            arm = label(config, agent, mode)
            selected = [row for row in rows if row['task'] == config['task'] and row['arm'] == arm
                        and row['nominal_step'] in result['nominal_points']]
            if [row['nominal_step'] for row in selected] != result['nominal_points']:
                raise ValueError('pilot fixture is missing points')
            source = next(run for run in summary['runs'] if (run['configuration_id'], run['arm']) == (config['id'], arm))
            result['configurations'].append(dict(task=config['task'], cap=config['cap'], agent=agent, mode=mode, arm=arm,
                proportion=selected[0]['agent_parameters']['relabeled_proportion'], source_rows_sha256=source['source_rows_sha256'],
                rows=[field_bits({key: value for key, value in row.items() if key not in OMIT}) for row in selected]))
    atomic_json(fixture, result)
    (root / 'pilot-rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    atomic_json(root / 'pilot-summary.json', summary)
    evidence = dict(runs=60, allocations=summary['allocations'], rows=len(rows), identity_rows=summary['identity_rows'],
        allocation_cpu_seconds=summary['allocation_cpu_seconds'],
        previous_pilot_root=summary['previous_pilot']['root'], previous_pilot=summary['previous_pilot'], rounds=summary['rounds'],
        schedule_binary_fixture_rows=summary['schedule_binary_fixture_rows'],
        cells=[dict(configuration_id=key[0], arm=key[1], **cost) for key, cost in cells.items()],
        references_sha256=hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest(),
        base_selection_sha256=hashlib.sha256((root / 'base-selections.json').read_bytes()).hexdigest(),
        rows_sha256=hashlib.sha256((root / 'pilot-rows.jsonl').read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((root / 'pilot-summary.json').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
        generators={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('protocol.py', 'execution.py', 'archive.py', 'prepare_plan.py', 'tuning.py', 'previous_pilot.py')})
    plan = study(cells, summary['allocation_cpu_seconds'], identity, evidence)
    atomic_json(output, plan)
    return dict(plan_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(), forecast=plan['forecast'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('plan.json'))
    parser.add_argument('--fixture', type=Path, default=ROOT / 'fixtures/replay_schedule_and_order_regression.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.out, args.fixture)))
