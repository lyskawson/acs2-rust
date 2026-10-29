import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from protocol import SCHEDULES
from grid_identity import field_bits
from grid_protocol import atomic_json, stop_phase


def verify(binary, fixture, task, output):
    fixture = json.loads(Path(fixture).read_text())
    selected = [config for config in fixture['configurations'] if config['task'] == task and config['mode'] in SCHEDULES]
    if len(selected) != 6:
        raise ValueError('binary fixture gate requires all six schedule settings')
    output = Path(output)
    output.mkdir(exist_ok=False)
    rows_checked = 0
    for index, config in enumerate(selected):
        path = output / f'{index}.jsonl'
        command = [str(binary), '--task', task, '--cap', str(config['cap']), '--protocol', 'maze-pairs-v1',
            '--agents', config['agent'], '--seeds', '42', '--targets', '1000,3000',
            '--evaluation-blocks', 'training,validation,test', '--replay-schedule', config['mode'],
            '--replay-order', 'independent', '--out', str(path)]
        if config['agent'] == 'acs2her_shortened':
            command.extend(['--relabel-proportion', str(config['proportion'])])
        subprocess.run(command, check=True, capture_output=True, text=True)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        actual = [field_bits({key: value for key, value in row.items() if key not in fixture['excluded']}) for row in rows]
        if actual != config['rows']:
            raise ValueError('new measurement binary differs from the original schedule fixture')
        rows_checked += len(rows)
    return dict(task=task, settings=len(selected), rows=rows_checked, verified=True,
                binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest())


def allocation(root, assignment, index):
    root = Path(root)
    plan = json.loads((root / 'plan.json').read_text())
    if plan['pilot_index'] != 9:
        raise ValueError('binary fixture gate belongs to the default identity pilot allocation')
    assignment = json.loads(Path(assignment).read_text())
    entry = assignment['runs'][index]
    config = next(config for config in plan['configurations'] if config['id'] == entry['configuration_id'])
    output = root / 'binary-fixture' / config['task']
    output.parent.mkdir(exist_ok=True)
    try:
        result = verify(assignment['binary'], Path(assignment['measurement_repository']) / 'fixtures/replay_schedule_and_order_regression.json',
                        config['task'], output)
        if result['binary_sha256'] != plan['binary_sha256'] or result['rows'] != 12:
            raise ValueError('binary fixture gate identity or row count differs')
        atomic_json(output / 'verification.json', result)
    except BaseException as error:
        stop_phase(Path(plan['stop_path']), dict(error=str(error), purpose='schedule binary fixture identity', task=config['task']))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('assignment')
    parser.add_argument('index', type=int)
    args = parser.parse_args()
    allocation(args.root, args.assignment, args.index)
