import argparse
import json
from pathlib import Path
import subprocess
import time

from protocol import ROOT, SHORTENED, ER, configurations, setting
from collect import outside_repository, named_row, validate_row
from grid_protocol import atomic_json


def run(output):
    output = outside_repository(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cells = []
    for base in configurations():
        if base['task'] not in ('maze4', 'woods14'):
            continue
        for agent, mode in [(SHORTENED, '3to1at30000'), (SHORTENED, 'backward'), (ER, 'backward')]:
            config = setting(base, agent, mode, ['validation'])
            config['targets'] = [1000, 3000, 30_100]
            path = output / (config['id'] + '.jsonl')
            settings = config['agent_settings'][agent]
            command = [str(ROOT / 'target/release/acs2-measure'), '--task', base['task'], '--cap', str(base['cap']),
                '--protocol', 'maze-pairs-v1', '--agents', agent, '--seeds', '42', '--targets', ','.join(map(str, config['targets'])),
                '--evaluation-blocks', 'validation', '--replay-schedule', settings['replay_schedule'],
                '--replay-order', settings['replay_order'], '--out', str(path)]
            if agent == SHORTENED:
                command += ['--relabel-proportion', str(settings['relabeled_proportion'])]
            subprocess.run(command, check=True, capture_output=True, text=True)
            rows = [named_row(json.loads(line), config) for line in path.read_bytes().splitlines()]
            if len(rows) != 3:
                raise ValueError('local check is missing points')
            for row, point in zip(rows, config['targets']):
                validate_row(dict(row, source_state='clean'), config, agent, 42, row['commit'], point)
            cells.append(dict(task=base['task'], arm=config['row_arm'], final=rows[-1]))
    result = dict(runs=6, rows=18, sequential=True, elapsed_seconds=time.monotonic() - started,
        evaluation_blocks=['validation'], interpretation='Short mechanics check only; no ranking.', cells=cells)
    atomic_json(output / 'summary.json', result)
    return {key: result[key] for key in ('runs', 'rows', 'elapsed_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out)))
