import argparse
import json
from pathlib import Path
import subprocess
import time

from protocol import AGENTS, ROOT, configurations
from collect import outside_repository, validate_row
from grid_protocol import atomic_json


def run(output):
    output = outside_repository(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cells = []
    for config in configurations():
        if config['task'] not in ('maze4', 'woods14'):
            continue
        config = dict(config, evaluation_blocks=['validation'], targets=[1000, 10000, 20000])
        for agent in AGENTS:
            path = output / (config['task'] + '_' + agent + '.jsonl')
            command = [str(ROOT / 'target/release/acs2-measure'), '--task', config['task'], '--cap', str(config['cap']),
                '--protocol', 'maze-pairs-v1', '--agents', agent, '--seeds', '42', '--targets', '1000,10000,20000',
                '--evaluation-blocks', 'validation', '--out', str(path)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != 3:
                raise ValueError('local check is missing points')
            for row, target in zip(rows, config['targets']):
                validate_row(row, config, agent, 42, row['commit'], target)
                if row['match_formations_train'] != 10 * row['actual_steps']:
                    raise ValueError('local check has incorrect full learning volume')
            final = rows[-1]
            cells.append(dict(task=config['task'], agent=agent, actual_steps=final['actual_steps'],
                replay_updates=final['replay_updates'], replay_diagnostics=final['replay_diagnostics'],
                source_commit=final['commit'], population=final['population_classifiers']))
    summary = dict(runs=4, rows=12, sequential=True, evaluation_blocks=['validation'],
        elapsed_seconds=time.monotonic() - started, interpretation='Mechanics and diagnostic check only; no ranking or parameter selection.', cells=cells)
    atomic_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({key: result[key] for key in ('runs', 'rows', 'elapsed_seconds')}))
