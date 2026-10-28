import argparse
import json
from pathlib import Path
import subprocess
import time

from protocol import HER, SHORTENED, ROOT, configurations, setting
from grid_protocol import atomic_json
from collect import outside_repository, named_row, validate_row


def run(output):
    output = outside_repository(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cells = []
    for base in configurations():
        if base['task'] not in ('maze4', 'woods14'):
            continue
        for agent, p in [(SHORTENED, 1 / 3), (SHORTENED, 0.5), (HER, 0.5)]:
            config = setting(base, agent, p, ['validation'])
            config['targets'] = [1000, 3000, 10000]
            path = output / (config['id'] + '.jsonl')
            command = [str(ROOT / 'target/release/acs2-measure'), '--task', config['task'], '--cap', str(config['cap']),
                '--protocol', 'maze-pairs-v1', '--agents', agent, '--relabel-proportion', str(p), '--seeds', '42',
                '--targets', ','.join(map(str, config['targets'])), '--evaluation-blocks', 'validation', '--out', str(path)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            rows = [named_row(json.loads(line), config) for line in path.read_text().splitlines()]
            if len(rows) != len(config['targets']):
                raise ValueError('local check is missing points')
            for row, target in zip(rows, config['targets']):
                checked = dict(row, source_state='clean')
                validate_row(checked, config, agent, 42, row['commit'], target)
                diag = row['replay_diagnostics']
                copies = diag['routes']['relabeled']
                requested = (copies['draws'] + diag['fallbacks']) / diag['draws']
                if abs(requested - p) > 0.04:
                    raise ValueError('requested proportion does not match route counters')
                if agent == SHORTENED and (copies['already_reached'] or copies['after_counterfactual_end'] or not diag['fallbacks'] or diag['realized_relabeled_share'] >= p):
                    raise ValueError('invalid shortened route composition')
            cells.append(dict(task=config['task'], arm=config['row_arm'], final=rows[-1]))
    result = dict(runs=6, rows=18, sequential=True, evaluation_blocks=['validation'],
        elapsed_seconds=time.monotonic() - started, interpretation='Mechanics only, no ranking or parameter selection.', cells=cells)
    atomic_json(output / 'summary.json', result)
    return {key: result[key] for key in ('runs', 'rows', 'elapsed_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out)))
