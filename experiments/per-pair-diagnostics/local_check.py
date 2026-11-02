import argparse
import json
from pathlib import Path
import subprocess
import time

from protocol import ROOT, configurations, recorded
from collect import outside_repository, validate_row
from grid_protocol import atomic_json


def run(output):
    output = outside_repository(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = recorded(configurations()[0])
    config['targets'] = [1000, 3000]
    path = output / 'rows.jsonl'
    command = [str(ROOT / 'target/release/acs2-measure'), '--task', 'maze4', '--cap', '5', '--protocol', 'maze-pairs-v1',
               '--agents', 'acs2her', '--seeds', '42', '--targets', '1000,3000', '--per-start', '--out', str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(rows) != 2:
        raise ValueError('short local check is incomplete')
    for row, point in zip(rows, config['targets']):
        validate_row(row, config, 'acs2her', 42, row['commit'], point)
    result = dict(runs=1, rows=2, elapsed_seconds=time.monotonic() - started, bytes=path.stat().st_size,
                  command=command, interpretation='Short measurement check only; no ranking.')
    atomic_json(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    print(json.dumps(run(parser.parse_args().out)))
