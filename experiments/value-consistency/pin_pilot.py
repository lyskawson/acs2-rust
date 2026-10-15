import argparse
import hashlib
import json
from pathlib import Path
import struct

from archive import audit
from protocol import ROOT
from grid_protocol import atomic_json


def bits(value):
    if isinstance(value, float):
        return 'f64:' + struct.pack('>d', value).hex()
    if isinstance(value, list):
        return list(map(bits, value))
    if isinstance(value, dict):
        return {key: bits(item) for key, item in value.items()}
    return value


def pin(root, output):
    root = Path(root)
    summary, _, _ = audit(root)
    original = json.loads((ROOT / 'fixtures/vcp_pair_regression.json').read_text())
    fixture = {key: value for key, value in original.items() if key != 'configurations'}
    fixture['scope'] = 'All completed stage-1 cost-pilot arms and mazes, first two registered validation points; every failed and cancelled attempt remains in the external pilot inventory.'
    fixture['source_pilot_allocation_cpu_seconds'] = summary['allocation_cpu_seconds']
    fixture['configurations'] = []
    excluded = set(fixture['excluded'])
    for run in summary['selected_runs']:
        source = root / run['directory'] / 'rows.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        chosen = [row for row in rows if row['nominal_step'] in fixture['nominal_points']]
        if len(chosen) != len(fixture['nominal_points']):
            raise ValueError('pilot is missing the regression points')
        for row in chosen:
            heads = row['ensemble']['heads']
            head_bytes = {head['head_known_bytes_lower_bound'] - head['population_known_bytes_lower_bound'] for head in heads}
            agent_bytes = row['ensemble']['agent_known_bytes_lower_bound'] - row['ensemble']['heads_known_bytes_lower_bound'] - row['trajectory_logical_bytes']
            if head_bytes != {fixture['source_inline_memory_layout']['head_inline_bytes']} or agent_bytes != fixture['source_inline_memory_layout']['agent_inline_without_store_bytes']:
                raise ValueError('pilot uses an unmeasured source memory layout')
        fixture['configurations'].append(dict(task=chosen[0]['task'], cap=chosen[0]['cap'],
            agent=run['agent'], temperature=run['temperature'], source_rows_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            rows=[bits({key: value for key, value in row.items() if key not in excluded}) for row in chosen]))
    atomic_json(output, fixture)
    return dict(configurations=len(fixture['configurations']), rows=sum(len(config['rows']) for config in fixture['configurations']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(pin(args.root, args.out)))
