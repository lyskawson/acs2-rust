import argparse
import hashlib
import json
from pathlib import Path

from archive import audit
from protocol import ARMS, BLOCKS, configurations
from grid_identity import field_bits, fields


def pin(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('fixture already pinned')
    entries, summary = audit(root)
    selected = []
    distance_rows = []
    goal_protocols = {}
    for config in configurations():
        for arm in ARMS:
            entry = entries[config['id'], arm, 42]
            row = entry['rows'][0]
            samples = {name: [dict(index=index, record=field_bits(block['starts'][index]))
                             for index in (0, len(block['starts']) - 1)] for name, block in row['evaluation_blocks'].items()}
            compact = dict(row, evaluation_blocks={name: dict(block, starts=None) for name, block in row['evaluation_blocks'].items()})
            shared = fields(compact, BLOCKS)
            goal_protocol = shared.pop('goal_protocol')
            if goal_protocols.setdefault(config['task'], goal_protocol) != goal_protocol:
                raise ValueError('pair protocol differs across arms')
            selected.append(dict(task=config['task'], cap=config['cap'], agent=arm, nominal_step=row['nominal_step'],
                source_rows_sha256=entry['sha256'], shared_fields=shared, records=samples))
            if arm == 'acs2':
                distance_rows.append(dict(task=config['task'], cap=config['cap'], source_rows_sha256=entry['sha256'],
                    distances={name: [record['shortest_distance'] for record in block['starts']] for name, block in row['evaluation_blocks'].items()}))
    identity = json.loads(Path(__file__).with_name('measurement.json').read_text())
    fixture = dict(scope='Thirty seed42 pilot runs, first checkpoint, first and last pair of each block; every pair distance from six archived ACS2 rows.',
        measurement_commit=identity['measurement_commit'], binary_sha256=identity['binary_sha256'],
        pilot_registration_sha256=hashlib.sha256((root / 'pilot-registration.json').read_bytes()).hexdigest(),
        excluded=['commit', 'source_state', 'host', 'cpu_model', 'wall_seconds_train', 'wall_seconds_eval', 'wall_seconds_total', 'arm'],
        configurations=selected, distance_rows=distance_rows, goal_protocols=goal_protocols)
    output.write_text(json.dumps(fixture, indent=2) + '\n')
    return dict(bytes=output.stat().st_size, runs=len(selected), sampled_pair_records=180, distance_rows=6,
                pilot_identity_rows=summary['identity_rows'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(pin(args.root, args.out)))
