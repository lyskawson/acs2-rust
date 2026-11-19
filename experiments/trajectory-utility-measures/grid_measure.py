import argparse
import json
import os
from pathlib import Path
import sys

from registration import checksum, read
import collect
from grid_identity import compare
from grid_protocol import atomic_json
import pair_schema
from recovery import checked_completion
from schema import per_pair_enabled
from validate import CPU_FIELDS, validate_row


def measure(path, parent):
    from grid_run import BINARY, ROOT
    expected_root = ROOT / 'attempts'
    if expected_root not in path.resolve().parents or path.parent / 'measurement' != parent:
        raise ValueError('measurement path differs from the assigned attempt')
    manifest = read(path)
    job, array, index, step = (os.environ[key] for key in
        ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_STEP_ID'))
    if (manifest['allocation_job_id'] != f'{array}_{index}'
            or manifest['allocation_job_id_raw'] != job or not step.isdigit()):
        raise ValueError('measurement step allocation differs')
    atomic_json(path.parent / 'step.json', dict(job_id=f'{array}_{index}', job_id_raw=job, step_id=step,
                                               limit_seconds=manifest['allocation_limit_seconds']))
    pair_schema.enabled = per_pair_enabled
    collect.validate_row = validate_row
    collect.run(argparse.Namespace(manifest=str(path), binary=str(BINARY), commit=manifest['measurement_commit'],
                                    output=str(parent), index=0, sample_interval=0.25))


def compare_pilot_prefix(pilot_root, run, config, new_directory):
    source = pilot_root / run['pilot']['rows']
    if pilot_root.resolve() not in source.resolve().parents or checksum(source) != run['pilot']['rows_sha256']:
        raise ValueError('seed42 stage-one prefix differs from its sealed archive')
    with source.open() as stream:
        old = [json.loads(line) for line in stream]
    targets = [row['nominal_step'] for row in old]
    _, new = checked_completion(new_directory, config['targets'])
    if targets != config['targets'][:len(targets)]:
        raise ValueError('the archived seed42 checkpoints do not form a registered prefix')
    for row in (*old, *new):
        for field in CPU_FIELDS:
            del row['trajectory_utility']['cost'][field]
    identity = compare(new, old, [run['seed']], targets, config['evaluation_blocks'])
    atomic_json(new_directory / 'prefix-identity.json',
                dict(identity, pilot_sha256=run['pilot']['rows_sha256'], excluded_utility_cpu_fields=CPU_FIELDS))


if __name__ == '__main__':
    measure(Path(sys.argv[1]), Path(sys.argv[2]))
