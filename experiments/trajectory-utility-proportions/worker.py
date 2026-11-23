import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from proportions import configuration, checksum, read
from pilot import BINARY, ROOT, batches, clean_head, run_directory, wave_dir, wave_plan
from grid_identity import fields
from grid_launch import allocation_fields
from grid_protocol import atomic_json, load_plan, seconds, stop_phase
from submit import execute
from validate_proportions import validate_row
import collect
import pair_schema
from schema import per_pair_enabled


def deterministic_fields(row, blocks):
    caps = row['test_caps']['extended']
    del caps['cumulative_cpu_seconds']
    cost = row['trajectory_utility']['cost']
    for name in tuple(cost):
        if name.endswith('cpu_seconds'):
            del cost[name]
    return fields(row, blocks)


def compare_prior(result, previous, blocks):
    before = ROOT / previous['relative_directory'] / 'rows.jsonl'
    if not before.exists():
        return 0
    old, _ = collect.complete_rows(before)
    new, incomplete = collect.complete_rows(result / 'rows.jsonl')
    if incomplete or len(old) > len(new):
        raise ValueError('retry does not cover the preserved checkpoints')
    for earlier, later in zip(old, new):
        if deterministic_fields(earlier, blocks) != deterministic_fields(later, blocks):
            raise ValueError('retry changed a preserved learning or evaluation field')
    atomic_json(result / 'prefix-identity.json', dict(rows=len(old), source_sha256=checksum(before)))
    return len(old)


def allocation(array, index, job):
    for retry in range(3):
        try:
            details = execute(['scontrol', 'show', 'job', '-o', f'{array}_{index}'])
            return details, allocation_fields(details, job, array, index)
        except subprocess.CalledProcessError as error:
            if 'Socket timed out' not in error.output or retry == 2:
                raise
            time.sleep(2 ** retry)


def run(folder, path, index):
    if (folder / 'STOP.json').exists():
        raise RuntimeError('pilot wave stopped')
    assignment = read(path)
    plan = load_plan(folder / 'plan.json', assignment['plan_sha256'])
    if (folder != wave_dir(plan['wave']) or path.parent != folder
            or plan != wave_plan(plan['wave'], checksum(BINARY), clean_head(),
                                 plan['build_cpu_seconds'], plan['earlier'])
            or assignment['runs'] != batches(plan['runs'])[assignment['batch']][0]):
        raise ValueError('node source, scope or assignment differs from the registration')
    cell = assignment['runs'][index]
    array, slot, job = (os.environ[name] for name in
                        ('SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_ID'))
    if index != int(slot):
        raise ValueError('SLURM slot differs')
    details, actual = allocation(array, slot, job)
    limit = seconds(actual['TimeLimit'])
    timeout = int(limit - seconds(actual['RunTime']) - 30)
    if (actual['Partition'] != 'lem-cpu-normal' or actual['NumCPUs'] != '1'
            or os.environ.get('SLURM_MEM_PER_NODE') != str(plan['memory_mib'])
            or limit != cell['limit_seconds'] or timeout < 60):
        raise ValueError('node allocation differs from the registered pilot')
    directory = run_directory(folder, cell, plan['configurations'])
    attempt = directory.parents[1]
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / 'allocation-start.json', dict(response=details, fields=actual))
    config = configuration(cell, plan['configurations'])
    manifest = dict(configuration=config, agent=cell['agent'], seed=42, attempt_id=cell['attempt_id'],
                    stop_path=str(folder / 'STOP.json'), measurement_commit=plan['commit'],
                    expected_binary_sha256=plan['binary_sha256'], expected_preset=plan['preset'],
                    expected_cpu_model=plan['cpu_model'], process_timeout_seconds=timeout,
                    allocation_limit_seconds=limit, first_row_timeout_seconds=60)
    manifest_path = attempt / 'manifest.json'
    atomic_json(manifest_path, manifest)
    pair_schema.enabled = per_pair_enabled
    collect.validate_row = validate_row
    collect.run(argparse.Namespace(manifest=str(manifest_path), binary=str(BINARY), commit=plan['commit'],
                                   output=str(directory.parent), index=0, sample_interval=0.2))
    previous = plan['earlier']['attempts'].get(cell['run_id'])
    if previous:
        compare_prior(directory, previous, config['evaluation_blocks'])


if __name__ == '__main__':
    folder, assignment, index = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    try:
        if '--failed-step' in sys.argv:
            raise RuntimeError('measurement step exited unsuccessfully')
        run(folder, assignment, index)
    except BaseException as error:
        if folder.parent == ROOT and not (folder / 'STOP.json').exists():
            stop_phase(folder / 'STOP.json', dict(error=str(error), assignment=str(assignment),
                                                 index=index, job=os.environ.get('SLURM_JOB_ID')))
        raise
