import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from pilot import BINARY, ROOT, batches, registered_plan, source_identity
from registration import read
import collect
from grid_launch import allocation_fields
from grid_protocol import atomic_json, load_plan, seconds, stop_phase
import pair_schema
from schema import per_pair_enabled
from submit import execute
from validate import validate_row


def allocation(array, task, job):
    for retry in range(3):
        try:
            details = execute(['scontrol', 'show', 'job', '-o', f'{array}_{task}'])
            return details, allocation_fields(details, job, array, task)
        except subprocess.CalledProcessError as error:
            if 'Socket timed out' not in error.output or retry == 2:
                raise
            time.sleep(2 ** retry)


def launch(root, assignment, index):
    if root.resolve() != ROOT or assignment.parent != ROOT:
        raise ValueError('pilot result path differs')
    stop = root / 'STOP.json'
    if stop.exists():
        raise SystemExit('pilot stopped')
    assigned = read(assignment)
    commit, digest = source_identity()
    plan = load_plan(root / 'plan.json', assigned['plan_sha256'])
    registered = registered_plan(commit, digest, plan['build_cpu_seconds'])
    if plan != registered or assigned['runs'] != batches(registered)[assigned['batch']][0]:
        raise ValueError('node assignment, source or binary differs from the registered pilot')
    run = assigned['runs'][index]
    array, task, job = (os.environ[name] for name in ('SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_ID'))
    if int(task) != index:
        raise ValueError('SLURM index differs')
    details, fields = allocation(array, task, job)
    limit = seconds(fields['TimeLimit'])
    timeout = limit - seconds(fields['RunTime']) - 30
    if (fields['Partition'] != 'lem-cpu-normal' or fields['NumCPUs'] != '1'
            or os.environ.get('SLURM_MEM_PER_NODE') != str(plan['memory_mib'])
            or limit != run['limit_seconds'] or timeout < 60):
        raise ValueError('node allocation differs from the registered pilot')
    directory = root / run['relative_directory']
    atomic_json(directory.parent.parent / 'allocation-start.json', dict(response=details, fields=fields))
    config = plan['configurations'][run['configuration_id']]
    manifest = dict(configuration=config, agent=run['agent'], seed=run['seed'], attempt_id=run['attempt_id'],
                    stop_path=str(stop), measurement_commit=commit, expected_binary_sha256=digest,
                    expected_preset=plan['preset'], expected_cpu_model=plan['cpu_model'],
                    process_timeout_seconds=timeout, allocation_limit_seconds=limit,
                    first_row_timeout_seconds=60)
    path = directory.parent.parent / 'manifest.json'
    atomic_json(path, manifest)
    pair_schema.enabled = per_pair_enabled
    collect.validate_row = validate_row
    collect.run(argparse.Namespace(manifest=str(path), binary=str(BINARY), commit=commit,
                                    output=str(directory.parent), index=0, sample_interval=0.1))


if __name__ == '__main__':
    root, assignment, index = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    try:
        if '--failed-step' in sys.argv:
            raise RuntimeError('srun or its launcher exited unsuccessfully')
        launch(root, assignment, index)
    except BaseException as error:
        stop = root / 'STOP.json'
        if root.resolve() == ROOT and not stop.exists():
            stop_phase(stop, dict(error=str(error), assignment=str(assignment), index=index,
                                  job=os.environ.get('SLURM_JOB_ID')))
        raise
