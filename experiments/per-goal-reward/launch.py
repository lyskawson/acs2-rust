import json
import os
from pathlib import Path
import subprocess
import sys

import protocol
from execution import require_authorization, registered_runs
from grid_launch import allocation_fields
from grid_protocol import atomic_json, load_plan, seconds, stop_phase
from grid_scope import stop_path, verify_operations
from grid_attempts import select_attempts
from schema import validate_sequence


def launch(root, assignment, index):
    root, assignment = Path(root), Path(assignment)
    active_stop = root / 'STOP.json'
    if active_stop.exists():
        raise ValueError('study stopped; no measurement started')
    try:
        assigned = json.loads(assignment.read_text())
        plan = load_plan(root / 'plan.json', assigned['plan_sha256'])
        active_stop = stop_path(root, plan)
        if active_stop.exists():
            raise ValueError('study stopped; no measurement started')
        require_authorization(root, plan)
        verify_operations(plan)
        if assigned['runs'] != registered_runs(plan) or assigned['operations_commit'] != plan['operations_commit']:
            raise ValueError('node assignment differs from the twelve registered pilot runs')
        run = assigned['runs'][index]
        repository = Path(assigned['measurement_repository'])
        commit = subprocess.check_output(['/usr/bin/git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['/usr/bin/git', '-C', str(repository), 'status', '--porcelain'], text=True).strip()
        if commit != plan['measurement_commit'] or dirty:
            raise ValueError('node measurement source identity differs')
        array, task = os.environ['SLURM_ARRAY_JOB_ID'], os.environ['SLURM_ARRAY_TASK_ID']
        if int(task) != index:
            raise ValueError('node array task differs from assignment')
        selector = f'{array}_{task}'
        details = subprocess.check_output(['scontrol', 'show', 'job', '-o', selector], text=True)
        fields = allocation_fields(details, os.environ['SLURM_JOB_ID'], array, task)
        output = (root / run['relative_directory']).parent
        atomic_json(output.parent / 'allocation-start.json', dict(selector=selector, response=details, fields=fields))
        limit = seconds(fields['TimeLimit'])
        timeout = limit - seconds(fields['RunTime']) - 30
        if timeout < 60 or limit != run['limit_seconds']:
            raise ValueError('node allocation has insufficient remaining time or a different limit')
        config = next(item for item in plan['configurations'] if item['id'] == run['configuration_id'])
        manifest = dict(configuration=config, agent=run['agent'], seed=run['seed'], expected_host=None,
            attempt_id=run['attempt_id'], expected_cpu_model=plan['cpu_model'],
            expected_binary_sha256=plan['binary_sha256'], expected_preset=plan['preset'], measurement_commit=commit,
            operations_commit=assigned['operations_commit'], plan_sha256=assigned['plan_sha256'],
            first_row_timeout_seconds=60, process_timeout_seconds=timeout, allocation_limit_seconds=limit,
            stop_path=str(active_stop))
        manifests = output.parent / 'resolved-manifests'
        manifests.mkdir(exist_ok=True)
        path = manifests / f"{config['id']}_{run['agent']}_s{run['seed']}.json"
        atomic_json(path, manifest)
        command = [sys.executable, '-B', str(Path(__file__).with_name('collector.py')),
            '--manifest', str(path), '--binary', assigned['binary'], '--commit', commit,
            '--output', str(output), '--index', '0']
        subprocess.run(command, check=True)
        rows = [json.loads(line) for line in (root / run['relative_directory'] / 'rows.jsonl').read_text().splitlines()]
        validate_sequence(rows, config, run['agent'], run['seed'], commit)
        atomic_json(root / run['relative_directory'] / 'identity-checks.json', [])
        select_attempts(root, json.loads((root / 'ledger.json').read_text()), config)
    except BaseException as error:
        if not active_stop.exists():
            stop_phase(active_stop, dict(error=str(error), job=os.environ.get('SLURM_JOB_ID'),
                                       assignment=str(assignment), index=index))
        raise


if __name__ == '__main__':
    launch(sys.argv[1], sys.argv[2], int(sys.argv[3]))
