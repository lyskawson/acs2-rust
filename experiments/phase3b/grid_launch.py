import json
import os
from pathlib import Path
import subprocess
import sys

from grid_protocol import atomic_json, load_plan, seconds, stop_phase
from grid_attempts import select_attempts


def main():
    root, assignment, index = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    if (root / 'STOP.json').exists():
        raise SystemExit('phase stopped; no measurement started')
    try:
        plan = load_plan(root / 'approved-plan.json')
        assignments = json.loads(assignment.read_text())
        entry = assignments['runs'][index]
        repo = Path(assignments['measurement_repository'])
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
        if commit != plan['measurement_commit'] or dirty:
            raise ValueError('measurement source identity differs')
        details = subprocess.check_output(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True)
        fields = dict(field.split('=', 1) for field in details.split() if '=' in field)
        limit = seconds(fields['TimeLimit'])
        runtime = seconds(fields['RunTime'])
        timeout = limit - runtime - 30
        if timeout < 60:
            raise ValueError('insufficient allocation remaining for the process watchdog')
        config = next(item for item in plan['configurations'] if item['id'] == entry['configuration_id'])
        reference = json.loads((root / 'references.json').read_text())[config['id']]
        manifest = dict(configuration=config, agent=entry['agent'], seed=entry['seed'],
                        expected_host=entry.get('host'), attempt_id=entry['attempt_id'], expected_cpu_model=plan['cpu_model'],
                        expected_binary_sha256=plan['binary_sha256'], expected_preset=reference['preset'],
                        measurement_commit=commit, operations_commit=assignments['operations_commit'],
                        plan_sha256=assignments['plan_sha256'], first_row_timeout_seconds=60,
                        process_timeout_seconds=timeout, allocation_limit_seconds=limit,
                        stop_path=str(root / 'STOP.json'))
        output = root / entry['relative_directory']
        output = output.parent
        manifests = output.parent / 'resolved-manifests'
        manifests.mkdir(exist_ok=True)
        path = manifests / f"{config['id']}_{entry['agent']}_s{entry['seed']}.json"
        atomic_json(path, manifest)
        command = [sys.executable, '-B', str(Path(__file__).with_name('collect.py')),
                   '--manifest', str(path), '--binary', assignments['binary'], '--commit', commit,
                   '--output', str(output), '--index', '0']
        result = subprocess.run(command)
        if result.returncode:
            raise ValueError(f'collector exited {result.returncode}')
        ledger = json.loads((root / 'ledger.json').read_text())
        select_attempts(root, ledger, config)
    except BaseException as error:
        stop_phase(root / 'STOP.json', {'error': str(error), 'job': os.environ.get('SLURM_JOB_ID'), 'assignment': str(assignment), 'index': index})
        raise


if __name__ == '__main__':
    main()
