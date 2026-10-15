import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from execution import APPROVED_SHA256, BINARY_SHA256, MEASUREMENT_COMMIT
from grid_authorization import require_authorization
from grid_attempts import inspect_attempt, attempts
from grid_protocol import atomic_json, load_plan, require_budget
from grid_scope import snapshot, verify_operations
from submit import execute, shared_reservation, submission_command

RECOVERY_SHA256 = 'afae09a7e1e96ba48c35169cf7bd45ca6452c2a6dc303e2bb8db1d377e989615'
RETRY_SEEDS = [44, 46, 49, 52, 53, 59, 61]


def approved_recovery(proposal, approval):
    proposal = load_plan(proposal, RECOVERY_SHA256)
    approval = json.loads(Path(approval).read_text())
    expected = dict(approved=True, confirmed_on='2026-10-16', plan_sha256=APPROVED_SHA256,
                    recovery_sha256=RECOVERY_SHA256, scope='seven_incomplete_maze4_k2_retries_at_ten_minutes_only',
                    phase_cpu_hour_limit=110)
    if approval != expected:
        raise ValueError('retry requires the exact approved operational recovery')
    return proposal


def retry(root, proposal_path, approval_path):
    root = Path(root).resolve()
    proposal = approved_recovery(proposal_path, approval_path)
    batch = root / 'maze4_c5_pairs_acs2vcp_k2'
    digest = (batch / 'plan.sha256').read_text().strip()
    plan = load_plan(batch / 'plan.json', digest)
    require_authorization(batch, plan, digest)
    verify_operations(plan)
    stop = root / 'STOP.json'
    cause = json.loads(stop.read_text())
    if (not cause['run'].endswith('maze4_c5_pairs_acs2vcp_k2_s44')
            or cause['result']['failure'] != 'TimeoutError: process deadline exceeded'):
        raise ValueError('STOP is not the approved process timeout')
    ledger = json.loads((batch / 'ledger.json').read_text())
    if len(ledger) != 1 or ledger[0]['array_id'] != 6109875:
        raise ValueError('recovery is single use and must follow the original array')
    origin = json.loads((batch / 'assignment-00.json').read_text())
    repo = Path(origin['measurement_repository'])
    if execute(['git', '-C', str(repo), 'rev-parse', 'HEAD']).strip() != MEASUREMENT_COMMIT or execute(['git', '-C', str(repo), 'status', '--porcelain']).strip():
        raise ValueError('frozen measurement source differs')
    if hashlib.sha256(Path(origin['binary']).read_bytes()).hexdigest() != BINARY_SHA256:
        raise ValueError('frozen measurement binary differs')
    failed = []
    for attempt in attempts(batch, ledger):
        config = plan['configurations'][0]
        item, _ = inspect_attempt(attempt, config)
        if item['status'] != 'complete':
            failed.append(attempt['seed'])
    if sorted(failed) != RETRY_SEEDS:
        raise ValueError('only the seven approved incomplete seeds may be retried')
    state = snapshot(plan, execute)
    expected = proposal['accounting']
    if state['spent_seconds'] != expected['grid_spent_cpu_seconds'] or state['reserved_seconds'] != 0:
        raise ValueError('recovery accounting differs from the approved terminal state')
    bound = require_budget(16041, state, 4200 + expected['all_remaining_registered_hard_seconds'], 110)
    if bound != expected['whole_study_upper_bound_seconds']:
        raise ValueError('recovery whole-study hard bound differs')
    runs = []
    for seed in RETRY_SEEDS:
        identifier = f'maze4_c5_pairs_acs2vcp_k2_s{seed}_a02'
        directory = f'attempts/{identifier}/measurement/maze4_c5_pairs_acs2vcp_k2_s{seed}'
        if (batch / directory).exists():
            raise ValueError('recovery attempt already has artifacts')
        runs.append(dict(configuration_id='maze4_c5_pairs', agent='acs2vcp_k2', seed=seed,
                         attempt_id=identifier, relative_directory=directory, limit_seconds=600,
                         cause='approved timeout recovery ' + RECOVERY_SHA256))
    assignment = batch / 'assignment-recovery-01.json'
    command = submission_command(batch, assignment, 10, 7, 20, plan['memory_mib'], [], 'tu-vcp-recovery')
    grant = execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n'])
    queue = execute(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
    if float(grant.strip()) + shared_reservation(queue) + 4200 > 15000 * 3600:
        raise ValueError('shared grant cannot cover recovery')
    tested = execute(command[:1] + ['--test-only'] + command[1:])
    recovery = root / 'recovery'
    recovery.mkdir(exist_ok=True)
    atomic_json(recovery / 'approved-recovery.json', proposal)
    (recovery / 'approval.json').write_bytes(Path(approval_path).read_bytes())
    (recovery / 'STOP-first-batch.json').write_bytes(stop.read_bytes())
    for run in runs:
        (batch / run['relative_directory']).parent.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(assignment, dict(origin, runs=runs))
    entry = dict(configuration_id='maze4_c5_pairs', runs=runs, command=command, grant=grant, queue=queue,
                 test_only=tested, balance=state, bound_seconds=bound, submitted_unix=time.time(), array_id=None,
                 recovery_sha256=RECOVERY_SHA256, operations_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    ledger.append(entry)
    atomic_json(batch / 'ledger.json', ledger)
    event = dict(recovery_sha256=RECOVERY_SHA256, event='approved_timeout_resolution', state=state,
                 new_limit_seconds=4200, whole_study_hard_bound_seconds=bound,
                 grant=grant, queue=queue, test_only=tested, command=command, unix=time.time())
    atomic_json(recovery / 'resolution.json', event)
    stop.unlink()
    try:
        response = execute(command).strip()
    except subprocess.CalledProcessError:
        stop.write_bytes((recovery / 'STOP-first-batch.json').read_bytes())
        raise
    entry.update(array_id=int(response.split(';')[0]), submission_response=response)
    atomic_json(batch / 'ledger.json', ledger)
    print(json.dumps(dict(array_id=entry['array_id'], runs=7, recovery_sha256=RECOVERY_SHA256, bound_cpu_hours=bound/3600)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--proposal', required=True, type=Path)
    parser.add_argument('--approval', required=True, type=Path)
    args = parser.parse_args()
    retry(args.root, args.proposal, args.approval)
