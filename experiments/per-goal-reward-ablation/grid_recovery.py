import hashlib
import json
from pathlib import Path

from execution import checksum
from grid_protocol import atomic_json, require_budget
from submit import shared_reservation


REASONS = {
    'scontrol timeout before launch': 'FAILED',
    'launch refused by STOP': 'FAILED',
    'cancelled before allocation start': 'CANCELLED',
}
ARCHIVED_STOP = 'STOP-infrastructure.json'


def retry_run(run):
    attempt = run['attempt_id']
    if not attempt.endswith('_a01'):
        raise ValueError('infrastructure recovery requires a first attempt')
    retry = attempt[:-2] + '02'
    return dict(run, attempt_id=retry, cause='retry after diagnosed SLURM infrastructure failure',
                relative_directory=run['relative_directory'].replace(attempt, retry, 1))


def checked_recovery(root, ledger):
    receipt = root / 'infrastructure-recovery.json'
    if not receipt.exists() or not (root / ARCHIVED_STOP).exists():
        return {}
    value = json.loads(receipt.read_text())
    stop = root / ARCHIVED_STOP
    prefix = json.dumps(ledger[:value['original_arrays']], indent=2) + '\n'
    if (checksum(stop) != value['stop_sha256']
            or hashlib.sha256(prefix.encode()).hexdigest() != value['original_ledger_sha256']):
        raise ValueError('archived infrastructure STOP or original admission ledger differs')
    return value['excluded']


def recovered_record(account, job, proof):
    return bool(proof and account.get('JobIDRaw') == proof['job_id_raw']
                and account.get('State', '').split()[0] == proof['state']
                and job == proof['job'] and float(account['CPUTimeRAW']) == proof['allocation_cpu_seconds'])


def evidence_matches(root, account, job, reason):
    log = root / 'logs' / f'{job}.err'
    if reason == 'cancelled before allocation start':
        return account['Start'] == 'None' and float(account['CPUTimeRAW']) == 0 and not log.exists()
    text = log.read_text() if log.exists() else ''
    if reason == 'launch refused by STOP':
        return 'ValueError: grid stopped' in text
    return 'Socket timed out' in text and "'scontrol', 'show', 'job'" in text


def start_recovery(root, ledger, records, state, runs, diagnostics, limit_hours, executor):
    stop = root / 'STOP.json'
    archived_stop = root / ARCHIVED_STOP
    receipt = root / 'infrastructure-recovery.json'
    if archived_stop.exists() or not stop.exists() or state['reserved_seconds']:
        raise ValueError('recovery requires an original STOP and no outstanding allocations')
    evidence = json.loads(Path(diagnostics).read_text())
    if evidence['stop'] != json.loads(stop.read_text()):
        raise ValueError('diagnosis does not describe the current STOP')
    accounts = {record['JobID']: record for record in records}
    registered = {run['attempt_id']: run for run in runs}
    excluded = {}
    triggers = set()
    for reason, allocations in evidence['allocations'].items():
        for item in allocations:
            attempt = item['attempt']
            if (attempt in excluded or item['state'] != REASONS.get(reason)
                    or item['launch_record'] or item['measurement_bytes'] or item['resources_bytes']):
                raise ValueError('recovery evidence must describe a unique, pre-measurement infrastructure loss')
            excluded[attempt] = dict(item, reason=reason)
            if reason == 'scontrol timeout before launch':
                triggers.add(item['job_id_raw'])
    if evidence['stop'].get('job') not in triggers:
        raise ValueError('the first STOP did not originate in a diagnosed infrastructure timeout')

    seen = set()
    affected_arrays = {item['job'].split('_')[0] for item in excluded.values()
                       if item['reason'] == 'scontrol timeout before launch'}
    for entry in ledger:
        blocked_dependency = any(str(parent) in affected_arrays for parent in entry.get('depends_on', ()))
        if blocked_dependency:
            affected_arrays.add(str(entry['array_id']))
        for index, run in enumerate(entry['runs']):
            attempt = run['attempt_id']
            job = f"{entry['array_id']}_{index}"
            if attempt in seen or registered.get(attempt) != run or job not in accounts:
                raise ValueError('original allocation ledger or registration differs')
            seen.add(attempt)
            account = accounts[job]
            if account['State'] == 'COMPLETED' and account['ExitCode'] == '0:0' and attempt not in excluded:
                continue
            proof = excluded.get(attempt)
            if not recovered_record(account, job, proof) or not evidence_matches(root, account, job, proof['reason']):
                raise ValueError(f'no matching infrastructure diagnosis for {job}')
            if proof['state'] == 'CANCELLED' and not blocked_dependency:
                raise ValueError(f'cancelled allocation {job} lacks a dependency on the infrastructure failure')
            if (root / run['relative_directory']).exists():
                raise ValueError(f'measurement artifacts exist for {job}')
    if not excluded.keys() <= seen:
        raise ValueError('diagnosis contains an allocation outside the ledger')

    remaining = [run for run in runs if run['attempt_id'] not in seen or run['attempt_id'] in excluded]
    reservation = sum(run['limit_seconds'] for run in remaining)
    bound = require_budget(0, state, reservation, limit_hours)
    grant = executor(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n'])
    queue = executor(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
    atomic_json(receipt, dict(original_arrays=len(ledger), original_ledger_sha256=checksum(root / 'ledger.json'),
                              stop_sha256=checksum(stop), diagnostics_sha256=checksum(diagnostics),
                              excluded=excluded, remaining_reservation_seconds=reservation,
                              bound_seconds=bound, grant=grant, queue=queue,
                              shared_grant_headroom_seconds=15000 * 3600 - float(grant.strip()) - shared_reservation(queue)))
    stop.rename(archived_stop)
    return dict(recovered=len(excluded), remaining=len(remaining), bound_cpu_hours=bound / 3600)
