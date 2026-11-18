import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

from registration import HERE, REPOSITORY, STUDY, checksum, pilot_registration, read
from grid_protocol import TERMINAL, accounting, allocation_rows, atomic_json, load_plan, require_budget
from grid_scope import SACCT_FIELDS
from submit import execute, shared_reservation, submission_command


BINARY = REPOSITORY / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
ROOT = (Path.home() / 'tu_runs' / STUDY).resolve()


def source_identity():
    if REPOSITORY != (Path.home() / 'acs2-tu').resolve():
        raise ValueError('pilot requires the dedicated cluster clone')
    git = ['/usr/bin/git', '-C', str(REPOSITORY)]
    commit = execute(git + ['rev-parse', 'HEAD']).strip()
    if execute(git + ['status', '--porcelain']).strip():
        raise ValueError('pilot clone must be clean')
    return commit, checksum(BINARY)


def registered_plan(commit, binary_sha256):
    settings, preset, configs, runs = pilot_registration()
    return dict(
        study=STUDY, measurement_commit=commit, binary_sha256=binary_sha256,
        registration_sha256=checksum(HERE / 'plan.json'), cells_sha256=checksum(HERE / 'pilot-cells.json'),
        preset=preset, cpu_model=settings['cpu_model'], memory_mib=settings['memory_mib'],
        limit_seconds=settings['max_cpu_seconds'], concurrency=settings['max_running_jobs'],
        configurations=configs, runs=runs,
    )


def batches(plan):
    long, short = defaultdict(list), defaultdict(list)
    for run in plan['runs']:
        limit = run['limit_seconds']
        (long if limit >= 30 * 60 else short)[limit].append(run)
    spare = plan['concurrency'] - sum(map(len, long.values()))
    if spare <= 0:
        raise ValueError('long jobs leave no capacity for the short-job chain')
    return [(runs, len(runs), False) for _, runs in sorted(long.items(), reverse=True)] + [
        (runs, min(len(runs), spare), True) for _, runs in sorted(short.items(), reverse=True)
    ]


def accounts(ledger):
    if any(entry['array_id'] is None for entry in ledger):
        raise ValueError('reconcile the pending submission intent first')
    command = ['sacct', '--array', '-S', '2026-11-18', '-P', '--format=' + SACCT_FIELDS]
    raw = execute(command + ['-j', ','.join(str(entry['array_id']) for entry in ledger)]) if ledger else ''
    records = list({row['JobIDRaw']: row for row in allocation_rows(raw)}.values())
    return raw, records, accounting(ledger, records)


def submit():
    commit, digest = source_identity()
    plan = registered_plan(commit, digest)
    ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT / 'logs').mkdir()
    path = ROOT / 'plan.json'
    atomic_json(path, plan)
    frozen = checksum(path)
    (ROOT / 'plan.sha256').write_text(frozen + '\n')
    ledger = []
    atomic_json(ROOT / 'ledger.json', ledger)
    remaining = sum(run['limit_seconds'] for run in plan['runs'])
    previous_short = []
    for number, (runs, throttle, short_chain) in enumerate(batches(plan)):
        if (ROOT / 'STOP.json').exists():
            raise ValueError('pilot STOP blocks further submission')
        if load_plan(path, frozen) != registered_plan(*source_identity()):
            raise ValueError('pilot registration, source or binary changed')
        _, _, balance = accounts(ledger)
        bound = require_budget(0, balance, remaining, plan['limit_seconds'] / 3600)
        grant = execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n']).strip()
        queue = execute(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
        new = sum(run['limit_seconds'] for run in runs)
        if float(grant) + shared_reservation(queue) + remaining > 15_000 * 3600:
            raise ValueError('shared grant cannot cover the queue and registered remaining limits')
        assignment = ROOT / f'assignment-{number:02d}.json'
        atomic_json(assignment, dict(plan_sha256=frozen, batch=number, runs=runs))
        for run in runs:
            (ROOT / run['relative_directory']).parent.parent.mkdir(parents=True, exist_ok=False)
        limit = runs[0]['limit_seconds'] // 60
        command = submission_command(ROOT, assignment, limit, len(runs), throttle, plan['memory_mib'],
                                     previous_short if short_chain else [], 'tu-utility-pilot')
        command[-4:] = [str(HERE / 'job.sh'), str(HERE), str(ROOT), str(assignment)]
        command.insert(1, '--kill-on-invalid-dep=yes')
        event = dict(array_id=None, runs=runs, grant_cpu_seconds=float(grant), queue=queue,
                     balance=balance, remaining_seconds=remaining, upper_bound_seconds=bound,
                     command=command, assignment_sha256=checksum(assignment), unix=time.time())
        try:
            event['test_only'] = execute(command[:1] + ['--test-only'] + command[1:])
        except subprocess.CalledProcessError as error:
            atomic_json(ROOT / f'rejected-{number:02d}.json', dict(event, test_only_error=error.output))
            raise
        ledger.append(event)
        atomic_json(ROOT / 'ledger.json', ledger)
        response = execute(command).strip()
        event.update(array_id=int(response.split(';')[0]), submission_response=response)
        atomic_json(ROOT / 'ledger.json', ledger)
        if short_chain:
            previous_short = [event['array_id']]
        remaining -= new
        print(json.dumps(dict(array=event['array_id'], cells=len(runs), limit_minutes=limit,
                              concurrency=throttle, bound_cpu_hours=bound / 3600)), flush=True)


def state_name(record):
    return record['State'].split()[0].split('+')[0]


def status(seal=False):
    ledger = read(ROOT / 'ledger.json')
    raw, records, balance = accounts(ledger)
    by_id = {record['JobID']: record for record in records}
    submitted = sum(len(entry['runs']) for entry in ledger)
    states = defaultdict(int)
    for entry in ledger:
        for task in range(len(entry['runs'])):
            record = by_id.get(f"{entry['array_id']}_{task}")
            states[state_name(record) if record else 'awaiting_accounting'] += 1
    stopped = (ROOT / 'STOP.json').exists()
    summary = dict(submitted=submitted, states=dict(states), stopped=stopped, **balance)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if seal:
        if any(state not in TERMINAL for state in states):
            raise ValueError('seal only after the submitted allocations are terminal')
        plan = load_plan(ROOT / 'plan.json', (ROOT / 'plan.sha256').read_text().strip())
        for batch, entry in enumerate(ledger):
            assignment = ROOT / f'assignment-{batch:02d}.json'
            if checksum(assignment) != entry['assignment_sha256']:
                raise ValueError('assignment changed since admission')
            for task, run in enumerate(entry['runs']):
                record = by_id[f"{entry['array_id']}_{task}"]
                if state_name(record) != 'COMPLETED':
                    continue
                directory = ROOT / run['relative_directory']
                completion = read(directory / 'completion.json')
                if (completion['failure'] or completion['exit_code'] or completion['incomplete_final_line']
                        or completion['rows_seen'] != len(plan['configurations'][run['configuration_id']]['targets'])
                        or any(checksum(directory / name) != digest for name, digest in completion['sha256'].items())):
                    raise ValueError(f'invalid completion: {run["attempt_id"]}')
        (ROOT / 'accounting.psv').write_text(raw)
        atomic_json(ROOT / 'accounting-summary.json', summary)
        manifest = 'files.sha256.json'
        atomic_json(ROOT / manifest, {str(path.relative_to(ROOT)): checksum(path)
            for path in sorted(ROOT.rglob('*')) if path.is_file() and path.name != manifest})
        print(checksum(ROOT / manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('submit', 'status', 'seal'))
    args = parser.parse_args()
    if args.command == 'submit':
        submit()
    else:
        status(seal=args.command == 'seal')
