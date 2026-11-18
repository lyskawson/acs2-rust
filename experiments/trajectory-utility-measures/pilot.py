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


MEASUREMENT_REPOSITORY = (Path.home() / 'acs2-tu').resolve()
BINARY = MEASUREMENT_REPOSITORY / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
ROOT = (Path.home() / 'tu_runs' / STUDY).resolve()
BUILD_LIMIT = 600


def clean_commit():
    git = ['/usr/bin/git', '-C', str(MEASUREMENT_REPOSITORY)]
    commit = execute(git + ['rev-parse', 'HEAD']).strip()
    if execute(git + ['status', '--porcelain']).strip():
        raise ValueError('pilot clone must be clean')
    return commit


def source_identity():
    return clean_commit(), checksum(BINARY)


def registered_plan(commit, binary_sha256, build_cpu_seconds=0):
    settings, preset, configs, runs = pilot_registration()
    return dict(
        study=STUDY, measurement_commit=commit, binary_sha256=binary_sha256,
        registration_sha256=checksum(HERE / 'plan.json'), cells_sha256=checksum(HERE / 'pilot-cells.json'),
        preset=preset, cpu_model=settings['cpu_model'], memory_mib=settings['memory_mib'],
        limit_seconds=settings['max_cpu_seconds'], concurrency=settings['max_running_jobs'],
        configurations=configs, runs=runs, build_cpu_seconds=build_cpu_seconds,
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


def check_grant(remaining):
    used = float(execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n']).strip())
    queue = execute(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
    if used + shared_reservation(queue) + remaining > 15_000 * 3600:
        raise ValueError('shared grant cannot cover the queue and registered remaining limits')
    return used, queue


def build():
    if REPOSITORY != MEASUREMENT_REPOSITORY:
        raise ValueError('pilot build requires the dedicated cluster clone')
    commit = clean_commit()
    settings, _, _, runs = pilot_registration()
    remaining = sum(run['limit_seconds'] for run in runs) + BUILD_LIMIT
    bound = require_budget(0, dict(spent_seconds=0, reserved_seconds=0), remaining, settings['max_cpu_seconds'] / 3600)
    grant, queue = check_grant(remaining)
    ROOT.mkdir(parents=True, exist_ok=False)
    evidence = ROOT / 'build'
    evidence.mkdir()
    command = ['sbatch', '--parsable', '--partition=lem-cpu-normal', '--cpus-per-task=1', '--mem=8192M',
               f'--time={BUILD_LIMIT // 60}', '--no-requeue', '--job-name=tu-utility-build',
               '--output=' + str(evidence / 'build.out'), '--error=' + str(evidence / 'build.err'),
               str(HERE / 'build.sh'), str(REPOSITORY), commit]
    event = dict(job_id=None, commit=commit, limit_seconds=BUILD_LIMIT, upper_bound_seconds=bound,
                 grant_cpu_seconds=grant, queue=queue, command=command, unix=time.time())
    try:
        event['test_only'] = execute(command[:1] + ['--test-only'] + command[1:])
    except subprocess.CalledProcessError as error:
        atomic_json(evidence / 'rejected.json', dict(event, test_only_error=error.output))
        raise
    path = evidence / 'submission.json'
    atomic_json(path, event)
    response = execute(command).strip()
    event.update(job_id=int(response.split(';')[0]), submission_response=response)
    atomic_json(path, event)
    print(json.dumps(dict(job=event['job_id'], bound_cpu_hours=bound / 3600)), flush=True)


def completed_build(commit, binary_sha256):
    evidence = ROOT / 'build'
    submission = read(evidence / 'submission.json')
    job = str(submission['job_id'])
    raw = execute(['sacct', '-S', '2026-11-18', '-P', '--format=' + SACCT_FIELDS, '-j', job])
    records = {row['JobIDRaw']: row for row in allocation_rows(raw) if row['JobID'] == job}
    record = records.get(job)
    if (submission['commit'] != commit or not record or record['State'] != 'COMPLETED'
            or record['ExitCode'] != '0:0' or record['AllocCPUS'] != '1'
            or not 0 < float(record['CPUTimeRAW']) <= submission['limit_seconds']
            or binary_sha256 not in (evidence / 'build.out').read_text()):
        raise ValueError('binary build is incomplete or differs from the recorded allocation')
    (evidence / 'accounting.psv').write_text(raw)
    return float(record['CPUTimeRAW'])


def submit():
    if REPOSITORY != MEASUREMENT_REPOSITORY:
        raise ValueError('initial pilot submission requires the dedicated cluster clone')
    commit, digest = source_identity()
    build_cpu_seconds = completed_build(commit, digest)
    plan = registered_plan(commit, digest, build_cpu_seconds)
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
        if load_plan(path, frozen) != registered_plan(*source_identity(), build_cpu_seconds):
            raise ValueError('pilot registration, source or binary changed')
        _, _, balance = accounts(ledger)
        bound = require_budget(build_cpu_seconds, balance, remaining, plan['limit_seconds'] / 3600)
        grant, queue = check_grant(remaining)
        new = sum(run['limit_seconds'] for run in runs)
        assignment = ROOT / f'assignment-{number:02d}.json'
        atomic_json(assignment, dict(plan_sha256=frozen, batch=number, runs=runs))
        for run in runs:
            (ROOT / run['relative_directory']).parent.parent.mkdir(parents=True, exist_ok=False)
        limit = runs[0]['limit_seconds'] // 60
        command = submission_command(ROOT, assignment, limit, len(runs), throttle, plan['memory_mib'],
                                     previous_short if short_chain else [], 'tu-utility-pilot')
        command[-4:] = [str(HERE / 'job.sh'), str(HERE), str(ROOT), str(assignment)]
        command.insert(1, '--kill-on-invalid-dep=yes')
        event = dict(array_id=None, runs=runs, grant_cpu_seconds=grant, queue=queue,
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
    plan = load_plan(ROOT / 'plan.json', (ROOT / 'plan.sha256').read_text().strip())
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
    summary = dict(submitted=submitted, states=dict(states), stopped=stopped,
                   build_cpu_seconds=plan['build_cpu_seconds'],
                   total_charged_cpu_seconds=plan['build_cpu_seconds'] + balance['spent_seconds'], **balance)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if seal:
        if any(state not in TERMINAL for state in states):
            raise ValueError('seal only after the submitted allocations are terminal')
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
    parser.add_argument('command', choices=('build', 'submit', 'status', 'seal'))
    args = parser.parse_args()
    if args.command == 'build':
        build()
    elif args.command == 'submit':
        submit()
    else:
        status(seal=args.command == 'seal')
