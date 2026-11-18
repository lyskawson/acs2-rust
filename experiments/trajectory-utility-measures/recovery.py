import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pilot import ROOT, check_grant, registered_plan, source_identity, state_name
from registration import HERE, REPOSITORY, TRIAL, checksum, read
from grid_identity import compare
from grid_protocol import TERMINAL, allocation_rows, atomic_json, load_plan, require_budget, stop_phase
from grid_scope import SACCT_FIELDS
from launch import launch_run
from submit import execute, submission_command
from validate import CPU_FIELDS


RECOVERY = ROOT / 'recovery'
SHORT = 8
LONG = 30
TIMEOUT = 'TimeoutError: process deadline exceeded'
SHARED_STOP = 'RuntimeError: shared study STOP prevents further measurement'


def pilot_accounting(ledger, raw):
    requested = {}
    for entry in ledger:
        if entry['array_id'] is None:
            raise ValueError('unresolved pilot submission intent')
        for index, run in enumerate(entry['runs']):
            job_id = f"{entry['array_id']}_{index}"
            if job_id in requested:
                raise ValueError('duplicate allocation in the pilot ledger')
            requested[job_id] = run
    records = {row['JobID']: row for row in allocation_rows(raw)}
    if set(records) - set(requested):
        raise ValueError('unexpected allocations in the pilot accounting')
    allocations = {}
    attempts = []
    states = Counter()
    reserved = 0
    for job_id, run in requested.items():
        record = records.get(job_id)
        if record is None:
            states['awaiting_accounting'] += 1
            reserved += run['limit_seconds']
            continue
        raw_id = record['JobIDRaw']
        state = state_name(record)
        cpu = float(record['CPUTimeRAW'])
        if cpu and (raw_id in allocations or cpu > run['limit_seconds'] or record['AllocCPUS'] != '1'):
            raise ValueError('duplicated or invalid charged pilot allocation')
        if cpu:
            allocations[raw_id] = cpu
        if state not in TERMINAL:
            reserved += max(0, run['limit_seconds'] - cpu)
        states[state] += 1
        attempts.append(dict(run, job_id=job_id, job_id_raw=raw_id, state=state, allocated_cpu_seconds=cpu))
    return dict(spent_seconds=sum(allocations.values()), reserved_seconds=reserved,
                states=dict(states), attempts=attempts)


def accounting_text(ledger):
    if not ledger:
        return ''
    if any(entry['array_id'] is None for entry in ledger):
        raise ValueError('unresolved submission intent')
    ids = ','.join(str(entry['array_id']) for entry in ledger)
    return execute(['sacct', '--array', '-S', '2026-11-18', '-P', '--format=' + SACCT_FIELDS, '-j', ids])


def checked_completion(path, expected):
    completion = read(path / 'completion.json')
    if (completion['rows_expected'] != len(expected)
            or any(checksum(path / name) != digest for name, digest in completion['sha256'].items())):
        raise ValueError(f'invalid preserved completion: {path.name}')
    rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
    if (len(rows) != completion['rows_seen']
            or [row['nominal_step'] for row in rows] != expected[:len(rows)]):
        raise ValueError(f'invalid preserved checkpoints: {path.name}')
    return completion, rows


def first_wave(plan, raw):
    ledger = read(ROOT / 'ledger.json')
    audit = pilot_accounting(ledger, raw)
    if (audit['reserved_seconds'] or set(audit['states']) - TERMINAL
            or {run['attempt_id'] for run in audit['attempts']} != {run['attempt_id'] for run in plan['runs']}):
        raise ValueError('the whole initial wave must be terminal and accounted for')
    stop = read(ROOT / 'STOP.json')
    trigger = Path(stop['run'])
    if ('woods102_alcs' not in trigger.name or TRIAL not in trigger.name or stop['result']['failure'] != TIMEOUT):
        raise ValueError('unexpected initial STOP trigger')
    for run in audit['attempts']:
        config = plan['configurations'][run['configuration_id']]
        path = ROOT / run['relative_directory']
        record = path / 'completion.json'
        if run['state'] == 'CANCELLED':
            if run['allocated_cpu_seconds'] or record.exists():
                raise ValueError('cancelled dependency was started or charged')
            continue
        completion, rows = checked_completion(path, config['targets'])
        if run['state'] == 'COMPLETED':
            if completion['failure'] or completion['exit_code'] or len(rows) != len(config['targets']):
                raise ValueError('initial complete allocation lacks a complete measurement')
        elif run['state'] != 'FAILED' or completion['failure'] not in (TIMEOUT, SHARED_STOP):
            raise ValueError('unexpected failure in the stopped wave')
        if path == trigger and (completion != stop['result'] or rows[-1]['nominal_step'] < 300000):
            raise ValueError('STOP evidence differs from the preserved timely prefix')
    return audit


def retry_runs(plan, initial):
    wanted = {run['attempt_id'] for run in initial['attempts'] if run['state'] != 'COMPLETED'}
    result = []
    for run in plan['runs']:
        if run['attempt_id'] not in wanted:
            continue
        cfg = plan['configurations'][run['configuration_id']]
        task, arm = cfg['task'], run['agent']
        limit = run['limit_seconds'] // 60
        if limit < LONG:
            limit = SHORT
            if task in ('woods101_5_alcs', 'woods102_alcs') and arm == TRIAL:
                limit = 14
            elif (task in ('woods101_5_alcs', 'woods102_alcs', 'woods14', 'maze4')
                  or task in ('mazea_alcs', 'mazef4_alcs') and arm == TRIAL):
                limit = 12
        old = run['attempt_id']
        new = old.removesuffix('_a01') + '_a02'
        result.append(dict(run, attempt_id=new, previous_attempt_id=old,
                           previous_directory=run['relative_directory'], limit_seconds=limit * 60,
                           relative_directory=run['relative_directory'].replace(old, new, 1)))
    if len(result) != len(wanted):
        raise ValueError('retry cells differ from incomplete registered cells')
    return result


def grouped_runs(runs, concurrency):
    groups = defaultdict(list)
    for run in runs:
        groups[run['limit_seconds']].append(run)
    ordinary = [(limit, group) for limit, group in sorted(groups.items(), reverse=True) if limit < LONG * 60]
    spare = concurrency - sum(len(group) for limit, group in ordinary if limit != SHORT * 60)
    if spare <= 0:
        raise ValueError('retry throttle is exhausted')
    return [(group, min(len(group), spare if limit == SHORT * 60 else concurrency), False)
            for limit, group in ordinary] + [
        (group, min(len(group), concurrency), True)
        for limit, group in sorted(groups.items(), reverse=True) if limit >= LONG * 60]


def first_plan():
    plan = load_plan(ROOT / 'plan.json', (ROOT / 'plan.sha256').read_text().strip())
    if plan != registered_plan(*source_identity(), plan['build_cpu_seconds']):
        raise ValueError('retry measurement or pilot registration differs from initial wave')
    return plan


def snapshot_files():
    return {str(path.relative_to(REPOSITORY)): checksum(path)
            for path in sorted(REPOSITORY.rglob('*')) if path.is_file()}


def submit():
    if REPOSITORY != RECOVERY / 'operations':
        raise ValueError('retry requires the frozen operations snapshot')
    if execute(['squeue', '-u', 'alelys2099', '-n', 'tu-utility-pilot', '-h']).strip():
        raise ValueError('the initial pilot still holds allocations')
    plan = first_plan()
    raw = accounting_text(read(ROOT / 'ledger.json'))
    initial = first_wave(plan, raw)
    runs = retry_runs(plan, initial)
    prior_cpu = plan['build_cpu_seconds'] + initial['spent_seconds']
    operations = snapshot_files()
    (RECOVERY / 'logs').mkdir(exist_ok=False)
    (RECOVERY / 'initial-accounting.psv').write_text(raw)
    wrapper = dict(plan, runs=runs, prior_cpu_seconds=prior_cpu, operations=operations,
                   initial_accounting_sha256=checksum(RECOVERY / 'initial-accounting.psv'),
                   stop_sha256=checksum(ROOT / 'STOP.json'))
    path = RECOVERY / 'plan.json'
    atomic_json(path, wrapper)
    frozen = checksum(path)
    (RECOVERY / 'plan.sha256').write_text(frozen + '\n')
    ledger, ordinary = [], []
    ledger_path = RECOVERY / 'ledger.json'
    atomic_json(ledger_path, ledger)
    remaining = sum(run['limit_seconds'] for run in runs)
    for index, (batch, throttle, gated) in enumerate(grouped_runs(runs, plan['concurrency'])):
        if (RECOVERY / 'STOP.json').exists():
            raise ValueError('retry STOP blocks submission')
        original = first_plan()
        verified = load_plan(path, frozen)
        if (verified != wrapper or verified['runs'] != retry_runs(original, initial)
                or verified['operations'] != snapshot_files()
                or verified['stop_sha256'] != checksum(ROOT / 'STOP.json')
                or verified['initial_accounting_sha256'] != checksum(RECOVERY / 'initial-accounting.psv')):
            raise ValueError('frozen retry manifest differs')
        account = pilot_accounting(ledger, accounting_text(ledger))
        bound = require_budget(prior_cpu, account, remaining, plan['limit_seconds'] / 3600)
        grant, queue = check_grant(remaining)
        assignment = RECOVERY / f'assignment-{index:02d}.json'
        atomic_json(assignment, dict(plan_sha256=frozen, runs=batch, batch=index))
        for run in batch:
            (RECOVERY / run['relative_directory']).parent.parent.mkdir(parents=True, exist_ok=False)
        command = submission_command(RECOVERY, assignment, batch[0]['limit_seconds'] // 60,
                                     len(batch), throttle, plan['memory_mib'], ordinary if gated else [],
                                     'tu-utility-retry')
        command[-4:] = [str(HERE / 'job.sh'), str(HERE), str(RECOVERY), str(assignment), 'recovery.py']
        command.insert(1, '--kill-on-invalid-dep=yes')
        entry = dict(array_id=None, runs=batch, command=command, balance=account,
                     upper_bound_seconds=bound, remaining_seconds=remaining, grant_cpu_seconds=grant,
                     queue=queue, assignment_sha256=checksum(assignment), unix=time.time())
        try:
            entry['test_only'] = execute(command[:1] + ['--test-only'] + command[1:])
        except subprocess.CalledProcessError as error:
            atomic_json(RECOVERY / f'rejected-{index:02d}.json', dict(entry, test_only_error=error.output))
            raise
        ledger.append(entry)
        atomic_json(ledger_path, ledger)
        response = execute(command).strip()
        entry.update(array_id=int(response.split(';')[0]), submission_response=response)
        atomic_json(ledger_path, ledger)
        if not gated:
            ordinary.append(entry['array_id'])
        remaining -= sum(run['limit_seconds'] for run in batch)
        print(json.dumps(dict(array=entry['array_id'], cells=len(batch), concurrency=throttle,
                              depends_on=ordinary if gated else [], bound_cpu_hours=bound / 3600)), flush=True)


def prefix_identity(run, plan):
    previous = ROOT / run['previous_directory']
    output = RECOVERY / run['relative_directory']
    if not (previous / 'completion.json').is_file():
        return
    config = plan['configurations'][run['configuration_id']]
    _, old = checked_completion(previous, config['targets'])
    _, new = checked_completion(output, config['targets'])
    for row in (*old, *new):
        for field in CPU_FIELDS:
            del row['trajectory_utility']['cost'][field]
    if old:
        identity = compare(new, old, [42], [row['nominal_step'] for row in old], config['evaluation_blocks'])
        atomic_json(output / 'prefix-identity.json', dict(identity, excluded_utility_cpu_fields=CPU_FIELDS))


def worker(root, assignment, index):
    if root.resolve() != RECOVERY or assignment.parent != RECOVERY:
        raise ValueError('retry worker addresses the wrong root')
    if (root / 'STOP.json').exists():
        raise SystemExit('retry stopped')
    assigned = read(assignment)
    plan = load_plan(root / 'plan.json', assigned['plan_sha256'])
    original = first_plan()
    saved = root / 'initial-accounting.psv'
    if (plan['operations'] != snapshot_files() or plan['stop_sha256'] != checksum(ROOT / 'STOP.json')
            or plan['initial_accounting_sha256'] != checksum(saved)
            or plan['measurement_commit'] != original['measurement_commit']
            or plan['binary_sha256'] != original['binary_sha256']
            or plan['configurations'] != original['configurations']):
        raise ValueError('node retry snapshot or pilot source differs')
    if assigned['runs'] != grouped_runs(plan['runs'], plan['concurrency'])[assigned['batch']][0]:
        raise ValueError('node retry assignment differs')
    run = assigned['runs'][index]
    launch_run(root, plan, run, index)
    prefix_identity(run, plan)


def status(seal=False):
    first = first_plan()
    plan = load_plan(RECOVERY / 'plan.json', (RECOVERY / 'plan.sha256').read_text().strip())
    raw = accounting_text(read(RECOVERY / 'ledger.json'))
    audit = pilot_accounting(read(RECOVERY / 'ledger.json'), raw)
    initial = first_wave(first, (RECOVERY / 'initial-accounting.psv').read_text())
    states = Counter(audit['states']) + Counter(initial['states'])
    attempt_cpu = initial['spent_seconds'] + audit['spent_seconds']
    charged = first['build_cpu_seconds'] + attempt_cpu
    stopped = (RECOVERY / 'STOP.json').exists()
    summary = dict(submitted=sum(states.values()), states=dict(states), stopped=stopped,
                   initial_attempt_cpu_seconds=initial['spent_seconds'], retry_attempt_cpu_seconds=audit['spent_seconds'],
                   build_cpu_seconds=first['build_cpu_seconds'], spent_seconds=attempt_cpu,
                   total_charged_cpu_seconds=charged, reserved_seconds=audit['reserved_seconds'])
    print(json.dumps(summary, sort_keys=True), flush=True)
    if seal:
        if audit['reserved_seconds'] or stopped or states['COMPLETED'] != 84 or charged > first['limit_seconds']:
            raise ValueError('recovered pilot is incomplete')
        for run in audit['attempts']:
            path = RECOVERY / run['relative_directory']
            completion, rows = checked_completion(path, plan['configurations'][run['configuration_id']]['targets'])
            if (run['state'] != 'COMPLETED' or completion['failure'] or completion['exit_code']
                    or len(rows) != completion['rows_expected']
                    or ((ROOT / run['previous_directory'] / 'completion.json').exists()
                        and not (path / 'prefix-identity.json').exists())):
                raise ValueError('retry did not complete or lost prefix evidence')
        manifest = 'files.sha256.json'
        (RECOVERY / 'accounting.psv').write_text(raw)
        atomic_json(ROOT / 'accounting-summary.json', summary)
        attempts = [dict(run, wave='initial') for run in initial['attempts']] + [
            dict(run, wave='recovery', relative_directory='recovery/' + run['relative_directory'])
            for run in audit['attempts']]
        atomic_json(ROOT / 'attempt-accounting.json', attempts)
        atomic_json(ROOT / manifest, {str(path.relative_to(ROOT)): checksum(path)
            for path in sorted(ROOT.rglob('*')) if path.is_file() and path.name != manifest})
        print(checksum(ROOT / manifest), flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        root, assignment, index = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
        try:
            if '--failed-step' in sys.argv:
                raise RuntimeError('retry srun or launcher failed')
            worker(root, assignment, index)
        except BaseException as error:
            if root.resolve() == RECOVERY and not (root / 'STOP.json').exists():
                stop_phase(root / 'STOP.json', dict(error=str(error), assignment=str(assignment),
                                                    index=index, job=os.environ.get('SLURM_JOB_ID')))
            raise
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=('submit', 'status', 'seal'))
        args = parser.parse_args()
        if args.command == 'submit':
            submit()
        else:
            status(seal=args.command == 'seal')
