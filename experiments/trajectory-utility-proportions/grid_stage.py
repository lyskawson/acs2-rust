import argparse
import csv
import io
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from proportions import HERE, checksum, configuration, read
from cost_forecast import dependent_limit
from grid_plan import (BINARY_SHA256, GRANT_SECONDS, MAX_SECONDS, MEASUREMENT_COMMIT, PILOT_CHARGE,
                       derive, schedule, schedule_totals)
from grid_protocol import TERMINAL, atomic_json, load_plan, require_budget, seconds, stop_phase
from pilot import BINARY, CLONE, ROOT as PILOT, sacct, state
from recovery_plan import (FIRST_PLAN, PREFIX_COUNT, derive_recovery, recovery_schedule)
from submit import execute, shared_reservation, submission_command
from validate_proportions import validate_row
from worker import allocation, deterministic_fields
import collect
import pair_schema
from schema import per_pair_enabled

INITIAL_GRID = PILOT / 'grid'
RECOVERY = PILOT / 'grid-recovery'
ROOT = RECOVERY if HERE == RECOVERY / 'operations/experiments/trajectory-utility-proportions' else INITIAL_GRID
JOB = 'tu-proportions-recovery' if ROOT == RECOVERY else 'tu-proportions-grid'


def operations():
    return {str(path.relative_to(ROOT / 'operations')): checksum(path)
            for path in sorted((ROOT / 'operations').rglob('*')) if path.is_file()}


def measurement_identity():
    git = ['/usr/bin/git', '-C', str(CLONE)]
    if (execute(git + ['rev-parse', 'HEAD']).strip() != MEASUREMENT_COMMIT
            or execute(git + ['status', '--porcelain']).strip() or checksum(BINARY) != BINARY_SHA256):
        raise ValueError('the cluster must run the unchanged measurement binary and checkout')


def registered_schedule(plan_sha):
    final = load_plan(HERE / 'final-plan.json', plan_sha)
    manifest = load_plan(PILOT / 'files.sha256.json', final['pilot_manifest_sha256'])
    summary = load_plan(PILOT / 'pilot-summary.json', manifest['pilot-summary.json'])
    plan = derive(final, manifest, summary)
    scheduled = schedule(plan)
    totals = schedule_totals(scheduled)
    registered = final['costs']['scheduler_reservation']
    if (registered != dict(totals, pilot_cpu_seconds=PILOT_CHARGE,
                           authorization_cpu_seconds=MAX_SECONDS,
                           cumulative_cpu_seconds=totals['reserved_seconds'] + PILOT_CHARGE)
            or registered['cumulative_cpu_seconds'] > MAX_SECONDS):
        raise ValueError('the actual scheduler envelope differs from registration or exceeds approval')
    return plan, scheduled


def admitted_schedule(plan_sha):
    plan, scheduled = registered_schedule(plan_sha)
    if ROOT == RECOVERY:
        if plan_sha != FIRST_PLAN:
            raise ValueError('recovery must use the original approved grid plan')
        plan = derive_recovery(plan, INITIAL_GRID, read(HERE / 'recovery-registration.json'))
        scheduled = recovery_schedule(plan)
    return plan, scheduled


def prepare(plan_sha, operations_commit):
    if HERE != ROOT / 'operations/experiments/trajectory-utility-proportions':
        raise ValueError('run grid operations from a frozen results snapshot')
    measurement_identity()
    admitted_schedule(plan_sha)
    (ROOT / 'logs').mkdir(exist_ok=False)
    atomic_json(ROOT / 'operations.json', dict(plan_sha256=plan_sha,
                                              operations_commit=operations_commit, files=operations()))
    atomic_json(ROOT / 'ledger.json', [])
    print(checksum(ROOT / 'operations.json'), flush=True)


def current(operations_sha, verify_files=True):
    if HERE != ROOT / 'operations/experiments/trajectory-utility-proportions':
        raise ValueError('invalid grid operations location')
    frozen = load_plan(ROOT / 'operations.json', operations_sha)
    if verify_files and frozen['files'] != operations():
        raise ValueError('the grid operations snapshot changed')
    return admitted_schedule(frozen['plan_sha256'])


def allocation_status(ledger, gate=None):
    if any(entry['array_id'] is None for entry in ledger):
        raise ValueError('a submission intent requires reconciliation')
    raw, _ = sacct([entry['array_id'] for entry in ledger])
    records = {}
    charged = set()
    for row in csv.DictReader(io.StringIO(raw), delimiter='|'):
        identifier = row['JobIDRaw']
        if '.' in identifier or '[' in row['JobID']:
            continue
        if row['JobID'] in records:
            raise ValueError('duplicate Slurm allocation slot')
        if not (state(row) in ('PENDING', 'CANCELLED') and row['CPUTimeRAW'] == '0'
                and row['Start'] in ('None', 'Unknown', 'N/A', '')):
            if identifier in charged:
                raise ValueError('duplicate Slurm allocation raw ID')
            charged.add(identifier)
        records[row['JobID']] = row
    by_job, states = {}, Counter()
    spent = reserved = 0
    for entry in ledger:
        limit = entry['limit_seconds']
        if gate and entry['phase'] == 'conditional':
            limit = gate['arrays'][str(entry['array_id'])]['limit_seconds']
        for slot in range(entry['allocation_count']):
            job = f"{entry['array_id']}_{slot}"
            row = records.get(job)
            status = state(row)
            states[status] += 1
            cost = int(row['CPUTimeRAW']) if row else 0
            if (cost < 0 or cost and row['AllocCPUS'] != '1'
                    or status == 'COMPLETED' and cost > limit):
                raise ValueError(f'Slurm CPU charge differs from the admitted allocation {job}')
            spent += cost
            reserved += max(0, limit - cost) if status not in TERMINAL else 0
            by_job[job] = dict(state=status, charged_cpu_seconds=cost,
                               raw_id=row['JobIDRaw'] if row else None,
                               exit_code=row['ExitCode'] if row else None)
    return raw, dict(spent_seconds=spent, reserved_seconds=reserved, states=dict(states), jobs=by_job)


def failures(account):
    return {job: row['state'] for job, row in account['jobs'].items()
            if row['state'] in TERMINAL and (row['state'] != 'COMPLETED' or row['exit_code'] != '0:0')}


def check_grant(needed, slots):
    used = int(execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n']).strip())
    queue = execute(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
    qos = execute(['sacctmgr', '-n', '-P', 'show', 'qos', 'hpc-alelys2099-1784823245',
                   'format=GrpTRESMins'])
    if 'cpu=1200000' not in qos or used + shared_reservation(queue) + needed > GRANT_SECONDS:
        raise ValueError('the confirmed shared grant cannot cover this submission')
    values = execute(['sacctmgr', '-n', '-P', 'show', 'assoc', 'user=alelys2099',
                      'format=MaxSubmitJobs']).splitlines()
    limits = [int(value) for row in values if (value := row.strip().strip('|')).isdigit()]
    if not limits or len(queue.splitlines()) + slots > min(limits):
        raise ValueError('the entire grid would exceed the available Slurm association slots')
    return dict(used_cpu_seconds=used, queue=queue, association_limit=min(limits), qos=qos)


def charge_and_reservation(account, planned=0, prior=PILOT_CHARGE):
    return require_budget(prior, account, planned, MAX_SECONDS / 3600)


def assignment(index):
    return ROOT / f'assignment-{index:03d}.json'


def submit(operations_sha):
    measurement_identity()
    plan, scheduled = current(operations_sha)
    path = ROOT / 'ledger.json'
    if read(path):
        raise ValueError('a submitted grid cannot be submitted again')
    totals = schedule_totals(scheduled)
    remaining, slots = totals['reserved_seconds'], totals['allocations']
    arrays, ledger = {}, []
    for index, batch in enumerate(scheduled):
        if (ROOT / 'STOP.json').exists() or current(operations_sha)[1][index] != batch:
            raise ValueError('STOP or changed assignment blocks further admission')
        _, account = allocation_status(ledger)
        if failures(account):
            raise ValueError('a submitted allocation failed')
        bound = charge_and_reservation(account, remaining, plan.get('prior_cpu_seconds', PILOT_CHARGE))
        grant = check_grant(remaining, slots)
        descriptor = assignment(index)
        atomic_json(descriptor, dict(operations_sha256=operations_sha, index=index, batch=batch))
        jobs = len(batch['jobs'])
        deps = [arrays[name] for name in batch['dependencies']]
        command = submission_command(ROOT, descriptor, batch['limit_seconds'] // 60, jobs,
                                     batch['throttle'], plan['memory_mib'], deps, JOB)
        command[-4:] = [str(HERE / 'grid_stage_job.sh'), str(HERE), str(descriptor)]
        if deps:
            command.insert(1, '--kill-on-invalid-dep=yes')
        event = dict(array_id=None, name=batch['name'], phase=batch['phase'], command=command,
                     limit_seconds=batch['limit_seconds'], allocation_count=jobs,
                     run_count=sum(map(len, batch['jobs'])), dependencies=deps,
                     assignment_sha256=checksum(descriptor), grant=grant, upper_bound_seconds=bound)
        try:
            event['test_only'] = execute(command[:1] + ['--test-only'] + command[1:])
        except subprocess.CalledProcessError as error:
            atomic_json(ROOT / f'rejected-{index:03d}.json', dict(event, test_only_error=error.output))
            raise
        ledger.append(event)
        atomic_json(path, ledger)
        reply = execute(command).strip()
        event.update(array_id=int(reply.split(';')[0]), response=reply, submitted_unix=time.time())
        atomic_json(path, ledger)
        arrays[batch['name']] = event['array_id']
        remaining -= batch['limit_seconds'] * jobs
        slots -= jobs
        print(json.dumps(dict(array=event['array_id'], phase=event['phase'], allocations=jobs,
                              runs=event['run_count'], dependencies=deps, cpu_hours_bound=bound/3600)),
              flush=True)
    atomic_json(ROOT / 'all-submitted.json', dict(runs=len(plan['runs']), ledger_sha256=checksum(path)))


def gate_result():
    path = ROOT / 'scout-gate.json'
    return read(path) if path.exists() else None


def output_directory(run, config):
    return ROOT / 'attempts' / run['run_id'] / 'measurement' / f"{config['id']}_{run['agent']}_s{run['seed']}"


def checked_result(directory, targets):
    complete = read(directory / 'completion.json')
    if (complete['failure'] or complete['exit_code'] or complete['incomplete_final_line']
            or complete['rows_seen'] != len(targets)
            or any(checksum(directory / name) != digest for name, digest in complete['sha256'].items())):
        raise ValueError(f'incomplete or changed measurement: {directory}')
    rows, incomplete = collect.complete_rows(directory / 'rows.jsonl')
    if incomplete or [row['nominal_step'] for row in rows] != targets:
        raise ValueError('measurement checkpoints differ')
    return complete, rows


def matched_prefix(reference, folder, rows, blocks):
    path = folder / reference['path']
    if checksum(path) != reference['sha256']:
        raise ValueError('preserved row checksum differs')
    old, incomplete = collect.complete_rows(path)
    if incomplete or len(old) != reference['checkpoints'] or len(old) >= len(rows):
        raise ValueError('saved rows are not the registered prefix')
    if any(deterministic_fields(a, blocks) != deterministic_fields(b, blocks)
           for a, b in zip(old, rows)):
        raise ValueError('full-budget run failed bitwise identity against its preserved prefix')
    return len(old)


def prefix_source(run):
    if run.get('previous'):
        return run['previous'], INITIAL_GRID, 'recovery-prefix-identity.json', 'previous_sha256'
    if run['pilot']:
        return run['pilot'], PILOT, 'prefix-identity.json', 'pilot_sha256'
    return None


def compare_prefix(run, config, directory):
    if source := prefix_source(run):
        reference, folder, name, key = source
        _, rows = checked_result(directory, config['targets'])
        count = matched_prefix(reference, folder, rows, config['evaluation_blocks'])
        result = {'rows': count, 'differences': 0, key: reference['sha256']}
        atomic_json(directory / name, result)


def expect_pending(array, count, limit):
    text = execute(['squeue', '--array', '-h', '-j', str(array), '-o', '%i|%T|%l'])
    actual = {job: (status, seconds(wall)) for job, status, wall in
              (line.split('|') for line in text.splitlines())}
    expected = {f'{array}_{i}': ('PENDING', limit) for i in range(count)}
    if actual != expected:
        raise ValueError('a dependent allocation was not pending with its registered limit')


def scout_gate(plan, scheduled):
    ledger_file = ROOT / 'ledger.json'
    ledger = read(ledger_file)
    admitted = read(ROOT / 'all-submitted.json')
    if admitted != dict(runs=len(plan['runs']), ledger_sha256=checksum(ledger_file)):
        raise ValueError('gate requires the complete admitted grid')
    scouts = [(batch['jobs'][0][0], entry) for batch, entry in zip(scheduled, ledger) if batch['phase'] == 'scout']
    for _ in range(12):
        _, scouting = allocation_status([entry for _, entry in scouts])
        if failures(scouting):
            raise ValueError('a scout failed')
        if scouting['states'] == {'COMPLETED': 12}:
            break
        time.sleep(5)
    else:
        raise ValueError('scout accounting did not settle')
    observations = {}
    for run, entry in scouts:
        cfg = configuration(run, plan['configs'])
        directory = output_directory(run, cfg)
        prior = run['pilot']
        proof = read(directory / 'prefix-identity.json')
        if proof != dict(rows=prior['checkpoints'], differences=0, pilot_sha256=prior['sha256']):
            raise ValueError('a scout did not reproduce its pilot prefix')
        job = f"{entry['array_id']}_0"
        cost = scouting['jobs'][job]['charged_cpu_seconds']
        if not 0 < cost <= run['limit_seconds']:
            raise ValueError('invalid scout allocation cost')
        observations[run['task'], run['arm']] = dict(job=job, charged_cpu_seconds=cost,
                                                   time_limit_seconds=run['limit_seconds'])
    updates = {}
    for batch, entry in zip(scheduled, ledger):
        if batch['phase'] != 'conditional':
            continue
        run = batch['jobs'][0][0]
        key = run['task'], run['arm']
        scout = observations[key]
        new = dependent_limit(scout['charged_cpu_seconds'], run['seed_spread'])
        if new > entry['limit_seconds']:
            raise ValueError(f'scout-based time limit exceeds the initial envelope for {key}')
        updates[str(entry['array_id'])] = dict(task=key[0], arm=key[1], limit_seconds=new, scout=scout)
        expect_pending(entry['array_id'], entry['allocation_count'], entry['limit_seconds'])
    _, account = allocation_status(ledger)
    if failures(account) or (ROOT / 'STOP.json').exists():
        raise ValueError('a failure blocks the scout gate')
    bound = charge_and_reservation(account)
    for entry in ledger:
        if entry['phase'] == 'conditional':
            array = entry['array_id']
            new = updates[str(array)]['limit_seconds']
            execute(['scontrol', 'update', f'JobId={array}', f'TimeLimit={new//60}'])
            expect_pending(array, entry['allocation_count'], new)
    atomic_json(ROOT / 'scout-gate.json', dict(arrays=updates, upper_bound_seconds=bound))


def own_ledger(index, array, path):
    for _ in range(15):
        ledger = read(ROOT / 'ledger.json')
        if len(ledger) > index and ledger[index]['array_id'] is not None:
            entry = ledger[index]
            if entry['array_id'] != int(array) or entry['assignment_sha256'] != checksum(path):
                raise ValueError('this allocation is not in the admitted ledger')
            return entry
        time.sleep(1)
    raise ValueError('allocation admission did not settle')


def worker(path):
    if (ROOT / 'STOP.json').exists():
        raise SystemExit('grid stopped')
    assigned = read(path)
    plan, scheduled = current(assigned['operations_sha256'])
    index, batch = assigned['index'], assigned['batch']
    if path != assignment(index) or batch != scheduled[index]:
        raise ValueError('node assignment differs from registration')
    array, slot, job = [os.environ[name] for name in ('SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_ID')]
    entry = own_ledger(index, array, path)
    started = time.monotonic()
    detail, actual = allocation(array, slot, job)
    limit = entry['limit_seconds']
    if batch['phase'] == 'conditional':
        gate = gate_result()
        if not gate:
            raise ValueError('dependent job has no successful scout gate')
        limit = gate['arrays'][array]['limit_seconds']
    if (actual['Partition'] != 'lem-cpu-normal' or actual['NumCPUs'] != '1'
            or os.environ.get('SLURM_MEM_PER_NODE') != str(plan['memory_mib'])
            or seconds(actual['TimeLimit']) != limit):
        raise ValueError('node allocation differs from the admitted grid')
    if batch['phase'] == 'gate':
        scout_gate(plan, scheduled)
        return
    measurement_identity()
    for run in batch['jobs'][int(slot)]:
        if (ROOT / 'STOP.json').exists():
            raise SystemExit('grid stopped')
        run_limit = limit if batch['phase'] == 'conditional' else run['limit_seconds']
        elapsed = seconds(actual['RunTime']) + time.monotonic() - started
        timeout = int(min(run_limit, limit - elapsed) - 30)
        if timeout < 60:
            raise ValueError('insufficient allocation time for the registered run')
        cfg = configuration(run, plan['configs'])
        directory = output_directory(run, cfg)
        attempt = directory.parents[1]
        attempt.mkdir(parents=True, exist_ok=False)
        atomic_json(attempt / 'allocation-start.json', dict(response=detail, fields=actual))
        manifest = dict(configuration=cfg, agent=run['agent'], seed=run['seed'], attempt_id=run.get('attempt_id', run['run_id']),
                        stop_path=str(ROOT/'STOP.json'), measurement_commit=MEASUREMENT_COMMIT,
                        expected_binary_sha256=BINARY_SHA256, expected_preset=plan['preset'],
                        expected_cpu_model=plan['cpu_model'], process_timeout_seconds=timeout,
                        allocation_limit_seconds=run_limit, first_row_timeout_seconds=60,
                        allocation_job_id=f'{array}_{slot}', allocation_job_id_raw=job)
        spec = attempt / 'manifest.json'
        atomic_json(spec, manifest)
        subprocess.run(['srun', '--ntasks=1', '--cpu-bind=cores', f'--time={run_limit // 60}',
                        sys.executable, '-B', str(Path(__file__)), 'measure', str(spec)], check=True)
        compare_prefix(run, cfg, directory)


def measure(path):
    spec = read(path)
    if path.resolve().parents[2] != ROOT or spec['stop_path'] != str(ROOT / 'STOP.json'):
        raise ValueError('invalid measurement path or STOP marker')
    array, slot, job, step = [os.environ[name] for name in
                             ('SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_ID', 'SLURM_STEP_ID')]
    if spec['allocation_job_id'] != f'{array}_{slot}' or spec['allocation_job_id_raw'] != job or not step.isdigit():
        raise ValueError('measurement step does not belong to its allocation')
    atomic_json(path.parent / 'step.json', dict(job_id=f'{array}_{slot}', job_id_raw=job, step_id=step,
                                               limit_seconds=spec['allocation_limit_seconds']))
    pair_schema.enabled = per_pair_enabled
    collect.validate_row = validate_row
    collect.run(argparse.Namespace(manifest=str(path), binary=str(BINARY), commit=MEASUREMENT_COMMIT,
                                   output=str(path.parent / 'measurement'), index=0, sample_interval=0.25))


def status(operations_sha, seal=False):
    plan, scheduled = current(operations_sha)
    ledger = read(ROOT / 'ledger.json')
    gate = gate_result()
    raw, account = allocation_status(ledger, gate)
    if (bad := failures(account)) and not (ROOT / 'STOP.json').exists():
        stop_phase(ROOT / 'STOP.json', dict(failed_allocations=bad))
    stopped = (ROOT / 'STOP.json').exists()
    completed = sum(len(runs) for batch, event in zip(scheduled, ledger)
                    for slot, runs in enumerate(batch['jobs'])
                    if account['jobs'][f"{event['array_id']}_{slot}"]['state'] == 'COMPLETED')
    prior = plan.get('prior_cpu_seconds', PILOT_CHARGE)
    bound = charge_and_reservation(account, prior=prior)
    summary = dict(states=account['states'], completed_runs=completed, stopped=stopped, reused=len(plan['reused']),
                   submitted_runs=sum(entry['run_count'] for entry in ledger), pilot_cpu_seconds=PILOT_CHARGE,
                   grid_cpu_seconds=account['spent_seconds'], reserved_seconds=account['reserved_seconds'],
                   upper_bound_seconds=bound, grid_total_pilot_and_spent_cpu_seconds=prior+account['spent_seconds'])
    if plan.get('recovery'):
        summary.update(earlier_completed_runs=plan['earlier_completed_runs'],
                       earlier_grid_cpu_seconds=plan['earlier_grid_cpu_seconds'])
    print(json.dumps(summary), flush=True)
    if seal:
        if account['reserved_seconds'] or set(account['states']) - TERMINAL:
            raise ValueError('seal requires terminal Slurm accounting')
        if not stopped and (completed != len(plan['runs']) or (ROOT == INITIAL_GRID and not gate)):
            raise ValueError('the entire admitted grid and gate must complete successfully')
        if not stopped:
            reproduced = 0
            for run in plan['runs']:
                cfg = configuration(run, plan['configs'])
                directory = output_directory(run, cfg)
                checked_result(directory, cfg['targets'])
                if source := prefix_source(run):
                    reference, _, name, key = source
                    proof = read(directory / name)
                    if proof != {'rows': reference['checkpoints'], 'differences': 0, key: reference['sha256']}:
                        raise ValueError('the preserved prefix proof differs')
                    reproduced += proof['rows']
            if plan.get('recovery'):
                if reproduced != PREFIX_COUNT:
                    raise ValueError('the recovery did not reproduce every saved checkpoint')
                summary['reproduced_checkpoints'] = reproduced
        (ROOT / 'accounting.psv').write_text(raw)
        atomic_json(ROOT / 'accounting-summary.json', summary)
        atomic_json(ROOT / 'allocations.json', account['jobs'])
        name = 'files.sha256.json'
        atomic_json(ROOT / name, {str(path.relative_to(ROOT)): checksum(path)
                                  for path in sorted(ROOT.rglob('*')) if path.is_file() and path.name != name})
        print(checksum(ROOT / name), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'submit', 'worker', 'measure', 'status', 'seal', 'failed'))
    parser.add_argument('argument')
    parser.add_argument('--operations-commit')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.argument, args.operations_commit)
        elif args.command == 'submit':
            submit(args.argument)
        elif args.command == 'worker':
            worker(Path(args.argument))
        elif args.command == 'measure':
            measure(Path(args.argument))
        elif args.command == 'failed':
            raise RuntimeError('Slurm allocation or worker failed')
        else:
            status(args.argument, args.command == 'seal')
    except BaseException as error:
        if args.command in ('submit', 'worker', 'measure', 'failed') and not (ROOT / 'STOP.json').exists():
            stop_phase(ROOT / 'STOP.json', dict(command=args.command, argument=args.argument,
                                               job=os.environ.get('SLURM_JOB_ID'), error=str(error)))
        raise
