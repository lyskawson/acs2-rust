import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from registration import HERE, REPOSITORY, checksum, read
from grid_limits import (LIMIT_HOURS, PILOT_CPU_SECONDS, PILOT_SHA256, PLAN_SHA256,
                         batches, cell_key, conditional_limit, derive)
from grid_protocol import TERMINAL, atomic_json, load_plan, require_budget, seconds, stop_phase
from grid_retry import RETRY_LIMIT_SECONDS, STOPPED_MANIFEST_SHA256, retry_batches, retry_plan, stopped_evidence
from launch import allocation
from pilot import BINARY, ROOT as PILOT, check_grant, source_identity
from recovery import accounting_text, checked_completion, pilot_accounting
from submit import execute, submission_command


ROOT = PILOT / 'grid'
RETRY = PILOT / 'grid-recovery'
MEASUREMENT_COMMIT = 'bb9a9e4c219fefc879b4d66e3cdc3775e57bf0d5'
BINARY_SHA256 = '9bd61888fbae98ff38e339c1184216db5a60ac90a9709a9016fad61a0710541f'
MEMORY_MIB = 2048
CPU_MODEL = 'AMD EPYC 9554 64-Core Processor'
JOB_NAME = 'tu-measures-grid'


def evidence_sha(path):
    with path.open('rb') as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def verify_measurement():
    if source_identity() != (MEASUREMENT_COMMIT, BINARY_SHA256):
        raise ValueError('the approved pilot binary or measurement source differs')


def snapshot():
    return {str(path.relative_to(REPOSITORY)): checksum(path)
            for path in sorted(REPOSITORY.rglob('*')) if path.is_file()}


def verify_pilot():
    manifest = load_plan(PILOT / 'files.sha256.json', PILOT_SHA256)
    account = load_plan(PILOT / 'accounting-summary.json', manifest['accounting-summary.json'])
    if account['reserved_seconds'] or account['total_charged_cpu_seconds'] != PILOT_CPU_SECONDS:
        raise ValueError('sealed pilot accounting differs')


def plan_for(wrapper, root=ROOT):
    if (REPOSITORY != root / 'operations'
            or wrapper['plan_sha256'] != PLAN_SHA256 or wrapper['operations'] != snapshot()
            or wrapper['measurement_commit'] != MEASUREMENT_COMMIT or wrapper['binary_sha256'] != BINARY_SHA256
            or wrapper['total_cpu_hour_limit'] != LIMIT_HOURS or wrapper['pilot_cpu_seconds'] != PILOT_CPU_SECONDS):
        raise ValueError('pinned operations snapshot or final plan differs')
    verify_measurement()
    verify_pilot()
    if root == RETRY:
        if (wrapper['stopped_manifest_sha256'] != STOPPED_MANIFEST_SHA256
                or wrapper['retry_limit_seconds'] != RETRY_LIMIT_SECONDS):
            raise ValueError('the staged retry does not match its archived wave or limits')
        return retry_plan(ROOT, wrapper['inputs_sha256'])
    return derive(load_plan(root / 'grid-inputs.json', wrapper['inputs_sha256']))


def scheduled_batches(plan):
    if 'earlier_grid_cpu_seconds' in plan:
        return retry_batches(plan['runs'])
    return batches(plan)


def prior_cpu_seconds(plan):
    return PILOT_CPU_SECONDS + plan.get('earlier_grid_cpu_seconds', 0)


def bound_current(plan):
    return sum(len(batch['jobs']) * batch['limit_seconds'] for batch in plan)


def prepare(inputs_sha, root=ROOT):
    if (root / 'grid-plan.json').exists():
        raise ValueError('the grid already has a frozen plan')
    wrapper = dict(plan_sha256=PLAN_SHA256, inputs_sha256=inputs_sha, operations=snapshot(),
                   measurement_commit=MEASUREMENT_COMMIT, binary_sha256=BINARY_SHA256,
                   total_cpu_hour_limit=LIMIT_HOURS, pilot_cpu_seconds=PILOT_CPU_SECONDS)
    if root == RETRY:
        wrapper.update(stopped_manifest_sha256=STOPPED_MANIFEST_SHA256, retry_limit_seconds=RETRY_LIMIT_SECONDS)
    plan = plan_for(wrapper, root)
    reservations = bound_current(scheduled_batches(plan))
    require_budget(prior_cpu_seconds(plan), dict(spent_seconds=0, reserved_seconds=0), reservations, LIMIT_HOURS)
    (root / 'logs').mkdir(exist_ok=False)
    atomic_json(root / 'grid-plan.json', wrapper)
    (root / 'grid-plan.sha256').write_text(checksum(root / 'grid-plan.json') + '\n')
    atomic_json(root / 'ledger.json', [])


def current_plan(root=ROOT):
    digest = (root / 'grid-plan.sha256').read_text().strip()
    wrapper = load_plan(root / 'grid-plan.json', digest)
    return digest, plan_for(wrapper, root)


def effective_limit(entry, gate):
    if entry['phase'] == 'conditional' and gate:
        return gate['arrays'][str(entry['array_id'])]['limit_seconds']
    return entry['limit_seconds']


def accounting(ledger, gate=None):
    tracked = []
    for entry in ledger:
        limit = effective_limit(entry, gate)
        tracked.append(dict(array_id=entry['array_id'], runs=[
            dict(job_slot=f"{entry['name']}:{i}", phase=entry['phase'], limit_seconds=limit)
            for i in range(entry['allocation_count'])]))
    raw = accounting_text(tracked)
    return raw, pilot_accounting(tracked, raw)


def faulty(account):
    return [dict(job=run['job_id'], state=run['state'])
            for run in account['attempts'] if run['state'] in TERMINAL and run['state'] != 'COMPLETED']


def association_slots():
    response = execute(['sacctmgr', '-n', '-P', 'show', 'assoc', 'user=alelys2099', 'format=MaxSubmitJobs'])
    limits = [int(value) for row in response.splitlines() if (value := row.strip().strip('|')).isdigit()]
    if not limits:
        raise ValueError('could not verify the WCSS association submission limit')
    return min(limits)


def assignment_name(index, root=ROOT):
    return root / f'assignment-{index:03d}.json'


def submit(root=ROOT):
    digest, plan = current_plan(root)
    scheduled = scheduled_batches(plan)
    ledger_path = root / 'ledger.json'
    if read(ledger_path):
        raise ValueError('grid submission requires an empty ledger')
    if root == RETRY:
        manifest, summary = stopped_evidence(ROOT)
        previous = load_plan(ROOT / 'ledger.json', manifest['ledger.json'])
        _, live = accounting(previous, gate_result())
        if (live['reserved_seconds'] or live['spent_seconds'] != plan['earlier_grid_cpu_seconds']
                or live['states'] != summary['allocations']):
            raise ValueError('the previous grid accounting changed after STOP')
    remaining = bound_current(scheduled)
    remaining_jobs = sum(len(batch['jobs']) for batch in scheduled)
    ledger = []
    ids = {}
    for index, batch in enumerate(scheduled):
        if (root / 'STOP.json').exists():
            raise ValueError('study STOP blocks submission')
        fresh_digest, fresh_plan = current_plan(root)
        if (fresh_digest, scheduled_batches(fresh_plan)[index]) != (digest, batch):
            raise ValueError('array manifest differs from the registration')
        _, account = accounting(ledger)
        if failures := faulty(account):
            raise ValueError(f'a grid allocation failed during submission: {failures}')
        bound = require_budget(prior_cpu_seconds(plan), account, remaining, LIMIT_HOURS)
        grant, queue = check_grant(remaining)
        if len(queue.splitlines()) + remaining_jobs > association_slots():
            raise ValueError('the full grid exceeds the available Slurm association slots')
        path = assignment_name(index, root)
        atomic_json(path, dict(index=index, grid_plan_sha256=digest, batch=batch))
        dependencies = [ids[name] for name in batch['dependencies']]
        jobs = len(batch['jobs'])
        job_name = JOB_NAME if root == ROOT else JOB_NAME + '-retry'
        command = submission_command(root, path, batch['limit_seconds'] // 60, jobs, batch['throttle'],
                                     MEMORY_MIB, dependencies, job_name)
        command[-4:] = [str(HERE / 'grid_job.sh'), str(HERE), str(path), batch['phase']]
        if dependencies:
            command.insert(1, '--kill-on-invalid-dep=yes')
        entry = dict(array_id=None, name=batch['name'], phase=batch['phase'],
                     allocation_count=jobs, run_count=sum(len(job['runs']) for job in batch['jobs']),
                     limit_seconds=batch['limit_seconds'], assignment_sha256=checksum(path),
                     dependencies=dependencies, command=command, grant_cpu_seconds=grant, queue=queue,
                     upper_bound_seconds=bound, submitted_unix=time.time())
        try:
            entry['test_only'] = execute(command[:1] + ['--test-only'] + command[1:])
        except subprocess.CalledProcessError as error:
            atomic_json(root / f'rejected-{index:03d}.json', dict(entry, test_only_error=error.output))
            raise
        ledger.append(entry)
        atomic_json(ledger_path, ledger)
        response = execute(command).strip()
        array = int(response.split(';')[0])
        entry.update(array_id=array, submission_response=response)
        atomic_json(ledger_path, ledger)
        ids[batch['name']] = array
        remaining -= jobs * batch['limit_seconds']
        remaining_jobs -= jobs
        print(json.dumps(dict(array=array, phase=entry['phase'], allocations=jobs, runs=entry['run_count'],
                              dependencies=dependencies, bound_cpu_hours=bound / 3600)), flush=True)
    atomic_json(root / 'all-submitted.json', dict(arrays=len(ledger), runs=sum(entry['run_count'] for entry in ledger),
                                                ledger_sha256=checksum(ledger_path)))


def ledger_for_assignment(index, root):
    array_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
    for _ in range(10):
        ledger = read(root / 'ledger.json')
        if index < len(ledger) and ledger[index]['array_id'] is not None:
            entry = ledger[index]
            if entry['array_id'] != array_id:
                raise ValueError('allocation array identity differs')
            return entry
        time.sleep(1)
    raise ValueError('submission record is missing')


def gate_result(root=ROOT):
    path = root / 'scout-gate.json'
    return read(path) if path.exists() else None


def confirm_pending(array, count, expected_limit):
    rows = execute(['squeue', '--array', '-h', '-j', str(array), '-o', '%i|%T|%l']).splitlines()
    values = dict((job, (state, seconds(limit))) for job, state, limit in (row.split('|') for row in rows))
    expected = {f'{array}_{i}': ('PENDING', expected_limit) for i in range(count)}
    if values != expected or len(rows) != count:
        raise ValueError(f'gated array does not have the expected pending time limit: {array}')


def checked_result(run, config, root=ROOT):
    directory = root / run['relative_directory']
    completion, rows = checked_completion(directory, config['targets'])
    if (completion['exit_code'] or completion['failure'] or completion['incomplete_final_line']
            or len(rows) != len(config['targets'])):
        raise ValueError(f'full run did not complete: {run["run_id"]}')
    return completion


def scout_gate(plan):
    ledger_path = ROOT / 'ledger.json'
    admitted = read(ROOT / 'all-submitted.json')
    if admitted['runs'] != len(plan['runs']) or admitted['ledger_sha256'] != checksum(ledger_path):
        raise ValueError('scout gate requires the entire admitted grid')
    ledger = read(ledger_path)
    scout_entries = [(index, entry) for index, entry in enumerate(ledger) if entry['phase'] == 'scout']
    for _ in range(12):
        _, scout_account = accounting([entry for _, entry in scout_entries])
        if scout_account['states'] == {'COMPLETED': 6}:
            break
        if faulty(scout_account):
            raise ValueError('a scout allocation failed')
        time.sleep(5)
    else:
        raise ValueError('six completed scouts are required for the gate')
    by_job = {run['job_id']: run['allocated_cpu_seconds'] for run in scout_account['attempts']}
    scout = {}
    for index, entry in scout_entries:
        assigned = read(assignment_name(index))
        run = assigned['batch']['jobs'][0]['runs'][0]
        checked_result(run, plan['configs'][run['task']])
        cost = by_job[f"{entry['array_id']}_0"]
        if not 0 < cost <= entry['limit_seconds'] or not (ROOT / run['relative_directory'] / 'prefix-identity.json').is_file():
            raise ValueError('the scout lacks admissible accounting or prefix identity')
        scout[cell_key(run)] = dict(allocation_cpu_seconds=cost, archive_spread=run['archive_spread'],
                                    job_id=f"{entry['array_id']}_0", attempt_id=run['attempt_id'])
    limits = {}
    drop = 0
    for index, entry in enumerate(ledger):
        if entry['phase'] != 'conditional':
            continue
        assigned = read(assignment_name(index))
        run = assigned['batch']['jobs'][0]['runs'][0]
        actual = scout[cell_key(run)]
        limit = conditional_limit(actual['allocation_cpu_seconds'], actual['archive_spread'])
        if limit > entry['limit_seconds']:
            raise ValueError('scout-derived limit exceeds the preapproved reservation')
        drop += (entry['limit_seconds'] - limit) * entry['allocation_count']
        limits[str(entry['array_id'])] = dict(task=run['task'], agent=run['agent'], limit_seconds=limit, **actual)
    _, account = accounting(ledger)
    if faulty(account) or (ROOT / 'STOP.json').exists():
        raise ValueError('study stopped before the scout gate')
    bound = require_budget(PILOT_CPU_SECONDS, account, -drop, LIMIT_HOURS)
    for entry in ledger:
        if entry['phase'] != 'conditional':
            continue
        array = entry['array_id']
        limit = limits[str(array)]['limit_seconds']
        confirm_pending(array, entry['allocation_count'], entry['limit_seconds'])
        execute(['scontrol', 'update', f'JobId={array}', f'TimeLimit={limit // 60}'])
        confirm_pending(array, entry['allocation_count'], limit)
    atomic_json(ROOT / 'scout-gate.json', dict(upper_bound_seconds=bound, arrays=limits, scouts=list(scout.values())))


def worker(path, root):
    if (root / 'STOP.json').exists():
        raise ValueError('study stopped')
    digest, plan = current_plan(root)
    manifest = read(path)
    index = manifest['index']
    if manifest['grid_plan_sha256'] != digest or path != assignment_name(index, root):
        raise ValueError('node assignment lacks the pinned grid plan')
    batch = scheduled_batches(plan)[index]
    if manifest['batch'] != batch:
        raise ValueError('node assignment differs from its registration')
    entry = ledger_for_assignment(index, root)
    if checksum(path) != entry['assignment_sha256']:
        raise ValueError('submitted assignment bytes differ')
    array, task, job = (os.environ[name] for name in ('SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_ID'))
    detail, fields = allocation(array, task, job)
    gate = gate_result(root)
    limit = effective_limit(entry, gate)
    if (fields['Partition'] != 'lem-cpu-normal' or fields['NumCPUs'] != '1'
            or os.environ.get('SLURM_MEM_PER_NODE') != str(MEMORY_MIB)
            or seconds(fields['TimeLimit']) != limit or batch['phase'] == 'conditional' and not gate):
        raise ValueError('node allocation differs from the approved grid')
    if batch['phase'] == 'gate':
        scout_gate(plan)
        return
    started = time.monotonic()
    elapsed_before = seconds(fields['RunTime'])
    for run in batch['jobs'][int(task)]['runs']:
        if (root / 'STOP.json').exists():
            raise ValueError('study stopped before the next run')
        run_limit = limit if batch['phase'] == 'conditional' else run['limit_seconds']
        elapsed = elapsed_before + time.monotonic() - started
        timeout = min(run_limit, limit - elapsed) - 30
        if timeout < 60:
            raise ValueError('insufficient allocation time before the measurement')
        path = root / run['relative_directory']
        path.parent.parent.mkdir(parents=True, exist_ok=False)
        config = dict(plan['configs'][run['task']])
        if run['per_pair']:
            config['per_start_diagnostics'] = True
        manifest = dict(configuration=config, agent=run['agent'], seed=run['seed'], attempt_id=run['attempt_id'],
                        stop_path=str(root / 'STOP.json'), measurement_commit=MEASUREMENT_COMMIT,
                        expected_binary_sha256=BINARY_SHA256, expected_preset=plan['preset'], expected_cpu_model=CPU_MODEL,
                        process_timeout_seconds=timeout, allocation_limit_seconds=run_limit,
                        first_row_timeout_seconds=60, allocation_job_id=f'{array}_{task}', allocation_job_id_raw=job)
        manifest_path = path.parent.parent / 'manifest.json'
        atomic_json(manifest_path, manifest)
        atomic_json(path.parent.parent / 'allocation-start.json', dict(response=detail, fields=fields))
        subprocess.run(['srun', '--ntasks=1', '--cpu-bind=cores', f'--time={run_limit // 60}',
                        sys.executable, '-B', str(HERE / 'grid_measure.py'), str(manifest_path), str(path.parent)],
                       check=True)
        if run['pilot']:
            from grid_measure import compare_pilot_prefix
            compare_pilot_prefix(PILOT, run, config, path)
        if run.get('previous'):
            from grid_measure import compare_retry_prefix
            compare_retry_prefix(ROOT, run, config, path)


def status(seal=False, root=ROOT):
    _, plan = current_plan(root)
    ledger = read(root / 'ledger.json')
    gate = gate_result(root)
    raw, account = accounting(ledger, gate)
    failed = faulty(account)
    stop = root / 'STOP.json'
    if failed and not stop.exists():
        stop_phase(stop, dict(terminal_failures=failed))
    states = Counter()
    recorded = {attempt['job_slot']: attempt for attempt in account['attempts']}
    for index, entry in enumerate(ledger):
        batch = read(assignment_name(index, root))['batch']
        for slot, job in enumerate(batch['jobs']):
            if attempt := recorded.get(f"{entry['name']}:{slot}"):
                if attempt['state'] == 'COMPLETED':
                    states[entry['phase']] += len(job['runs'])
    summary = dict(completed_runs=dict(states), submitted_runs=sum(entry['run_count'] for entry in ledger),
                   pilot_reused=len(plan['reused']), pilot_cpu_seconds=PILOT_CPU_SECONDS,
                   grid_cpu_seconds=plan.get('earlier_grid_cpu_seconds', 0) + account['spent_seconds'],
                   reserved_seconds=account['reserved_seconds'], states=account['states'],
                   stopped=stop.exists(), scout_gate=gate,
                   total_charged_cpu_seconds=prior_cpu_seconds(plan) + account['spent_seconds'])
    if root == RETRY:
        summary.update(retry_cpu_seconds=account['spent_seconds'],
                       earlier_completed_runs=plan['earlier_completed_runs'])
    print(json.dumps(summary, sort_keys=True), flush=True)
    if seal:
        if (summary['stopped'] or (root == ROOT and gate is None) or account['reserved_seconds']
                or sum(states.values()) != len(plan['runs'])
                or summary['total_charged_cpu_seconds'] > LIMIT_HOURS * 3600):
            raise ValueError('full grid is not complete under its authorization')
        for run in plan['runs']:
            checked_result(run, plan['configs'][run['task']], root)
            if previous := run.get('previous'):
                proof = read(root / run['relative_directory'] / 'retry-prefix-identity.json')
                if (proof['differences'] or proof['previous_sha256'] != previous['rows_sha256']
                        or proof['rows'] != previous['checkpoints']):
                    raise ValueError('retry prefix differs from its earlier attempt')
        (root / 'accounting.psv').write_text(raw)
        atomic_json(root / 'accounting-summary.json', summary)
        atomic_json(root / 'allocation-accounting.json', account['attempts'])
        files = {str(path.relative_to(root)): evidence_sha(path)
                 for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'files.sha256.json'}
        atomic_json(root / 'files.sha256.json', files)
        print(checksum(root / 'files.sha256.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('command', choices=('prepare', 'submit', 'worker', 'failed', 'status', 'seal'))
    parser.add_argument('parameter', nargs='?')
    args = parser.parse_args()
    node_command = args.command in ('worker', 'failed')
    root = Path(args.parameter).parent if node_command else RETRY if args.retry else ROOT
    if root not in (ROOT, RETRY):
        raise ValueError('grid command addresses an unknown run directory')
    try:
        if args.command == 'prepare':
            prepare(args.parameter, root)
        elif args.command == 'submit':
            submit(root)
        elif args.command == 'worker':
            worker(Path(args.parameter), root)
        elif args.command == 'failed':
            raise RuntimeError('grid allocation terminated or launcher failed')
        else:
            status(seal=args.command == 'seal', root=root)
    except BaseException as error:
        if args.command in ('submit', 'worker', 'failed') and not (root / 'STOP.json').exists():
            stop_phase(root / 'STOP.json', dict(error=str(error), command=args.command, assignment=args.parameter,
                                               job=os.environ.get('SLURM_JOB_ID')))
        raise
