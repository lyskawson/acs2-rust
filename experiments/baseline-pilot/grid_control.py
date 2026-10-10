import argparse
import fcntl
import json
import hashlib
import re
from pathlib import Path
import subprocess
import time

from collect import complete_rows, outside_repository
from grid_protocol import ORDER, PLAN_SHA256, TERMINAL, accounting, allocation_rows, atomic_json, load_plan, require_budget
from grid_attempts import attempts, inspect_attempt, select_attempts

SACCT_FIELDS = 'JobID,JobIDRaw,CPUTimeRAW,TotalCPU,AllocCPUS,ElapsedRaw,MaxRSS,State,ExitCode,NodeList,Partition,Start,End,TimelimitRaw'


def execute(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def snapshot(root, ledger):
    unresolved = [entry for entry in ledger if not entry.get('array_id')]
    if unresolved:
        raise ValueError('unresolved submission intent; inspect ledger and reconcile without resubmitting')
    identifiers = [str(entry['array_id']) for entry in ledger]
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    directory = root / 'accounting'
    directory.mkdir(exist_ok=True)
    if identifiers:
        raw = execute(['sacct', '--array', '-S', '2026-09-30', '-j', ','.join(identifiers), '-P', '--format=' + SACCT_FIELDS])
        queue = execute(['squeue', '-j', ','.join(identifiers), '-o', '%.24i %.14T %.15M %.16l %.40R'])
    else:
        raw, queue = SACCT_FIELDS.replace(',', '|') + '|\n', ''
    (directory / f'sacct-{stamp}.psv').write_text(raw)
    (directory / 'sacct-latest.psv').write_text(raw)
    (directory / f'squeue-{stamp}.txt').write_text(queue)
    records = allocation_rows(raw)
    known = {row['JobID']: row for row in records}
    for entry in ledger:
        for index, run in enumerate(entry['runs']):
            row = known.get(f"{entry['array_id']}_{index}")
            if row and row['State'].split()[0].split('+')[0] in TERMINAL:
                run['terminal_accounting'] = row
    atomic_json(root / 'ledger.json', ledger)
    state = accounting(ledger, records)
    atomic_json(directory / 'balance.json', state)
    return state, records, queue


def monitor(root, plan, ledger):
    state, records, queue = snapshot(root, ledger)
    configs = {item['id']: item for item in plan['configurations']}
    observed, failures, historical = [], [], []
    known = {row['JobID']: row for row in records}
    for attempt in attempts(root, ledger):
        job = f"{attempt['array_id']}_{attempt['task_id']}"
        row = known.get(job, attempt.get('terminal_accounting', {}))
        status = row.get('State', '').split('+')[0].split(' ')[0]
        try:
            item, rows = inspect_attempt(attempt, configs[attempt['configuration_id']])
            for value in rows:
                if value['commit'] != plan['measurement_commit'] or value['source_state'] != 'clean' or value['cpu_model'] != plan['cpu_model']:
                    raise ValueError('row provenance mismatch')
            if rows:
                item['first_row'] = rows[0]
            observed.append(item)
            bad = status in TERMINAL and (status != 'COMPLETED' or item['status'] != 'complete')
            bad = bad or bool(item.get('completion', {}).get('failure'))
            if bad:
                audit = attempt.get('accounted_failure')
                raw_path = attempt['path'] / 'rows.jsonl'
                preserved = not raw_path.exists()
                if audit and audit.get('rows_sha256'):
                    preserved = raw_path.exists() and audit['rows_sha256'] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
                if audit and audit['state'] == status and audit['cpu_seconds'] == float(row['CPUTimeRAW']) and preserved:
                    historical.append(dict(attempt_id=attempt['attempt_id'], job=job, audit=audit))
                else:
                    failures.append(dict(attempt_id=attempt['attempt_id'], job=job, state=status, status=item['status']))
        except (ValueError, KeyError, OSError) as error:
            failures.append(dict(attempt_id=attempt['attempt_id'], job=job, error=str(error)))
    report = dict(unix=time.time(), attempts=observed, failures=failures, accounted_historical_failures=historical,
                  balance=state, queue=queue, stop=(root / 'STOP.json').exists())
    atomic_json(root / 'monitor-latest.json', report)
    if not failures and not report['stop']:
        atomic_json(root / 'submission-check.json', {'unix': time.time(), 'healthy': True})
    print(json.dumps(dict(failures=failures, historical_failures=len(historical), complete=sum(item['status'] == 'complete' for item in observed),
                          stop=report['stop'], balance=state, queue=queue)), flush=True)
    return report


def chain_plan(root, plan, ledger):
    batches = []
    for config_id in ORDER:
        config = next(item for item in plan['configurations'] if item['id'] == config_id)
        selected, inventory = select_attempts(root, ledger, config)
        groups = []
        for agent in plan['agents']:
            missing = [seed for seed in plan['seeds'] if (agent, seed) not in selected]
            for lane in (0, 1):
                runs = []
                for seed in missing[lane::2]:
                    previous = [item for item in inventory if item['agent'] == agent and item['seed'] == seed]
                    number = len(previous) + 1
                    attempt_id = f'{config_id}_{agent}_s{seed}_a{number:02d}'
                    name = f'{config_id}_{agent}_s{seed}'
                    runs.append(dict(configuration_id=config_id, agent=agent, seed=seed,
                                     attempt_id=attempt_id, attempt_number=number,
                                     cause='recovery after: ' + '; '.join(item['cause'] for item in previous) if previous else 'registered first attempt',
                                     previous_attempts=[item['attempt_id'] for item in previous],
                                     relative_directory=f'attempts/{attempt_id}/measurement/{name}',
                                     limit_seconds=config['limits_minutes'][agent] * 60))
                if runs:
                    groups.append(dict(agent=agent, lane=lane, runs=runs))
        if groups:
            batches.append(dict(configuration_id=config_id, groups=groups))
    return batches


def submission_command(operations, root, assignment, logdir, config, group, dependencies):
    command = ['sbatch', '--parsable', '--hold', '--partition=lem-cpu-normal', '--cpus-per-task=1', '--mem=512M',
               '--time=' + str(config['limits_minutes'][group['agent']]),
               '--array=0-' + str(len(group['runs']) - 1) + '%1', '--no-requeue',
               '--job-name=tu3b-' + config['task'] + '-' + group['agent'],
               '--output=' + str(logdir / '%A_%a.out'), '--error=' + str(logdir / '%A_%a.err')]
    if dependencies:
        command.append('--dependency=afterok:' + ':'.join(map(str, dependencies)))
    return command + [str(operations / 'grid_job.sh'), str(operations), str(root), str(assignment)]


def submit_chain(root, plan, ledger, operations, repo, chain_id=None):
    if chain_id is not None and not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', chain_id):
        raise ValueError('invalid chain identifier')
    chain_directory = root / 'chains' / chain_id if chain_id is not None else root
    chain_path = chain_directory / 'chain.json'
    assignment_directory = chain_directory / 'assignments' if chain_id is not None else root / 'chain-assignments'
    report = monitor(root, plan, ledger)
    if report['stop'] or report['failures'] or report['balance']['reserved_seconds']:
        raise ValueError('STOP, unexplained failures or outstanding historical reservations')
    if chain_path.exists():
        raise ValueError('chain already planned; reconcile it without resubmitting')
    approval = json.loads((root / 'approval.json').read_text())
    rules = json.loads((root / 'execution-rules-v2.json').read_text())
    if approval['notes_commit'] != '2a28372528907325303e30ecdccf49cbb829b7a3' or approval['plan_sha256'] != PLAN_SHA256:
        raise ValueError('approval provenance differs')
    if rules['plan_sha256'] != PLAN_SHA256 or rules['supervision_lease'] or rules['host_pinning'] or not rules['chain_authorized']:
        raise ValueError('execution rule approval differs')
    batches = chain_plan(root, plan, ledger)
    remaining = sum(run['limit_seconds'] for batch in batches for group in batch['groups'] for run in group['runs'])
    bound = require_budget(plan['pilot_allocation_cpu_hours'] * 3600, report['balance'], remaining)
    chain_directory.mkdir(parents=True, exist_ok=True)
    chain = dict(chain_id=chain_id, plan_sha256=PLAN_SHA256, new_limit_seconds=remaining, initial_balance=report['balance'],
                 pilot_seconds=plan['pilot_allocation_cpu_hours'] * 3600, bound_seconds=bound,
                 batches=batches, released=False, started_unix=time.time())
    atomic_json(chain_path, chain)
    operations_commit = json.loads((root / 'operations.json').read_text())['commit']
    dependencies = []
    for batch in batches:
        config_id = batch['configuration_id']
        config = next(item for item in plan['configurations'] if item['id'] == config_id)
        directory = assignment_directory / config_id
        (directory / 'logs').mkdir(parents=True, exist_ok=False)
        current = []
        for group in batch['groups']:
            if (root / 'STOP.json').exists():
                raise ValueError('phase stopped during submission; all new arrays remain held')
            state, records, queue = snapshot(root, ledger)
            bound = require_budget(plan['pilot_allocation_cpu_hours'] * 3600, state, remaining)
            grant = float(execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n']).strip())
            if grant + state['reserved_seconds'] + remaining > 15000 * 3600:
                raise ValueError('shared grant cannot cover all remaining caps')
            for run in group['runs']:
                attempt_dir = (root / run['relative_directory']).parent.parent
                attempt_dir.mkdir(parents=True, exist_ok=False)
                atomic_json(attempt_dir / 'attempt.json', run)
            assignment = directory / f"{group['agent']}-{group['lane']}.json"
            data = dict(configuration_id=config_id, runs=group['runs'], plan_sha256=PLAN_SHA256,
                        operations_commit=operations_commit, measurement_repository=str(repo),
                        binary=str(repo / 'target/x86_64-unknown-linux-musl/release/acs2-measure'))
            atomic_json(assignment, data)
            command = submission_command(operations, root, assignment, directory / 'logs', config, group, dependencies)
            test = execute(command[:1] + ['--test-only'] + command[1:])
            entry = dict(data, array_id=None, command=command, submitted_unix=time.time(),
                         grant_raw_usage=grant, test_only=test, phase_bound_hours=bound / 3600, dependencies=dependencies.copy())
            ledger.append(entry)
            atomic_json(root / 'ledger.json', ledger)
            response = execute(command).strip()
            entry['submission_response'] = response
            entry['array_id'] = int(response.split(';')[0].splitlines()[-1])
            atomic_json(root / 'ledger.json', ledger)
            group['array_id'] = entry['array_id']
            current.append(entry['array_id'])
            remaining -= sum(run['limit_seconds'] for run in group['runs'])
            atomic_json(chain_path, chain)
            print(json.dumps(dict(configuration=config_id, array_id=entry['array_id'], runs=len(group['runs']), dependencies=dependencies, bound_hours=bound / 3600)), flush=True)
        dependencies = current
    state, records, queue = snapshot(root, ledger)
    require_budget(plan['pilot_allocation_cpu_hours'] * 3600, state, 0)
    if (root / 'STOP.json').exists():
        raise ValueError('phase stopped; new arrays remain held')
    chain['release_started_unix'] = time.time()
    atomic_json(chain_path, chain)
    for batch in batches:
        for group in batch['groups']:
            execute(['scontrol', 'release', str(group['array_id'])])
    chain.update(released=True, released_unix=time.time())
    atomic_json(chain_path, chain)


def seal(root, plan, ledger, config_id):
    config = next(item for item in plan['configurations'] if item['id'] == config_id)
    selected, inventory = select_attempts(root, ledger, config)
    if len(selected) != 40:
        raise ValueError('configuration does not yet have 40 complete runs')
    state, records, queue = snapshot(root, ledger)
    known = {row['JobID']: row for row in records}
    sources = []
    for attempt in attempts(root, ledger, config_id):
        row = known.get(f"{attempt['array_id']}_{attempt['task_id']}", attempt.get('terminal_accounting', {}))
        if row.get('State', '').split()[0].split('+')[0] not in TERMINAL:
            raise ValueError('attempt still active; cannot seal')
        sources.extend(path for path in attempt['path'].rglob('*') if path.is_file())
    for directory in [root / 'chain-assignments' / config_id, *list((root / 'batches').glob(f'*-{config_id}')), *list((root / 'chains').glob(f'*/assignments/{config_id}'))]:
        sources.extend(path for path in directory.rglob('*') if path.is_file())
    for attempt in attempts(root, ledger, config_id):
        if 'relative_directory' in attempt and attempt['path'].parts[-2] == 'measurement':
            sources.extend(path for path in attempt['path'].parent.parent.rglob('*') if path.is_file())
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (root / 'integrity').mkdir(exist_ok=True)
    atomic_json(root / 'integrity' / f'{config_id}-chain.json', dict(configuration_id=config_id, files=hashes, attempts=inventory, unix=time.time()))
    print(json.dumps(dict(configuration=config_id, files=len(hashes))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['monitor', 'submit-chain', 'seal'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--configuration')
    parser.add_argument('--chain-id')
    parser.add_argument('--repository', type=Path, default=Path.home() / 'acs2-tu')
    args = parser.parse_args()
    root = outside_repository(args.root)
    with (root / '.control.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = load_plan(root / 'approved-plan.json')
        ledger = json.loads((root / 'ledger.json').read_text())
        if args.action == 'monitor':
            monitor(root, plan, ledger)
        elif args.action == 'seal':
            seal(root, plan, ledger, args.configuration)
        else:
            submit_chain(root, plan, ledger, Path(__file__).resolve().parent, args.repository, args.chain_id)


if __name__ == '__main__':
    main()
