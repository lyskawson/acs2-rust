import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from collect import complete_rows, outside_repository
from grid_protocol import ORDER, PLAN_SHA256, accounting, allocation_rows, atomic_json, load_plan, require_budget

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
        raw = execute(['sacct', '-S', '2026-09-30', '-j', ','.join(identifiers), '-P', '--format=' + SACCT_FIELDS])
        queue = execute(['squeue', '-j', ','.join(identifiers), '-o', '%.24i %.14T %.15M %.16l %.40R'])
    else:
        raw, queue = SACCT_FIELDS.replace(',', '|') + '|\n', ''
    (directory / f'sacct-{stamp}.psv').write_text(raw)
    (directory / 'sacct-latest.psv').write_text(raw)
    (directory / f'squeue-{stamp}.txt').write_text(queue)
    records = allocation_rows(raw)
    state = accounting(ledger, records)
    atomic_json(directory / 'balance.json', state)
    return state, records, queue


def monitor(root, plan, ledger):
    state, records, queue = snapshot(root, ledger)
    observed = []
    incomplete = []
    observations_path = root / 'first-observations.json'
    first_observations = json.loads(observations_path.read_text()) if observations_path.exists() else {}
    for path in sorted((root / 'batches').glob('*/runs/*/rows.jsonl')):
        rows, tail = complete_rows(path)
        if rows:
            first = rows[0]
            if first['commit'] != plan['measurement_commit'] or first['source_state'] != 'clean' or first['cpu_model'] != plan['cpu_model']:
                raise ValueError(f'first row provenance mismatch: {path}')
            observed.append({'path': str(path.relative_to(root)), 'rows': len(rows), 'first_row': first})
            name = str(path.relative_to(root))
            if name not in first_observations:
                launch = json.loads(path.with_name('launch.json').read_text())
                first_observations[name] = {'observed_unix': time.time(), 'seconds_since_launch': time.time() - launch['started_unix'], 'first_row': first}
        if tail:
            incomplete.append(str(path.relative_to(root)))
    failures = []
    for path in sorted((root / 'batches').glob('*/runs/*/completion.json')):
        result = json.loads(path.read_text())
        if result['failure'] or result['exit_code'] or result['incomplete_final_line'] or result['rows_seen'] != result['rows_expected']:
            failures.append(str(path.relative_to(root)))
    for row in records:
        if row['State'].split()[0].split('+')[0] in {'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL'}:
            failures.append(row['JobID'])
    report = {'unix': time.time(), 'observed': observed, 'incomplete_files': incomplete, 'failures': failures,
              'balance': state, 'queue': queue, 'stop': (root / 'STOP.json').exists()}
    atomic_json(observations_path, first_observations)
    atomic_json(root / 'monitor-latest.json', report)
    if failures or report['stop']:
        if failures and not report['stop']:
            atomic_json(root / 'STOP.json', {'unix': time.time(), 'failures': failures})
        print(json.dumps({'failures': failures, 'stop': report['stop'], 'balance': state, 'queue': queue}))
        return
    atomic_json(root / 'supervision.json', {'unix': time.time(), 'observer': 'external_grid_control', 'observed_runs': len(observed)})
    print(json.dumps({'observed_runs': len(observed), 'incomplete_files': incomplete, 'balance': state, 'queue': queue}))


def submit(root, plan, ledger, config_id, operations, repo):
    if (root / 'STOP.json').exists():
        raise ValueError('phase stopped')
    watch = json.loads((root / 'supervision.json').read_text())
    if time.time() - watch['unix'] > 300:
        raise ValueError('fresh external supervision required before submitting')
    existing = list(dict.fromkeys(entry['configuration_id'] for entry in ledger))
    if existing != ORDER[:len(existing)] or config_id != ORDER[len(existing)]:
        raise ValueError('batch order or duplicate submission violates approval')
    for previous in existing:
        result = json.loads((root / 'verified' / f'{previous}.json').read_text())
        if result['plan_sha256'] != PLAN_SHA256 or result['runs'] != 40 or result['pilot_identity']['matched_runs'] != 2:
            raise ValueError('previous batch has not passed the complete local analysis')
    config = next(item for item in plan['configurations'] if item['id'] == config_id)
    batch = root / 'batches' / f'{len(existing) + 1:02d}-{config_id}'
    batch.mkdir(parents=True, exist_ok=False)
    (batch / 'logs').mkdir()
    approval = json.loads((root / 'approval.json').read_text())
    if approval['notes_commit'] != '2a28372528907325303e30ecdccf49cbb829b7a3' or approval['plan_sha256'] != PLAN_SHA256:
        raise ValueError('approval provenance differs')
    state, records, queue = snapshot(root, ledger)
    if state['reserved_seconds']:
        raise ValueError('previous allocations are not yet terminal')
    operations_commit = json.loads((root / 'operations.json').read_text())['commit']
    for agent in plan['agents']:
        for parity in (0, 1):
            state, records, queue = snapshot(root, ledger)
            host = plan['even_seed_host' if parity == 0 else 'odd_seed_host']
            runs = [{'configuration_id': config_id, 'agent': agent, 'seed': seed, 'host': host,
                     'limit_seconds': config['limits_minutes'][agent] * 60}
                    for seed in plan['seeds'] if seed % 2 == parity]
            new_seconds = sum(run['limit_seconds'] for run in runs)
            bound = require_budget(plan['pilot_allocation_cpu_hours'] * 3600, state, new_seconds)
            grant_text = execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n'])
            grant = float(grant_text.strip())
            if grant + state['reserved_seconds'] + new_seconds > 15000 * 3600:
                raise ValueError('shared grant cannot cover new allocation limits')
            assignment = batch / f'{agent}-{parity}.json'
            data = {'configuration_id': config_id, 'runs': runs, 'plan_sha256': PLAN_SHA256,
                    'operations_commit': operations_commit, 'measurement_repository': str(repo),
                    'binary': str(repo / 'target/x86_64-unknown-linux-musl/release/acs2-measure'),
                    'output': str(batch / 'runs')}
            atomic_json(assignment, data)
            command = ['sbatch', '--parsable', '--partition=lem-cpu-normal', '--cpus-per-task=1', '--mem=512M',
                       '--time=' + str(config['limits_minutes'][agent]), '--array=0-9%1', '--no-requeue',
                       '--nodelist=' + host, '--job-name=tu3b-' + config['task'] + '-' + agent,
                       '--output=' + str(batch / 'logs/%A_%a.out'), '--error=' + str(batch / 'logs/%A_%a.err'),
                       str(operations / 'grid_job.sh'), str(operations), str(root), str(assignment)]
            test = execute(command[:1] + ['--test-only'] + command[1:])
            entry = dict(data, array_id=None, command=command, submitted_unix=time.time(),
                         grant_raw_usage=grant, test_only=test, phase_bound_hours=bound / 3600)
            ledger.append(entry)
            atomic_json(root / 'ledger.json', ledger)
            response = execute(command).strip()
            entry['submission_response'] = response
            entry['array_id'] = int(response.split(';')[0].splitlines()[-1])
            atomic_json(root / 'ledger.json', ledger)
            print(json.dumps({'array_id': entry['array_id'], 'runs': len(runs), 'agent': agent, 'host': host, 'bound_hours': bound / 3600}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['monitor', 'submit'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--configuration')
    parser.add_argument('--repository', type=Path, default=Path.home() / 'acs2-tu')
    args = parser.parse_args()
    root = outside_repository(args.root)
    with (root / '.control.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = load_plan(root / 'approved-plan.json')
        ledger = json.loads((root / 'ledger.json').read_text())
        if args.action == 'monitor':
            monitor(root, plan, ledger)
        else:
            submit(root, plan, ledger, args.configuration, Path(__file__).resolve().parent, args.repository)


if __name__ == '__main__':
    main()
