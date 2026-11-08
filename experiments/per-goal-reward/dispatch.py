import argparse
import json
from pathlib import Path
import time

from execution_grid import (APPROVED_SHA256, approved, batch_name, checksum, manifest_for,
                            registered_runs, require_authorization)
from grid_protocol import atomic_json, require_budget, load_plan
from grid_scope import snapshot, stop_path, verify_operations, require_batch_root
from grid_attempts import attempts, inspect_attempt
from schema import validate_sequence
from submit import execute, shared_reservation, submission_command


def manifest(root, index):
    root = Path(root)
    plan, _ = approved(root / 'approved-plan.json')
    operations = root / 'execution-operations/manifest.json'
    return dict(manifest_for(plan, index), authorization_plan=str(root / 'approved-plan.json'),
        authorization_sha256=APPROVED_SHA256, budget_scope=str(root / 'continuation/scope.json'),
        budget_scope_sha256=checksum(root / 'continuation/scope.json'), stop_path=str(root / 'continuation/STOP.json'),
        operations_commit=json.loads(operations.read_text())['commit'], operations_manifest=str(operations),
        operations_manifest_sha256=checksum(operations), job_name='tu-per-goal-reward')


def completed(root, definition):
    root = Path(root)
    if not root.exists():
        return False
    ledger = json.loads((root / 'ledger.json').read_text())
    found = set()
    for entry in attempts(root, ledger):
        config = next(config for config in definition['configurations'] if config['id'] == entry['configuration_id'])
        status, rows = inspect_attempt(entry, config)
        if status['status'] != 'complete':
            return False
        validate_sequence(rows, config, entry['agent'], entry['seed'], definition['measurement_commit'])
        found.add((entry['configuration_id'], entry['agent'], entry['seed']))
    return found == {(config['id'], agent, seed) for config in definition['configurations']
                     for agent in definition['agents'] for seed in definition['seeds']}


def submit(root, repository, index, executor=execute):
    root, repository = Path(root), Path(repository)
    registered, _ = approved(root / 'approved-plan.json')
    definition = manifest(root, index)
    batch = root / batch_name(registered['forecast']['arrays'][index])
    require_authorization(batch, definition)
    require_batch_root(batch, definition)
    verify_operations(definition)
    if stop_path(batch, definition).exists():
        raise ValueError('shared STOP prevents further submissions')
    if batch.exists():
        raise ValueError('batch already exists; preserve and reconcile every prior attempt')
    binary = repository / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
    if (executor(['/usr/bin/git', '-C', str(repository), 'rev-parse', 'HEAD']).strip() != definition['measurement_commit']
            or executor(['/usr/bin/git', '-C', str(repository), 'status', '--porcelain']).strip()
            or checksum(binary) != definition['binary_sha256']):
        raise ValueError('pinned measurement source or binary changed; rebuilding is forbidden')
    state = snapshot(definition, executor)
    active = [name for name, value in state['batch_states'].items() if value['reserved_seconds'] > 0]
    if len(active) >= 2:
        raise ValueError('two active streams already reserve forty running slots')
    previous = [i for i, array in enumerate(registered['forecast']['arrays'][:index])
                if array['agent'] == definition['agents'][0]]
    for earlier in previous:
        earlier_root = root / batch_name(registered['forecast']['arrays'][earlier])
        if (state['batch_states'].get(str(earlier_root), {}).get('reserved_seconds', 1)
                or not completed(earlier_root, manifest(root, earlier))):
            raise ValueError('previous array of this arm must finish and validate first')
    runs = registered_runs(definition)
    if not 20 <= len(runs) <= 40 or len({run['limit_seconds'] for run in runs}) != 1:
        raise ValueError('registered arrays require twenty to forty equal-limit jobs')
    new = sum(run['limit_seconds'] for run in runs)
    bound = require_budget(0, state, new, 29)
    grant = executor(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n'])
    queue = executor(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
    if float(grant.strip()) + shared_reservation(queue) + new > 15000 * 3600:
        raise ValueError('shared grant cannot cover queued and new reservations')
    assignment = batch / 'assignment-00.json'
    command = submission_command(batch, assignment, runs[0]['limit_seconds'] // 60, len(runs), 20,
        definition['memory_mib'], [], 'tu-per-goal-reward')
    operations = Path(definition['operations_manifest']).parent / 'experiments/per-goal-reward'
    command[-4:] = [str(operations / 'job_grid.sh'), str(operations), str(batch), str(assignment)]
    tested = executor(command[:1] + ['--test-only'] + command[1:])
    require_authorization(batch, definition)
    batch.mkdir(parents=True, exist_ok=False)
    (batch / 'logs').mkdir()
    atomic_json(batch / 'plan.json', definition)
    digest = checksum(batch / 'plan.json')
    (batch / 'plan.sha256').write_text(digest + '\n')
    (batch / 'approval.json').write_bytes((root / 'approval.json').read_bytes())
    for run in runs:
        (batch / run['relative_directory']).parent.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(assignment, dict(measurement_repository=str(repository), binary=str(binary),
        operations_commit=definition['operations_commit'], plan_sha256=digest, runs=runs))
    event = dict(root=str(batch), forecast_index=index, new_limit_seconds=new, balance=state,
        bound_seconds=bound, grant=grant, queue=queue, test_only=tested, unix=time.time())
    entry = dict(configuration_id=None, runs=runs, command=command, grant=grant, queue=queue,
        test_only=tested, balance=state, bound_seconds=bound, submitted_unix=time.time(), array_id=None)
    atomic_json(batch / 'ledger.json', [entry])
    with (root / 'continuation/admission-decisions.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    response = executor(command).strip()
    entry.update(array_id=int(response.split(';')[0]), submission_response=response)
    atomic_json(batch / 'ledger.json', [entry])
    return dict(array_id=entry['array_id'], forecast_index=index, runs=len(runs), bound_cpu_hours=bound / 3600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--array', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.root, args.repo, args.array)))
