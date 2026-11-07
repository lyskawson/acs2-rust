import argparse
import hashlib
import json
from pathlib import Path
import time

from protocol import ROOT, NEW_ARMS
from execution import require_authorization, registered_runs
from collect import outside_repository
from grid_protocol import atomic_json, require_budget
from grid_scope import require_batch_root, snapshot, stop_path, verify_operations
from submit import execute, shared_reservation, submission_command


def submit_pilot(root, manifest, repository, executor=execute):
    root, manifest, repository = outside_repository(root), Path(manifest), Path(repository).resolve()
    plan = json.loads(manifest.read_text())
    require_authorization(root, plan)
    require_batch_root(root, plan)
    verify_operations(plan)
    if root.exists() or stop_path(root, plan).exists():
        raise ValueError('pilot already exists or is stopped; reconcile every attempt before a retry')
    if executor(['/usr/bin/git', '-C', str(repository), 'rev-parse', 'HEAD']).strip() != plan['measurement_commit']:
        raise ValueError('pilot measurement commit differs')
    if executor(['/usr/bin/git', '-C', str(repository), 'status', '--porcelain']).strip():
        raise ValueError('pilot measurement clone must be clean')
    binary = repository / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != plan['binary_sha256']:
        raise ValueError('pilot static binary changed')
    oracle = json.loads((repository / 'fixtures/goal_measurement_regression.json').read_text())
    if plan['preset'] != oracle['configurations'][0]['rows'][0]['preset'] or plan['preset']['do_ga'] is not False:
        raise ValueError('pilot preset differs or GA is enabled')
    operations_identity = json.loads(Path(plan['operations_manifest']).read_text())
    if operations_identity['commit'] != plan['operations_commit']:
        raise ValueError('operations commit differs from the frozen snapshot')
    runs = registered_runs(plan)
    if len(runs) != 12 or len({run['limit_seconds'] for run in runs}) != 1:
        raise ValueError('pilot requires twelve equal-limit allocations')
    new_seconds = sum(run['limit_seconds'] for run in runs)
    state = snapshot(plan, executor)
    bound = require_budget(plan['pilot_allocation_cpu_seconds'], state, new_seconds, plan['phase_cpu_hour_limit'])
    grant = executor(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n'])
    queue = executor(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
    if float(grant.strip()) + shared_reservation(queue) + new_seconds > 15000 * 3600:
        raise ValueError('shared grant cannot cover queued reservations and this pilot')
    assignment = root / 'assignment-00.json'
    command = submission_command(root, assignment, runs[0]['limit_seconds'] // 60, len(runs),
        plan['concurrency'], plan['memory_mib'], [], 'tu-per-goal-reward')
    operations = Path(plan['operations_manifest']).parent / 'experiments/per-goal-reward'
    command[-4:] = [str(operations / 'job.sh'), str(operations), str(root), str(assignment)]
    tested = executor(command[:1] + ['--test-only'] + command[1:])
    require_authorization(root, plan)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    atomic_json(root / 'plan.json', plan)
    digest = hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest()
    (root / 'plan.sha256').write_text(digest + '\n')
    for run in runs:
        (root / run['relative_directory']).parent.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(assignment, dict(measurement_repository=str(repository), binary=str(binary),
        operations_commit=plan['operations_commit'], plan_sha256=digest, runs=runs))
    entry = dict(configuration_id=None, runs=runs, command=command, grant=grant, queue=queue,
        test_only=tested, balance=state, bound_seconds=bound, submitted_unix=time.time(), array_id=None)
    atomic_json(root / 'ledger.json', [entry])
    event = dict(root=str(root), assignment=0, new_limit_seconds=new_seconds, remaining_limit_seconds=new_seconds,
        balance=state, bound_seconds=bound, grant=grant, queue=queue, test_only=tested, unix=time.time())
    with (Path(plan['budget_scope']).parent / 'admission-decisions.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    response = executor(command).strip()
    entry.update(array_id=int(response.split(';')[0]), submission_response=response)
    atomic_json(root / 'ledger.json', [entry])
    return dict(array_id=entry['array_id'], runs=12, bound_cpu_hours=bound / 3600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()
    print(json.dumps(submit_pilot(args.root, args.manifest, args.repo)))
