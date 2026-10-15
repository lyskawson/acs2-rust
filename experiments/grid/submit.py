import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from collect import outside_repository
from grid_authorization import require_authorization
from grid_protocol import accounting, allocation_rows, atomic_json, require_budget, seconds
from grid_scope import SACCT_FIELDS, snapshot, stop_path, verify_operations


def execute(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def shared_reservation(queue):
    return sum(seconds(line.split('|')[3]) * int(line.split('|')[4])
               for line in queue.splitlines() if line.strip())


def submission_command(root, assignment, limit_minutes, count, concurrency, memory_mib, dependencies, job_name='tu-maze-baselines'):
    operations = Path(__file__).resolve().parent
    command = ['sbatch', '--parsable', '--partition=lem-cpu-normal', '--cpus-per-task=1',
               f'--mem={memory_mib}M', f'--time={limit_minutes}', f'--array=0-{count - 1}%{concurrency}',
               '--no-requeue', '--job-name=' + job_name,
               '--output=' + str(root / 'logs/%A_%a.out'), '--error=' + str(root / 'logs/%A_%a.err')]
    if dependencies:
        command.append('--dependency=afterok:' + ':'.join(map(str, dependencies)))
    return command + [str(operations / 'grid_job.sh'), str(operations), str(root), str(assignment)]


def prepare(root, manifest, repo, approval=None):
    root = outside_repository(root)
    commit = execute(['git', '-C', str(repo), 'rev-parse', 'HEAD']).strip()
    if execute(['git', '-C', str(repo), 'status', '--porcelain']).strip():
        raise ValueError('measurement clone must be clean')
    binary = repo / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
    plan = json.loads(manifest.read_text())
    verify_operations(plan)
    oracle = json.loads((repo / 'fixtures/goal_measurement_regression.json').read_text())
    identities = dict(measurement_commit=commit, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      preset=oracle['configurations'][0]['rows'][0]['preset'])
    for key, value in identities.items():
        if key in plan and plan[key] != value:
            raise ValueError(f'registered {key} differs from the pinned clone')
    plan.update(identities)
    digest = hashlib.sha256((json.dumps(plan, indent=2) + '\n').encode()).hexdigest()
    require_authorization(root, plan, digest, approval)
    maximum = 60 if plan.get('experiment') == 'her-variants' else 40
    if plan.get('experiment') == 'value-consistency':
        maximum = 5 if plan['stage'] == 'authorized_cost_pilot' else plan['phase_cpu_hour_limit']
    if plan['partition'] != 'lem-cpu-normal' or not 20 <= plan['concurrency'] <= 40 or not 0 < plan['phase_cpu_hour_limit'] <= maximum:
        raise ValueError('wrong partition, concurrency or CPU-hour limit')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    atomic_json(root / 'plan.json', plan)
    if approval is not None:
        (root / 'approval.json').write_bytes(Path(approval).read_bytes())
    (root / 'plan.sha256').write_text(digest + '\n')
    atomic_json(root / 'ledger.json', [])
    return plan, digest, binary


def balance(plan, ledger, executor=execute):
    if 'budget_scope' in plan:
        return snapshot(plan, executor)
    raw = executor(['sacct', '--array', '-S', plan.get('accounting_since', '2026-10-10'), '-j',
                   ','.join(str(entry['array_id']) for entry in ledger), '-P', '--format=' + SACCT_FIELDS]) if ledger else SACCT_FIELDS.replace(',', '|') + '|\n'
    return accounting(ledger, allocation_rows(raw))


def run_batches(plan):
    if plan['stage'] == 'authorized_cost_pilot':
        return [[(config, agent, seed) for config in plan['configurations']
                 for agent in plan['agents'] for seed in plan['seeds']]]
    if plan.get('batch_by') == 'group':
        groups = {}
        for config in plan['configurations']:
            groups.setdefault(config['batch_group'], []).extend((config, agent, seed) for agent in plan['agents'] for seed in plan['seeds'])
        if any(not 20 <= len(group) <= 40 for group in groups.values()):
            raise ValueError('each batch group must contain 20--40 jobs')
        return list(groups.values())
    if 'batch_size' in plan:
        size = plan['batch_size']
        if not 20 <= size <= 40:
            raise ValueError('batch size must be within 20--40 jobs')
        runs = [(config, agent, seed) for config in plan['configurations'] for agent in plan['agents'] for seed in plan['seeds']]
        return [runs[index:index + size] for index in range(0, len(runs), size)]
    return [[(config, agent, seed) for agent in plan['agents'] for seed in plan['seeds']]
            for config in plan['configurations']]


def submit(root, manifest, repo, approval=None):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    if root.exists():
        raise ValueError('batch already exists; reconcile instead of resubmitting')
    preview = json.loads(Path(manifest).read_text())
    require_authorization(root, preview, hashlib.sha256(Path(manifest).read_bytes()).hexdigest(), approval)
    preview_batches = run_batches(preview)
    preview_limits = sum(max(config['limits_minutes'][agent] for config, agent, _ in batch) * 60 * len(batch)
                         for batch in preview_batches)
    require_budget(preview['pilot_allocation_cpu_seconds'], balance(preview, [], execute), preview_limits,
                   preview['phase_cpu_hour_limit'])
    plan, digest, binary = prepare(root, Path(manifest), repo, approval)
    batches = run_batches(plan)
    remaining = sum(max(config['limits_minutes'][agent] for config, agent, _ in batch) * 60 * len(batch)
                    for batch in batches)
    require_budget(plan['pilot_allocation_cpu_seconds'], balance(plan, [], execute), remaining,
                   plan['phase_cpu_hour_limit'])
    ledger, dependencies = [], list(plan.get('initial_dependencies', []))
    for number, batch in enumerate(batches):
        if stop_path(root, plan).exists():
            raise ValueError('node failure stopped the experiment')
        limit = max(config['limits_minutes'][agent] for config, agent, _ in batch)
        runs = []
        for config, agent, seed in batch:
            identifier = f"{config['id']}_{agent}_s{seed}_a01"
            directory = f"attempts/{identifier}/measurement/{config['id']}_{agent}_s{seed}"
            (root / directory).parent.parent.mkdir(parents=True, exist_ok=False)
            runs.append(dict(configuration_id=config['id'], agent=agent, seed=seed, attempt_id=identifier,
                             cause='registered first attempt', relative_directory=directory, limit_seconds=limit * 60))
        assignment = root / f'assignment-{number:02d}.json'
        atomic_json(assignment, dict(measurement_repository=str(repo), binary=str(binary), operations_commit=plan.get('operations_commit', plan['measurement_commit']),
                                    plan_sha256=digest, runs=runs))
        state = balance(plan, ledger, execute)
        bound = require_budget(plan['pilot_allocation_cpu_seconds'], state, remaining, plan['phase_cpu_hour_limit'])
        grant_text = execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n'])
        queue = execute(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
        if float(grant_text.strip()) + shared_reservation(queue) + remaining > 15_000 * 3600:
            raise ValueError('shared grant cannot cover queue plus new limits')
        command = submission_command(root, assignment, limit, len(runs), plan['concurrency'], plan['memory_mib'], dependencies,
                                     plan.get('job_name', 'tu-maze-baselines'))
        tested = execute(command[:1] + ['--test-only'] + command[1:])
        entry = dict(configuration_id=batch[0][0]['id'] if len({config['id'] for config, _, _ in batch}) == 1 else None,
                     runs=runs, command=command, grant=grant_text, queue=queue, test_only=tested,
                     balance=state, bound_seconds=bound, submitted_unix=time.time(), array_id=None)
        ledger.append(entry)
        atomic_json(root / 'ledger.json', ledger)
        if 'budget_scope' in plan:
            event = dict(root=str(root), assignment=number, new_limit_seconds=limit * 60 * len(runs),
                         remaining_limit_seconds=remaining, balance=state, bound_seconds=bound,
                         grant=grant_text, queue=queue, test_only=tested, unix=time.time())
            with (Path(plan['budget_scope']).parent / 'admission-decisions.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
        response = execute(command).strip()
        entry['array_id'] = int(response.split(';')[0])
        entry['submission_response'] = response
        atomic_json(root / 'ledger.json', ledger)
        dependencies = [entry['array_id']]
        remaining -= limit * 60 * len(runs)
    print(json.dumps({'arrays': dependencies, 'runs': sum(len(entry['runs']) for entry in ledger), 'plan_sha256': digest}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--approval')
    args = parser.parse_args()
    submit(args.root, args.manifest, args.repo, args.approval)
