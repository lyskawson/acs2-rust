import hashlib
import json
from pathlib import Path
import subprocess
import time

from collect import complete_rows, outside_repository, validate_row
from grid_attempts import attempts, inspect_attempt
from grid_authorization import require_authorization
from grid_identity import field_bits, verify_references
from grid_protocol import atomic_json, load_plan, require_budget, stop_phase
from grid_scope import require_batch_root, snapshot, stop_path, verify_operations
from submit import execute, prepare, run_batches, shared_reservation, submission_command


def scoped_state(plan, executor=execute):
    scope = load_plan(plan['budget_scope'], plan['budget_scope_sha256'])
    frozen = []
    for name in scope['batch_roots']:
        path = Path(name)
        if (path / 'plan.json').exists():
            registered = json.loads((path / 'plan.json').read_text())
            if registered['stage'] == 'authorized_cost_pilot':
                frozen.append(path.resolve())
    return snapshot(plan, executor, preserve_roots=frozen)


def require_stream_slot(root, state):
    active = [name for name, value in state['batch_states'].items() if value['reserved_seconds'] > 0]
    if str(Path(root).resolve()) in active:
        raise ValueError('the previous array of this stream is not terminal')
    if len(active) >= 2:
        raise ValueError('two streams are already active; forty-job bound prevents another array')


def require_prefix(root, plan, ledger):
    batches = run_batches(plan)
    if len(ledger) > len(batches):
        raise ValueError('ledger exceeds the registered arrays')
    for entry, batch in zip(ledger, batches):
        expected = [(config['id'], agent, seed) for config, agent, seed in batch]
        actual = [(run['configuration_id'], run['agent'], run['seed']) for run in entry['runs']]
        limit = max(config['limits_minutes'][agent] for config, agent, _ in batch) * 60
        if actual != expected or any(run['limit_seconds'] != limit for run in entry['runs']):
            raise ValueError('ledger is not a prefix of the registered array sequence')
    return batches


def verify_completed(root, plan, ledger):
    for attempt in attempts(root, ledger):
        config = next(config for config in plan['configurations'] if config['id'] == attempt['configuration_id'])
        inspected, rows = inspect_attempt(attempt, config)
        if inspected['status'] != 'complete':
            raise ValueError('terminal previous array contains an incomplete or failed attempt')
        for row, target in zip(rows, config['targets']):
            validate_row(row, config, attempt['agent'], attempt['seed'], plan['measurement_commit'], target)
        identities = verify_references(plan, attempt, rows)
        recorded = json.loads((attempt['path'] / 'identity-checks.json').read_text())
        if field_bits(recorded) != field_bits(identities):
            raise ValueError('previous node identity record differs from independent comparison')


def dispatch(root, manifest, repository, approval, executor=execute):
    root, repository, manifest = outside_repository(root), Path(repository).resolve(), Path(manifest)
    preview = json.loads(manifest.read_text())
    require_batch_root(root, preview)
    verify_operations(preview)
    if stop_path(root, preview).exists():
        raise ValueError('shared STOP prevents a new array')
    require_authorization(root, preview, hashlib.sha256(manifest.read_bytes()).hexdigest(), approval)
    if root.exists():
        plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
        if field_bits(plan) != field_bits(preview):
            raise ValueError('saved stream manifest differs from current mechanical derivation')
        if (root / 'approval.json').read_bytes() != Path(approval).read_bytes():
            raise ValueError('stream approval differs')
        binary = repository / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
    else:
        plan, _, binary = prepare(root, manifest, repository, approval)
    if executor(['git', '-C', str(repository), 'rev-parse', 'HEAD']).strip() != plan['measurement_commit'] or executor(['git', '-C', str(repository), 'status', '--porcelain']).strip():
        raise ValueError('the frozen measurement clone changed')
    if hashlib.sha256(binary.read_bytes()).hexdigest() != plan['binary_sha256']:
        raise ValueError('the approved pilot binary changed; do not rebuild')
    ledger = json.loads((root / 'ledger.json').read_text())
    batches = require_prefix(root, plan, ledger)
    state = scoped_state(plan, executor)
    require_stream_slot(root, state)
    try:
        verify_completed(root, plan, ledger)
    except BaseException as error:
        stop_phase(stop_path(root, plan), dict(error=str(error), stream=str(root)))
        raise
    number = len(ledger)
    if number == len(batches):
        return dict(complete=True, submitted=False, arrays=number, runs=sum(len(entry['runs']) for entry in ledger))
    batch = batches[number]
    if not 20 <= len(batch) <= 40 or plan['concurrency'] != 20:
        raise ValueError('lean arrays require twenty running jobs and twenty to forty registered attempts')
    limit = max(config['limits_minutes'][agent] for config, agent, _ in batch)
    new_seconds = limit * 60 * len(batch)
    bound = require_budget(plan['pilot_allocation_cpu_seconds'], state, new_seconds, plan['phase_cpu_hour_limit'])
    grant = executor(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n'])
    queue = executor(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
    if float(grant.strip()) + shared_reservation(queue) + new_seconds > 15000 * 3600:
        raise ValueError('shared grant cannot cover the queue and this array')
    runs = []
    for config, agent, seed in batch:
        identifier = f"{config['id']}_{agent}_s{seed}_a01"
        directory = f"attempts/{identifier}/measurement/{config['id']}_{agent}_s{seed}"
        if (root / directory).exists():
            raise ValueError('unsubmitted assignment already has measurements; reconcile it')
        runs.append(dict(configuration_id=config['id'], agent=agent, seed=seed, attempt_id=identifier,
                         cause='registered first attempt', relative_directory=directory, limit_seconds=limit * 60))
    assignment = root / f'assignment-{number:02d}.json'
    payload = dict(measurement_repository=str(repository), binary=str(binary), operations_commit=plan['operations_commit'],
                   plan_sha256=(root / 'plan.sha256').read_text().strip(), runs=runs)
    if assignment.exists() and field_bits(json.loads(assignment.read_text())) != field_bits(payload):
        raise ValueError('prepared assignment differs from the registered next array')
    atomic_json(assignment, payload)
    command = submission_command(root, assignment, limit, len(runs), 20, plan['memory_mib'], [], plan['job_name'])
    event = dict(root=str(root), assignment=number, new_limit_seconds=new_seconds, remaining_limit_seconds=new_seconds,
                 balance=state, bound_seconds=bound, grant=grant, queue=queue, unix=time.time())
    try:
        tested = executor(command[:1] + ['--test-only'] + command[1:])
    except subprocess.CalledProcessError as error:
        event.update(admitted=False, returncode=error.returncode, error_output=error.output, allocated_cpu_seconds=0)
        with (Path(plan['budget_scope']).parent / 'submission-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        raise
    event.update(admitted=True, test_only=tested)
    require_authorization(root, plan, (root / 'plan.sha256').read_text().strip(), approval)
    for run in runs:
        (root / run['relative_directory']).parent.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(configuration_id=batch[0][0]['id'] if len({config['id'] for config, _, _ in batch}) == 1 else None,
                 runs=runs, command=command, grant=grant, queue=queue, test_only=tested, balance=state,
                 bound_seconds=bound, submitted_unix=time.time(), array_id=None)
    ledger.append(entry)
    atomic_json(root / 'ledger.json', ledger)
    with (Path(plan['budget_scope']).parent / 'admission-decisions.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    response = executor(command).strip()
    entry.update(array_id=int(response.split(';')[0]), submission_response=response)
    atomic_json(root / 'ledger.json', ledger)
    return dict(complete=False, submitted=True, array_id=entry['array_id'], assignment=number,
                runs=len(runs), bound_cpu_hours=bound / 3600)
