import hashlib
import json
from pathlib import Path
import subprocess
import time

from collect import complete_rows, outside_repository, validate_row
from grid_attempts import attempts, inspect_attempt, select_attempts
from grid_authorization import require_authorization
from grid_identity import compare, field_bits, verify_references
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
    previous = -1
    for position, entry in enumerate(ledger):
        logical = entry.get('registered_assignment', position)
        if type(logical) is not int or not 0 <= logical < len(batches) or logical not in (previous, previous + 1):
            raise ValueError('ledger exceeds or skips the registered arrays')
        if logical == previous and entry.get('retry_of') != ledger[position - 1].get('array_id'):
            raise ValueError('repeated registered array lacks its retry provenance')
        previous = logical
        batch = batches[logical]
        expected = [(config['id'], agent, seed) for config, agent, seed in batch]
        actual = [(run['configuration_id'], run['agent'], run['seed']) for run in entry['runs']]
        limit = max(config['limits_minutes'][agent] for config, agent, _ in batch) * 60
        if actual != expected or any(run['limit_seconds'] != limit for run in entry['runs']):
            raise ValueError('ledger is not a prefix of the registered array sequence')
    return batches


def verify_completed(root, plan, ledger, recovered=()):
    partials = []
    complete = {}
    for attempt in attempts(root, ledger):
        config = next(config for config in plan['configurations'] if config['id'] == attempt['configuration_id'])
        inspected, rows = inspect_attempt(attempt, config)
        if inspected['status'] != 'complete':
            if attempt['array_id'] not in recovered:
                raise ValueError('terminal previous array contains an incomplete or failed attempt')
            completion_path = attempt['path'] / 'completion.json'
            if completion_path.exists():
                completion = json.loads(completion_path.read_text())
                for name, digest in completion['sha256'].items():
                    if hashlib.sha256((attempt['path'] / name).read_bytes()).hexdigest() != digest:
                        raise ValueError('failed attempt artifact changed')
            for row, target in zip(rows, config['targets']):
                validate_row(row, config, attempt['agent'], attempt['seed'], plan['measurement_commit'], target)
            partials.append((attempt, config, rows))
            continue
        for row, target in zip(rows, config['targets']):
            validate_row(row, config, attempt['agent'], attempt['seed'], plan['measurement_commit'], target)
        identities = verify_references(plan, attempt, rows)
        recorded = json.loads((attempt['path'] / 'identity-checks.json').read_text())
        if field_bits(recorded) != field_bits(identities):
            raise ValueError('previous node identity record differs from independent comparison')
        complete[attempt['configuration_id'], attempt['agent'], attempt['seed']] = rows
    for attempt, config, rows in partials:
        key = attempt['configuration_id'], attempt['agent'], attempt['seed']
        if rows and key in complete:
            compare(complete[key], rows, [attempt['seed']], [row['nominal_step'] for row in rows], config.get('evaluation_blocks', ['training', 'validation', 'test']))


def recovery_arrays(path, plan):
    if path is None:
        return set()
    value = json.loads(Path(path).read_text())
    stop = Path(path).parent / value['archived_stop']
    if (value['plan_sha256'] != plan['authorization_sha256'] or value['binary_sha256'] != plan['binary_sha256']
            or value['phase_cpu_hour_limit'] != plan['phase_cpu_hour_limit'] or value['resolved'] is not True
            or hashlib.sha256(stop.read_bytes()).hexdigest() != value['stop_sha256']):
        raise ValueError('recovery evidence or unchanged scientific identity differs')
    return set(value['failed_arrays'])


def pending_assignment(root, plan, ledger, batches):
    selected = {}
    for config in plan['configurations']:
        selected[config['id']] = select_attempts(root, ledger, config)[0]
    for index, batch in enumerate(batches):
        if any((agent, seed) not in selected[config['id']] for config, agent, seed in batch):
            return index
    return len(batches)


def dispatch(root, manifest, repository, approval, executor=execute, recovery=None):
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
    recovered = recovery_arrays(recovery, plan)
    try:
        verify_completed(root, plan, ledger, recovered)
    except BaseException as error:
        stop_phase(stop_path(root, plan), dict(error=str(error), stream=str(root)))
        raise
    number = len(ledger)
    logical = pending_assignment(root, plan, ledger, batches) if ledger else 0
    if logical == len(batches):
        return dict(complete=True, submitted=False, arrays=number, runs=sum(len(entry['runs']) for entry in ledger))
    retry_of = next((entry['array_id'] for position, entry in reversed(list(enumerate(ledger))) if entry.get('registered_assignment', position) == logical), None)
    if retry_of is not None and retry_of not in recovered:
        raise ValueError('failed array requires diagnosed recovery evidence before retry')
    batch = batches[logical]
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
        attempt_number = 1 + sum((run['configuration_id'], run['agent'], run['seed']) == (config['id'], agent, seed) for entry in ledger for run in entry['runs'])
        identifier = f"{config['id']}_{agent}_s{seed}_a{attempt_number:02d}"
        directory = f"attempts/{identifier}/measurement/{config['id']}_{agent}_s{seed}"
        if (root / directory).exists():
            raise ValueError('unsubmitted assignment already has measurements; reconcile it')
        runs.append(dict(configuration_id=config['id'], agent=agent, seed=seed, attempt_id=identifier,
                         cause='retry after diagnosed SLURM metadata timeout' if retry_of is not None else 'registered first attempt', relative_directory=directory, limit_seconds=limit * 60))
    assignment = root / f'assignment-{number:02d}.json'
    payload = dict(measurement_repository=str(repository), binary=str(binary), operations_commit=plan['operations_commit'],
                   plan_sha256=(root / 'plan.sha256').read_text().strip(), runs=runs)
    if assignment.exists() and field_bits(json.loads(assignment.read_text())) != field_bits(payload):
        raise ValueError('prepared assignment differs from the registered next array')
    atomic_json(assignment, payload)
    command = submission_command(root, assignment, limit, len(runs), 20, plan['memory_mib'], [], plan['job_name'])
    node_operations = Path(plan['operations_manifest']).parent / 'experiments/grid'
    command[-4:] = [str(node_operations / 'grid_job.sh'), str(node_operations), str(root), str(assignment)]
    event = dict(root=str(root), assignment=number, registered_assignment=logical, retry_of=retry_of, new_limit_seconds=new_seconds, remaining_limit_seconds=new_seconds,
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
                 runs=runs, registered_assignment=logical, retry_of=retry_of, dispatcher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), command=command, grant=grant, queue=queue, test_only=tested, balance=state,
                 bound_seconds=bound, submitted_unix=time.time(), array_id=None)
    ledger.append(entry)
    atomic_json(root / 'ledger.json', ledger)
    with (Path(plan['budget_scope']).parent / 'admission-decisions.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    response = executor(command).strip()
    entry.update(array_id=int(response.split(';')[0]), submission_response=response)
    atomic_json(root / 'ledger.json', ledger)
    return dict(complete=False, submitted=True, array_id=entry['array_id'], assignment=number, registered_assignment=logical,
                runs=len(runs), bound_cpu_hours=bound / 3600)
