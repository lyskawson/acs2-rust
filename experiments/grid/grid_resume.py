import json
from pathlib import Path
import subprocess
import time

from grid_protocol import atomic_json, load_plan, require_budget
from grid_scope import snapshot, stop_path, verify_operations
from submit import run_batches, shared_reservation


def resume(root, execute, command_builder):
    root = Path(root)
    digest = (root / 'plan.sha256').read_text().strip()
    plan = load_plan(root / 'plan.json', digest)
    if stop_path(root, plan).exists():
        raise ValueError('shared STOP prevents resumption')
    verify_operations(plan)
    approval = json.loads((root / 'approval.json').read_text())
    if approval != json.loads((Path(plan['authorization_plan']).parent / 'approval.json').read_text()) or not approval['approved']:
        raise ValueError('resumption requires the same explicit approval')
    if approval['plan_sha256'] != plan['authorization_sha256']:
        raise ValueError('resumption approval has a different plan digest')
    ledger = json.loads((root / 'ledger.json').read_text())
    batches = run_batches(plan)
    if len(ledger) > len(batches):
        raise ValueError('ledger exceeds the registered batch scope')
    for entry, batch in zip(ledger, batches):
        actual = [(run['configuration_id'], run['agent'], run['seed']) for run in entry['runs']]
        expected = [(config['id'], agent, seed) for config, agent, seed in batch]
        if actual != expected:
            raise ValueError('submitted batches differ from the registered prefix')
    origin = json.loads((root / 'assignment-00.json').read_text())
    repo = origin['measurement_repository']
    if execute(['git', '-C', repo, 'rev-parse', 'HEAD']).strip() != plan['measurement_commit'] or execute(['git', '-C', repo, 'status', '--porcelain']).strip():
        raise ValueError('frozen measurement clone differs')
    remaining = sum(len(batch) * 60 * max(config['limits_minutes'][agent] for config, agent, _ in batch)
                    for batch in batches[len(ledger):])
    require_budget(5445, snapshot(plan, execute), remaining, 60)
    for number in range(len(ledger), len(batches)):
        if stop_path(root, plan).exists():
            raise ValueError('shared STOP prevents another submission')
        batch = batches[number]
        limit = max(config['limits_minutes'][agent] for config, agent, _ in batch)
        runs = []
        for config, agent, seed in batch:
            identifier = f"{config['id']}_{agent}_s{seed}_a01"
            directory = f"attempts/{identifier}/measurement/{config['id']}_{agent}_s{seed}"
            output = root / directory
            if output.exists():
                raise ValueError('unsubmitted assignment already has measurement artifacts')
            output.parent.parent.mkdir(parents=True, exist_ok=True)
            runs.append(dict(configuration_id=config['id'], agent=agent, seed=seed, attempt_id=identifier,
                             cause='registered first attempt', relative_directory=directory, limit_seconds=limit * 60))
        assignment = root / f'assignment-{number:02d}.json'
        payload = dict(origin, runs=runs)
        if assignment.exists() and json.loads(assignment.read_text()) != payload:
            raise ValueError('prepared assignment differs from the registered remaining scope')
        atomic_json(assignment, payload)
        state = snapshot(plan, execute)
        bound = require_budget(5445, state, remaining, 60)
        grant = execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n'])
        queue = execute(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
        if float(grant.strip()) + shared_reservation(queue) + remaining > 15000 * 3600:
            raise ValueError('shared grant cannot cover remaining reservations')
        command = command_builder(root, assignment, limit, len(runs), 20, plan['memory_mib'],
                                  [ledger[-1]['array_id']], plan.get('job_name', 'tu-her-variants'))
        event = dict(event='resumption_preflight', assignment=number, command=command, balance=state,
                     bound_seconds=bound, new_limit_seconds=len(runs) * limit * 60,
                     remaining_limit_seconds=remaining, grant=grant, queue=queue, unix=time.time())
        try:
            tested = execute(command[:1] + ['--test-only'] + command[1:])
        except subprocess.CalledProcessError as error:
            event.update(admitted=False, error_output=error.output, returncode=error.returncode, allocated_cpu_seconds=0)
            with (Path(plan['budget_scope']).parent / 'submission-events.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            return dict(complete=False, next_assignment=number, submitted_arrays=len(ledger), error=error.output)
        event.update(admitted=True, test_only=tested)
        with (Path(plan['budget_scope']).parent / 'admission-decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        entry = dict(configuration_id=batch[0][0]['id'], runs=runs, command=command, grant=grant, queue=queue,
                     test_only=tested, balance=state, bound_seconds=bound, submitted_unix=time.time(), array_id=None)
        ledger.append(entry)
        atomic_json(root / 'ledger.json', ledger)
        response = execute(command).strip()
        entry.update(array_id=int(response.split(';')[0]), submission_response=response)
        atomic_json(root / 'ledger.json', ledger)
        remaining -= len(runs) * limit * 60
    return dict(complete=True, submitted_arrays=len(ledger), runs=sum(len(entry['runs']) for entry in ledger))
