import argparse
import hashlib
import json
from pathlib import Path

from protocol import pilot_roots
from execution import measurement_identity, pilot_manifest, scope
from grid_attempts import attempts, select_attempts
from grid_identity import verify_references
from grid_protocol import allocation_rows, accounting, load_plan, require_budget
from collect import validate_row


def catalog(batch):
    batch = Path(batch)
    plan = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    ledger = json.loads((batch / 'ledger.json').read_text())
    inventory = attempts(batch, ledger)
    registered = {(config['id'], agent, seed) for config in plan['configurations'] for agent in plan['agents'] for seed in plan['seeds']}
    if any((attempt['configuration_id'], attempt['agent'], attempt['seed']) not in registered for attempt in inventory):
        raise ValueError('unregistered attempt in batch ledger')
    for allocation in ledger:
        expected_limit = 60 * max(next(config for config in plan['configurations'] if config['id'] == run['configuration_id'])['limits_minutes'][run['agent']] for run in allocation['runs'])
        if any(run['limit_seconds'] != expected_limit for run in allocation['runs']):
            raise ValueError('attempt hard reservation differs from registered array limit')
    records = allocation_rows((batch / 'accounting.psv').read_text())
    by_id = {row['JobID']: row for row in records}
    if len(by_id) != len(records):
        raise ValueError('duplicate allocation accounting')
    costs = accounting(ledger, records)
    if costs['reserved_seconds']:
        raise ValueError('unresolved reservations in completed batch')
    for attempt in inventory:
        record = by_id.get(f"{attempt['array_id']}_{attempt['task_id']}")
        if record is None or not record['State'].startswith(('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED')):
            raise ValueError('missing terminal accounting for an attempt')
    entries = {}
    for config in plan['configurations']:
        selected, _ = select_attempts(batch, ledger, config)
        if set(selected) != {(agent, seed) for agent in plan['agents'] for seed in plan['seeds']}:
            raise ValueError('incomplete registered batch')
        for (agent, seed), attempt in selected.items():
            path = attempt['path']
            rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
            launch = json.loads((path / 'launch.json').read_text())
            manifest = launch['manifest']
            expected = dict(configuration=config, agent=agent, seed=seed, attempt_id=attempt['attempt_id'],
                measurement_commit=plan['measurement_commit'], expected_binary_sha256=plan['binary_sha256'],
                expected_preset=plan['preset'], allocation_limit_seconds=attempt['limit_seconds'],
                plan_sha256=(batch / 'plan.sha256').read_text().strip())
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError('resolved launch manifest differs')
            expected_launch = dict(commit=plan['measurement_commit'], binary_sha256=plan['binary_sha256'],
                partition='lem-cpu-normal', array_job_id=str(attempt['array_id']), array_task_id=str(attempt['task_id']))
            if any(launch.get(key) != value for key, value in expected_launch.items()):
                raise ValueError('launch source, binary or allocation identity changed')
            for row, target in zip(rows, config['targets']):
                validate_row(row, config, agent, seed, plan['measurement_commit'], target)
                if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model'] or row['match_formations_train'] != 10 * row['actual_steps']:
                    raise ValueError('preset, hardware or full-update matching volume changed')
            declared = Path(plan['authorization_plan']).parent
            def resolve(value):
                candidate = Path(value)
                if not candidate.is_relative_to(declared):
                    raise ValueError('identity reference outside study')
                return batch.parent / candidate.relative_to(declared)
            identities = verify_references(plan, attempt, rows, resolve)
            if field_json(json.loads((path / 'identity-checks.json').read_text())) != field_json(identities):
                raise ValueError('node identity records differ from independent comparison')
            record = by_id[f"{attempt['array_id']}_{attempt['task_id']}"]
            if record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0' or record['Partition'] != 'lem-cpu-normal' or record['AllocCPUS'] != '1':
                raise ValueError('selected allocation resources or terminal state differ')
            completion = json.loads((path / 'completion.json').read_text())
            entries[config['id'], agent, seed] = dict(rows=rows, file=str(path / 'rows.jsonl'),
                sha256=hashlib.sha256((path / 'rows.jsonl').read_bytes()).hexdigest(),
                cost=dict(body_wall_seconds=completion['elapsed_seconds'], process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                    allocation_cpu_seconds=int(record['CPUTimeRAW']), peak_rss_kib=completion['peak_rss_kib']))
    return entries, dict(costs, attempts=len(inventory))


def field_json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def audit(root):
    root = Path(root)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    digest = hashlib.sha256((root / 'pilot-registration.json').read_bytes()).hexdigest()
    frozen_scope = json.loads((root / 'pilot-scope.json').read_text())
    declared = Path(frozen_scope['batch_roots'][0]).parent
    if frozen_scope != scope(digest, declared):
        raise ValueError('pilot scope changed')
    if hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest() != registration['reference_index_sha256']:
        raise ValueError('pilot reference index changed')
    cells, runs, rows = {}, [], []
    spent = attempts_count = identity_count = 0
    arrays = set()
    for index, name in enumerate(pilot_roots()):
        batch = root / name
        manifest = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
        expected = pilot_manifest(index, root, declared)
        if registration['manifests'][index] != expected or any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError('pilot manifest is not its mechanically registered arm')
        if (manifest['authorization_sha256'] != digest or manifest['budget_scope_sha256'] != hashlib.sha256((root / 'pilot-scope.json').read_bytes()).hexdigest()
                or manifest['stop_path'] != str(declared / 'STOP.json')):
            raise ValueError('pilot accounting or STOP differs from registration')
        operations = load_plan(root / 'operations/manifest.json', manifest['operations_manifest_sha256'])
        for file, checksum in operations['files'].items():
            path = root / 'operations' / file
            if not path.resolve().is_relative_to((root / 'operations').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError('pilot operation snapshot changed')
        ledger = json.loads((batch / 'ledger.json').read_text())
        for entry in ledger:
            if not entry.get('array_id') or entry['array_id'] in arrays:
                raise ValueError('unresolved or duplicated pilot allocation')
            arrays.add(entry['array_id'])
        entries, state = catalog(batch)
        spent += state['spent_seconds']
        attempts_count += state['attempts']
        for config in manifest['configurations']:
            agent = manifest['agents'][0]
            entry = entries[config['id'], agent, 42]
            cells[config['registered_id'], config['row_arm']] = entry['cost']
            final = entry['rows'][-1]
            runs.append(dict(configuration_id=config['registered_id'], task=config['task'], agent=agent,
                arm=config['row_arm'], proportion=config['agent_settings'][agent]['relabeled_proportion'],
                evaluation_blocks=config['evaluation_blocks'], source_rows_sha256=entry['sha256'],
                cost=entry['cost'], final=final))
            rows.extend(entry['rows'])
            if index == 4:
                identity_count += len(entry['rows'])
    require_budget(0, dict(spent_seconds=spent, reserved_seconds=0), 0, 5)
    if attempts_count < 30:
        raise ValueError('pilot is missing registered attempts')
    return dict(runs=runs, rows=len(rows), allocations=attempts_count, allocation_cpu_seconds=int(spent),
                identity_rows=identity_count, **measurement_identity()), rows, cells


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary, _, _ = audit(args.root)
    print(json.dumps({key: summary[key] for key in ('rows', 'allocations', 'allocation_cpu_seconds', 'identity_rows')}))
