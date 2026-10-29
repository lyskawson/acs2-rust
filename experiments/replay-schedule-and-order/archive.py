import argparse
import hashlib
import json
from pathlib import Path

from protocol import CORRECTED_PILOT_INDICES, SCHEDULES, configurations, pilot_roots, PILOT_SETTINGS
from previous_pilot import local_reference, previous_path, verify_previous
from execution import measurement_identity, pilot_manifest, scope
from grid_attempts import attempts, select_attempts, inspect_attempt
from grid_identity import compare, field_bits, verify_references
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
                if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model'] or row['match_formations_train'] != row['actual_steps'] + 3 * row['learning_update_counts']['full']:
                    raise ValueError('preset, hardware or full-update matching volume changed')
            declared = Path(plan['authorization_plan']).parent
            def resolve(value):
                return local_reference(batch.parent, declared, value)
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
    return json.dumps(field_bits(value), sort_keys=True, allow_nan=False)


def binary_fixture_audit(root, previous):
    fixture = json.loads((previous / 'registration/replay_schedule_and_order_regression.json').read_text())
    checked = 0
    for base in configurations():
        directory = root / pilot_roots()[-1] / 'binary-fixture' / base['task']
        proof = json.loads((directory / 'verification.json').read_text())
        if proof != dict(task=base['task'], settings=6, rows=12, verified=True,
                         binary_sha256=measurement_identity()['binary_sha256']):
            raise ValueError('new binary schedule fixture gate is incomplete')
        selected = [config for config in fixture['configurations'] if config['task'] == base['task'] and config['mode'] in SCHEDULES]
        for index, config in enumerate(selected):
            rows = [json.loads(line) for line in (directory / f'{index}.jsonl').read_text().splitlines()]
            actual = [field_bits({key: value for key, value in row.items() if key not in fixture['excluded']}) for row in rows]
            if actual != config['rows']:
                raise ValueError('new binary differs from an unchanged schedule fixture row')
            checked += len(rows)
    if checked != 72:
        raise ValueError('schedule binary identity requires seventy-two fixture rows')
    return checked


def audit(root):
    root = Path(root)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    digest = hashlib.sha256((root / 'pilot-registration.json').read_bytes()).hexdigest()
    frozen_scope = json.loads((root / 'pilot-scope.json').read_text())
    declared = Path(registration['destination'])
    previous = previous_path(root)
    checked_previous = verify_previous(previous)
    if any(registration['previous_pilot'].get(key) != value for key, value in checked_previous.items()):
        raise ValueError('previous pilot differs from the pinned first round')
    if frozen_scope != scope(digest, declared, registration['previous_pilot']['root']):
        raise ValueError('pilot accounting scope changed')
    if registration['pilot_indices'] != CORRECTED_PILOT_INDICES or any(
            registration['manifests'][index] is not None for index in range(10) if index not in CORRECTED_PILOT_INDICES):
        raise ValueError('the corrected round must contain exactly twenty-four runs')
    for name, key in [('reference-index.json', 'reference_index_sha256'), ('base-selections.json', 'base_selection_sha256')]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != registration[key]:
            raise ValueError('pilot reference or base selection changed')
    old_summary = json.loads((previous / 'pilot-summary.json').read_text())
    runs = [run for run in old_summary['runs'] if run['mode'] in SCHEDULES]
    arms = {run['arm'] for run in runs}
    rows = [json.loads(line) for line in (previous / 'pilot-rows.jsonl').read_text().splitlines()
            if json.loads(line)['arm'] in arms]
    cells = {(run['configuration_id'], run['arm']): run['cost'] for run in runs}
    spent, attempts_count = checked_previous['allocation_cpu_seconds'], checked_previous['allocations']
    identity_count = new_rows = 0
    arrays = {allocation['array_id'] for name in pilot_roots()
              for allocation in json.loads((previous / name / 'ledger.json').read_text())}
    for index in CORRECTED_PILOT_INDICES:
        name = pilot_roots()[index]
        batch = root / name
        manifest = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
        expected = pilot_manifest(index, root, declared)
        if registration['manifests'][index] != expected or any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError('pilot arm differs from its exact mechanical registration')
        if (manifest['authorization_sha256'] != digest or manifest['budget_scope_sha256'] != hashlib.sha256((root / 'pilot-scope.json').read_bytes()).hexdigest()
                or manifest['stop_path'] != str(declared / 'STOP.json')):
            raise ValueError('pilot accounting or shared STOP changed')
        operations = load_plan(root / 'operations/manifest.json', manifest['operations_manifest_sha256'])
        for file, checksum in operations['files'].items():
            path = root / 'operations' / file
            if not path.resolve().is_relative_to((root / 'operations').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError('pilot operation snapshot changed')
        ledger = json.loads((batch / 'ledger.json').read_text())
        for allocation in ledger:
            if not allocation.get('array_id') or allocation['array_id'] in arrays:
                raise ValueError('unresolved or duplicate pilot allocation')
            arrays.add(allocation['array_id'])
        entries, state = catalog(batch)
        spent += state['spent_seconds']
        attempts_count += state['attempts']
        for config in manifest['configurations']:
            agent = manifest['agents'][0]
            entry = entries[config['id'], agent, 42]
            cells[config['registered_id'], config['row_arm']] = entry['cost']
            runs.append(dict(configuration_id=config['registered_id'], task=config['task'], agent=agent,
                mode=PILOT_SETTINGS[index][1], arm=config['row_arm'], source_rows_sha256=entry['sha256'],
                cost=entry['cost'], final=entry['rows'][-1]))
            rows.extend(entry['rows'])
            new_rows += len(entry['rows'])
            if PILOT_SETTINGS[index][1] == 'base':
                identity_count += len(entry['rows'])
        for attempt in attempts(batch, ledger):
            config = next(config for config in manifest['configurations'] if config['id'] == attempt['configuration_id'])
            status, partial = inspect_attempt(attempt, config)
            if status['status'] != 'complete' and partial:
                current = entries[config['id'], attempt['agent'], attempt['seed']]['rows']
                compare(current, partial, [attempt['seed']], [row['nominal_step'] for row in partial], config['evaluation_blocks'])
    require_budget(0, dict(spent_seconds=spent, reserved_seconds=0), 0, 12)
    if len(runs) != 60 or new_rows != 196 or attempts_count < 84:
        raise ValueError('pilot is incomplete')
    return dict(runs=runs, rows=len(rows), allocations=attempts_count, allocation_cpu_seconds=int(spent),
                identity_rows=identity_count, previous_pilot=registration['previous_pilot'],
                schedule_binary_fixture_rows=binary_fixture_audit(root, previous),
                rounds=[dict(round='length_weighted', **checked_previous),
                        dict(round='uniform_episode', runs=24, rows=new_rows,
                             allocations=attempts_count - checked_previous['allocations'],
                             allocation_cpu_seconds=int(spent) - checked_previous['allocation_cpu_seconds'],
                             identity_rows=identity_count)], **measurement_identity()), rows, cells


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary, _, _ = audit(args.root)
    print(json.dumps({key: summary[key] for key in ('rows', 'allocations', 'allocation_cpu_seconds', 'identity_rows')}))
