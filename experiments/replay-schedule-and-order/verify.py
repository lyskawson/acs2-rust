import argparse
import hashlib
import json
from pathlib import Path

import archive
from approved_execution import APPROVED_SHA256, approved, catalog_cache
from execution import choose, derive, measurement_identity, read_index, scope, study_references
from failed_pilot import failed_path
from previous_pilot import previous_path
from protocol import EXPERIMENT, SHORTENED, pilot_roots
from grid_attempts import attempts, inspect_attempt
from grid_dispatch import require_prefix
from grid_identity import compare, field_bits
from grid_protocol import allocation_rows, accounting, load_plan, require_budget, seconds

FROZEN_PILOT_SHA256 = 'd8782780579e462c1b71589d5c26d979574607e5f6a890ee0fd084ebe9cb1eca'


def resolved_batch(root, declared):
    root, declared = Path(root), Path(declared)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    locations = [(Path(registration['destination']), root),
                 (Path(registration['failed_pilot']['root']), failed_path(root)),
                 (Path(registration['previous_pilot']['root']), previous_path(root))]
    for source, destination in locations:
        if declared.is_relative_to(source):
            return destination / declared.relative_to(source)
    raise ValueError('batch is outside the three registered archive locations')


def verify_pilots(root):
    root = Path(root)
    frozen = root.parent / 'uniform-episode-pilot-frozen'
    inventory = frozen / 'checksums.json'
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != FROZEN_PILOT_SHA256:
        raise ValueError('the original corrected pilot inventory changed')
    for name, digest in json.loads(inventory.read_text())['files'].items():
        data = (frozen / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('sealed corrected pilot artifact changed: ' + name)
        if name == 'admission-decisions.jsonl':
            if not (root / name).read_bytes().startswith(data):
                raise ValueError('pilot admission prefix changed')
        elif name != 'shared-accounting.psv' and (root / name).read_bytes() != data:
            raise ValueError('active immutable pilot artifact changed: ' + name)
    original = allocation_rows((frozen / 'shared-accounting.psv').read_text())
    current = {row['JobID']: row for row in allocation_rows((root / 'shared-accounting.psv').read_text())}
    fields = ('JobIDRaw', 'CPUTimeRAW', 'AllocCPUS', 'State', 'ExitCode', 'Partition')
    if any(row['JobID'] not in current or any(current[row['JobID']][key] != row[key] for key in fields) for row in original):
        raise ValueError('original terminal accounting differs from the shared scope')
    pilot, _, _ = archive.audit(root)
    if (pilot['allocations'], pilot['allocation_cpu_seconds'], pilot['identity_rows'], pilot['retry_identity_rows']) != (90, 8147, 49, 37):
        raise ValueError('approved pilot accounting or identity changed')
    return pilot


def verify_manifest(root, batch, plan, selection=None):
    root, batch = Path(root), Path(batch)
    manifest = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    declared = Path(manifest['authorization_plan']).parent
    expected = dict(derive(plan, manifest['purpose'], selection, manifest.get('configuration_id'), manifest.get('axis'), manifest.get('slot')),
        experiment=EXPERIMENT, partition=plan['execution']['partition'], cpu_model=plan['execution']['cpu_model'],
        memory_mib=plan['forecast']['memory_mib'], concurrency=20, phase_cpu_hour_limit=60,
        pilot_allocation_cpu_seconds=0, accounting_since='2026-10-28', **measurement_identity())
    expected['identity_references'] = study_references(expected, root, declared)
    operations_path = root / 'execution-operations/manifest.json'
    operations = load_plan(operations_path, manifest['operations_manifest_sha256'])
    for name, digest in operations['files'].items():
        path = root / 'execution-operations' / name
        if not path.resolve().is_relative_to((root / 'execution-operations').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('immutable execution operations changed')
    expected.update(authorization_plan=str(declared / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
        budget_scope=str(declared / 'scope.json'), budget_scope_sha256=hashlib.sha256((root / 'scope.json').read_bytes()).hexdigest(),
        stop_path=str(declared / 'STOP.json'), operations_manifest=str(declared / 'execution-operations/manifest.json'),
        operations_manifest_sha256=hashlib.sha256(operations_path.read_bytes()).hexdigest(),
        operations_commit=operations['commit'], job_name='tu-replay-' + manifest['purpose'])
    if field_bits(manifest) != field_bits(expected):
        raise ValueError('manifest differs from the exact approved mechanical derivation')
    if (batch / 'approval.json').read_bytes() != (root / 'approval.json').read_bytes():
        raise ValueError('stream approval differs')
    require_prefix(batch, manifest, json.loads((batch / 'ledger.json').read_text()))
    return manifest


def allocation_inventory(root):
    root = Path(root)
    value = json.loads((root / 'scope.json').read_text())
    result, identifiers, arrays, states = [], set(), set(), {}
    for declared in value['batch_roots']:
        batch = resolved_batch(root, declared)
        if not batch.exists():
            continue
        ledger = json.loads((batch / 'ledger.json').read_text())
        records = allocation_rows((batch / 'accounting.psv').read_text())
        known = {row['JobID']: row for row in records}
        state = accounting(ledger, records)
        if state['reserved_seconds']:
            raise ValueError('final archive has outstanding reservations')
        states[declared] = state
        for entry in ledger:
            if not entry.get('array_id') or entry['array_id'] in arrays:
                raise ValueError('unresolved or duplicate allocation in the scope')
            arrays.add(entry['array_id'])
        for attempt in attempts(batch, ledger):
            record = known[f"{attempt['array_id']}_{attempt['task_id']}"]
            if record['JobIDRaw'] in identifiers or record['State'].split()[0].split('+')[0] not in (
                    'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED'):
                raise ValueError('allocation is duplicated or not terminal')
            identifiers.add(record['JobIDRaw'])
            if record['AllocCPUS'] != '1' or record['Partition'] != 'lem-cpu-normal':
                raise ValueError('allocation resources differ')
            completion = attempt['path'] / 'completion.json'
            measured = json.loads(completion.read_text()) if completion.exists() else None
            purpose = 'pilot' if batch.name.startswith('pilot-') else 'screening' if batch.name == 'screening' else 'test'
            result.append(dict(root=declared, purpose=purpose, attempt_id=attempt['attempt_id'],
                job_id=f"{attempt['array_id']}_{attempt['task_id']}", job_id_raw=record['JobIDRaw'],
                state=record['State'], exit_code=record['ExitCode'], allocation_cpu_seconds=int(record['CPUTimeRAW']),
                limit_seconds=attempt['limit_seconds'], agent=attempt['agent'], seed=attempt['seed'],
                process_cpu_seconds=measured['user_cpu_seconds'] + measured['system_cpu_seconds'] if measured and all(
                    name in measured for name in ('user_cpu_seconds', 'system_cpu_seconds')) else None,
                peak_rss_kib=measured.get('peak_rss_kib') if measured else None))
    state = dict(spent_seconds=sum(value['spent_seconds'] for value in states.values()), reserved_seconds=0)
    require_budget(0, state, 0, 60)
    if sum(entry['allocation_cpu_seconds'] for entry in result if entry['purpose'] == 'pilot') != 8147:
        raise ValueError('pilot allocations were omitted or charged twice')
    return result, dict(state, array_ids=sorted(arrays), batch_states=states)


def verify_admissions(root, inventory):
    root = Path(root)
    scope_value = json.loads((root / 'scope.json').read_text())
    ledgers = {}
    for declared in scope_value['batch_roots']:
        batch = resolved_batch(root, declared)
        if (batch / 'ledger.json').exists():
            for number, entry in enumerate(json.loads((batch / 'ledger.json').read_text())):
                ledgers[declared, number] = entry
    observed = set()
    max_bound = max_streams = 0
    for location in (previous_path(root), failed_path(root), root):
        for event in map(json.loads, (location / 'admission-decisions.jsonl').read_text().splitlines()):
            key = event['root'], event['assignment']
            if key not in ledgers or key in observed:
                raise ValueError('missing, duplicate or unregistered admission')
            observed.add(key)
            entry, balance = ledgers[key], event['balance']
            new = sum(run['limit_seconds'] for run in entry['runs'])
            bound = balance['spent_seconds'] + balance['reserved_seconds'] + new
            pilot = Path(key[0]).name.startswith('pilot-')
            if event['new_limit_seconds'] != new or event['bound_seconds'] != bound or bound > (12 if pilot else 60) * 3600:
                raise ValueError('admission exceeds its cumulative allocation limit')
            if not event.get('test_only') or event['test_only'] != entry['test_only'] or event['grant'] != entry['grant'] or event['queue'] != entry['queue']:
                raise ValueError('grant, queue or scheduler test-only proof differs')
            reserved = sum(seconds(line.split('|')[3]) * int(line.split('|')[4]) for line in event['queue'].splitlines() if line.strip())
            if float(event['grant'].strip()) + reserved + new > 15000 * 3600:
                raise ValueError('admission exceeded the shared grant')
            active = [name for name, value in balance['batch_states'].items() if value['reserved_seconds'] > 0]
            if len(active) >= 2 or key[0] in active:
                raise ValueError('admission exceeds two streams or overlaps its own previous array')
            before = {value['array_id'] for other, value in ledgers.items() if other != key and value['submitted_unix'] < event['unix']}
            if set(balance['array_ids']) != before:
                raise ValueError('admission omitted or duplicated a prior array')
            if not pilot and (not 20 <= len(entry['runs']) <= 40 or not any(part.startswith('--array=') and part.endswith('%20') for part in entry['command'])):
                raise ValueError('lean array size or concurrency differs')
            if any(part.startswith(('--dependency=', '--nodelist=')) for part in entry['command']):
                raise ValueError('dependent queues and host pinning are not registered')
            max_bound, max_streams = max(max_bound, bound), max(max_streams, len(active) + 1)
    if observed != set(ledgers):
        raise ValueError('not every attempt has exactly one admission')
    return dict(admissions=len(observed), maximum_admission_bound_seconds=max_bound, maximum_streams=max_streams,
                running_job_hard_bound=40, pilot_allocations=sum(entry['purpose'] == 'pilot' for entry in inventory))


def verify_study(root):
    root = Path(root)
    plan = approved(root / 'approved-plan.json')
    verify_pilots(root)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    declared = Path(registration['destination'])
    if json.loads((root / 'scope.json').read_text()) != scope(APPROVED_SHA256, declared,
            registration['previous_pilot']['root'], registration['failed_pilot']['root']):
        raise ValueError('scope differs from all 46 registered roots')
    verify_manifest(root, root / 'screening', plan)
    screening, _ = archive.catalog(root / 'screening')
    defaults = archive.catalog(root / pilot_roots()[-1])[0]
    base = json.loads((root / pilot_roots()[-1] / 'plan.json').read_text())
    default_rows = {(config['registered_id'], 42): defaults[config['id'], SHORTENED, 42]['rows'] for config in base['configurations']}
    selection = choose(plan, screening, read_index(root), default_rows)
    if field_bits(selection) != field_bits(json.loads((root / 'selections.json').read_text())):
        raise ValueError('saved selection differs from complete validation-only rows')
    proof = json.loads((root / 'selection-verification.json').read_text())
    if (proof['plan_sha256'] != APPROVED_SHA256 or proof['selection_sha256'] != hashlib.sha256((root / 'selections.json').read_bytes()).hexdigest()
            or proof['input_rows_sha256'] != {'|'.join(map(str, key)): entry['sha256'] for key, entry in screening.items()}):
        raise ValueError('selection provenance differs')
    tests, expected = {}, set()
    for config in plan['configurations']:
        for axis in ('B', 'C'):
            if selection['selections'][config['id']][axis]['setting'] == 'base':
                continue
            for slot in (['shortened', 'er'] if axis == 'B' else ['shortened', 'er', 'order-control']):
                name = f'test-{axis}-{config["id"]}-{slot}'
                expected.add(name)
                verify_manifest(root, root / name, plan, selection)
                entries, _ = archive.catalog(root / name)
                if len(entries) != 20:
                    raise ValueError('conditional arm requires twenty complete trajectories')
                tests.update(entries)
    if {path.name for path in root.glob('test-*') if path.is_dir()} != expected or len(screening) != 120:
        raise ValueError('missing or extra conditional tests or screening rows')
    inventory, state = allocation_inventory(root)
    admissions = verify_admissions(root, inventory)
    return tests, screening, selection, dict(state, allocations=len(inventory), admissions=admissions), inventory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    with catalog_cache():
        tests, screening, selection, state, _ = verify_study(args.root)
    print(json.dumps(dict(test_runs=len(tests), screening_runs=len(screening), execution=state)))
