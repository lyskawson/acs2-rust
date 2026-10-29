import hashlib
import json
from pathlib import Path

from protocol import BLOCKS, SHORTENED, pilot_roots
from collect import validate_row
from grid_attempts import attempts, inspect_attempt
from grid_identity import compare
from grid_protocol import accounting, allocation_rows, load_plan


def verify_failed(root):
    root = Path(root)
    inventory = root / 'checksums.json'
    for name, digest in json.loads(inventory.read_text())['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('failed pilot artifact changed')
    stop = json.loads((root / 'STOP.json').read_text())
    if stop.get('purpose') != 'schedule binary fixture identity':
        raise ValueError('failed pilot requires the diagnosed CLI fixture gate failure')
    if any((root / name).exists() for name in pilot_roots()[:-1]):
        raise ValueError('C must not have started before default identity')
    batch = root / pilot_roots()[-1]
    plan = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    identity = json.loads(Path(__file__).with_name('measurement.json').read_text())
    if any(plan[key] != value for key, value in identity.items()):
        raise ValueError('retry must preserve the failed attempt measurement identity')
    ledger = json.loads((batch / 'ledger.json').read_text())
    records = allocation_rows((batch / 'accounting.psv').read_text())
    costs = accounting(ledger, records)
    inventory_attempts = attempts(batch, ledger)
    if len(inventory_attempts) != 6 or costs['reserved_seconds']:
        raise ValueError('failed pilot must have six fully accounted terminal allocations')
    known = {record['JobID']: record for record in records}
    references = json.loads((root / 'reference-index.json').read_text())
    partial_rows = 0
    for attempt in inventory_attempts:
        record = known[f"{attempt['array_id']}_{attempt['task_id']}"]
        if record['State'] != 'FAILED' or record['ExitCode'] != '1:0' or record['AllocCPUS'] != '1':
            raise ValueError('failed pilot allocation differs from the diagnosed failure')
        config = next(config for config in plan['configurations'] if config['id'] == attempt['configuration_id'])
        _, rows = inspect_attempt(attempt, config)
        for row in rows:
            validate_row(row, config, SHORTENED, 42, identity['measurement_commit'], row['nominal_step'])
        reference = next(ref for ref in references if (ref['purpose'], ref['configuration_id'], ref['seed']) == ('test', config['registered_id'], 42))
        prior = [json.loads(line) for line in (root / reference['file']).read_text().splitlines()]
        if rows:
            compare(rows, prior, [42], [row['nominal_step'] for row in rows], BLOCKS)
        partial_rows += len(rows)
    return dict(allocations=6, allocation_cpu_seconds=int(costs['spent_seconds']), partial_rows=partial_rows,
        inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        registration_sha256=hashlib.sha256((root / 'pilot-registration.json').read_bytes()).hexdigest())


def failed_path(root):
    root = Path(root)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    failed = registration['failed_pilot']
    return Path(failed['root']) if root.resolve() == Path(registration['destination']).resolve() else (root / failed['mirror_relative']).resolve()


def compare_retry(root, entries):
    batch = failed_path(root) / pilot_roots()[-1]
    plan = json.loads((batch / 'plan.json').read_text())
    ledger = json.loads((batch / 'ledger.json').read_text())
    checked = 0
    for attempt in attempts(batch, ledger):
        config = next(config for config in plan['configurations'] if config['id'] == attempt['configuration_id'])
        _, partial = inspect_attempt(attempt, config)
        if partial:
            compare(entries[config['id'], SHORTENED, 42]['rows'], partial, [42],
                    [row['nominal_step'] for row in partial], BLOCKS)
        checked += len(partial)
    return checked
