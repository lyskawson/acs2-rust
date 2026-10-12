import hashlib
import json
from pathlib import Path

from grid_protocol import TERMINAL, accounting, allocation_rows, load_plan

SACCT_FIELDS = 'JobID,JobIDRaw,CPUTimeRAW,TotalCPU,AllocCPUS,ElapsedRaw,MaxRSS,State,ExitCode,NodeList,Partition,Start,End'


def stop_path(root, plan):
    return Path(plan.get('stop_path', Path(root) / 'STOP.json'))


def verify_operations(plan):
    if 'operations_manifest' not in plan:
        return
    path = Path(plan['operations_manifest'])
    manifest = load_plan(path, plan['operations_manifest_sha256'])
    for name, digest in manifest['files'].items():
        file = path.parent / name
        if not file.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('operations file is outside its snapshot')
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f'operations snapshot changed: {name}')


def snapshot(plan, execute):
    path = Path(plan['budget_scope'])
    scope = load_plan(path, plan['budget_scope_sha256'])
    base = path.parent.resolve()
    roots = [Path(name).resolve() for name in scope['batch_roots']]
    if len(roots) != len(set(roots)) or not all(root.is_relative_to(base) for root in roots):
        raise ValueError('invalid accounting scope roots')
    if scope['pilot_allocation_cpu_seconds'] != plan['pilot_allocation_cpu_seconds']:
        raise ValueError('pilot accounting differs from the shared scope')
    if scope['authorization_sha256'] != plan['authorization_sha256']:
        raise ValueError('accounting scope belongs to a different approved plan')
    ledgers, arrays = {}, set()
    for root in roots:
        if not root.exists():
            continue
        ledger = json.loads((root / 'ledger.json').read_text())
        for entry in ledger:
            identifier = entry.get('array_id')
            if not identifier or identifier in arrays:
                raise ValueError('unresolved submission intent or duplicate allocation in the study')
            arrays.add(identifier)
        ledgers[root] = ledger
    raw = execute(['sacct', '--array', '-S', scope['accounting_since'], '-j',
                   ','.join(map(str, sorted(arrays))), '-P', '--format=' + SACCT_FIELDS]) if arrays else SACCT_FIELDS.replace(',', '|') + '|\n'
    records = allocation_rows(raw)
    states = {str(root): accounting(ledger, records) for root, ledger in ledgers.items()}
    for root in ledgers:
        (root / 'accounting.psv').write_text(raw)
    return dict(spent_seconds=sum(state['spent_seconds'] for state in states.values()),
                reserved_seconds=sum(state['reserved_seconds'] for state in states.values()),
                batch_states=states, array_ids=sorted(arrays), pilot_allocation_cpu_seconds=scope['pilot_allocation_cpu_seconds'])


def archived_snapshot(root, plan):
    root = Path(root)
    scope = load_plan(root / 'scope.json', plan['budget_scope_sha256'])
    if scope['authorization_sha256'] != plan['authorization_sha256']:
        raise ValueError('archived accounting scope belongs to another registration')
    if scope['pilot_allocation_cpu_seconds'] != plan['pilot_allocation_cpu_seconds']:
        raise ValueError('archived pilot accounting differs from the execution plan')
    base = Path(plan['budget_scope']).parent
    paths = [Path(name) for name in scope['batch_roots']]
    if len(paths) != len(set(paths)) or any(path.parent != base for path in paths):
        raise ValueError('invalid archived accounting scope roots')
    states, arrays = {}, set()
    for path in paths:
        batch = root / path.name
        ledger = json.loads((batch / 'ledger.json').read_text()) if batch.exists() else []
        records = allocation_rows((batch / 'accounting.psv').read_text()) if ledger else []
        known = {record['JobID']: record for record in records}
        for entry in ledger:
            identifier = entry.get('array_id')
            if not identifier or identifier in arrays:
                raise ValueError('unresolved or duplicate archived allocation')
            arrays.add(identifier)
            for index, _ in enumerate(entry['runs']):
                record = known.get(f'{identifier}_{index}')
                if record is None or record['State'].split()[0].split('+')[0] not in TERMINAL:
                    raise ValueError('archived allocation has missing or nonterminal accounting')
        states[path.name] = accounting(ledger, records)
    return dict(spent_seconds=sum(state['spent_seconds'] for state in states.values()),
                reserved_seconds=sum(state['reserved_seconds'] for state in states.values()),
                batch_states=states, array_ids=sorted(arrays),
                pilot_allocation_cpu_seconds=scope['pilot_allocation_cpu_seconds'])
