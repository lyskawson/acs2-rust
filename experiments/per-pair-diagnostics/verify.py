import argparse
import hashlib
import json
from pathlib import Path

from archive import audit, catalog
from execution import identity_references, replication, scope
from prepare_execution import approved, APPROVED_SHA256, OPERATIONS_COMMIT
from grid_authorization import require_derivation
from grid_dispatch import require_prefix
from grid_protocol import allocation_rows, atomic_json, load_plan
from submit import shared_reservation


def verify_admissions(events, ledgers, accounting, declared, limit=15):
    expected = []
    for name in ('pilot', 'replication'):
        for entry in ledgers[name]:
            expected.append((name, entry))
    if len(events) != len(expected):
        raise ValueError('every allocation needs exactly one admission')
    spent = 0
    for event, (name, allocation) in zip(events, expected):
        new = sum(run['limit_seconds'] for run in allocation['runs'])
        if (event['root'] != str(Path(declared) / name) or not event['test_only'] or not event['grant']
                or event['balance']['spent_seconds'] != spent or event['balance']['reserved_seconds'] != 0
                or event['new_limit_seconds'] != new or event['bound_seconds'] != spent + new
                or event['bound_seconds'] > (4 if name == 'pilot' else limit) * 3600
                or float(event['grant'].strip()) + shared_reservation(event['queue']) + new > 15000 * 3600
                or not 20 <= len(allocation['runs']) <= 40):
            raise ValueError('admission differs from cumulative cost, grant or hard reservations')
        records = [accounting[f"{allocation['array_id']}_{index}"] for index in range(len(allocation['runs']))]
        spent += sum(int(record['CPUTimeRAW']) for record in records)
    return spent


def verify(root):
    root = Path(root)
    plan, approval = approved()
    if load_plan(root / 'approved-plan.json', APPROVED_SHA256) != plan or json.loads((root / 'approval.json').read_text()) != approval:
        raise ValueError('archive differs from user-approved plan or limit')
    frozen = root.parent / 'pilot-frozen'
    inventory = json.loads((frozen / 'raw-checksums.json').read_text())
    for name, digest in inventory['files'].items():
        if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen pilot changed: ' + name)
        active = root / name
        if name == 'admission-decisions.jsonl':
            if not active.read_bytes().startswith((frozen / name).read_bytes()):
                raise ValueError('shared admissions lost the original pilot prefix')
        elif hashlib.sha256(active.read_bytes()).hexdigest() != digest:
            raise ValueError('active pilot file changed: ' + name)
    pilot, pilot_summary = audit(frozen, retain_rows=False)
    if pilot_summary != json.loads((frozen / 'pilot-summary.json').read_text()):
        raise ValueError('recomputed pilot summary differs')
    active, _ = catalog(root / 'pilot', root, retain_rows=False)
    if {key: item['sha256'] for key, item in active.items()} != {key: item['sha256'] for key, item in pilot.items()}:
        raise ValueError('active and frozen pilot rows differ')
    declared = Path(json.loads((root / 'pilot-registration.json').read_text())['destination'])
    scope_path = root / 'scope.json'
    if json.loads(scope_path.read_text()) != scope(APPROVED_SHA256, declared):
        raise ValueError('one cumulative accounting scope required')
    manifest = load_plan(root / 'replication/plan.json', (root / 'replication/plan.sha256').read_text().strip())
    expected = replication(plan, 15)
    expected.update(identity_references=identity_references(expected, root, declared),
        budget_scope=str(declared / 'scope.json'), budget_scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
        stop_path=str(declared / 'STOP.json'))
    require_derivation(manifest, expected)
    if manifest['authorization_sha256'] != APPROVED_SHA256 or manifest['operations_commit'] != OPERATIONS_COMMIT:
        raise ValueError('execution authorization or pinned operations differ')
    ops = root / 'execution-operations'
    pinned = json.loads((root / 'operations/manifest.json').read_text())
    actual = load_plan(ops / 'manifest.json', manifest['operations_manifest_sha256'])
    if actual['commit'] != OPERATIONS_COMMIT or actual['base_manifest_sha256'] != hashlib.sha256((root / 'operations/manifest.json').read_bytes()).hexdigest():
        raise ValueError('operations snapshot is not derived from the approved original')
    for name, digest in actual['files'].items():
        if not (ops / name).resolve().is_relative_to(ops.resolve()) or hashlib.sha256((ops / name).read_bytes()).hexdigest() != digest:
            raise ValueError('execution operation changed')
    if any(actual['files'].get(name) != digest for name, digest in pinned['files'].items()):
        raise ValueError('a pinned operation was replaced')
    if set(actual['files']) - set(pinned['files']) != {'experiments/per-pair-diagnostics/plan.json', 'experiments/per-pair-diagnostics/approval.json'}:
        raise ValueError('unexpected additions to pinned operations')
    more, state = catalog(root / 'replication', root, retain_rows=False)
    ledgers = {name: json.loads((root / name / 'ledger.json').read_text()) for name in ('pilot', 'replication')}
    require_prefix(root / 'replication', manifest, ledgers['replication'])
    if len(ledgers['replication']) != 6 or len(more) != 120:
        raise ValueError('exactly six registered arrays and 120 replications required')
    records = allocation_rows((root / 'replication/accounting.psv').read_text())
    accounts = {record['JobID']: record for record in records}
    required = {f"{entry['array_id']}_{index}" for ledger in ledgers.values() for entry in ledger for index in range(len(entry['runs']))}
    if set(accounts) != required:
        raise ValueError('accounting has missing or unregistered allocations')
    events = [json.loads(line) for line in (root / 'admission-decisions.jsonl').read_text().splitlines()]
    spent = verify_admissions(events, ledgers, accounts, declared)
    if spent != pilot_summary['allocation_cpu_seconds'] + state['spent_seconds'] or spent > 15 * 3600:
        raise ValueError('cumulative allocation cost differs or exceeds approval')
    entries = dict(active)
    entries.update(more)
    if len(entries) != 150:
        raise ValueError('the five-seed analysis grid is incomplete')
    summary = dict(verified=True, runs=150, replication_runs=120, allocations=150, failures=0, retries=0,
        pilot_cpu_seconds=pilot_summary['allocation_cpu_seconds'], replication_cpu_seconds=state['spent_seconds'],
        allocation_cpu_seconds=spent, reserved_seconds=state['reserved_seconds'],
        rows=sum(item['row_count'] for item in entries.values()), identity_rows=sum(item['identity_rows'] for item in entries.values()),
        pair_records=sum(item['pair_records'] for item in entries.values()), plan_sha256=APPROVED_SHA256,
        runs_index=[dict(configuration_id=key[0], arm=key[1], seed=key[2], file=str(item['path'].relative_to(root) / 'rows.jsonl'),
                        sha256=item['sha256'], row_file_bytes=item['row_file_bytes'], rows=item['row_count'], cost=item['cost'])
                    for key, item in sorted(entries.items())])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = verify(args.root)
    atomic_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'runs_index'}))
