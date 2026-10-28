import argparse
import hashlib
import json
from pathlib import Path

from approved_execution import APPROVED_SHA256, approved, authorization_manifest
from archive import audit, catalog
from execution import choose, registered_references, scope
from protocol import pilot_roots
from grid_attempts import attempts, inspect_attempt
from grid_authorization import require_derivation
from grid_identity import compare, field_bits
from grid_protocol import load_plan, require_budget
from grid_scope import archived_snapshot
from grid_dispatch import require_prefix, recovery_arrays


def verify_manifest(root, batch):
    root, batch = Path(root), Path(batch)
    manifest = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    require_derivation(manifest, authorization_manifest(manifest, root / 'approved-plan.json'))
    declared = Path(manifest['authorization_plan']).parent
    expected_scope = scope(APPROVED_SHA256, declared)
    if json.loads((root / 'scope.json').read_text()) != expected_scope:
        raise ValueError('scope differs from all thirteen registered roots')
    operations = load_plan(root / 'execution-operations/manifest.json', manifest['operations_manifest_sha256'])
    for name, digest in operations['files'].items():
        path = root / 'execution-operations' / name
        if not path.resolve().is_relative_to((root / 'execution-operations').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('execution operation snapshot changed')
    expected = dict(authorization_plan=str(declared / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
        budget_scope=str(declared / 'scope.json'), budget_scope_sha256=hashlib.sha256((root / 'scope.json').read_bytes()).hexdigest(),
        stop_path=str(declared / 'STOP.json'), operations_manifest=str(declared / 'execution-operations/manifest.json'),
        operations_commit=operations['commit'], job_name='tu-hindsight-proportion-' + manifest['purpose'])
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('execution transport, STOP or accounting differs from approved derivation')
    if (batch / 'approval.json').read_bytes() != (root / 'approval.json').read_bytes():
        raise ValueError('stream approval differs')
    return manifest


def verify_admissions(root, plan, manifests):
    root = Path(root)
    records = [json.loads(line) for line in (root / 'admission-decisions.jsonl').read_text().splitlines()]
    ledgers = {}
    declared = None
    for name in pilot_roots() + list(manifests):
        batch = root / name
        ledger = json.loads((batch / 'ledger.json').read_text())
        manifest = manifests.get(name) or json.loads((batch / 'plan.json').read_text())
        declared = Path(manifest['authorization_plan']).parent
        ledgers[str(declared / name)] = ledger
    expected_keys = {(name, index) for name, ledger in ledgers.items() for index in range(len(ledger))}
    observed = set()
    for event in records:
        key = event['root'], event['assignment']
        if key not in expected_keys or key in observed:
            raise ValueError('missing, duplicate or unregistered allocation admission')
        observed.add(key)
        entry = ledgers[key[0]][key[1]]
        new = sum(run['limit_seconds'] for run in entry['runs'])
        balance = event['balance']
        bound = balance['spent_seconds'] + balance['reserved_seconds'] + new
        is_pilot = Path(key[0]).name in pilot_roots()
        cap = 5 if is_pilot else 45
        if event['new_limit_seconds'] != new or event['bound_seconds'] != bound or bound > cap * 3600:
            raise ValueError('admission does not enforce the complete allocation bound')
        if not event.get('test_only') or not entry.get('test_only') or not entry.get('grant') or 'queue' not in entry:
            raise ValueError('admission is missing grant, queue or scheduler test-only evidence')
        if not is_pilot:
            pilot_cost = sum(value['spent_seconds'] for name, value in balance['batch_states'].items() if Path(name).name in pilot_roots())
            if pilot_cost != 4784:
                raise ValueError('pilot allocations were omitted or charged twice')
            active = [name for name, state in balance['batch_states'].items() if state['reserved_seconds'] > 0]
            if key[0] in active or len(active) >= 2:
                raise ValueError('lean admission exceeds two streams or overlaps its own previous array')
            prior = {item['array_id'] for ledger in ledgers.values() for item in ledger if item['submitted_unix'] < event['unix']}
            if set(balance['array_ids']) != prior:
                raise ValueError('admission scope omits or duplicates a prior array')
            if len(entry['runs']) not in range(20, 41) or not any(value.endswith('%20') for value in entry['command'] if value.startswith('--array=')):
                raise ValueError('lean allocation size or running-job cap differs')
            if any(value.startswith('--dependency=') or value.startswith('--nodelist=') for value in entry['command']):
                raise ValueError('lean arrays must not enqueue dependent arrays or pin a host')
    if observed != expected_keys:
        raise ValueError('not every allocation has one admission record')
    return dict(admissions=len(observed), pilot_admissions=5, new_admissions=len(observed) - 5)


def verify_pilot_inventory(root):
    root = Path(root)
    frozen = json.loads((root / 'checksums.json').read_text())
    for name, digest in frozen['files'].items():
        if name == 'admission-decisions.jsonl':
            prefix = (root / 'pilot-admission-decisions.jsonl').read_bytes()
            if hashlib.sha256(prefix).hexdigest() != digest or not (root / name).read_bytes().startswith(prefix):
                raise ValueError('pilot admission prefix changed')
        elif hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen pilot artifact changed: ' + name)
    return frozen


def verify_study(root):
    root = Path(root)
    plan = approved(root / 'approved-plan.json')
    frozen = verify_pilot_inventory(root)
    pilot, _, _ = audit(root)
    if pilot['allocation_cpu_seconds'] != 4784 or pilot['identity_rows'] != 49:
        raise ValueError('approved pilot accounting or identity scope differs')
    manifests = {}
    tuning_manifest = verify_manifest(root, root / 'tuning')
    manifests['tuning'] = tuning_manifest
    tuning, tuning_state = catalog(root / 'tuning')
    references = registered_references(plan, root)
    selection = choose(plan, tuning, references)
    if field_bits(json.loads((root / 'selections.json').read_text())) != field_bits(selection):
        raise ValueError('saved selection differs from complete validation tuning')
    proof = json.loads((root / 'selection-verification.json').read_text())
    if proof['plan_sha256'] != APPROVED_SHA256 or proof['selection_sha256'] != hashlib.sha256((root / 'selections.json').read_bytes()).hexdigest():
        raise ValueError('selection provenance differs')
    if proof['input_rows_sha256'] != {'|'.join(map(str, key)): entry['sha256'] for key, entry in tuning.items()}:
        raise ValueError('selection used different tuning inputs')
    manifests['control'] = verify_manifest(root, root / 'control')
    control, control_state = catalog(root / 'control')
    if len(tuning) != 120 or len(control) != 120:
        raise ValueError('incomplete tuning or dose control')
    entries = {}
    states = dict(tuning=tuning_state, control=control_state)
    for name, manifest in [('control', manifests['control'])]:
        for key, entry in control.items():
            config = next(config for config in manifest['configurations'] if config['id'] == key[0])
            entries[config['registered_id'], config['row_arm'], key[2]] = dict(entry, purpose='control', reused=False)
    for config in plan['configurations']:
        name = 'test-' + config['id']
        p = selection['selections'][config['id']]['proportion']
        if p == 2 / 3:
            if (root / name / 'ledger.json').exists() and json.loads((root / name / 'ledger.json').read_text()):
                raise ValueError('coincident two-thirds arm was rerun')
            continue
        manifests[name] = verify_manifest(root, root / name)
        selected, state = catalog(root / name)
        if len(selected) != 20:
            raise ValueError('conditional test requires twenty seeds')
        states[name] = state
        for key, entry in selected.items():
            item = manifests[name]['configurations'][0]
            entries[config['id'], item['row_arm'], key[2]] = dict(entry, purpose='test', reused=False)
    state = archived_snapshot(root, tuning_manifest)
    require_budget(0, state, 0, 45)
    if state['reserved_seconds'] or state['spent_seconds'] != 4784 + sum(value['spent_seconds'] for value in states.values()):
        raise ValueError('terminal scope costs do not reconcile with pilot and new purposes')
    admissions = verify_admissions(root, plan, manifests)
    recovered = recovery_arrays(root / 'recovery.json', tuning_manifest) if (root / 'recovery.json').exists() else set()
    identities, inventory, partial_identity_rows = [], [], 0
    purpose_cost = {'pilot': 4784, 'tuning': tuning_state['spent_seconds'], 'control': control_state['spent_seconds'],
                    'test': sum(value['spent_seconds'] for name, value in states.items() if name.startswith('test-'))}
    for name, manifest in manifests.items():
        ledger = json.loads((root / name / 'ledger.json').read_text())
        require_prefix(root / name, manifest, ledger)
        selected_entries, _ = catalog(root / name)
        for attempt in attempts(root / name, ledger):
            config = next(item for item in manifest['configurations'] if item['id'] == attempt['configuration_id'])
            status, partial = inspect_attempt(attempt, config)
            identity_path = attempt['path'] / 'identity-checks.json'
            if status['status'] != 'complete':
                if attempt['array_id'] not in recovered:
                    raise ValueError('failed attempt lacks diagnosed recovery evidence')
                completion_path = attempt['path'] / 'completion.json'
                if completion_path.exists():
                    completion = json.loads(completion_path.read_text())
                    for file, digest in completion['sha256'].items():
                        if hashlib.sha256((attempt['path'] / file).read_bytes()).hexdigest() != digest:
                            raise ValueError('failed attempt raw checksum changed')
                if partial:
                    current = selected_entries[config['id'], attempt['agent'], attempt['seed']]['rows']
                    compare(current, partial, [attempt['seed']], [row['nominal_step'] for row in partial], config['evaluation_blocks'])
                    partial_identity_rows += len(partial)
                records = []
            else:
                records = json.loads(identity_path.read_text())
            identities.extend(records)
            inventory.append(dict(root=name, attempt_id=attempt['attempt_id'], array_id=attempt['array_id'], task_id=attempt['task_id'],
                                  agent=attempt['agent'], seed=attempt['seed'], cause=attempt['cause']))
    summary = dict(plan_sha256=APPROVED_SHA256, allocated_cpu_seconds=int(state['spent_seconds']), reserved_cpu_seconds=0,
        new_complete_runs=len(tuning) + len(entries), new_attempts=len(inventory), pilot_attempts=30,
        purpose_allocation_cpu_seconds=purpose_cost, identity_comparison_rows=sum(item['rows'] for item in identities),
        identity_records=len(identities), two_thirds_tuning_identity_rows=sum(len(config['targets']) * 5 for config in plan['configurations']),
        pilot_identity_rows=49, frozen_pilot_files=len(frozen['files']), partial_retry_identity_rows=partial_identity_rows, accounting=state, admissions=admissions,
        attempt_inventory=inventory, selected_proportions={key: value['proportion'] for key, value in selection['selections'].items()})
    return entries, tuning, selection, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    _, _, _, summary = verify_study(args.root)
    print(json.dumps(summary))
