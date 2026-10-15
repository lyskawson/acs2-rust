import hashlib
import json
from pathlib import Path
import sys

from execution import APPROVED_SHA256, approved, authorization_manifest, batches, study_scope
from protocol import ROOT
from retry_batch import RECOVERY_SHA256, RETRY_SEEDS, approved_recovery

sys.path.append(str(ROOT / 'experiments/grid'))
from collect import complete_rows, validate_row
from grid_attempts import attempts, inspect_attempt, select_attempts, verify_artifacts
from grid_authorization import require_derivation
from grid_identity import compare, verify_references
from grid_protocol import TERMINAL, allocation_rows, load_plan
from grid_scope import archived_snapshot


def verify_snapshot(study, child):
    path = Path(study) / 'operations/manifest.json'
    manifest = load_plan(path, child['operations_manifest_sha256'])
    for name, digest in manifest['files'].items():
        file = path.parent / name
        if not file.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('operations file is outside its snapshot')
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f'operations snapshot changed: {name}')
    return manifest


def audit_batch(root, study):
    root, study = Path(root), Path(study)
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    require_derivation(plan, authorization_manifest(plan, study / 'approved-plan.json'))
    verify_snapshot(study, plan)
    ledger = json.loads((root / 'ledger.json').read_text())
    verify_attempt_scope(plan, ledger, study)
    known = {row['JobID']: row for row in allocation_rows((root / 'accounting.psv').read_text())}
    details, seconds, rows_checked, identities = [], 0, 0, 0
    resolver = reference_resolver(study, plan)
    for attempt in attempts(root, ledger):
        config = next(item for item in plan['configurations'] if item['id'] == attempt['configuration_id'])
        item, rows = inspect_attempt(attempt, config)
        if 'completion' in item:
            verify_artifacts(attempt['path'], item['completion'])
            verify_launch(attempt['path'], plan, attempt)
        record = known.get(f"{attempt['array_id']}_{attempt['task_id']}")
        if record is None or record['State'].split()[0].split('+')[0] not in TERMINAL:
            raise ValueError('audit requires terminal accounting for every attempt')
        if record['Partition'] != 'lem-cpu-normal' or record['AllocCPUS'] != '1':
            raise ValueError('allocation partition or CPU count differs')
        seconds += int(record['CPUTimeRAW'])
        if [row['nominal_step'] for row in rows] != config['targets'][:len(rows)]:
            raise ValueError('attempt is not a registered checkpoint prefix')
        for row in rows:
            validate_row(row, config, attempt['agent'], attempt['seed'], plan['measurement_commit'], row['nominal_step'])
            if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model']:
                raise ValueError('VCP row hardware or preset differs')
        rows_checked += len(rows)
        if item['status'] == 'complete':
            checks = verify_references(plan, attempt, rows, resolver)
            if json.loads((attempt['path'] / 'identity-checks.json').read_text()) != checks:
                raise ValueError('VCP node identity record differs from independent verification')
            identities += sum(check['rows'] for check in checks)
        details.append(dict(attempt_id=attempt['attempt_id'], agent=attempt['agent'], seed=attempt['seed'],
                            status=item['status'], rows=len(rows), failure=item.get('completion', {}).get('failure'),
                            allocation_cpu_seconds=int(record['CPUTimeRAW']), allocation_state=record['State']))
    return dict(attempts=len(details), rows_checked=rows_checked, pilot_identity_rows=identities,
                allocation_cpu_seconds=seconds, details=details)


def reference_resolver(root, plan):
    declared = Path(plan['authorization_plan']).parent
    def resolve(value):
        path = Path(value)
        if not path.is_relative_to(declared):
            raise ValueError('identity reference is outside this VCP study')
        return Path(root) / path.relative_to(declared)
    return resolve


def verify_attempt_scope(plan, ledger, study):
    config, agent = plan['configurations'][0], plan['agents'][0]
    if not ledger or len(plan['configurations']) != 1 or len(plan['agents']) != 1:
        raise ValueError('VCP batch must have exactly one registered configuration and arm')
    for index, entry in enumerate(ledger):
        if index == 0:
            seeds, limit, suffix = plan['seeds'], 60 * config['limits_minutes'][agent], 'a01'
        elif index == 1 and config['id'] == 'maze4_c5_pairs' and agent == 'acs2vcp_k2':
            recovery = Path(study) / 'recovery'
            approved_recovery(recovery / 'approved-recovery.json', recovery / 'approval.json')
            if entry.get('recovery_sha256') != RECOVERY_SHA256:
                raise ValueError('retry ledger lacks the approved operational recovery')
            seeds, limit, suffix = RETRY_SEEDS, 600, 'a02'
        else:
            raise ValueError('additional VCP attempts have no approval')
        expected = [(config['id'], agent, seed, limit, f"{config['id']}_{agent}_s{seed}_{suffix}") for seed in seeds]
        actual = [(run['configuration_id'], run['agent'], run['seed'], run['limit_seconds'], run['attempt_id']) for run in entry['runs']]
        if actual != expected:
            raise ValueError('attempt scope or hard limits differ from the approved registration')


def verify_launch(path, plan, attempt):
    launch = json.loads((path / 'launch.json').read_text())
    expected = dict(commit=plan['measurement_commit'], binary_sha256=plan['binary_sha256'],
                    partition='lem-cpu-normal', array_job_id=str(attempt['array_id']), array_task_id=str(attempt['task_id']))
    if any(launch.get(name) != value for name, value in expected.items()):
        raise ValueError('recorded VCP launch identity differs from the frozen measurement')
    manifest = launch['manifest']
    config = next(item for item in plan['configurations'] if item['id'] == attempt['configuration_id'])
    expected = dict(configuration=config, agent=attempt['agent'], seed=attempt['seed'], attempt_id=attempt['attempt_id'],
                    measurement_commit=plan['measurement_commit'], expected_binary_sha256=plan['binary_sha256'],
                    expected_preset=plan['preset'], allocation_limit_seconds=attempt['limit_seconds'])
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError('resolved VCP launch differs from its registered attempt')


def verify_batch(root, study):
    root, study = Path(root), Path(study)
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    require_derivation(plan, authorization_manifest(plan, study / 'approved-plan.json'))
    verify_snapshot(study, plan)
    ledger = json.loads((root / 'ledger.json').read_text())
    verify_attempt_scope(plan, ledger, study)
    records = allocation_rows((root / 'accounting.psv').read_text())
    known = {row['JobID']: row for row in records}
    entries, seconds, identities, retry_identities = {}, 0, 0, 0
    resolver = reference_resolver(study, plan)
    for config in plan['configurations']:
        selected, inventory = select_attempts(root, ledger, config)
        if set(selected) != {(agent, seed) for agent in plan['agents'] for seed in plan['seeds']}:
            raise ValueError('incomplete registered VCP batch')
        for attempt in inventory:
            record = known.get(f"{attempt['array_id']}_{attempt['task_id']}")
            if record is None or record['State'].split()[0].split('+')[0] not in TERMINAL:
                raise ValueError('VCP allocation has missing or nonterminal accounting')
            if record['Partition'] != 'lem-cpu-normal' or record['AllocCPUS'] != '1':
                raise ValueError('VCP allocation partition or CPU count differs')
            seconds += int(record['CPUTimeRAW'])
            if attempt['selected'] and (record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0'):
                raise ValueError('selected VCP attempt did not complete successfully')
        for (agent, seed), attempt in selected.items():
            path = attempt['path']
            verify_launch(path, plan, attempt)
            rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
            if [row['nominal_step'] for row in rows] != config['targets']:
                raise ValueError('VCP checkpoint sequence differs')
            for row in rows:
                validate_row(row, config, agent, seed, plan['measurement_commit'], row['nominal_step'])
                if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model']:
                    raise ValueError('VCP row hardware or preset differs')
            for previous in inventory:
                if previous['agent'] == agent and previous['seed'] == seed and previous['status'] != 'complete':
                    prefix, _ = complete_rows(root / previous['directory'] / 'rows.jsonl')
                    if prefix:
                        checked = compare(rows, prefix, [seed], [row['nominal_step'] for row in prefix],
                                          ['training', 'validation', 'test'])
                        retry_identities += checked['rows']
            checks = verify_references(plan, dict(configuration_id=config['id'], agent=agent, seed=seed), rows, resolver)
            if json.loads((path / 'identity-checks.json').read_text()) != checks:
                raise ValueError('VCP node identity record differs from independent verification')
            identities += sum(check['rows'] for check in checks)
            entries[config['id'], agent, seed] = dict(path=path, targets=config['targets'], task=config['task'],
                array_id=attempt['array_id'], task_id=attempt['task_id'],
                allocation_cpu_seconds=int(known[f"{attempt['array_id']}_{attempt['task_id']}"]['CPUTimeRAW']))
    return entries, dict(runs=len(entries), allocation_cpu_seconds=seconds, pilot_identity_rows=identities,
                         retry_prefix_identity_rows=retry_identities)


def verify_study(root):
    root = Path(root)
    if (root / 'STOP.json').exists():
        raise ValueError('STOP blocks VCP analysis')
    plan = approved(root / 'approved-plan.json')
    declared = Path(json.loads((root / 'scope.json').read_text())['batch_roots'][0]).parent
    if json.loads((root / 'scope.json').read_text()) != study_scope(plan, declared):
        raise ValueError('VCP accounting scope differs')
    entries, details = {}, {}
    for config, agent in batches(plan):
        name = config['id'] + '_' + agent
        audit = audit_batch(root / name, root)
        values, checked = verify_batch(root / name, root)
        entries.update(values)
        details[name] = dict(checked, attempts=audit['attempts'], raw_rows=audit['rows_checked'],
                             incomplete_attempts=[item for item in audit['details'] if item['status'] != 'complete'])
    first = root / (batches(plan)[0][0]['id'] + '_' + batches(plan)[0][1])
    child = load_plan(first / 'plan.json', (first / 'plan.sha256').read_text().strip())
    state = archived_snapshot(root, child)
    total = state['spent_seconds'] + state['pilot_allocation_cpu_seconds']
    if total > 110 * 3600 or state['reserved_seconds'] != 0 or len(entries) != 480:
        raise ValueError('VCP cumulative accounting or run scope differs')
    if state['spent_seconds'] != sum(item['allocation_cpu_seconds'] for item in details.values()):
        raise ValueError('VCP allocation accounting disagrees across roots')
    return entries, dict(runs=len(entries), allocated_cpu_seconds=total, pilot_allocated_cpu_seconds=16041,
                         grid_allocated_cpu_seconds=state['spent_seconds'],
                         pilot_identity_rows=sum(item['pilot_identity_rows'] for item in details.values()),
                         retry_prefix_identity_rows=sum(item['retry_prefix_identity_rows'] for item in details.values()),
                         batches=details)
