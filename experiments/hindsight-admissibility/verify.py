import json
from collections import Counter
from pathlib import Path

from artifacts import reference_resolver, verify_launch, verify_snapshot
from execution import PILOT_SECONDS, approved, authorization_manifest, batches, study_scope
from collect import validate_row
from grid_attempts import attempts, inspect_attempt, select_attempts, verify_artifacts
from grid_authorization import require_derivation
from grid_identity import verify_references
from grid_protocol import TERMINAL, allocation_rows, load_plan
from grid_scope import archived_snapshot


def verify_attempt_scope(plan, ledger):
    if len(ledger) != 1 or len(plan['configurations']) != 1 or len(plan['agents']) != 1:
        raise ValueError('each registered batch requires one first-attempt twenty-seed allocation')
    config, agent = plan['configurations'][0], plan['agents'][0]
    expected = [(config['id'], agent, seed, 60 * config['limits_minutes'][agent],
                 f"{config['id']}_{agent}_s{seed}_a01") for seed in plan['seeds']]
    actual = [(run['configuration_id'], run['agent'], run['seed'], run['limit_seconds'], run['attempt_id'])
              for run in ledger[0]['runs']]
    if actual != expected:
        raise ValueError('attempt scope or hard reservations differ from the registered batch')
    receipt = ledger[0]
    if str(receipt['array_id']) != receipt['submission_response'].split(';')[0]:
        raise ValueError('allocation differs from its submission receipt')
    required = ['--partition=lem-cpu-normal', '--cpus-per-task=1', '--no-requeue', '--array=0-19%20',
                '--time=' + str(config['limits_minutes'][agent])]
    if any(option not in receipt['command'] for option in required) or any('nodelist' in option for option in receipt['command']):
        raise ValueError('scheduler resources differ from the approved execution')


def verify_batch(root, study):
    root, study = Path(root), Path(study)
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    require_derivation(plan, authorization_manifest(plan, study / 'approved-plan.json'))
    operations = verify_snapshot(study, plan)
    if operations['commit'] != plan['operations_commit']:
        raise ValueError('operations source differs from the frozen batch')
    ledger = json.loads((root / 'ledger.json').read_text())
    verify_attempt_scope(plan, ledger)
    records = allocation_rows((root / 'accounting.psv').read_text())
    known = {row['JobID']: row for row in records}
    config = plan['configurations'][0]
    selected, _ = select_attempts(root, ledger, config)
    if set(selected) != {(agent, seed) for agent in plan['agents'] for seed in plan['seeds']}:
        raise ValueError('batch is missing a complete registered trajectory')
    entries, seconds, identities, checked_rows = {}, 0, 0, 0
    resolver = reference_resolver(study, plan)
    for attempt in attempts(root, ledger):
        record = known.get(f"{attempt['array_id']}_{attempt['task_id']}")
        if record is None or record['State'].split()[0].split('+')[0] not in TERMINAL:
            raise ValueError('every attempt requires terminal allocation accounting')
        if record['Partition'] != 'lem-cpu-normal' or record['AllocCPUS'] != '1':
            raise ValueError('wrong allocation partition or CPU count')
        seconds += int(record['CPUTimeRAW'])
        item, rows = inspect_attempt(attempt, config)
        if item['status'] != 'complete' or record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0':
            raise ValueError('an unsuccessful attempt blocks analysis')
        verify_artifacts(attempt['path'], item['completion'])
        verify_launch(attempt['path'], plan, attempt)
        if [row['nominal_step'] for row in rows] != config['targets']:
            raise ValueError('measurement points differ from the registered sequence')
        for row in rows:
            validate_row(row, config, attempt['agent'], attempt['seed'], plan['measurement_commit'], row['nominal_step'])
            if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model']:
                raise ValueError('measurement preset or hardware differs')
            if row['match_formations_train'] != 10 * row['actual_steps']:
                raise ValueError('full replay match volume differs')
        checked_rows += len(rows)
        checks = verify_references(plan, attempt, rows, resolver)
        if json.loads((attempt['path'] / 'identity-checks.json').read_text()) != checks:
            raise ValueError('node identity result differs from independent bit comparison')
        identities += sum(check['rows'] for check in checks)
        entries[config['id'], attempt['agent'], attempt['seed']] = dict(path=attempt['path'], task=config['task'],
            targets=config['targets'], allocation_cpu_seconds=int(record['CPUTimeRAW']),
            array_id=attempt['array_id'], task_id=attempt['task_id'])
    return entries, dict(runs=len(entries), attempts=len(ledger[0]['runs']), rows=checked_rows,
                         allocation_cpu_seconds=seconds, pilot_identity_rows=identities)


def verify_admissions(root, scope):
    admissions = [json.loads(line) for line in (Path(root) / 'admission-decisions.jsonl').read_text().splitlines()]
    if Counter(item['root'] for item in admissions) != Counter(scope['batch_roots']):
        raise ValueError('every registered root must have exactly one admission')
    for event in admissions:
        expected = PILOT_SECONDS + event['balance']['spent_seconds'] + event['balance']['reserved_seconds'] + event['remaining_limit_seconds']
        if event['bound_seconds'] != expected or expected > 30 * 3600 or event['new_limit_seconds'] != event['remaining_limit_seconds']:
            raise ValueError('admission violates the approved cumulative allocation bound')


def verify_study(root):
    root = Path(root)
    if (root / 'STOP.json').exists():
        raise ValueError('shared STOP blocks analysis')
    plan = approved(root / 'approved-plan.json')
    scope = json.loads((root / 'scope.json').read_text())
    declared = Path(scope['batch_roots'][0]).parent
    if scope != study_scope(plan, declared):
        raise ValueError('accounting scope differs from all twelve approved roots')
    verify_admissions(root, scope)
    entries, details, child = {}, {}, None
    for config, agent in batches(plan):
        name = config['id'] + '_' + agent
        values, summary = verify_batch(root / name, root)
        entries.update(values)
        details[name] = summary
        if child is None:
            child = load_plan(root / name / 'plan.json', (root / name / 'plan.sha256').read_text().strip())
    state = archived_snapshot(root, child)
    seconds = sum(item['allocation_cpu_seconds'] for item in details.values())
    identities = sum(item['pilot_identity_rows'] for item in details.values())
    if (state['spent_seconds'] != seconds or state['reserved_seconds'] or seconds + PILOT_SECONDS > 30 * 3600
            or len(entries) != 240 or identities != plan['cost_pilot']['rows']):
        raise ValueError('completed grid, identity coverage or allocation accounting differs')
    return entries, dict(runs=len(entries), attempts=sum(item['attempts'] for item in details.values()),
        rows=sum(item['rows'] for item in details.values()), grid_allocated_cpu_seconds=seconds,
        pilot_allocated_cpu_seconds=PILOT_SECONDS, allocated_cpu_seconds=seconds + PILOT_SECONDS,
        pilot_identity_rows=identities, batches=details)
