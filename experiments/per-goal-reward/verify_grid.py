import argparse
import json
import math
from pathlib import Path

from protocol import NEW_ARMS, configurations
from execution_grid import (APPROVED_SHA256, approved, batch_name, checksum, manifest_for, scope, validate_inputs)
from archive import audit, deduplicate, models
from analyze import checked_path
from analysis import attach
from schema import validate_sequence
from grid_attempts import attempts, inspect_attempt
from grid_authorization import require_derivation
from grid_protocol import allocation_rows, atomic_json, load_plan, TERMINAL
from grid_scope import snapshot
from submit import execute, shared_reservation


def update_accounting(root):
    root = Path(root)
    from dispatch import manifest
    value = snapshot(manifest(root, 0), execute)
    atomic_json(root / 'continuation/final-accounting.json', value)
    return value


def validate_aggregates(block, truth):
    records = block['starts']
    for field, expected in [('success', block['success']),
            ('estimated_first_action_value', block['value_diagnostics']['all']['mean_first_action_estimate']),
            ('discounted_return', block['value_diagnostics']['all']['mean_discounted_return'])]:
        measured = math.fsum(record['weight'] * record[field] for record in records)
        if not math.isclose(measured, expected, rel_tol=2e-12, abs_tol=1e-12):
            raise ValueError('pair records disagree with evaluation aggregates')
    attach(records, truth)


def verify(root):
    root = Path(root)
    registered, approval = approved(root / 'approved-plan.json')
    validate_inputs(root, registered)
    preparation = json.loads((root / 'execution-preparation.json').read_text())
    if preparation['measurement_rebuilt'] is not False or checksum(root / 'pilot-inventory.json') != preparation['pilot_inventory_sha256']:
        raise ValueError('pilot inventory or unchanged binary declaration differs')
    for name, digest in json.loads((root / 'pilot-inventory.json').read_text()).items():
        checked_path(root, dict(file=name, sha256=digest))
    pilot = audit(root)
    truth = models(root)
    operations = json.loads((root / 'execution-operations/manifest.json').read_text())
    if operations['commit'] != preparation['operations_commit']:
        raise ValueError('frozen execution operations commit differs')
    for name, digest in operations['files'].items():
        checked_path(root / 'execution-operations', dict(file=name, sha256=digest))
    records = deduplicate(allocation_rows((root / 'continuation/shared-accounting.psv').read_text()))
    by_job = {row['JobID']: row for row in records}
    if len(by_job) != len(records):
        raise ValueError('accounting job identities are not unique')
    jobs = {f"{entry['array_id']}_{index}" for entry in json.loads((root / 'pilot/ledger.json').read_text())
            for index, _ in enumerate(entry['runs'])}
    runs = list(pilot['runs'])
    receipts = [json.loads(line) for line in (root / 'continuation/admission-decisions.jsonl').read_text().splitlines()]
    if len(receipts) != len(registered['forecast']['arrays']):
        raise ValueError('one admission per registered array is required')
    failed = pilot['failed_attempts']
    allocations = pilot['attempts']
    for index, array in enumerate(registered['forecast']['arrays']):
        batch = root / batch_name(array)
        definition = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
        declared = Path(definition['authorization_plan']).parent
        if json.loads((root / 'continuation/scope.json').read_text()) != scope(declared, registered):
            raise ValueError('cumulative accounting scope changed')
        require_derivation(definition, dict(manifest_for(registered, index),
            budget_scope=str(declared / 'continuation/scope.json'), budget_scope_sha256=checksum(root / 'continuation/scope.json'),
            stop_path=str(declared / 'continuation/STOP.json')))
        if (definition['authorization_sha256'] != APPROVED_SHA256
                or definition['operations_commit'] != preparation['operations_commit']
                or definition['operations_manifest_sha256'] != checksum(root / 'execution-operations/manifest.json')
                or json.loads((batch / 'approval.json').read_text()) != approval):
            raise ValueError('batch provenance differs from approval')
        ledger = json.loads((batch / 'ledger.json').read_text())
        if len(ledger) != 1:
            raise ValueError('unexpected attempts require explicit reconciliation')
        event = next((event for event in receipts if event['forecast_index'] == index), None)
        entry = ledger[0]
        new = sum(run['limit_seconds'] for run in entry['runs'])
        if (event is None or event['root'] != str(declared / batch.name)
                or event['new_limit_seconds'] != new or new != array['reservation_cpu_seconds']
                or event['bound_seconds'] != event['balance']['spent_seconds'] + event['balance']['reserved_seconds'] + new
                or event['bound_seconds'] > 29 * 3600 or not event['test_only']
                or float(event['grant'].strip()) + shared_reservation(event['queue']) + new > 15000 * 3600):
            raise ValueError('admission lacks exact budget, grant or scheduler evidence')
        selected = set()
        inventory = attempts(batch, ledger)
        allocations += len(inventory)
        for attempt in inventory:
            job = f"{attempt['array_id']}_{attempt['task_id']}"
            if job in jobs:
                raise ValueError('one allocation appears in multiple ledgers')
            jobs.add(job)
            account = by_job.get(job)
            if account is None or account['State'].split()[0].split('+')[0] not in TERMINAL:
                raise ValueError('every attempt requires terminal accounting')
            config = next(config for config in definition['configurations'] if config['id'] == attempt['configuration_id'])
            status, rows = inspect_attempt(attempt, config)
            if status['status'] != 'complete':
                failed += 1
                continue
            key = attempt['configuration_id'], attempt['agent'], attempt['seed']
            if key in selected:
                raise ValueError('duplicate complete registered run')
            selected.add(key)
            if (account['State'] != 'COMPLETED' or account['ExitCode'] != '0:0' or account['AllocCPUS'] != '1'
                    or account['Partition'] != 'lem-cpu-normal' or attempt['limit_seconds'] != array['limit_minutes'] * 60):
                raise ValueError('allocation differs from the registered array')
            launch = json.loads((attempt['path'] / 'launch.json').read_text())
            if (launch['binary_sha256'] != registered['binary_sha256'] or launch['commit'] != registered['measurement_commit']
                    or launch['manifest']['configuration'] != config
                    or ('--per-start' in launch['command']) != (array['kind'] == 'per_pair')
                    or launch['array_job_id'] != str(attempt['array_id']) or launch['array_task_id'] != str(attempt['task_id'])):
                raise ValueError('launch source, recording or allocation identity differs')
            validate_sequence(rows, config, attempt['agent'], attempt['seed'], registered['measurement_commit'])
            for row in rows:
                if row['cpu_model'] != definition['cpu_model']:
                    raise ValueError('measurement hardware differs')
                if array['kind'] == 'per_pair':
                    for block in row['evaluation_blocks'].values():
                        validate_aggregates(block, truth[config['task']])
            completion = json.loads((attempt['path'] / 'completion.json').read_text())
            runs.append(dict(configuration_id=config['id'], agent=attempt['agent'], seed=attempt['seed'],
                rows=len(rows), pairs=sum(len(block['starts'] or []) for row in rows for block in row['evaluation_blocks'].values()),
                file=str((attempt['path'] / 'rows.jsonl').relative_to(root)), sha256=checksum(attempt['path'] / 'rows.jsonl'),
                cost=dict(allocation_cpu_seconds=int(account['CPUTimeRAW']), body_wall_seconds=completion['elapsed_seconds'],
                    process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'], peak_rss_kib=completion['peak_rss_kib'])))
        if selected != {(config['id'], array['agent'], seed) for config in definition['configurations'] for seed in array['seeds']}:
            raise ValueError('missing or additional registered replications')
    if set(by_job) != jobs:
        raise ValueError('unmapped allocation accounting')
    spent = sum(float(row['CPUTimeRAW']) for row in records)
    if spent > 29 * 3600:
        raise ValueError('cumulative CPU-hour limit exceeded')
    if {(run['configuration_id'], run['agent'], run['seed']) for run in runs} != {
            (config['id'], arm, seed) for config in configurations() for arm in NEW_ARMS for seed in range(42, 62)}:
        raise ValueError('the complete registered twenty-seed grid is required')
    return dict(verified=True, plan_sha256=APPROVED_SHA256, runs_index=runs, rows=sum(run['rows'] for run in runs),
        pairs_checked=sum(run['pairs'] for run in runs), allocation_cpu_seconds=spent, reserved_seconds=0,
        allocations=allocations, failed_attempts=failed, phase_cpu_hour_limit=29, measurement_rebuilt=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--update-accounting', action='store_true')
    args = parser.parse_args()
    if args.update_accounting:
        print(json.dumps(update_accounting(args.root)))
    else:
        result = verify(args.root)
        atomic_json(Path(args.root) / 'verification.json', result)
        print(json.dumps({key: value for key, value in result.items() if key != 'runs_index'}))
