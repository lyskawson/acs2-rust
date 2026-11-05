import argparse
import hashlib
import json
import math
from pathlib import Path

from protocol import ARM, BLOCKS, configurations, pilot
from dose import shares
from random_rule_schema import probability
from execution import identity, scope
from analysis import attach, truth_index
from collect import validate_row
from grid_attempts import attempts, inspect_attempt
from grid_authorization import require_derivation
from grid_identity import field_bits
from grid_protocol import accounting, allocation_rows, atomic_json, load_plan, TERMINAL
from submit import shared_reservation


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, entry):
    path = root / entry['file']
    if not path.resolve().is_relative_to(root.resolve()) or checksum(path) != entry['sha256']:
        raise ValueError('input escaped its root or changed checksum')
    return path


def models(root):
    result = {}
    for entry in json.loads((root / 'model-index.json').read_text()):
        value = json.loads(checked_path(root, entry).read_text())
        if value['task'] != entry['task'] or entry['task'] in result:
            raise ValueError('duplicated or inconsistent model')
        result[entry['task']] = truth_index(value)
    if set(result) != {config['task'] for config in configurations()}:
        raise ValueError('six environment models required')
    return result


def deduplicate(records):
    unique = {}
    for record in records:
        key = record['JobIDRaw']
        if key in unique and record != unique[key]:
            raise ValueError('conflicting accounting rows for one allocation')
        unique[key] = record
    return list(unique.values())


def validate_aggregates(block):
    records = block['starts']
    for name, value in [
        ('success', math.fsum(record['weight'] * record['success'] for record in records)),
        ('mean_first_action_estimate', math.fsum(record['weight'] * record['estimated_first_action_value'] for record in records)),
        ('mean_discounted_return', math.fsum(record['weight'] * record['discounted_return'] for record in records))]:
        expected = block['success'] if name == 'success' else block['value_diagnostics']['all'][name]
        if not math.isclose(value, expected, rel_tol=2e-12, abs_tol=1e-12):
            raise ValueError('per-pair records disagree with block aggregates')


def audit(root):
    root = Path(root)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    digest = checksum(root / 'pilot-registration.json')
    declared = Path(registration['destination'])
    if registration['manifest'] != pilot(identity()):
        raise ValueError('pilot measurement differs from frozen registration')
    for name in ('reference-index', 'model-index'):
        if checksum(root / (name + '.json')) != registration[name + '_sha256']:
            raise ValueError('pilot input index changed')
    if json.loads((root / 'pilot-scope.json').read_text()) != scope(digest, declared):
        raise ValueError('pilot scope changed')
    for entry in json.loads((root / 'reference-index.json').read_text()):
        checked_path(root / 'references', entry)
    batch = root / 'pilot'
    plan = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    expected = dict(registration['manifest'], budget_scope=str(declared / 'pilot-scope.json'),
        budget_scope_sha256=checksum(root / 'pilot-scope.json'), stop_path=str(declared / 'STOP.json'))
    require_derivation(plan, expected)
    if plan['authorization_sha256'] != digest:
        raise ValueError('pilot authorization changed')
    ops = load_plan(root / 'operations/manifest.json', plan['operations_manifest_sha256'])
    for name, sha in ops['files'].items():
        path = root / 'operations' / name
        if not path.resolve().is_relative_to((root / 'operations').resolve()) or checksum(path) != sha:
            raise ValueError('frozen operations changed')
    ledger = json.loads((batch / 'ledger.json').read_text())
    records = deduplicate(allocation_rows((batch / 'accounting.psv').read_text()))
    inventory = attempts(batch, ledger)
    expected_keys = {(config['id'], ARM, 42) for config in configurations()}
    if any((entry['configuration_id'], entry['agent'], entry['seed']) not in expected_keys for entry in inventory):
        raise ValueError('unregistered pilot allocation')
    state = accounting(ledger, records)
    if state['reserved_seconds'] or state['spent_seconds'] > 7200:
        raise ValueError('unresolved reservations or exceeded pilot CPU-hour limit')
    by_id = {record['JobID']: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {f"{entry['array_id']}_{entry['task_id']}" for entry in inventory}:
        raise ValueError('unmapped or conflicting pilot allocation accounting')
    for entry in inventory:
        account = by_id.get(f"{entry['array_id']}_{entry['task_id']}")
        if account is None or account['State'].split()[0].split('+')[0] not in TERMINAL:
            raise ValueError('every attempt needs terminal accounting')
    truth = models(root)
    runs, selected = [], set()
    for entry in inventory:
        config = next(config for config in plan['configurations'] if config['id'] == entry['configuration_id'])
        inspected, rows = inspect_attempt(entry, config)
        if inspected['status'] != 'complete':
            continue
        key = entry['configuration_id'], entry['agent'], entry['seed']
        if key in selected:
            raise ValueError('multiple complete attempts require explicit reconciliation')
        selected.add(key)
        account = by_id[f"{entry['array_id']}_{entry['task_id']}"]
        if (account['State'] != 'COMPLETED' or account['ExitCode'] != '0:0' or account['AllocCPUS'] != '1'
                or account['Partition'] != 'lem-cpu-normal' or entry['limit_seconds'] != 1200):
            raise ValueError('allocation identity differs from the registered pilot')
        launch = json.loads((entry['path'] / 'launch.json').read_text())
        if (launch['binary_sha256'] != plan['binary_sha256'] or launch['commit'] != plan['measurement_commit']
                or launch['manifest']['configuration'] != config or '--per-start' not in launch['command']
                or launch['array_job_id'] != str(entry['array_id']) or launch['array_task_id'] != str(entry['task_id'])):
            raise ValueError('launch differs from frozen source or assignment')
        previous = None
        for row, target in zip(rows, config['targets']):
            validate_row(row, config, ARM, 42, plan['measurement_commit'], target)
            counts = row['replay_diagnostics']['reward_updates']
            if previous and any(counts[origin][key] < previous[origin][key] for origin in counts for key in counts[origin]):
                raise ValueError('cumulative reward counters decreased')
            previous = counts
            if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model']:
                raise ValueError('hardware or preset changed')
            for block in row['evaluation_blocks'].values():
                validate_aggregates(block)
                attach(block['starts'], truth[config['task']])
        completion = json.loads((entry['path'] / 'completion.json').read_text())
        dose = shares(rows[-1])['relabeled']
        if abs(dose - probability(config['task'])) > 0.01:
            raise ValueError('STOP: pilot relabeled dose exceeds tolerance: ' + config['task'])
        runs.append(dict(dose=dict(relabeled=dose, registered_probability=probability(config['task']), all=shares(rows[-1])['all']), configuration_id=config['id'], agent=ARM, seed=42, rows=len(rows),
            pairs=sum(len(block['starts']) for row in rows for block in row['evaluation_blocks'].values()),
            file=str((entry['path'] / 'rows.jsonl').relative_to(root)), sha256=checksum(entry['path'] / 'rows.jsonl'),
            cost=dict(body_wall_seconds=completion['elapsed_seconds'], allocation_cpu_seconds=int(account['CPUTimeRAW']),
                process_cpu_seconds=completion['user_cpu_seconds']+completion['system_cpu_seconds'], peak_rss_kib=completion['peak_rss_kib'])))
    if selected != expected_keys:
        raise ValueError('all six complete pilot runs required')
    events = [json.loads(line) for line in (root / 'admission-decisions.jsonl').read_text().splitlines()]
    if len(events) != len(ledger):
        raise ValueError('each allocation needs an admission decision')
    for event, entry in zip(events, ledger):
        new = sum(run['limit_seconds'] for run in entry['runs'])
        bound = event['balance']['spent_seconds'] + event['balance']['reserved_seconds'] + new
        if (event['root'] != str(declared / 'pilot') or event['new_limit_seconds'] != new
                or event['bound_seconds'] != bound or bound > 7200 or not event['test_only'] or not event['grant']
                or float(event['grant'].strip()) + shared_reservation(event['queue']) + new > 15000*3600):
            raise ValueError('admission lacks exact reservation, grant or scheduler evidence')
    return dict(verified=True, runs=runs, rows=sum(run['rows'] for run in runs), pairs_checked=sum(run['pairs'] for run in runs),
        allocation_cpu_seconds=state['spent_seconds'], reserved_seconds=0, attempts=len(inventory),
        failed_attempts=len(inventory)-len(runs), measurement_commit=plan['measurement_commit'], binary_sha256=plan['binary_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    result = audit(args.root)
    atomic_json(Path(args.root) / 'pilot-summary.json', result)
    print(json.dumps({key:value for key,value in result.items() if key != 'runs'}))
