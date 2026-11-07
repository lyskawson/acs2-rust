import argparse
import json
import math
from pathlib import Path

from protocol import NEW_ARMS, E1, E, S, configurations, pilot
from execution import identity, scope
from freeze_references import checksum
from analyze import checked_path
from analysis import attach, truth_index
from schema import validate_sequence
from grid_attempts import attempts, inspect_attempt
from grid_authorization import require_derivation
from grid_protocol import accounting, allocation_rows, atomic_json, load_plan, TERMINAL
from submit import shared_reservation


def deduplicate(records):
    unique = {}
    for row in records:
        key = row['JobIDRaw']
        if key in unique and unique[key] != row:
            raise ValueError('conflicting accounting rows for one allocation')
        unique[key] = row
    return list(unique.values())


def models(root):
    result = {}
    for entry in json.loads((root / 'model-index.json').read_text()):
        value = json.loads(checked_path(root, entry).read_text())
        if entry['task'] in result or value['task'] != entry['task']:
            raise ValueError('duplicate or inconsistent environment model')
        result[entry['task']] = truth_index(value)
    if set(result) != {config['task'] for config in configurations()}:
        raise ValueError('all six environment models required')
    return result


def audit(root):
    root = Path(root)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    declared = Path(registration['destination'])
    digest = checksum(root / 'pilot-registration.json')
    if registration['manifest'] != pilot(identity()):
        raise ValueError('pilot measurement differs from its registration')
    for name in ('reference-index', 'model-index'):
        if checksum(root / (name + '.json')) != registration[name.replace('-', '_') + '_sha256']:
            raise ValueError('pilot frozen input index changed')
    if json.loads((root / 'pilot-scope.json').read_text()) != scope(digest, declared):
        raise ValueError('pilot accounting scope changed')
    references = json.loads((root / 'reference-index.json').read_text())
    for entry in references:
        checked_path(root / 'references', entry)
    batch = root / 'pilot'
    plan = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    expected = dict(registration['manifest'], budget_scope=str(declared / 'pilot-scope.json'),
        budget_scope_sha256=checksum(root / 'pilot-scope.json'), stop_path=str(declared / 'STOP.json'))
    require_derivation(plan, expected)
    if plan['authorization_sha256'] != digest:
        raise ValueError('pilot authorization changed')
    operations = load_plan(root / 'operations/manifest.json', plan['operations_manifest_sha256'])
    for name, sha in operations['files'].items():
        checked_path(root / 'operations', dict(file=name, sha256=sha))
    ledger = json.loads((batch / 'ledger.json').read_text())
    records = deduplicate(allocation_rows((batch / 'accounting.psv').read_text()))
    inventory = attempts(batch, ledger)
    expected_keys = {(config['id'], arm, 42) for config in configurations() for arm in NEW_ARMS}
    if any((entry['configuration_id'], entry['agent'], entry['seed']) not in expected_keys for entry in inventory):
        raise ValueError('unregistered pilot allocation')
    state = accounting(ledger, records)
    if state['reserved_seconds'] or state['spent_seconds'] > 4 * 3600:
        raise ValueError('unresolved reservations or pilot CPU-hour limit exceeded')
    by_job = {row['JobID']: row for row in records}
    jobs = {f"{entry['array_id']}_{entry['task_id']}" for entry in inventory}
    if len(by_job) != len(records) or set(by_job) != jobs:
        raise ValueError('unmapped or conflicting pilot accounting')
    if any(row['State'].split()[0].split('+')[0] not in TERMINAL for row in records):
        raise ValueError('every pilot attempt requires terminal accounting')
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
        account = by_job[f"{entry['array_id']}_{entry['task_id']}"]
        if (account['State'] != 'COMPLETED' or account['ExitCode'] != '0:0' or account['AllocCPUS'] != '1'
                or account['Partition'] != 'lem-cpu-normal' or entry['limit_seconds'] != 1200):
            raise ValueError('pilot allocation differs from registration')
        launch = json.loads((entry['path'] / 'launch.json').read_text())
        if (launch['binary_sha256'] != plan['binary_sha256'] or launch['commit'] != plan['measurement_commit']
                or launch['manifest']['configuration'] != config or '--per-start' not in launch['command']
                or launch['array_job_id'] != str(entry['array_id']) or launch['array_task_id'] != str(entry['task_id'])):
            raise ValueError('pilot launch differs from frozen source and assignment')
        validate_sequence(rows, config, entry['agent'], 42, plan['measurement_commit'])
        for row in rows:
            if row['cpu_model'] != plan['cpu_model']:
                raise ValueError('pilot hardware differs')
            for block in row['evaluation_blocks'].values():
                records_for_block = block['starts']
                success = math.fsum(record['weight'] * record['success'] for record in records_for_block)
                if not math.isclose(success, block['success'], rel_tol=2e-12, abs_tol=1e-12):
                    raise ValueError('pilot per-pair success disagrees with its block')
                for field, aggregate in [('estimated_first_action_value', 'mean_first_action_estimate'),
                                         ('discounted_return', 'mean_discounted_return')]:
                    total = math.fsum(record['weight'] * record[field] for record in records_for_block)
                    if not math.isclose(total, block['value_diagnostics']['all'][aggregate], rel_tol=2e-12, abs_tol=1e-12):
                        raise ValueError('pilot per-pair values disagree with their block')
                attach(records_for_block, truth[config['task']])
        completion = json.loads((entry['path'] / 'completion.json').read_text())
        base_arm = E if entry['agent'] == E1 else S
        baseline = next(ref for ref in references if (ref['kind'], ref['configuration_id'], ref['agent'], ref['seed'])
                        == ('pairs', config['id'], base_arm, 42))
        cost = dict(body_wall_seconds=completion['elapsed_seconds'], allocation_cpu_seconds=int(account['CPUTimeRAW']),
            process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'], peak_rss_kib=completion['peak_rss_kib'])
        runs.append(dict(configuration_id=config['id'], task=config['task'], agent=entry['agent'], seed=42,
            rows=len(rows), pairs=sum(len(block['starts']) for row in rows for block in row['evaluation_blocks'].values()),
            file=str((entry['path'] / 'rows.jsonl').relative_to(root)), sha256=checksum(entry['path'] / 'rows.jsonl'),
            final_entries=rows[-1]['replay_diagnostics']['per_goal_rewards'], cost=cost,
            baseline_arm=base_arm, baseline_cost=baseline['cost'],
            cost_ratios={field: cost[field] / baseline['cost'][field] for field in ('body_wall_seconds', 'allocation_cpu_seconds', 'process_cpu_seconds')}))
    if selected != expected_keys:
        raise ValueError('all twelve complete pilot runs required')
    events = [json.loads(line) for line in (root / 'admission-decisions.jsonl').read_text().splitlines()]
    if len(events) != len(ledger):
        raise ValueError('every allocation requires an admission decision')
    for event, entry in zip(events, ledger):
        new = sum(run['limit_seconds'] for run in entry['runs'])
        bound = event['balance']['spent_seconds'] + event['balance']['reserved_seconds'] + new
        if (event['root'] != str(declared / 'pilot') or event['new_limit_seconds'] != new
                or event['bound_seconds'] != bound or bound > 4 * 3600 or not event['test_only']
                or float(event['grant'].strip()) + shared_reservation(event['queue']) + new > 15000 * 3600):
            raise ValueError('pilot admission lacks exact budget, grant or scheduler evidence')
    return dict(verified=True, runs=runs, rows=sum(run['rows'] for run in runs),
        pairs_checked=sum(run['pairs'] for run in runs), allocation_cpu_seconds=state['spent_seconds'],
        reserved_seconds=0, attempts=len(inventory), failed_attempts=len(inventory) - len(runs),
        measurement_commit=plan['measurement_commit'], binary_sha256=plan['binary_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    summary = audit(args.root)
    atomic_json(Path(args.root) / 'pilot-summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'runs'}))
