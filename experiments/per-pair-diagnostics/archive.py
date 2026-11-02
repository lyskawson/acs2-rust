import argparse
import hashlib
import json
from pathlib import Path

from protocol import ARMS, BLOCKS, configurations
from execution import pilot_manifest, scope
from references import read
from analysis import attach, truth_index
from collect import validate_row
from grid_attempts import attempts, select_attempts, inspect_attempt
from grid_identity import field_bits, verify_references
from grid_protocol import accounting, allocation_rows, atomic_json, load_plan, TERMINAL
from grid_authorization import require_derivation


def models(root):
    root = Path(root)
    result = {}
    for entry in json.loads((root / 'model-index.json').read_text()):
        path = root / entry['file']
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('environment model checksum changed')
        value = json.loads(path.read_text())
        if value['task'] != entry['task'] or entry['task'] in result:
            raise ValueError('model task differs or is duplicated')
        result[entry['task']] = truth_index(value)
    if set(result) != {config['task'] for config in configurations()}:
        raise ValueError('six environment models required')
    return result


def catalog(batch, root, retain_rows=True):
    batch, root = Path(batch), Path(root)
    plan = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    ledger = json.loads((batch / 'ledger.json').read_text())
    expected = {(config['id'], arm, seed) for config in plan['configurations'] for arm in plan['agents'] for seed in plan['seeds']}
    inventory = attempts(batch, ledger)
    if any((entry['configuration_id'], entry['agent'], entry['seed']) not in expected for entry in inventory):
        raise ValueError('unregistered allocation')
    records = allocation_rows((batch / 'accounting.psv').read_text())
    by_id = {record['JobID']: record for record in records}
    if len(by_id) != len(records):
        raise ValueError('duplicate accounting record')
    state = accounting(ledger, records)
    if state['reserved_seconds']:
        raise ValueError('unresolved hard reservations')
    for allocation in ledger:
        hard_limit = 60 * max(next(config for config in plan['configurations'] if config['id'] == entry['configuration_id'])['limits_minutes'][entry['agent']] for entry in allocation['runs'])
        if any(entry['limit_seconds'] != hard_limit for entry in allocation['runs']):
            raise ValueError('array hard reservation differs from registration')
    for entry in inventory:
        account = by_id.get(f"{entry['array_id']}_{entry['task_id']}")
        if account is None or account['State'].split()[0].split('+')[0] not in TERMINAL:
            raise ValueError('every attempt needs terminal sacct accounting')
    entries = {}
    ground_truth = models(root)
    for config in plan['configurations']:
        selected, _ = select_attempts(batch, ledger, config)
        if set(selected) != {(arm, seed) for arm in plan['agents'] for seed in plan['seeds']}:
            raise ValueError('incomplete registered grid')
        for (arm, seed), entry in selected.items():
            path = entry['path']
            rows = [json.loads(line) for line in (path / 'rows.jsonl').read_bytes().splitlines()]
            launch = json.loads((path / 'launch.json').read_text())
            if (launch['binary_sha256'] != plan['binary_sha256'] or launch['commit'] != plan['measurement_commit']
                    or launch['partition'] != 'lem-cpu-normal' or '--per-start' not in launch['command']
                    or launch['manifest']['configuration'] != config
                    or launch['array_job_id'] != str(entry['array_id']) or launch['array_task_id'] != str(entry['task_id'])):
                raise ValueError('launch identity differs')
            for row, target in zip(rows, config['targets']):
                validate_row(row, config, arm, seed, plan['measurement_commit'], target)
                if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model']:
                    raise ValueError('preset or hardware differs')
                for block in row['evaluation_blocks'].values():
                    attach(block['starts'], ground_truth[config['task']])
            declared = Path(plan['authorization_plan']).parent
            identities = verify_references(plan, entry, rows, lambda value: root / Path(value).relative_to(declared))
            if field_bits(json.loads((path / 'identity-checks.json').read_text())) != field_bits(identities):
                raise ValueError('node identity result differs from independent recomputation')
            record = by_id[f"{entry['array_id']}_{entry['task_id']}"]
            if (record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0' or record['AllocCPUS'] != '1'
                    or record['Partition'] != 'lem-cpu-normal'):
                raise ValueError('selected allocation did not complete on one Lem CPU')
            completion = json.loads((path / 'completion.json').read_text())
            entries[config['id'], arm, seed] = dict(rows=rows if retain_rows else None, row_count=len(rows), pair_records=sum(sum(block['evaluation_pairs'] for block in row['evaluation_blocks'].values()) for row in rows), path=path, sha256=hashlib.sha256((path / 'rows.jsonl').read_bytes()).hexdigest(),
                cost=dict(body_wall_seconds=completion['elapsed_seconds'], process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                          allocation_cpu_seconds=int(record['CPUTimeRAW']), peak_rss_kib=completion['peak_rss_kib']),
                row_file_bytes=(path / 'rows.jsonl').stat().st_size, identity_rows=sum(item['rows'] for item in identities))
    return entries, dict(state, attempts=len(inventory), allocations=len(inventory), arrays=[entry['array_id'] for entry in ledger])


def admissions_for(events, declared):
    return [event for event in events if Path(event['root']) == Path(declared) / 'pilot']


def audit(root, retain_rows=True):
    root = Path(root)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    digest = hashlib.sha256((root / 'pilot-registration.json').read_bytes()).hexdigest()
    declared = Path(registration['destination'])
    if registration['manifest'] != pilot_manifest(root, declared):
        raise ValueError('pilot differs from exact registered presets')
    for name, key in [('reference-index.json', 'reference_index_sha256'), ('model-index.json', 'model_index_sha256')]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != registration[key]:
            raise ValueError('registered pilot input changed: ' + name)
    read(root)
    if json.loads((root / 'pilot-scope.json').read_text()) != scope(digest, declared):
        raise ValueError('pilot accounting scope changed')
    manifest = load_plan(root / 'pilot/plan.json', (root / 'pilot/plan.sha256').read_text().strip())
    expected = dict(registration['manifest'], budget_scope=str(declared / 'pilot-scope.json'),
        budget_scope_sha256=hashlib.sha256((root / 'pilot-scope.json').read_bytes()).hexdigest(), stop_path=str(declared / 'STOP.json'))
    require_derivation(manifest, expected)
    if manifest['authorization_sha256'] != digest:
        raise ValueError('pilot belongs to a different authorization')
    operations = load_plan(root / 'operations/manifest.json', manifest['operations_manifest_sha256'])
    for name, checksum in operations['files'].items():
        path = root / 'operations' / name
        if not path.resolve().is_relative_to((root / 'operations').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('immutable operations changed')
    entries, state = catalog(root / 'pilot', root, retain_rows=retain_rows)
    if state['spent_seconds'] > 4 * 3600 or state['reserved_seconds']:
        raise ValueError('pilot exceeds four allocated CPU-hours')
    admissions = admissions_for([json.loads(line) for line in (root / 'admission-decisions.jsonl').read_text().splitlines()], declared)
    ledger = json.loads((root / 'pilot/ledger.json').read_text())
    if len(admissions) != len(ledger):
        raise ValueError('every submission needs an admission record')
    for event, allocation in zip(admissions, ledger):
        new = sum(run['limit_seconds'] for run in allocation['runs'])
        total = event['balance']['spent_seconds'] + event['balance']['reserved_seconds'] + new
        if (not event['test_only'] or not event['grant'] or event['new_limit_seconds'] != new
                or event['bound_seconds'] != total or total > 4 * 3600 or len(allocation['runs']) > 40):
            raise ValueError('pilot admission lacks scheduler, grant or budget evidence')
    runs = [dict(configuration_id=config, agent=arm, seed=seed, cost=entry['cost'], row_file_bytes=entry['row_file_bytes'],
                 row_bytes=[len(line) + 1 for line in (entry['path'] / 'rows.jsonl').read_bytes().splitlines()],
                 rows=entry['row_count'], identity_rows=entry['identity_rows'], source_rows_sha256=entry['sha256'])
            for (config, arm, seed), entry in sorted(entries.items())]
    summary = dict(verified=True, runs=runs, allocation_cpu_seconds=state['spent_seconds'], allocations=state['allocations'],
                   reserved_seconds=0, rows=sum(item['rows'] for item in runs), identity_rows=sum(item['identity_rows'] for item in runs),
                   reference_runs=150, pairs_checked=sum(entry['pair_records'] for entry in entries.values()))
    return entries, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    _, summary = audit(args.root)
    atomic_json(args.root / 'pilot-summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'runs'}))
