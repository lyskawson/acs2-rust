import argparse
import json
from pathlib import Path

from artifact_analysis import checksum, write_csv
from grid_attempts import attempts
from grid_execution import (
    APPROVED_SHA256, LIMIT_HOURS, PILOT_CPU_SECONDS, completion,
    permitted_attempts, require_authorization, scout_gate, state_name,
)
from grid_recovery import checked_recovery
from grid_protocol import accounting, allocation_rows, load_plan, require_budget
from protocol import HERE, recorded
from submit import shared_reservation


def scope(root):
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    operations = Path(plan['operations_manifest'])
    local_plan = dict(plan, registered_plan=str(root / 'registered-plan.json'),
        operations_manifest=str(root.parent / operations.parent.name / operations.name),
        pilots=[dict(pilot, root=str(root.parent / Path(pilot['root']).name)) for pilot in plan['pilots']])
    return local_plan, require_authorization(local_plan)


def verify_admissions(root, ledger, registered):
    expected = permitted_attempts(registered, checked_recovery(root, ledger))
    plan_sha = checksum(root / 'plan.json')
    operation_commits = {}
    for number, entry in enumerate(ledger):
        assignment = json.loads((root / f'assignment-{number:03d}.json').read_text())
        runs = entry['runs']
        if (assignment['plan_sha256'] != plan_sha or assignment['runs'] != runs
                or any(expected.get(run['attempt_id']) != run for run in runs)):
            raise ValueError('assignment or run identity differs from the approval')
        limits = {run['limit_seconds'] for run in runs}
        if len(limits) != 1 or not 1 <= len(runs) <= 40:
            raise ValueError('array must group one limit and at most forty attempts')
        limit = next(iter(limits))
        command = entry['command']
        required = {'--partition=lem-cpu-normal', '--mem=2048M', '--cpus-per-task=1',
                    f'--time={limit // 60}', f'--array=0-{len(runs) - 1}%20'}
        if not required.issubset(command):
            raise ValueError('scheduler allocation differs from the approved settings')
        reservation = limit * len(runs)
        bound = require_budget(0, entry['balance'], reservation, LIMIT_HOURS)
        grant_bound = float(entry['grant'].strip()) + shared_reservation(entry['queue']) + reservation
        if (bound != entry['bound_seconds'] or not entry['test_only']
                or grant_bound > 15000 * 3600):
            raise ValueError('missing or inconsistent admission evidence')
        if 'operations_manifest' in assignment:
            snapshot = Path(assignment['operations_manifest'])
            manifest = load_plan(root.parent / snapshot.parent.name / snapshot.name,
                                 assignment['operations_manifest_sha256'])
            if manifest['commit'] != assignment['operations_commit']:
                raise ValueError('assigned recovery operations differ from their snapshot')
        operation_commits[entry['array_id']] = assignment['operations_commit']
    return operation_commits


def read_resources(directory):
    path = directory / 'resources.jsonl'
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.endswith('\n')]


def verified_source(root, attempt, account, run, config, plan, registered, resources, report, operation_commit):
    directory = attempt['path']
    completion(root, run, registered)
    hashes = report['sha256']
    required = {'rows.jsonl', 'resources.jsonl', 'cpu_samples.csv', 'launch.json'}
    if (directory / 'binary-rows.jsonl').exists():
        required.add('binary-rows.jsonl')
    if set(hashes) != required or any(checksum(directory / name) != hashes[name] for name in required):
        raise ValueError(f'raw artifact checksum differs for {run["attempt_id"]}')
    if [row['nominal_step'] for row in resources] != config['targets']:
        raise ValueError('full run resource checkpoints differ from the registration')
    launch = json.loads((directory / 'launch.json').read_text())
    manifest_file = directory.parents[1] / 'manifest.json'
    manifest = launch['manifest']
    if (launch['manifest_sha256'] != checksum(manifest_file)
            or manifest != json.loads(manifest_file.read_text())
            or launch['binary_sha256'] != plan['binary_sha256']
            or launch['commit'] != plan['measurement_commit']
            or launch['partition'] != 'lem-cpu-normal'
            or launch['array_job_id'] != str(attempt['array_id'])
            or launch['array_task_id'] != str(attempt['task_id'])
            or account['AllocCPUS'] != '1' or account['ExitCode'] != '0:0'
            or manifest['configuration'] != recorded(config, run['seed'])
            or manifest['agent'] != run['agent'] or manifest['seed'] != run['seed']
            or manifest['attempt_id'] != run['attempt_id']
            or manifest['allocation_limit_seconds'] != run['limit_seconds']
            or manifest['operations_commit'] != operation_commit
            or ('--per-start' in launch['command']) != (run['seed'] in registered['per_pair_seeds'])):
        raise ValueError(f'launch or allocation identity differs for {run["attempt_id"]}')
    return dict(task=run['task'], agent=run['agent'], seed=run['seed'], kind='grid_full',
                source=str(directory / 'rows.jsonl'), sha256=hashes['rows.jsonl'])


def grid_snapshot(root):
    plan, registered = scope(root)
    ledger = json.loads((root / 'ledger.json').read_text())
    operation_commits = verify_admissions(root, ledger, registered)
    raw = allocation_rows((root / 'accounting-all.psv').read_text())
    records = list({row['JobIDRaw']: row for row in raw}.values())
    by_job = {row['JobID']: row for row in records}
    if len(by_job) != len(records):
        raise ValueError('a JobID maps to multiple allocation identifiers')
    pilot = [entry for pilot in plan['pilots'] for entry in json.loads((Path(pilot['root']) / 'ledger.json').read_text())]
    pilot_state = accounting(pilot, records)
    if pilot_state != dict(spent_seconds=PILOT_CPU_SECONDS, reserved_seconds=0):
        raise ValueError('pilot accounting differs from the frozen charge')
    state = accounting(ledger, records)
    bound = require_budget(PILOT_CPU_SECONDS, state, 0, LIMIT_HOURS)
    gate = scout_gate(root, registered, ledger, records)
    configs = {config['id']: config for config in registered['configurations']}
    excluded = checked_recovery(root, ledger)
    registered_runs_by_id = permitted_attempts(registered, excluded)
    names = load_plan(HERE / 'maze-names.json', registered['maze_names_sha256'])
    sources, run_costs, checkpoint_costs = [], [], []
    for attempt in attempts(root, ledger):
        run = registered_runs_by_id[attempt['attempt_id']]
        config = configs[run['configuration_id']]
        job = f'{attempt["array_id"]}_{attempt["task_id"]}'
        account = by_job.get(job, {})
        status = state_name(account)
        directory = attempt['path']
        resources = read_resources(directory)
        completion_file = directory / 'completion.json'
        report = json.loads(completion_file.read_text()) if completion_file.exists() else {}
        if status == 'COMPLETED':
            if run['attempt_id'] in excluded:
                raise ValueError('excluded infrastructure attempt became a completed measurement')
            sources.append(verified_source(root, attempt, account, run, config, plan, registered,
                                           resources, report, operation_commits[attempt['array_id']]))
        common = dict(task=run['task'], maze=names[run['task']], agent=run['agent'], seed=run['seed'],
                      attempt_id=run['attempt_id'], job_id=job, job_id_raw=account.get('JobIDRaw'), state=status)
        wait4 = report['user_cpu_seconds'] + report['system_cpu_seconds'] if report else None
        run_costs.append(dict(**common, phase=run['phase'], limit_seconds=run['limit_seconds'],
            allocation_cpu_seconds=float(account.get('CPUTimeRAW', 0)),
            last_checkpoint=resources[-1]['nominal_step'] if resources else None,
            process_cpu_seconds_wait4=wait4,
            peak_rss_kib=report.get('peak_rss_kib', resources[-1]['peak_rss_kib'] if resources else None),
            failure=report.get('failure')))
        checkpoint_costs.extend(dict(**common, **point) for point in resources)
    source_keys = {(run['task'], run['agent'], run['seed']) for run in sources}
    complete = (len(source_keys) == len(sources) == registered['new_full_runs'] and state['reserved_seconds'] == 0
                and gate['passed'] and not (root / 'STOP.json').exists())
    summary = dict(plan_sha256=APPROVED_SHA256, complete_grid=complete, verified_full_runs=len(sources),
        grid_attempts=len(run_costs), pilot_allocation_cpu_seconds=PILOT_CPU_SECONDS,
        grid_allocation_cpu_seconds=state['spent_seconds'], reserved_seconds=state['reserved_seconds'],
        cumulative_bound_cpu_seconds=bound, cumulative_cpu_hour_limit=LIMIT_HOURS, scout_gate=gate)
    return registered, names, summary, sources, run_costs, checkpoint_costs


def reference_sources(index, original_root, mirror_root, registered):
    entries = load_plan(index, registered['archive_reference_index_sha256'])
    return [dict(entry, source=str(mirror_root / Path(entry['source']).relative_to(original_root)))
            for entry in entries]


def write_results(root, output, archive_index=None, original_root=None, mirror_root=None, reference_costs=None):
    registered, names, summary, sources, runs, checkpoints = grid_snapshot(root)
    if summary['complete_grid']:
        if None in (archive_index, original_root, mirror_root, reference_costs):
            raise ValueError('complete analysis needs the original archive index and sealed cost profiles')
        references = reference_sources(archive_index, original_root, mirror_root, registered)
        profiles = load_plan(reference_costs, registered['cost_evidence_sha256'])
        costs = [dict(task=row['task'], maze=names[row['task']], agent=row['agent'], seed=row['seed'],
                      allocation_cpu_seconds=row['allocation_cpu_seconds'],
                      process_cpu_seconds_wait4=row['process_cpu_seconds_wait4'], peak_rss_kib=row['peak_rss_kib'])
                 for row in profiles if row['kind'] == 'archive_full']
        if len(references) != 600 or len(costs) != 600:
            raise ValueError('the registered analysis needs 600 original full archive runs')

    output.mkdir(parents=True, exist_ok=False)
    (output / 'grid-verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    if runs:
        write_csv(runs, output / 'grid-run-costs.csv')
    if checkpoints:
        write_csv(checkpoints, output / 'grid-checkpoint-costs.csv')
    if summary['complete_grid']:
        write_csv(costs, output / 'reference-run-costs.csv')
        (output / 'full-index.json').write_text(json.dumps(references + sources, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--archive-index', type=Path)
    parser.add_argument('--original-root', type=Path)
    parser.add_argument('--mirror-root', type=Path)
    parser.add_argument('--reference-costs', type=Path)
    args = parser.parse_args()
    print(json.dumps(write_results(args.root, args.out, args.archive_index, args.original_root,
                                   args.mirror_root, args.reference_costs)))
