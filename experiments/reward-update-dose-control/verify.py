import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from protocol import ARM, ROOT, configurations
from execution import approved, continuation, continuation_scope, validate_inputs, APPROVED_SHA256
from archive import audit, checksum, checked_path, deduplicate, models, validate_aggregates
from analysis import attach
from collect import validate_row
from grid_attempts import attempts, select_attempts
from grid_authorization import require_derivation
from grid_protocol import atomic_json, allocation_rows, load_plan, TERMINAL
from grid_scope import snapshot
from submit import execute, shared_reservation


def update_accounting(root):
    root=Path(root)
    manifest=json.loads((root/'continuation/compact-manifest.json').read_text())
    result=snapshot(manifest,execute,preserve_roots=(Path(manifest['authorization_plan']).parent/'pilot',))
    atomic_json(root/'continuation/final-accounting.json',result)
    return result


def verify(root):
    root=Path(root)
    registered,approval=approved(root/'approved-plan.json')
    validate_inputs(root,registered)
    preparation=json.loads((root/'execution-preparation.json').read_text())
    if preparation['pilot_inventory_sha256']!=checksum(root/'pilot-inventory.json'):
        raise ValueError('preserved pilot inventory changed')
    for name,digest in json.loads((root/'pilot-inventory.json').read_text()).items():
        checked_path(root,dict(file=name,sha256=digest))
    pilot_summary=audit(root)
    truth=models(root)
    accounting=deduplicate(allocation_rows((root/'continuation/shared-accounting.psv').read_text()))
    by_id={row['JobID']:row for row in accounting}
    if len(by_id)!=len(accounting):
        raise ValueError('accounting identifiers are not unique')
    expected_jobs=set()
    for entry in json.loads((root/'pilot/ledger.json').read_text()):
        expected_jobs.update(f"{entry['array_id']}_{index}" for index,_ in enumerate(entry['runs']))
    runs=list(pilot_summary['runs'])
    failed=0
    allocation_count=pilot_summary['attempts']
    receipt_count=0
    receipts=[json.loads(line) for line in (root/'continuation/admission-decisions.jsonl').read_text().splitlines()]
    operation_manifest=load_plan(root/'execution-operations/manifest.json',checksum(root/'execution-operations/manifest.json'))
    if operation_manifest['commit']!=preparation['operations_commit']:
        raise ValueError('execution operations commit changed')
    for name,digest in operation_manifest['files'].items():
        checked_path(root/'execution-operations',dict(file=name,sha256=digest))
    for purpose in ('diagnostics','compact'):
        batch=root/purpose
        manifest=load_plan(batch/'plan.json',(batch/'plan.sha256').read_text().strip())
        declared=Path(manifest['authorization_plan']).parent
        scope_path=declared/'continuation/scope.json'
        if json.loads((root/'continuation/scope.json').read_text())!=continuation_scope(declared):
            raise ValueError('cumulative accounting scope changed')
        expected=dict(continuation(registered,purpose),budget_scope=str(scope_path),
            budget_scope_sha256=checksum(root/'continuation/scope.json'),stop_path=str(declared/'continuation/STOP.json'))
        require_derivation(manifest,expected)
        if (manifest['authorization_sha256']!=APPROVED_SHA256 or manifest['operations_commit']!=preparation['operations_commit']
                or manifest['operations_manifest_sha256']!=checksum(root/'execution-operations/manifest.json')
                or json.loads((batch/'approval.json').read_text())!=approval):
            raise ValueError('execution provenance differs from approval')
        ledger=json.loads((batch/'ledger.json').read_text())
        receipt_count+=len(ledger)
        entries=attempts(batch,ledger)
        allowed={(config['id'],ARM,seed) for config in manifest['configurations'] for seed in manifest['seeds']}
        if any((entry['configuration_id'],entry['agent'],entry['seed']) not in allowed for entry in entries):
            raise ValueError('unregistered allocation')
        allocation_count+=len(entries)
        for entry in entries:
            job=f"{entry['array_id']}_{entry['task_id']}"
            expected_jobs.add(job)
            record=by_id.get(job)
            if record is None or record['State'].split()[0].split('+')[0] not in TERMINAL:
                raise ValueError('every attempt requires terminal accounting')
        for config in manifest['configurations']:
            selected,inventory=select_attempts(batch,ledger,config)
            if set(selected)!={(ARM,seed) for seed in manifest['seeds']}:
                raise ValueError('incomplete registered seed set')
            failed+=sum(item['status']!='complete' for item in inventory)
            for (arm,seed),entry in selected.items():
                path=entry['path']
                launch=json.loads((path/'launch.json').read_text())
                if (launch['binary_sha256']!=registered['binary_sha256'] or launch['commit']!=registered['measurement_commit']
                        or launch['partition']!='lem-cpu-normal' or launch['manifest']['configuration']!=config
                        or launch['array_job_id']!=str(entry['array_id']) or launch['array_task_id']!=str(entry['task_id'])
                        or ('--per-start' in launch['command'])!=(purpose=='diagnostics')):
                    raise ValueError('launch differs from registered assignment')
                record=by_id[f"{entry['array_id']}_{entry['task_id']}"]
                if record['State']!='COMPLETED' or record['ExitCode']!='0:0' or record['AllocCPUS']!='1' or record['Partition']!='lem-cpu-normal':
                    raise ValueError('selected allocation did not complete on one Lem CPU')
                rows=[json.loads(line) for line in (path/'rows.jsonl').read_text().splitlines()]
                previous=None
                pairs=0
                for row,point in zip(rows,config['targets']):
                    validate_row(row,config,arm,seed,registered['measurement_commit'],point)
                    if row['preset']!=registered['preset'] or row['cpu_model']!=manifest['cpu_model']:
                        raise ValueError('preset or CPU model changed')
                    counts=row['replay_diagnostics']['reward_updates']
                    if previous and any(counts[origin][key]<previous[origin][key] for origin in counts for key in counts[origin]):
                        raise ValueError('cumulative reward counts decreased')
                    previous=counts
                    if purpose=='diagnostics':
                        for block in row['evaluation_blocks'].values():
                            validate_aggregates(block)
                            attach(block['starts'],truth[config['task']])
                            pairs+=len(block['starts'])
                completion=json.loads((path/'completion.json').read_text())
                runs.append(dict(configuration_id=config['id'],agent=arm,seed=seed,rows=len(rows),pairs=pairs,
                    file=str((path/'rows.jsonl').relative_to(root)),sha256=checksum(path/'rows.jsonl'),
                    cost=dict(allocation_cpu_seconds=int(record['CPUTimeRAW']),body_wall_seconds=completion['elapsed_seconds'],
                        process_cpu_seconds=completion['user_cpu_seconds']+completion['system_cpu_seconds'],peak_rss_kib=completion['peak_rss_kib'])))
        events=[event for event in receipts if event['root']==str(declared/purpose)]
        if len(events)!=len(ledger):
            raise ValueError('each allocation needs one admission receipt')
        for event,allocation in zip(events,ledger):
            new=sum(run['limit_seconds'] for run in allocation['runs'])
            bound=event['balance']['spent_seconds']+event['balance']['reserved_seconds']+new
            if (event['new_limit_seconds']!=new or event['bound_seconds']!=bound or bound>50400
                    or not event['test_only'] or not event['grant'] or event.get('admitted') is not True
                    or float(event['grant'].strip())+shared_reservation(event['queue'])+new>15000*3600):
                raise ValueError('admission lacks exact budget, grant or scheduler evidence')
    if len(receipts)!=receipt_count:
        raise ValueError('unmapped admission receipt')
    if set(by_id)!=expected_jobs:
        raise ValueError('unmapped or missing accounting allocations')
    total=math.fsum(float(by_id[job]['CPUTimeRAW']) for job in expected_jobs)
    if total>50400:
        raise ValueError('cumulative CPU-hour limit exceeded')
    expected={(config['id'],ARM,seed) for config in configurations() for seed in range(42,62)}
    if {(run['configuration_id'],run['agent'],run['seed']) for run in runs}!=expected or len(runs)!=120:
        raise ValueError('analysis requires exactly all 120 registered runs')
    result=dict(verified=True,plan_sha256=APPROVED_SHA256,runs_index=runs,rows=sum(run['rows'] for run in runs),
        pairs_checked=sum(run['pairs'] for run in runs),allocation_cpu_seconds=total,reserved_seconds=0,
        allocations=allocation_count,failed_attempts=failed,phase_cpu_hour_limit=14,measurement_rebuilt=False)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--update-accounting',action='store_true')
    args=parser.parse_args()
    if args.update_accounting:
        print(json.dumps(update_accounting(args.root)))
    else:
        result=verify(args.root)
        atomic_json(Path(args.root)/'verification.json',result)
        print(json.dumps({key:value for key,value in result.items() if key!='runs_index'}))
