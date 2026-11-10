import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from protocol import HERE,ROOT,configurations,recorded
from register_study import build
from grid_protocol import atomic_json,allocation_rows,accounting,require_budget,TERMINAL
from grid_scope import verify_operations,SACCT_FIELDS
from submit import execute,submission_command,shared_reservation


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registration():
    value=json.loads((HERE/'registration.json').read_text())
    if value!=build():raise ValueError('pilot registration differs from the card-derived cells')
    return value


def prepare(root,repo):
    root,repo=Path(root).resolve(),Path(repo).resolve()
    if root.exists() or not root.is_relative_to(Path.home().resolve()/'tu_runs/per-goal-reward-ablation'):
        raise ValueError('new study-owned result root required')
    registered=registration()
    commit=execute(['/usr/bin/git','-C',str(repo),'rev-parse','HEAD']).strip()
    if execute(['/usr/bin/git','-C',str(repo),'status','--porcelain']).strip():raise ValueError('measurement checkout must be clean')
    binary=repo/'target/x86_64-unknown-linux-musl/release/acs2-measure'
    root.mkdir(parents=True);(root/'logs').mkdir()
    operations=root/'operations';operations.mkdir()
    files={}
    for name in execute(['/usr/bin/git','-C',str(repo),'ls-files','experiments','fixtures']).splitlines():
        target=operations/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target);files[name]=checksum(target)
    atomic_json(operations/'manifest.json',dict(commit=commit,files=files))
    preset=json.loads((repo/'fixtures/goal_measurement_regression.json').read_text())['configurations'][0]['rows'][0]['preset']
    plan=dict(result_root=str(root),registration=registered,registration_sha256=checksum(HERE/'registration.json'),measurement_commit=commit,
        binary_sha256=checksum(binary),preset=preset,measurement_repository=str(repo),binary=str(binary),
        operations_manifest=str(operations/'manifest.json'),operations_manifest_sha256=checksum(operations/'manifest.json'),
        phase_cpu_hour_limit=15,pilot_allocation_cpu_seconds=0,partition='lem-cpu-normal',concurrency=20,
        memory_mib=2048,cpu_model='AMD EPYC 9554 64-Core Processor',accounting_since='2026-11-09',
        configurations=[recorded(c,42) for c in configurations()],stop_path=str(root/'STOP.json'))
    atomic_json(root/'plan.json',plan);(root/'plan.sha256').write_text(checksum(root/'plan.json')+'\n')
    atomic_json(root/'ledger.json',[])
    return dict(commit=commit,binary_sha256=plan['binary_sha256'],registration_sha256=plan['registration_sha256'])


def require_authorization(plan):
    if (plan['registration']!=registration() or plan['registration_sha256']!=checksum(HERE/'registration.json')
            or plan['phase_cpu_hour_limit']!=15 or plan['concurrency']!=20 or plan['partition']!='lem-cpu-normal'
            or plan['configurations']!=[recorded(c,42) for c in configurations()]):
        raise ValueError('pilot exceeds or differs from the registered scope')
    verify_operations(plan)


def registered_runs(plan):
    require_authorization(plan)
    return [dict(c,attempt_id=f"{c['configuration_id']}_{c['agent']}_s42_a01",cause='registered first attempt',
        relative_directory=f"attempts/{c['configuration_id']}_{c['agent']}_s42_a01/measurement/{c['configuration_id']}_{c['agent']}_s42")
        for c in plan['registration']['pilot_runs']]


def balance(root,ledger,executor=execute):
    raw=executor(['sacct','--array','-S','2026-11-09','-j',','.join(str(e['array_id']) for e in ledger),'-P','--format='+SACCT_FIELDS]) if ledger else SACCT_FIELDS.replace(',','|')+'|\n'
    records=list({r['JobIDRaw']:r for r in allocation_rows(raw)}.values())
    (Path(root)/'accounting.psv').write_text(raw)
    return accounting(ledger,records),records


def dispatch(root,executor=execute):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());require_authorization(plan)
    if checksum(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('pilot manifest changed')
    if root.resolve()!=Path(plan['result_root']).resolve():raise ValueError('result root differs from the frozen pilot destination')
    if Path(plan['stop_path']).exists():raise ValueError('study STOP prevents submission')
    ledger=json.loads((root/'ledger.json').read_text());state,records=balance(root,ledger,executor)
    known={r['JobID']:r for r in records}
    active=sum(any(known.get(f"{e['array_id']}_{i}",{}).get('State','PENDING').split('+')[0].split()[0] not in TERMINAL for i in range(len(e['runs']))) for e in ledger)
    if active>=2:return dict(submitted=False,active_streams=active,**state)
    done={(r['configuration_id'],r['agent']) for e in ledger for r in e['runs']}
    remaining=[r for r in registered_runs(plan) if (r['configuration_id'],r['agent']) not in done]
    if not remaining:return dict(submitted=False,all_submitted=True,active_streams=active,**state)
    limit=max(r['limit_seconds'] for r in remaining)
    batch=[r for r in remaining if r['limit_seconds']==limit][:40]
    new=sum(r['limit_seconds'] for r in batch)
    bound=require_budget(0,state,new,15)
    grant=executor(['sshare','-U','-u','alelys2099','-o','RawUsage','-n'])
    queue=executor(['squeue','--array','-u','alelys2099','-h','-o','%i|%j|%t|%L|%C'])
    if float(grant.strip())+shared_reservation(queue)+new>15000*3600:raise ValueError('shared grant cannot cover this reservation')
    assignment=root/f'assignment-{len(ledger):02d}.json'
    atomic_json(assignment,dict(plan_sha256=checksum(root/'plan.json'),runs=batch,
        measurement_repository=plan['measurement_repository'],binary=plan['binary'],operations_commit=plan['measurement_commit']))
    command=submission_command(root,assignment,limit//60,len(batch),20,2048,[],job_name='tu-reward-ablation')
    operations=Path(plan['operations_manifest']).parent/'experiments/per-goal-reward-ablation'
    command[-4:]=[str(operations/'job.sh'),str(operations),str(root),str(assignment)]
    tested=executor(command[:1]+['--test-only']+command[1:])
    require_authorization(plan)
    for r in batch:(root/r['relative_directory']).parent.parent.mkdir(parents=True)
    entry=dict(array_id=None,runs=batch,command=command,test_only=tested,grant=grant,queue=queue,
        balance=state,bound_seconds=bound,submitted_unix=time.time())
    ledger.append(entry);atomic_json(root/'ledger.json',ledger)
    response=executor(command).strip();entry.update(array_id=int(response.split(';')[0]),submission_response=response)
    atomic_json(root/'ledger.json',ledger)
    return dict(submitted=True,array_id=entry['array_id'],runs=len(batch),bound_cpu_hours=bound/3600)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','dispatch','status']);parser.add_argument('root');parser.add_argument('--repo')
    args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.root,args.repo)
    elif args.action=='dispatch':result=dispatch(args.root)
    else:
        ledger=json.loads((Path(args.root)/'ledger.json').read_text());state,records=balance(args.root,ledger);result=dict(state=state,allocations=records)
    print(json.dumps(result))
