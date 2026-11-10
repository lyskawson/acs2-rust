import argparse
import json
from pathlib import Path
import time

from protocol import HERE,configurations
from cost_pilot_registration import build,configuration,PRIOR_CPU_SECONDS,MEASUREMENT_COMMIT,BINARY_SHA256
from execution import checksum
from grid_protocol import atomic_json,load_plan,allocation_rows,accounting,require_budget,TERMINAL
from grid_scope import verify_operations,SACCT_FIELDS
from submit import execute,submission_command,shared_reservation


def registration():
    value=json.loads((HERE/'cost-pilot.json').read_text())
    if value!=build():raise ValueError('30k cost-pilot registration differs from approval')
    return value


def verify_prior(root,expected=None):
    root=Path(root)
    plan=json.loads((root/'plan.json').read_text())
    ledger=json.loads((root/'ledger.json').read_text())
    if (plan['measurement_commit']!=MEASUREMENT_COMMIT or plan['binary_sha256']!=BINARY_SHA256
            or [e['array_id'] for e in ledger]!=[6296853,6296855]
            or sum(len(e['runs']) for e in ledger)!=18):
        raise ValueError('prior evidence differs from the 18 authorized stopped attempts')
    rows=list({r['JobIDRaw']:r for r in allocation_rows((root/'accounting.psv').read_text())}.values())
    state=accounting(ledger,rows)
    if state!=dict(spent_seconds=PRIOR_CPU_SECONDS,reserved_seconds=0):
        raise ValueError('prior attempts must be terminal and charged exactly once')
    value=dict(root=str(root.resolve()),files={name:checksum(root/name) for name in ['plan.json','ledger.json','accounting.psv','archive-checksums.json']})
    if expected is not None and value!=expected:raise ValueError('frozen prior evidence changed')
    return value


def prepare(root,repo,operations,prior):
    root,repo,operations=Path(root).resolve(),Path(repo).resolve(),Path(operations).resolve()
    if root.exists() or not root.is_relative_to(Path.home().resolve()/'tu_runs/per-goal-reward-ablation'):
        raise ValueError('new study-owned result root required')
    registered=registration();previous=verify_prior(prior)
    commit=execute(['/usr/bin/git','-C',str(repo),'rev-parse','HEAD']).strip()
    if commit!=MEASUREMENT_COMMIT or execute(['/usr/bin/git','-C',str(repo),'status','--porcelain']).strip():
        raise ValueError('measurement clone must remain clean at the approved measurement commit')
    binary=repo/'target/x86_64-unknown-linux-musl/release/acs2-measure'
    if checksum(binary)!=BINARY_SHA256:raise ValueError('measurement binary differs')
    manifest=operations/'manifest.json'
    ops=json.loads(manifest.read_text())
    plan=dict(result_root=str(root),registration=registered,registration_sha256=checksum(HERE/'cost-pilot.json'),
        prior=previous,measurement_commit=commit,binary_sha256=BINARY_SHA256,measurement_repository=str(repo),binary=str(binary),
        operations_manifest=str(manifest),operations_manifest_sha256=checksum(manifest),operations_commit=ops['commit'],
        preset=json.loads((repo/'fixtures/goal_measurement_regression.json').read_text())['configurations'][0]['rows'][0]['preset'],
        phase_cpu_hour_limit=15,partition='lem-cpu-normal',concurrency=20,memory_mib=2048,
        cpu_model='AMD EPYC 9554 64-Core Processor',accounting_since='2026-11-09',
        configurations=[configuration(c) for c in configurations()],stop_path=str(root/'STOP.json'))
    verify_operations(plan)
    root.mkdir(parents=True);(root/'logs').mkdir()
    atomic_json(root/'plan.json',plan);(root/'plan.sha256').write_text(checksum(root/'plan.json')+'\n')
    atomic_json(root/'ledger.json',[])
    return dict(measurement_commit=commit,operations_commit=ops['commit'],prior_cpu_seconds=PRIOR_CPU_SECONDS,
                maximum_combined_cpu_hours=registered['maximum_combined_cpu_hours'])


def require_authorization(plan):
    if (plan['registration']!=registration() or plan['registration_sha256']!=checksum(HERE/'cost-pilot.json')
            or plan['phase_cpu_hour_limit']!=15 or plan['concurrency']!=20 or plan['partition']!='lem-cpu-normal'
            or plan['measurement_commit']!=MEASUREMENT_COMMIT or plan['binary_sha256']!=BINARY_SHA256
            or plan['configurations']!=[configuration(c) for c in configurations()]):
        raise ValueError('cost-pilot scope or measurement identity differs')
    verify_prior(plan['prior']['root'],plan['prior']);verify_operations(plan)


def registered_runs(plan):
    require_authorization(plan)
    result=[]
    for run in plan['registration']['pilot_runs']:
        identifier=f"{run['configuration_id']}_{run['agent']}_s42_p30000_a01"
        result.append(dict(run,attempt_id=identifier,cause='approved fresh 30k cost measurement and prefix validation',
            relative_directory=f"attempts/{identifier}/measurement/{run['configuration_id']}_{run['agent']}_s42"))
    return result


def balance(root,plan,ledger,executor=execute):
    prior=json.loads((Path(plan['prior']['root'])/'ledger.json').read_text())
    arrays=[e['array_id'] for e in prior+ledger]
    if len(arrays)!=len(set(arrays)):raise ValueError('duplicate allocation array')
    raw=executor(['sacct','--array','-S','2026-11-09','-j',','.join(map(str,arrays)),'-P','--format='+SACCT_FIELDS])
    records=list({r['JobIDRaw']:r for r in allocation_rows(raw)}.values())
    old=accounting(prior,records);new=accounting(ledger,records)
    if old!=dict(spent_seconds=PRIOR_CPU_SECONDS,reserved_seconds=0):raise ValueError('prior accounting is missing, changed or nonterminal')
    (Path(root)/'accounting-all.psv').write_text(raw)
    return dict(spent_seconds=old['spent_seconds']+new['spent_seconds'],reserved_seconds=new['reserved_seconds'],
                prior_cpu_seconds=old['spent_seconds'],new_cpu_seconds=new['spent_seconds']),records


def dispatch(root,executor=execute):
    root=Path(root);plan=load_plan(root/'plan.json',(root/'plan.sha256').read_text().strip());require_authorization(plan)
    if root.resolve()!=Path(plan['result_root']).resolve():raise ValueError('result destination differs')
    if Path(plan['stop_path']).exists():raise ValueError('cost pilot stopped')
    ledger=json.loads((root/'ledger.json').read_text());state,records=balance(root,plan,ledger,executor)
    known={r['JobID']:r for r in records}
    active=sum(any(known.get(f"{e['array_id']}_{i}",{}).get('State','PENDING').split('+')[0].split()[0] not in TERMINAL for i in range(len(e['runs']))) for e in ledger)
    if active>=2:return dict(submitted=False,active_streams=active,**state)
    done={(r['configuration_id'],r['agent']) for e in ledger for r in e['runs']}
    remaining=[r for r in registered_runs(plan) if (r['configuration_id'],r['agent']) not in done]
    if not remaining:return dict(submitted=False,all_submitted=True,active_streams=active,**state)
    limit=max(r['limit_seconds'] for r in remaining);batch=[r for r in remaining if r['limit_seconds']==limit][:40]
    bound=require_budget(0,state,sum(r['limit_seconds'] for r in batch),15)
    grant=executor(['sshare','-U','-u','alelys2099','-o','RawUsage','-n'])
    queue=executor(['squeue','--array','-u','alelys2099','-h','-o','%i|%j|%t|%L|%C'])
    if float(grant.strip())+shared_reservation(queue)+sum(r['limit_seconds'] for r in batch)>15000*3600:
        raise ValueError('shared grant cannot cover the next array')
    assignment=root/f'assignment-{len(ledger):02d}.json'
    atomic_json(assignment,dict(plan_sha256=checksum(root/'plan.json'),runs=batch,
        measurement_repository=plan['measurement_repository'],binary=plan['binary'],operations_commit=plan['operations_commit']))
    command=submission_command(root,assignment,limit//60,len(batch),20,2048,[],job_name='tu-ablation-cost')
    operations=Path(plan['operations_manifest']).parent/'experiments/per-goal-reward-ablation'
    command[-4:]=[str(operations/'job_cost.sh'),str(operations),str(root),str(assignment)]
    tested=executor(command[:1]+['--test-only']+command[1:]);require_authorization(plan)
    for r in batch:(root/r['relative_directory']).parent.parent.mkdir(parents=True)
    entry=dict(array_id=None,runs=batch,command=command,test_only=tested,grant=grant,queue=queue,
               balance=state,bound_seconds=bound,submitted_unix=time.time())
    ledger.append(entry);atomic_json(root/'ledger.json',ledger)
    response=executor(command).strip();entry.update(array_id=int(response.split(';')[0]),submission_response=response)
    atomic_json(root/'ledger.json',ledger)
    return dict(submitted=True,array_id=entry['array_id'],runs=len(batch),bound_cpu_hours=bound/3600)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','dispatch','status']);parser.add_argument('root')
    parser.add_argument('--repo');parser.add_argument('--operations');parser.add_argument('--prior');args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.root,args.repo,args.operations,args.prior)
    elif args.action=='dispatch':result=dispatch(args.root)
    else:
        root=Path(args.root);plan=json.loads((root/'plan.json').read_text());state,records=balance(root,plan,json.loads((root/'ledger.json').read_text()));result=dict(state=state,allocations=records)
    print(json.dumps(result))
