import argparse
import json
from pathlib import Path
import subprocess
import time

from protocol import HERE,ARMS,A1,OLD_TASKS,SEEDS
from execution import checksum
from cost_pilot_registration import MEASUREMENT_COMMIT,BINARY_SHA256
from grid_protocol import atomic_json,load_plan,allocation_rows,accounting,require_budget,TERMINAL,stop_phase
from grid_scope import verify_operations,SACCT_FIELDS
from submit import execute,submission_command,shared_reservation


APPROVED_SHA256='7d9606634a9b114922c84c77d9acea358964ebe5e0e3ff27b590b3713684ac66'
REVIEW_TEST_COMMIT='a8e8c15909b147b03b6818f72688b21706989d61'
SCOUT_TASKS={'mazee2_alcs','mazee3_alcs'}
PILOT_CPU_SECONDS=12965
LIMIT_HOURS=3583


def registered_plan():
    return load_plan(HERE/'plan.json',APPROVED_SHA256)


def registered_runs(registered):
    configs={c['task']:c for c in registered['configurations']}
    cells=registered['forecast']['cells']
    expected={(task,arm) for task in configs for arm in ((A1,) if task in OLD_TASKS else ARMS)}
    if len(cells)!=138 or {(c['task'],c['agent']) for c in cells}!=expected or registered['seeds']!=SEEDS:
        raise ValueError('approved full-grid cells or seeds differ')
    result=[]
    for cell in cells:
        config=configs[cell['task']]
        if cell['registered_budget_steps']!=config['targets'][-1]:raise ValueError('approved budget differs')
        for seed in SEEDS:
            identifier=f"{config['id']}_{cell['agent']}_s{seed}_a01"
            phase='ordinary' if config['task'] not in SCOUT_TASKS else ('scout' if seed==42 else 'conditional')
            result.append(dict(configuration_id=config['id'],task=config['task'],agent=cell['agent'],seed=seed,
                phase=phase,limit_seconds=cell['proposed_time_limit_seconds'],
                forecast_high_cpu_seconds=cell['forecast_registered_high_cpu_seconds'],attempt_id=identifier,
                cause='approved full-budget first attempt',
                relative_directory=f"attempts/{identifier}/measurement/{config['id']}_{cell['agent']}_s{seed}"))
    if len(result)!=2760:raise ValueError('2760 full runs required')
    return result


def authorization():
    return dict(approved_plan_sha256=APPROVED_SHA256,review_test_commit=REVIEW_TEST_COMMIT,
        cumulative_cpu_hour_limit=LIMIT_HOURS,pilot_allocation_cpu_seconds=PILOT_CPU_SECONDS,
        memory_mib=2048,max_streams=2,concurrency=20,partition='lem-cpu-normal',
        ordinary='All seeds on the 26 mazes outside MazeE2 and MazeE3.',
        scout='Seed42 on all twelve MazeE2 and MazeE3 cells; may overlap ordinary runs.',
        conditional='Remaining 228 runs only after all twelve scouts completed within their own high CPU-allocation forecast.',
        failure='Collector STOP unchanged. No timeout retry or limit/budget increase without approval.',
        prefix_comparison=False)


def frozen_pilots(parent,registered):
    result=[]
    for name,key,account_file in [('stage1','prior_archive_manifest_sha256','accounting.psv'),
            ('pilot-30000','fresh_archive_manifest_sha256','accounting-all.psv')]:
        root=Path(parent)/name
        if checksum(root/'archive-checksums.json')!=registered[key]:raise ValueError('pilot archive manifest differs')
        archived=json.loads((root/'archive-checksums.json').read_text())
        files={n:checksum(root/n) for n in ['plan.json','ledger.json',account_file,'archive-checksums.json']}
        if any(digest!=archived[n] for n,digest in files.items() if n!='archive-checksums.json'):
            raise ValueError('pilot plan, ledger or accounting differs from the sealed archive')
        expected=18 if name=='stage1' else 138
        if sum(len(e['runs']) for e in json.loads((root/'ledger.json').read_text()))!=expected:
            raise ValueError('pilot attempt count differs')
        result.append(dict(root=str(root.resolve()),files=files))
    return result


def prepare(root,repo,operations):
    root,repo,operations=Path(root).resolve(),Path(repo).resolve(),Path(operations).resolve()
    if root.exists() or root.parent!=Path.home().resolve()/'tu_runs/per-goal-reward-ablation' or root.name!='grid':
        raise ValueError('new study-owned grid root required')
    registered=registered_plan();runs=registered_runs(registered)
    if (execute(['/usr/bin/git','-C',str(repo),'rev-parse','HEAD']).strip()!=MEASUREMENT_COMMIT
            or execute(['/usr/bin/git','-C',str(repo),'status','--porcelain']).strip()):
        raise ValueError('measurement clone must remain clean at the approved commit')
    binary=repo/'target/x86_64-unknown-linux-musl/release/acs2-measure'
    if checksum(binary)!=BINARY_SHA256:raise ValueError('approved measurement binary differs')
    manifest=operations/'manifest.json';ops=json.loads(manifest.read_text())
    plan=dict(result_root=str(root),registered_plan=str(root/'registered-plan.json'),authorization=authorization(),
        measurement_commit=MEASUREMENT_COMMIT,binary_sha256=BINARY_SHA256,measurement_repository=str(repo),binary=str(binary),
        operations_commit=ops['commit'],operations_manifest=str(manifest),operations_manifest_sha256=checksum(manifest),
        preset=registered['preset'],configurations=registered['configurations'],pilot_allocation_cpu_seconds=PILOT_CPU_SECONDS,
        phase_cpu_hour_limit=LIMIT_HOURS,partition='lem-cpu-normal',concurrency=20,memory_mib=2048,
        cpu_model='AMD EPYC 9554 64-Core Processor',accounting_since='2026-11-09',
        pilots=frozen_pilots(root.parent,registered),stop_path=str(root/'STOP.json'))
    verify_operations(plan)
    bound=require_budget(PILOT_CPU_SECONDS,dict(spent_seconds=0,reserved_seconds=0),sum(r['limit_seconds'] for r in runs),LIMIT_HOURS)
    root.mkdir(parents=True);(root/'logs').mkdir()
    (root/'registered-plan.json').write_bytes((HERE/'plan.json').read_bytes())
    atomic_json(root/'plan.json',plan);(root/'plan.sha256').write_text(checksum(root/'plan.json')+'\n');atomic_json(root/'ledger.json',[])
    return dict(runs=len(runs),hard_bound_cpu_hours=bound/3600,operations_commit=ops['commit'],approved_plan_sha256=APPROVED_SHA256)


def require_authorization(plan):
    registered=load_plan(plan['registered_plan'],APPROVED_SHA256)
    if (plan['authorization']!=authorization() or plan['measurement_commit']!=MEASUREMENT_COMMIT
            or plan['binary_sha256']!=BINARY_SHA256 or plan['phase_cpu_hour_limit']!=LIMIT_HOURS
            or plan['pilot_allocation_cpu_seconds']!=PILOT_CPU_SECONDS or plan['memory_mib']!=2048
            or plan['partition']!='lem-cpu-normal' or plan['concurrency']!=20
            or plan['configurations']!=registered['configurations'] or plan['preset']!=registered['preset']):
        raise ValueError('grid scope or measurement identity differs from approval')
    for pilot in plan['pilots']:
        for name,digest in pilot['files'].items():
            if checksum(Path(pilot['root'])/name)!=digest:raise ValueError('sealed pilot evidence changed')
    verify_operations(plan)
    if json.loads(Path(plan['operations_manifest']).read_text())['commit']!=plan['operations_commit']:
        raise ValueError('operations commit differs from its frozen manifest')
    return registered


def balance(root,plan,ledger,executor=execute):
    prior=[e for p in plan['pilots'] for e in json.loads((Path(p['root'])/'ledger.json').read_text())]
    arrays=[e['array_id'] for e in prior+ledger]
    if None in arrays or len(arrays)!=len(set(arrays)):raise ValueError('duplicate allocation or unresolved submission intent')
    raw=executor(['sacct','--array','-S',plan['accounting_since'],'-j',','.join(map(str,arrays)),'-P','--format='+SACCT_FIELDS])
    records=list({r['JobIDRaw']:r for r in allocation_rows(raw)}.values())
    old=accounting(prior,records);new=accounting(ledger,records)
    if old!=dict(spent_seconds=PILOT_CPU_SECONDS,reserved_seconds=0):raise ValueError('all 156 pilot attempts must be terminal and charged exactly once')
    (Path(root)/'accounting-all.psv').write_text(raw)
    return dict(spent_seconds=old['spent_seconds']+new['spent_seconds'],reserved_seconds=new['reserved_seconds'],
        pilot_cpu_seconds=old['spent_seconds'],grid_cpu_seconds=new['spent_seconds']),records


def state_name(record):
    return record.get('State','PENDING').split()[0].split('+')[0]


def completion(root,run,registered):
    path=Path(root)/run['relative_directory']/'completion.json';value=json.loads(path.read_text())
    targets=next(c['targets'] for c in registered['configurations'] if c['id']==run['configuration_id'])
    if (value['exit_code'] or value['failure'] or value['incomplete_final_line']
            or value['rows_seen']!=len(targets) or value['rows_expected']!=len(targets)):
        raise ValueError('full-budget measurement did not complete every registered checkpoint')
    return checksum(path)


def scout_gate(root,registered,ledger,records):
    expected={r['attempt_id']:r for r in registered_runs(registered) if r['phase']=='scout'}
    known={r['JobID']:r for r in records};checks=[]
    for entry in ledger:
        for index,run in enumerate(entry['runs']):
            if run['attempt_id'] not in expected:continue
            if run!=expected[run['attempt_id']]:raise ValueError('scout identity or limit differs')
            account=known.get(f"{entry['array_id']}_{index}",{})
            done=state_name(account)=='COMPLETED'
            cost=float(account.get('CPUTimeRAW',0))
            checks.append(dict(attempt_id=run['attempt_id'],task=run['task'],agent=run['agent'],
                completed=done,allocation_cpu_seconds=cost,forecast_high_cpu_seconds=run['forecast_high_cpu_seconds'],
                within_high=done and cost<=run['forecast_high_cpu_seconds'],
                completion_sha256=completion(root,run,registered) if done else None))
    if len({c['attempt_id'] for c in checks})!=len(checks):raise ValueError('duplicate scout attempt')
    complete=len(checks)==12 and all(c['completed'] for c in checks)
    return dict(complete=complete,passed=complete and all(c['within_high'] for c in checks),
        failed_cost=any(c['completed'] and not c['within_high'] for c in checks),checks=checks)


def inspect(root,plan,registered,ledger,records):
    known={r['JobID']:r for r in records};counts={};failed=[];active=[]
    for entry in ledger:
        stream_active=False
        for index,run in enumerate(entry['runs']):
            account=known.get(f"{entry['array_id']}_{index}",{});state=state_name(account)
            counts[state]=counts.get(state,0)+1
            if state not in TERMINAL:stream_active=True
            elif state!='COMPLETED':failed.append(dict(job=f"{entry['array_id']}_{index}",attempt=run['attempt_id'],state=state))
            elif account.get('AllocCPUS')!='1' or account.get('ExitCode')!='0:0':
                failed.append(dict(attempt=run['attempt_id'],state='allocation identity differs'))
        if stream_active:
            active.append(entry['array_id'])
    if failed:stop_phase(Path(root)/'STOP.json',dict(terminal_failures=failed))
    gate=scout_gate(root,registered,ledger,records)
    atomic_json(Path(root)/'scout-cost-gate.json',gate)
    return dict(active_streams=len(active),active_arrays=active,allocations=counts,submitted=sum(len(e['runs']) for e in ledger),
        terminal_failures=failed,scout_gate=gate)


def next_wave(ledger, active_arrays):
    if not ledger or 'wave' not in ledger[-1]:
        if len(active_arrays) > 2:
            raise ValueError('cannot queue behind more than two independent arrays')
        return len(ledger), active_arrays

    last = ledger[-1]
    siblings = [entry for entry in ledger if entry.get('wave') == last['wave']]
    if len(siblings) > 2 or siblings != ledger[-len(siblings):]:
        raise ValueError('array wave in the ledger is not contiguous or exceeds two arrays')
    if len(siblings) == 1:
        return last['wave'], last['wave_predecessors']
    return len(ledger), [entry['array_id'] for entry in siblings]


def dispatch(root,executor=execute,*,queue_wave=False):
    root=Path(root);plan=load_plan(root/'plan.json',(root/'plan.sha256').read_text().strip());registered=require_authorization(plan)
    if root.resolve()!=Path(plan['result_root']).resolve():raise ValueError('grid destination differs')
    ledger=json.loads((root/'ledger.json').read_text());state,records=balance(root,plan,ledger,executor)
    status=inspect(root,plan,registered,ledger,records)
    if (root/'STOP.json').exists():return dict(submitted=False,stopped=True,status=status,**state)
    if status['scout_gate']['failed_cost']:return dict(submitted=False,blocked='scout allocation exceeded its approved high forecast',status=status,**state)
    if not queue_wave and status['active_streams']>=2:return dict(submitted=False,status=status,**state)
    done={r['attempt_id'] for e in ledger for r in e['runs']}
    remaining=[r for r in registered_runs(registered) if r['attempt_id'] not in done]
    if not remaining:return dict(submitted=False,all_submitted=True,status=status,**state)
    eligible=next(([r for r in remaining if r['phase']==phase] for phase in ('ordinary','scout','conditional')
        if any(r['phase']==phase for r in remaining)),[])
    if eligible[0]['phase']=='conditional' and not status['scout_gate']['passed']:
        return dict(submitted=False,waiting_for_scout_gate=True,status=status,**state)
    limit=max(r['limit_seconds'] for r in eligible);batch=[r for r in eligible if r['limit_seconds']==limit][:40]
    dependencies = []
    wave_metadata = {}
    if queue_wave:
        wave, predecessors = next_wave(ledger, status['active_arrays'])
        dependencies = [array_id for array_id in predecessors if array_id in status['active_arrays']]
        wave_metadata = dict(wave=wave, wave_predecessors=predecessors, depends_on=dependencies,
                             controller_sha256=checksum(__file__))
    new=sum(r['limit_seconds'] for r in batch);bound=require_budget(0,state,new,LIMIT_HOURS)
    grant=executor(['sshare','-U','-u','alelys2099','-o','RawUsage','-n'])
    queue=executor(['squeue','--array','-u','alelys2099','-h','-o','%i|%j|%t|%L|%C'])
    if float(grant.strip())+shared_reservation(queue)+new>15000*3600:raise ValueError('shared grant cannot cover this array')
    assignment=root/f'assignment-{len(ledger):03d}.json'
    atomic_json(assignment,dict(plan_sha256=checksum(root/'plan.json'),runs=batch,
        measurement_repository=plan['measurement_repository'],binary=plan['binary'],operations_commit=plan['operations_commit']))
    command=submission_command(root,assignment,limit//60,len(batch),20,2048,dependencies,job_name='tu-ablation-grid')
    if dependencies:
        command.insert(1, '--kill-on-invalid-dep=yes')
    operations=Path(plan['operations_manifest']).parent/'experiments/per-goal-reward-ablation'
    command[-4:]=[str(operations/'job_grid.sh'),str(operations),str(root),str(assignment)]
    try:
        tested = executor(command[:1] + ['--test-only'] + command[1:])
    except subprocess.CalledProcessError as error:
        raise ValueError(f'array test-only failed: {error.output.strip()}') from error
    require_authorization(plan)
    if (root/'STOP.json').exists():raise ValueError('study stopped before submission')
    for run in batch:(root/run['relative_directory']).parent.parent.mkdir(parents=True)
    entry=dict(array_id=None,runs=batch,command=command,test_only=tested,grant=grant,queue=queue,
        balance=state,bound_seconds=bound,submitted_unix=time.time(),**wave_metadata)
    ledger.append(entry);atomic_json(root/'ledger.json',ledger)
    response=executor(command).strip();entry.update(array_id=int(response.split(';')[0]),submission_response=response)
    atomic_json(root/'ledger.json',ledger)
    return dict(submitted=True,array_id=entry['array_id'],runs=len(batch),phase=batch[0]['phase'],
        limit_seconds=limit,bound_cpu_hours=bound/3600,**wave_metadata)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','dispatch','enqueue','status']);parser.add_argument('root')
    parser.add_argument('--repo');parser.add_argument('--operations');parser.add_argument('--max-arrays',type=int,default=256);args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.root,args.repo,args.operations)
    elif args.action=='dispatch':result=dispatch(args.root)
    elif args.action == 'enqueue':
        if args.max_arrays < 1:
            parser.error('--max-arrays must be positive')
        for index in range(args.max_arrays):
            result = dispatch(args.root, queue_wave=True)
            if not result['submitted'] or index + 1 == args.max_arrays:
                break
            print(json.dumps(result), flush=True)
    else:
        root=Path(args.root);plan=load_plan(root/'plan.json',(root/'plan.sha256').read_text().strip());registered=require_authorization(plan)
        ledger=json.loads((root/'ledger.json').read_text());state,records=balance(root,plan,ledger)
        result=dict(state=state,**inspect(root,plan,registered,ledger,records))
    print(json.dumps(result))
