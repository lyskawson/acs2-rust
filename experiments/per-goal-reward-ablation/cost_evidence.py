import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from protocol import configurations,OLD_TASKS,A,E,S,E1,S1
from agent_names import canonical_run_name
from grid_attempts import attempts
from grid_protocol import allocation_rows


ROW_COST_FIELDS = ('actual_steps','population_classifiers','population_numerosity','population_logical_bytes','population_mark_entries')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def profile(path,metadata,allocation):
    path=Path(path);folder=path.parent
    completion=json.loads((folder/'completion.json').read_text())
    for name in ['rows.jsonl','resources.jsonl']:
        if checksum(folder/name)!=completion['sha256'][name]:raise ValueError('cost evidence checksum differs')
    resources=[json.loads(line) for line in (folder/'resources.jsonl').read_text().splitlines()]
    rows=[];per_pair=None
    with path.open() as stream:
        for line in stream:
            raw=json.loads(line)
            if (raw['task']!=metadata['task'] or canonical_run_name(raw['agent'])!=metadata['agent'] or raw['seed']!=metadata['seed']):
                raise ValueError('cost evidence arm, maze or seed differs')
            values={name:raw[name] for name in ROW_COST_FIELDS}
            values['step']=raw['nominal_step']
            values['per_goal_entries']=(raw.get('replay_diagnostics') or {}).get('per_goal_rewards')
            rows.append(values)
            if per_pair is None:
                per_pair=raw['evaluation_blocks']['test']['starts'] is not None
    if len(rows)!=len(resources) or [r['step'] for r in rows]!=[r['nominal_step'] for r in resources]:
        raise ValueError('cost evidence rows and resource checkpoints differ')
    train=evaluation=0.0
    for row,resource in zip(rows,resources):
        keys=('train_cpu_seconds_estimate','eval_cpu_seconds_estimate','process_cpu_seconds_at_observation','observed_elapsed_seconds','peak_rss_kib')
        if any(not finite(resource[k]) for k in keys):raise ValueError('invalid CPU or memory observation')
        train+=resource['train_cpu_seconds_estimate'];evaluation+=resource['eval_cpu_seconds_estimate']
        row.update(train_cpu_seconds_delta_estimate=resource['train_cpu_seconds_estimate'],
            eval_cpu_seconds_delta_estimate=resource['eval_cpu_seconds_estimate'],
            train_cpu_seconds_cumulative_estimate=train,eval_cpu_seconds_cumulative_estimate=evaluation,
            process_cpu_seconds_at_observation=resource['process_cpu_seconds_at_observation'],
            elapsed_seconds_at_observation=resource['observed_elapsed_seconds'],peak_rss_kib=resource['peak_rss_kib'],
            maximum_sample_gap_seconds=resource['maximum_sample_gap_seconds'],row_bytes=resource['row_bytes'])
    exact_cpu=completion['user_cpu_seconds']+completion['system_cpu_seconds']
    if not finite(exact_cpu) or exact_cpu<=0:raise ValueError('positive wait4 CPU cost required')
    return dict(**metadata,source=str(path),source_sha256=checksum(path),resources_sha256=checksum(folder/'resources.jsonl'),
        completion_sha256=checksum(folder/'completion.json'),per_pair=per_pair,checkpoints=rows,
        complete=completion['exit_code']==0 and not completion['failure'],failure=completion['failure'],
        process_cpu_seconds_wait4=exact_cpu,body_wall_seconds=completion['elapsed_seconds'],peak_rss_kib=completion['peak_rss_kib'],
        allocation_cpu_seconds=int(allocation['CPUTimeRAW']),job_id_raw=allocation['JobIDRaw'])


def accounts(root,filename='accounting.psv'):
    records=allocation_rows((Path(root)/filename).read_text())
    unique={r['JobIDRaw']:r for r in records}
    return {r['JobID']:r for r in unique.values()}


def archived(index):
    cache={};result=[]
    configs={c['task']:c for c in configurations()}
    for entry in index:
        path=Path(entry['source'])
        if checksum(path)!=entry['sha256']:raise ValueError('archived reference rows changed')
        root=next((p for p in path.parents if (p/'ledger.json').exists()),None)
        if root is None:raise ValueError('archived reference ledger missing')
        if root not in cache:
            ledger=json.loads((root/'ledger.json').read_text())
            cache[root]=({a['path'].resolve():a for a in attempts(root,ledger)},accounts(root))
        attempt_map,account_map=cache[root];attempt=attempt_map[path.parent.resolve()]
        account=account_map[f"{attempt['array_id']}_{attempt['task_id']}"]
        if account['State']!='COMPLETED' or account['AllocCPUS']!='1':raise ValueError('archived full allocation did not complete on one CPU')
        config=configs[entry['task']]
        value=profile(path,dict(task=entry['task'],agent=entry['agent'],seed=entry['seed'],
            configuration_id=config['id'],kind='archive_full',registered_budget=config['targets'][-1]),account)
        if not value['complete'] or [p['step'] for p in value['checkpoints']]!=config['targets']:
            raise ValueError('archived cost reference lacks the full registered horizon')
        result.append(value)
    expected={(task,arm,seed) for task in OLD_TASKS for arm in (A,E,S,E1,S1) for seed in range(42,62)}
    if len(result)!=600 or {(r['task'],r['agent'],r['seed']) for r in result}!=expected:
        raise ValueError('all 600 unique original full references required')
    return result


def batch(root,kind,accounting_file='accounting.psv'):
    root=Path(root);ledger=json.loads((root/'ledger.json').read_text());account_map=accounts(root,accounting_file)
    plan=json.loads((root/'plan.json').read_text());configs={c['id']:c for c in plan['configurations']}
    result=[]
    for entry in ledger:
        for i,run in enumerate(entry['runs']):
            config=configs[run['configuration_id']]
            result.append(profile(root/run['relative_directory']/'rows.jsonl',
                dict(task=config['task'],agent=run['agent'],seed=run['seed'],configuration_id=config['id'],kind=kind,
                     registered_budget=next(c['targets'][-1] for c in configurations() if c['task']==config['task'])),
                account_map[f"{entry['array_id']}_{i}"]))
    return result


def export_checkpoints(profiles,path):
    fields=['kind','task','agent','seed','step','actual_steps','train_cpu_seconds_delta_estimate','eval_cpu_seconds_delta_estimate',
        'train_cpu_seconds_cumulative_estimate','eval_cpu_seconds_cumulative_estimate','process_cpu_seconds_at_observation',
        'elapsed_seconds_at_observation','peak_rss_kib','population_classifiers','population_numerosity','population_logical_bytes',
        'per_goal_existing_pairs','per_goal_updated_entries','original_reward_updates','relabeled_reward_updates','maximum_sample_gap_seconds']
    with Path(path).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for profile in profiles:
            for point in profile['checkpoints']:
                row={k:profile[k] for k in ('kind','task','agent','seed')}
                row.update({k:point[k] for k in fields if k in point})
                entries=point['per_goal_entries'] or {}
                row.update(per_goal_existing_pairs=entries.get('existing_pairs'),per_goal_updated_entries=entries.get('updated_entries'),
                    original_reward_updates=entries.get('reward_updates',{}).get('original'),relabeled_reward_updates=entries.get('reward_updates',{}).get('relabeled'))
                writer.writerow(row)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',required=True);parser.add_argument('--prior',required=True)
    parser.add_argument('--pilot');parser.add_argument('--out',required=True);args=parser.parse_args()
    values=archived(json.loads(Path(args.index).read_text()))+batch(args.prior,'stopped_long')
    if args.pilot:values+=batch(args.pilot,'pilot_30000','accounting-all.psv')
    output=Path(args.out);output.write_text(json.dumps(values,indent=2)+'\n');export_checkpoints(values,output.with_suffix('.csv'))
    print(json.dumps(dict(profiles=len(values),cost_checkpoints=sum(len(p['checkpoints']) for p in values),success_summary=False)))
