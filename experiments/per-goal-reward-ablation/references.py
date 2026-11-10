import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from protocol import ROOT, A, E, S, E1, S1, OLD_TASKS
from agent_names import canonical_run_name


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_index(baseline, per_goal):
    baseline,per_goal=Path(baseline),Path(per_goal)
    verified=json.loads((baseline/'verification.json').read_text())
    if verified.get('approved_plan_bytes_unchanged') is not True or verified.get('grid_allocations_completed')!=240:
        raise ValueError('verified original 4b pair grid required')
    hashes={line.split(None,1)[1]:line.split()[0] for line in (baseline/'raw-files.sha256').read_text().splitlines()}
    records=[]
    for config in json.loads((baseline/'plan.json').read_text())['configurations']:
        for seed in range(42,62):
            paths=list(baseline.glob(f"attempts/{config['id']}_acs2_s{seed}_a*/measurement/*/rows.jsonl"))
            if len(paths)!=1:
                raise ValueError('original ACS2 archive is ambiguous')
            path=paths[0]
            if digest(path)!=hashes[str(path.relative_to(baseline))]:
                raise ValueError('original ACS2 archive checksum differs')
            records.append(dict(task=config['task'],agent=A,seed=seed,source=str(path),sha256=digest(path),kind='success',charged_to_this_study=False))
    v=json.loads((per_goal/'verification.json').read_text())
    if v.get('verified') is not True:
        raise ValueError('verified per-goal archive required')
    for entry in v['runs_index']:
        path=per_goal/entry['file']
        if digest(path)!=entry['sha256']:
            raise ValueError('per-goal archive changed')
        records.append(dict(task=entry.get('task',entry['configuration_id'].split('_c')[0]),agent=entry['agent'],seed=entry['seed'],source=str(path),sha256=entry['sha256'],kind='success',charged_to_this_study=False))
    for entry in json.loads((per_goal/'reference-index.json').read_text()):
        if entry['kind']!='success' or entry['agent'] not in (E,S):continue
        path=Path(entry['source'])
        if digest(path)!=entry['sha256']:
            raise ValueError('original E/S archive changed')
        task=entry['configuration_id'].split('_c')[0]
        records.append(dict(task=task,agent=entry['agent'],seed=entry['seed'],source=str(path),sha256=entry['sha256'],kind='success',charged_to_this_study=False))
    expected={(task,arm,seed) for task in OLD_TASKS for arm in (A,E,S,E1,S1) for seed in range(42,62)}
    if len(records)!=len(expected) or {(r['task'],r['agent'],r['seed']) for r in records}!=expected:
        raise ValueError('archive must cover all 600 registered reference runs')
    return sorted(records,key=lambda r:(r['task'],r['agent'],r['seed']))


def bits(value):
    if isinstance(value,float):return struct.pack('>d',value).hex()
    if isinstance(value,list):return [bits(v) for v in value]
    if isinstance(value,dict):return {k:bits(v) for k,v in value.items()}
    return value


def verify_short(index,binary,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    checked=[]
    excluded={'commit','source_state','host','cpu_model','wall_seconds_train','wall_seconds_eval','wall_seconds_total'}
    for task in OLD_TASKS:
        paths=[r for r in index if r['task']==task and r['seed']==42]
        original={r['agent']:[json.loads(line) for line in Path(r['source']).read_text().splitlines()][0] for r in paths}
        target=original[A]['nominal_step'];cap=original[A]['cap']
        path=output/(task+'.jsonl')
        subprocess.run([str(binary),'--task',task,'--cap',str(cap),'--protocol','maze-pairs-v1',
            '--agents',','.join([A,E,S,E1,S1]),'--seeds','42','--targets',str(target),'--out',str(path)],check=True,capture_output=True)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            expected=original[row['agent']]
            def learning(r):
                r={k:v for k,v in r.items() if k not in excluded}
                r['agent']=canonical_run_name(r['agent'])
                for block in r['evaluation_blocks'].values():block['starts']=None
                return bits(r)
            if learning(row)!=learning(expected):
                keys=[key for key in learning(row) if learning(row)[key]!=learning(expected).get(key)]
                raise ValueError(str((task,row['agent'],keys)))
            checked.append(dict(task=task,arm=row['agent'],seed=42,checkpoint=target,source_sha256=next(r['sha256'] for r in paths if r['agent']==row['agent']),learning_fields_equal=True))
    (output/'identity.json').write_text(json.dumps(checked,indent=2)+'\n')
    return checked


def check_archived_intervals(index, earlier_path):
    import statistics
    from analyze import interval,difference,check_reference_intervals
    values={}
    for entry in index:
        if entry['agent'] not in (E,S,E1,S1):continue
        rows=[json.loads(line) for line in Path(entry['source']).read_text().splitlines()]
        scores=[r['evaluation_blocks']['test']['success'] for r in rows]
        values[entry['task'],entry['agent'],entry['seed']]=dict(primary=statistics.mean(scores),final=scores[-1])
    current=[]
    for task in OLD_TASKS:
        for contrast in ('E+1 - E','S+1 - S','copies x values'):
            for measure in ('primary','final'):
                estimates=[difference(contrast,{arm:values[task,arm,seed][measure] for arm in (E,S,E1,S1)}) for seed in range(42,62)]
                current.append(dict(task=task,contrast=contrast,measure=measure,**interval(estimates)))
    earlier=json.loads(Path(earlier_path).read_text())
    if isinstance(earlier,dict):earlier=earlier['rows']
    return dict(checked=check_reference_intervals(current,earlier),intervals=current)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True);parser.add_argument('--per-goal',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--binary');parser.add_argument('--short-out')
    args=parser.parse_args()
    index=reference_index(args.baseline,args.per_goal)
    Path(args.out).write_text(json.dumps(index,indent=2)+'\n')
    if args.binary:print(json.dumps(verify_short(index,Path(args.binary),args.short_out)))
