import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from protocol import ARM, REFERENCES, ARMS, BLOCKS, configurations
from archive import checked_path, models
from analysis import attach, pool_blocks
from decision_metrics import decision_metrics
from post_hoc import uniform_pairs
from results import contrasts, decisions
from collect import outside_repository
from grid_protocol import atomic_json


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def calculate(configs, inputs, truth):
    success, mechanisms, curves, supplementary = {}, {}, [], []
    for config in configs:
        for arm in ARMS:
            for seed in range(42,62):
                key = config['id'],arm,seed
                path = inputs['new' if arm == ARM else 'success'][key]
                scores, points = [], []
                for row in read_rows(path):
                    points.append(row['nominal_step'])
                    scores.append(row['evaluation_blocks']['test']['success'])
                if points != config['targets']:
                    raise ValueError('success analysis requires every registered checkpoint in order')
                success[config['task'],arm,seed] = dict(primary=statistics.mean(scores),final=scores[-1])
                if seed >= 47:
                    continue
                pair_path = path if arm == ARM else inputs['pairs'][key]
                pair_points = []
                for row in read_rows(pair_path):
                    point = row['nominal_step']
                    pair_points.append(point)
                    blocks = {name:attach(row['evaluation_blocks'][name]['starts'],truth[config['task']]) for name in BLOCKS}
                    pooled = pool_blocks(blocks)
                    measured = decision_metrics(pooled)
                    curves.append(dict(maze=config['task'],arm=arm,seed=seed,nominal_step=point,view='pooled',post_hoc=True,
                        F=measured['F'],F_mod=measured['F_mod']))
                    if point == config['targets'][-1]:
                        mechanisms[config['task'],arm,seed] = measured
                        for view,records in [('pooled',pooled),('uniform_pairs',uniform_pairs(pooled)),*blocks.items()]:
                            supplementary.append(dict(maze=config['task'],arm=arm,seed=seed,nominal_step=point,view=view,
                                post_hoc=True,**decision_metrics(records)))
                if pair_points != config['targets']:
                    raise ValueError('mechanism analysis requires every registered checkpoint in order')
    registered = contrasts(success,mechanisms,configs)
    return dict(registered=registered,decisions=decisions(registered,[config['task'] for config in configs]),
        curves=curves,supplementary=supplementary,
        success=[dict(maze=maze,arm=arm,seed=seed,**values) for (maze,arm,seed),values in success.items()])


def read_inputs(root,verification):
    plan_path = Path(__file__).with_name('plan.json')
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    registered = json.loads(plan_path.read_text())
    for name in ('reference-index','model-index'):
        if hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest() != registered['cost_pilot'][name+'_sha256']:
            raise ValueError('registered reference or model index changed')
    checked = json.loads(Path(verification).read_text())
    approval = json.loads((root/'approval.json').read_text())
    if ((root/'approved-plan.json').read_bytes() != plan_path.read_bytes() or approval.get('approved') is not True
            or approval.get('plan_sha256') != digest or not isinstance(approval.get('phase_cpu_hour_limit'),(int,float))
            or approval['phase_cpu_hour_limit'] <= 0 or checked.get('verified') is not True or checked.get('plan_sha256') != digest):
        raise ValueError('registered analysis requires exact approved plan and independently verified complete runs')
    result = dict(new={},success={},pairs={})
    for kind,entries,base in [('new',checked['runs_index'],root),
            ('references',json.loads((root/'reference-index.json').read_text()),root/'references')]:
        for entry in entries:
            group = 'new' if kind == 'new' else entry['kind']
            key = entry['configuration_id'],entry['agent'],entry['seed']
            if group not in result or key in result[group]:
                raise ValueError('duplicate or unknown analysis input')
            result[group][key] = checked_path(base,entry)
    configs = configurations()
    for group,arms,seeds in [('new',[ARM],range(42,62)),('success',REFERENCES,range(42,62)),('pairs',REFERENCES,range(42,47))]:
        if set(result[group]) != {(config['id'],arm,seed) for config in configs for arm in arms for seed in seeds}:
            raise ValueError('missing or extra registered analysis input')
    return result,digest


def write_csv(path,rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root,verification,output):
    root,output=Path(root),outside_repository(output)
    inputs,digest=read_inputs(root,verification)
    result=calculate(configurations(),inputs,models(root))
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('registered-comparisons',result['registered']),('post-hoc-curves',result['curves']),
            ('post-hoc-seed-metrics',result['supplementary']),('success-seed-metrics',result['success'])]:
        atomic_json(output/(name+'.json'),dict(plan_sha256=digest,rows=rows))
        write_csv(output/(name+'.csv'),rows)
    atomic_json(output/'registered-decisions.json',dict(plan_sha256=digest,**result['decisions']))
    return dict(registered_comparisons=len(result['registered']),**result['decisions'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--verification',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(args.root,args.verification,args.out)))
