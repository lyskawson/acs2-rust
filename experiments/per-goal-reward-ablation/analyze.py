import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct
from fractions import Fraction

from protocol import (HERE, A, A1, E, S, E1, S1, ARMS, SEEDS, T19, CONTRASTS, CEILING_ARMS)


def interval(values):
    if len(values) != 20 or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('twenty finite paired seeds required')
    mean=statistics.mean(values)
    se=statistics.stdev(values)/math.sqrt(20)
    return dict(n=20,mean=mean,se=se,ci_low=mean-T19*se,ci_high=mean+T19*se)


def difference(name, values):
    if name=='memory x values':
        return (values[E1]-values[E])-(values[A1]-values[A])
    if name=='copies x values':
        return (values[S1]-values[S])-(values[E1]-values[E])
    positive,negative=CONTRASTS[name]
    return values[positive]-values[negative]


def verdict(positive,negative,n,contrast):
    if n==0:
        return 'not assessed'
    interaction=contrast in ('memory x values','copies x values')
    more,less=({'memory x values':('values gain more with memory','values gain less with memory'),
                'copies x values':('copies gain more from values','copies gain less from values')}.get(contrast,('improves','worsens')))
    if positive>=math.ceil(n/2) and negative==0:
        return more
    if negative>=math.ceil(n/2) and positive==0:
        return less
    if positive and negative:
        return 'mixed'
    return 'no clear difference' if interaction else 'no clear effect'


def sustained(targets, means):
    for i,target in enumerate(targets):
        if all(mean>=0.95 for mean in means[i:]):
            return target
    return 'not reached'


def calculate(configs, inputs, names=None):
    names=names or json.loads((HERE/'maze-names.json').read_text())
    scores,curves,labels,comparisons={},[],[],[]
    for config in configs:
        task=config['task'];targets=config['targets']
        if 1000 not in targets or config['group'] not in ('aliasing','no_aliasing'):
            raise ValueError('registered first checkpoint and maze group required')
        summaries={}
        for arm in ARMS:
            vectors=[]
            for seed in SEEDS:
                rows=inputs[task,arm,seed]
                if ([r['nominal_step'] for r in rows]!=targets or any(r['task']!=task or r['agent']!=arm or r['seed']!=seed for r in rows)):
                    raise ValueError('row identities, seed pairing or checkpoints differ')
                values=[r['evaluation_blocks']['test']['success'] for r in rows]
                if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1+1e-12 for v in values):
                    raise ValueError('invalid test success')
                scores[task,arm,seed]=dict(primary=statistics.mean(values),final=values[-1])
                vectors.append(values)
            summaries[arm]=dict(final=interval([v[-1] for v in vectors]),
                first=statistics.mean(v[targets.index(1000)] for v in vectors))
            means=[statistics.mean(v[i] for v in vectors) for i in range(len(targets))]
            curves.append(dict(task=task,maze=names[task],group=config['group'],arm=arm,
                sustained_095=sustained(targets,means),targets=targets,means=means))
        reference=config['registration']['blocks']['test']
        floor=Fraction(max(summaries[a]['final']['ci_low'] for a in ARMS))<=Fraction(reference.get('random_floor_exact',str(reference['random_floor'])))
        easy=summaries[A]['first']>=0.9*reference['reachable_within_cap']
        labels.append(dict(task=task,maze=names[task],group=config['group'],floor=floor,too_easy=easy,
            random_floor=reference['random_floor'],ceiling=reference['reachable_within_cap']))
        for contrast in CONTRASTS:
            ceiling=all(summaries[a]['final']['mean']>=0.99 for a in CEILING_ARMS[contrast])
            for measure in ('primary','final'):
                values=[difference(contrast,{a:scores[task,a,seed][measure] for a in ARMS}) for seed in SEEDS]
                excluded=floor or (measure=='primary' and easy) or (measure=='final' and ceiling)
                comparisons.append(dict(task=task,maze=names[task],group=config['group'],contrast=contrast,
                    measure=measure,**interval(values),floor=floor,too_easy=easy,ceiling=ceiling,counted=not excluded))
    grouped=[]
    for contrast in CONTRASTS:
        for measure in ('primary','final'):
            for group in ('no_aliasing','aliasing','all'):
                rows=[r for r in comparisons if r['contrast']==contrast and r['measure']==measure
                      and r['counted'] and (group=='all' or r['group']==group)]
                positive=sum(r['ci_low']>0 for r in rows);negative=sum(r['ci_high']<0 for r in rows)
                result=dict(group=group,contrast=contrast,measure=measure,A_plus=positive,A_minus=negative,n=len(rows))
                if group!='all':
                    result['verdict']=verdict(positive,negative,len(rows),contrast)
                grouped.append(result)
    return dict(comparisons=comparisons,labels=labels,grouped=grouped,sustained=curves)


def check_reference_intervals(current, earlier):
    mapping={'E+1 - E':'E+1 - E','S+1 - S':'S+1 - S','copies x values':'interaction'}
    checked=0
    for row in current:
        if row['task'] not in ('maze4','maze5','maze6','maze7','woods14','mazema') or row['contrast'] not in mapping:
            continue
        matches=[r for r in earlier if r['maze']==row['task'] and r.get('comparison',r.get('contrast'))==mapping[row['contrast']] and r.get('metric',r.get('measure'))==row['measure']]
        if len(matches)!=1:
            raise ValueError('earlier interval missing or duplicated')
        for key in ('mean','se','ci_low','ci_high'):
            if struct.pack('>d',row[key])!=struct.pack('>d',matches[0][key]):
                raise ValueError('earlier interval differs: '+str((row['task'],row['contrast'],row['measure'],key)))
        checked+=1
    if checked!=36:
        raise ValueError('expected 36 unchanged earlier intervals')
    return checked


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    from protocol import configurations
    from hashlib import sha256
    entries=json.loads(Path(args.index).read_text())
    inputs={}
    for entry in entries:
        path=Path(entry['source'])
        if sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('analysis input checksum differs')
        key=entry['task'],entry['agent'],entry['seed']
        if key in inputs:
            raise ValueError('duplicate run input')
        inputs[key]=[json.loads(line) for line in path.read_text().splitlines()]
    result=calculate(configurations(),inputs)
    output=Path(args.out);output.mkdir(parents=True,exist_ok=False)
    for name in ('comparisons','labels','grouped'):
        rows=result[name]
        with (output/(name+'.csv')).open('w') as stream:
            keys=list(dict.fromkeys(key for row in rows for key in row))
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
