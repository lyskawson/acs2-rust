import argparse
import json
import math
from pathlib import Path
import statistics

from protocol import AGENTS, configurations


def summarize(rows):
    configs = {config['task']: config for config in configurations()}
    expected = {(task, agent, seed, point) for task in configs for agent in AGENTS for seed in range(42,47) for point in [1000,10000,20000]}
    indexed = {}
    for row in rows:
        key = row['task'],row['agent'],row['seed'],row['nominal_step']
        if key in indexed or key not in expected:
            raise ValueError('duplicate or unexpected local check row')
        if set(row['evaluation_blocks']) != {'validation'} or row['source_state'] != 'clean':
            raise ValueError('local check requires clean source and validation only')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('incomplete local check')
    def estimate(values):
        return dict(n=5,mean=statistics.mean(values),se=statistics.stdev(values)/math.sqrt(5))
    result=[]
    for task in configs:
        for agent in AGENTS:
            for point in [1000,10000,20000]:
                group=[indexed[task,agent,seed,point] for seed in range(42,47)]
                cell=dict(task=task,agent=agent,point=point,
                    success=estimate([row['evaluation_blocks']['validation']['success'] for row in group]),
                    knowledge=estimate([row['knowledge'] for row in group]),
                    population=estimate([row['population_classifiers'] for row in group]),
                    memory_bytes=estimate([row['population_known_bytes_lower_bound']+row['replay_logical_bytes']+row['trajectory_logical_bytes'] for row in group]),
                    train_wall_seconds=estimate([row['wall_seconds_train'] for row in group]),
                    full_updates=estimate([row['learning_update_counts']['full'] for row in group]),
                    values_only_updates=estimate([row['learning_update_counts']['values_only'] for row in group]))
                if agent != 'acs2er_u1':
                    cell['realized_relabeled_share']=estimate([row['replay_diagnostics']['realized_relabeled_share'] for row in group])
                    cell['objective_cost_per_draw']={purpose:estimate([row['replay_diagnostics']['objective_evaluations_by_purpose'][purpose]['objective_evaluations']/row['replay_diagnostics']['draws'] for row in group]) for purpose in ['scoring','selection','provenance']}
                    cell['fallback_share']=estimate([row['replay_diagnostics']['fallbacks']/row['replay_diagnostics']['draws'] for row in group])
                result.append(cell)
    return dict(runs=150,rows=450,maximum_steps=20000,seeds=list(range(42,47)),evaluation_block='validation',interpretation='mechanics check; no ranking or selection',cells=result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    rows=[json.loads(line) for path in sorted(args.root.glob('*.jsonl')) for line in path.read_text().splitlines()]
    result=summarize(rows)
    (args.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['runs','rows','maximum_steps']}))
