import argparse
import csv
import json
from pathlib import Path
import statistics

from analyze import read_inputs, read_rows


def value_dose(row):
    routes=row['replay_diagnostics']['reward_updates']
    original,relabeled=routes['original'],routes['relabeled']
    ratio=lambda numerator,denominator:numerator/denominator if denominator else None
    return dict(relabeled_goal_action_set_share=ratio(relabeled['draws_with_goal'],relabeled['draws']),
        skipped_reward_opportunity_share=ratio(relabeled['skipped_without_goal'],original['action_set_rules']+relabeled['action_set_rules']),
        retained_blind_opportunity_share=ratio(original['without_goal'],original['without_goal']+relabeled['skipped_without_goal']))


def export(root,verification,directory):
    root,directory=Path(root),Path(directory)
    inputs,digest=read_inputs(root,verification)
    rows,finals=[],[]
    for (configuration,arm,seed),path in sorted(inputs['new'].items()):
        records=list(read_rows(path))
        for row in records:
            base=dict(post_hoc=True,configuration_id=configuration,arm=arm,seed=seed,nominal_step=row['nominal_step'])
            for origin,counts in row['replay_diagnostics']['reward_updates'].items():
                rows.append(dict(base,origin=origin,**counts))
        finals.append(dict(base,**value_dose(records[-1])))
    summary=[]
    for configuration in sorted({row['configuration_id'] for row in finals}):
        selected=[row for row in finals if row['configuration_id']==configuration]
        if {row['seed'] for row in selected}!=set(range(42,62)):
            raise ValueError('dose summary requires all twenty registered seeds')
        for metric in ('relabeled_goal_action_set_share','skipped_reward_opportunity_share','retained_blind_opportunity_share'):
            values=[row[metric] for row in selected]
            summary.append(dict(post_hoc=True,configuration_id=configuration,metric=metric,
                n=sum(value is not None for value in values),mean=None if any(value is None for value in values) else statistics.mean(values)))
    for name,values in [('post-hoc-reward-counts',rows),('post-hoc-final-dose',finals),('post-hoc-dose-summary',summary)]:
        with (directory/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return dict(post_hoc=True,plan_sha256=digest,counter_rows=len(rows),final_seed_rows=len(finals))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--verification',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.root,args.verification,args.out)))
